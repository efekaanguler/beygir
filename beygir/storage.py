"""Durable run metadata and an append-only log of API attempts."""

import json
import re
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

import fcntl


def encode(data) -> str:
    return json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def output_directory(database: Path, run_id: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", run_id):
        raise ValueError("Run ID must contain only letters, digits, underscores or hyphens")
    return database.parent / run_id


@contextmanager
def exclusive_runner(database: Path):
    """Prevent two processes from issuing the same requests or sharing stale quotas."""
    database.parent.mkdir(parents=True, exist_ok=True)
    with database.with_suffix(database.suffix + ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError(f"An experiment is already writing to {database}") from None
        yield


class RunStore:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, timeout=30)
        self.db.row_factory = sqlite3.Row
        self.db.executescript("""
            PRAGMA journal_mode = WAL;
            PRAGMA synchronous = FULL;
            PRAGMA foreign_keys = ON;
            CREATE TABLE IF NOT EXISTS runs (
                id TEXT PRIMARY KEY,
                spec TEXT NOT NULL,
                created REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS items (
                run_id TEXT NOT NULL REFERENCES runs(id),
                item_id TEXT NOT NULL,
                item TEXT NOT NULL,
                PRIMARY KEY (run_id, item_id)
            );
            CREATE TABLE IF NOT EXISTS attempts (
                id INTEGER PRIMARY KEY,
                run_id TEXT NOT NULL,
                item_id TEXT NOT NULL,
                name TEXT NOT NULL,
                rule TEXT NOT NULL,
                model TEXT NOT NULL,
                route TEXT NOT NULL,
                quota_group TEXT NOT NULL,
                started REAL NOT NULL,
                duration REAL NOT NULL,
                status TEXT NOT NULL,
                http_status INTEGER,
                charged_tokens INTEGER NOT NULL,
                request TEXT NOT NULL,
                response TEXT,
                FOREIGN KEY (run_id, item_id) REFERENCES items(run_id, item_id)
            );
            CREATE UNIQUE INDEX IF NOT EXISTS completed_task
                ON attempts(run_id, item_id, name, rule, model)
                WHERE status = 'response';
            CREATE INDEX IF NOT EXISTS quota_window ON attempts(model, quota_group, started);
        """)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.db.close()

    def start(self, run_id: str, spec: dict, items: list[dict]) -> None:
        saved = self.db.execute("SELECT spec FROM runs WHERE id = ?", (run_id,)).fetchone()
        if saved:
            previous = json.loads(saved["spec"])
            if previous == spec:
                return
            old_models = {model["model_id"]: model for model in previous.pop("endpoints")}
            new_models = {model["model_id"]: model for model in spec["endpoints"]}
            same_protocol = previous == {key: value for key, value in spec.items() if key != "endpoints"}
            if not same_protocol or not all(new_models.get(key) == model for key, model in old_models.items()):
                raise ValueError("Run ID already belongs to different data, prompts or endpoints")
            with self.db:
                self.db.execute("UPDATE runs SET spec = ? WHERE id = ?", (encode(spec), run_id))
            return
        with self.db:
            self.db.execute("INSERT INTO runs VALUES (?, ?, ?)", (run_id, encode(spec), time.time()))
            self.db.executemany("INSERT INTO items VALUES (?, ?, ?)",
                                ((run_id, item["id"], encode(item)) for item in items))

    def load(self, run_id: str) -> tuple[dict, list[dict]]:
        run = self.db.execute("SELECT spec FROM runs WHERE id = ?", (run_id,)).fetchone()
        if not run:
            raise ValueError(f"Unknown run: {run_id}")
        items = self.db.execute("SELECT item FROM items WHERE run_id = ? ORDER BY item_id", (run_id,))
        return json.loads(run["spec"]), [json.loads(row["item"]) for row in items]

    def record(self, row: dict) -> None:
        with self.db:
            self.db.execute("""
                INSERT INTO attempts(
                    run_id, item_id, name, rule, model, route, quota_group, started, duration,
                    status, http_status, charged_tokens, request, response
                ) VALUES (
                    :run_id, :item_id, :name, :rule, :model, :route, :quota_group, :started, :duration,
                    :status, :http_status, :charged_tokens, :request, :response
                )
            """, {**row, "request": encode(row["request"]), "response": encode(row["response"])})

    def completed(self, run_id: str) -> dict[tuple, object]:
        rows = self.db.execute("""
            SELECT item_id, name, rule, model, response FROM attempts
            WHERE run_id = ? AND status = 'response'
        """, (run_id,))
        return {(row["model"], row["item_id"], row["name"], row["rule"]): json.loads(row["response"])
                for row in rows}

    def attempts(self, run_id: str):
        rows = self.db.execute("SELECT * FROM attempts WHERE run_id = ? ORDER BY id", (run_id,))
        for row in rows:
            yield {**dict(row), "request": json.loads(row["request"]),
                   "response": json.loads(row["response"])}

    def recent_usage(self, model: str, quota_group: str, since: float,
                     route_labels: list[str] | None = None) -> list[tuple[float, int]]:
        labels = route_labels or []
        placeholders = ",".join("?" for _ in labels) or "NULL"
        return [(row["started"], row["charged_tokens"]) for row in self.db.execute(f"""
            SELECT started, charged_tokens FROM attempts
            WHERE model = ? AND started > ? AND (quota_group = ? OR route IN ({placeholders}))
            ORDER BY started
        """, (model, since, quota_group, *labels))]

    def progress(self, run_id: str) -> list[dict]:
        run = self.db.execute("SELECT spec FROM runs WHERE id = ?", (run_id,)).fetchone()
        if run is None:
            raise ValueError(f"Unknown run: {run_id}")
        spec = json.loads(run["spec"])
        size = self.db.execute("SELECT COUNT(*) FROM items WHERE run_id = ?", (run_id,)).fetchone()[0]
        planned = size * len(spec["names"]) * len(spec["rules"])
        rows = self.db.execute("""
            SELECT model, COUNT(*) AS attempts,
                SUM(CASE WHEN status = 'response' THEN 1 ELSE 0 END) AS saved,
                AVG(CASE WHEN status = 'response' THEN charged_tokens END) AS avg_tokens,
                SUM(CASE WHEN status = 'response' AND started >= ? THEN 1 ELSE 0 END) AS last_hour
            FROM attempts WHERE run_id = ? GROUP BY model
        """, (time.time() - 3600, run_id))
        totals = {row["model"]: dict(row) for row in rows}
        return [{"model": endpoint["model_id"], "planned": planned,
                 **totals.get(endpoint["model_id"], {"attempts": 0, "saved": 0,
                                                    "avg_tokens": None, "last_hour": 0})}
                for endpoint in spec["endpoints"]]

    def task_progress(self, run_id: str) -> list[dict]:
        run = self.db.execute("SELECT spec FROM runs WHERE id = ?", (run_id,)).fetchone()
        if run is None:
            raise ValueError(f"Unknown run: {run_id}")
        spec = json.loads(run["spec"])
        conditions = len(spec["names"]) * len(spec["rules"]) * len(spec["endpoints"])
        items = self.db.execute("""
            SELECT json_extract(item, '$.task') AS task, COUNT(*) AS n
            FROM items WHERE run_id = ? GROUP BY task ORDER BY task
        """, (run_id,))
        replies = self.db.execute("""
            SELECT json_extract(items.item, '$.task') AS task, COUNT(*) AS saved
            FROM attempts JOIN items ON items.run_id = attempts.run_id AND items.item_id = attempts.item_id
            WHERE attempts.run_id = ? AND attempts.status = 'response' GROUP BY task
        """, (run_id,))
        counts = {row["task"]: row["saved"] for row in replies}
        return [{"task": row["task"], "items": row["n"], "planned": row["n"] * conditions,
                 "saved": counts.get(row["task"], 0)} for row in items]
