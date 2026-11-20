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

    def recent_usage(self, model: str, quota_group: str, since: float) -> list[tuple[float, int]]:
        return [(row["started"], row["charged_tokens"]) for row in self.db.execute("""
            SELECT started, charged_tokens FROM attempts
            WHERE model = ? AND quota_group = ? AND started > ?
            ORDER BY started
        """, (model, quota_group, since))]
