"""Export cached scores, per-condition metrics and all matched contrasts."""

import csv
import itertools
from collections import defaultdict
from pathlib import Path

from .figures import write_figures
from .scoring import name_effects, paired, scored_item, summarize
from .storage import RunStore, encode, exclusive_runner, output_directory


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def write_report(store: RunStore, database: Path, run_id: str) -> Path:
    directory = output_directory(database, run_id)
    with exclusive_runner(directory / "report"):
        return export_report(store, directory, run_id)


def export_report(store: RunStore, directory: Path, run_id: str) -> Path:
    store.db.execute("BEGIN")
    with store.db:
        spec, items = store.load(run_id)
        completed = store.completed(run_id)
        attempts = list(store.attempts(run_id))
    scores = []
    conditions = list(itertools.product(spec["names"], spec["rules"]))
    condition_groups = defaultdict(list)
    matched_groups = defaultdict(list)
    for endpoint in spec["endpoints"]:
        model = endpoint["model_id"]
        for item, (name, rule) in itertools.product(items, conditions):
            key = model, item["id"], name, rule
            row = scored_item(item, model, name, rule, key in completed, completed.get(key))
            scores.append(row)
            for scope in ("all", item["group"]):
                condition_groups[model, item["task"], name, rule, scope].append(row)
                matched_groups[model, item["task"], scope].append(row)
            if row.get("common_mate"):
                matched_groups[model, item["task"], "common_mate"].append(row)
    metrics = [{"model": model, "task": task, "name": name, "rule": rule, "scope": scope, **summarize(rows)}
               for (model, task, name, rule, scope), rows in sorted(condition_groups.items())]
    pairs = [record for (model, task, scope), rows in sorted(matched_groups.items())
             for record in paired(rows, model, task, scope, conditions)]
    effects = [record for (model, task, scope), rows in sorted(matched_groups.items())
               for record in name_effects(rows, model, task, scope, spec["rules"])]
    write_csv(directory / "item_scores.csv", scores)
    write_csv(directory / "metrics.csv", metrics)
    write_csv(directory / "matched_pairs.csv", pairs)
    write_csv(directory / "name_effects.csv", effects)
    temp = directory / "raw_attempts.jsonl.tmp"
    with temp.open("w", encoding="utf-8") as handle:
        for attempt in attempts:
            handle.write(encode(attempt) + "\n")
    temp.replace(directory / "raw_attempts.jsonl")
    (directory / "run_spec.json").write_text(encode(spec) + "\n", encoding="utf-8")
    write_figures(directory, metrics, pairs, spec)
    print(f"{run_id}: {len(completed):,}/{len(scores):,} saved ({len(completed) / len(scores):.1%}); "
          f"{len(scores) - len(completed):,} remaining; reports in {directory}", flush=True)
    return directory
