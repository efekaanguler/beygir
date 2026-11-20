"""Run BEYGIR and regenerate reports from saved responses."""

import argparse
import asyncio
import sys
import time
from pathlib import Path

from httpx import HTTPError

from botocore.exceptions import BotoCoreError, ClientError
from dotenv import load_dotenv

from .batch import (TERMINAL, batch_directory, check_batch, collect_batches, import_batch_file,
                    load_batch_plan, prepare_batches, submit_batches)
from .clients import client_for
from .dataset import DEFAULT_ITEMS, SEED
from .endpoints import DEFAULT_MODELS, catalog
from .report import write_report
from .runner import execute
from .storage import RunStore, exclusive_runner, output_directory

DEFAULT_DB = Path("runs/beygir.sqlite3")


def positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def show_progress(store: RunStore, run_id: str, refresh: int | None) -> None:
    while True:
        with store.db:
            store.db.execute("BEGIN")
            models = store.progress(run_id)
            tasks = store.task_progress(run_id)
        saved = sum(row["saved"] for row in models)
        planned = sum(row["planned"] for row in models)
        phase = "complete" if saved == planned else "ready or waiting" if not saved else "collecting or paused"
        if refresh and sys.stdout.isatty():
            print("\033[2J\033[H", end="")
        print(f"{run_id}: {saved:,}/{planned:,} saved ({saved / planned:.1%}) | {phase}")
        print("\nModels")
        for row in models:
            average = "—" if row["avg_tokens"] is None else f"{row['avg_tokens']:.0f} tokens/reply"
            print(f"{row['model']:<31} {row['saved']:>5}/{row['planned']:<5}  "
                  f"{row['last_hour']:>4} last hour  {row['attempts'] - row['saved']:>3} retries/errors  {average}")
        print("\nTasks")
        for row in tasks:
            print(f"{row['task']:<15} {row['saved']:>5}/{row['planned']:<5}"
                  f" ({row['saved'] / row['planned']:>6.1%}) from {row['items']} items")
        if not refresh or saved == planned:
            return
        try:
            sys.stdout.flush()
            time.sleep(refresh)
        except KeyboardInterrupt:
            return


def main() -> None:
    load_dotenv(".env")
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="start or resume a benchmark run")
    report = commands.add_parser("report", help="rebuild CSV and JSONL exports without API calls")
    status = commands.add_parser("status", help="show live progress from the local database")
    status.add_argument("--watch", nargs="?", type=positive_int, const=10, metavar="SECONDS",
                        help="refresh the status until completion or Ctrl-C")
    prepare = commands.add_parser("batch-prepare", help="write offline Bedrock JSONL and a record manifest")
    submit = commands.add_parser("batch-submit", help="upload and submit a prepared Bedrock batch")
    collect = commands.add_parser("batch-collect", help="import completed Bedrock batch outputs")
    check = commands.add_parser("batch-check", help="check prepared inputs and AWS batch prerequisites")
    import_file = commands.add_parser("batch-import", help="import a Bedrock output downloaded from S3")
    import_file.add_argument("--model", required=True, help="aws-* alias in the prepared batch")
    import_file.add_argument("--output", required=True, type=Path, help="downloaded .jsonl.out file")
    for command in (run, report, status, prepare, submit, collect, check, import_file):
        command.add_argument("run_id")
        command.add_argument("--db", type=Path, default=DEFAULT_DB)
    models = catalog()
    for command in (run, prepare):
        command.add_argument("--dataset", type=Path, default=DEFAULT_ITEMS)
        command.add_argument("--split", choices=("pilot", "main"), default="pilot" if command is run else "main")
        command.add_argument("--models", nargs="+", choices=sorted(models), default=list(DEFAULT_MODELS) if command is run else None,
                             required=command is prepare)
        command.add_argument("--seed", type=int, default=SEED)
        command.add_argument("--per-group", type=positive_int, help="pick this many items per task and sample group")
    run.add_argument("--workers", type=positive_int, default=3, help="concurrent requests per model")
    run.add_argument("--serial", action="store_true", help="allow only one in-flight model request across all models")
    run.add_argument("--keep-waiting", action="store_true",
                     help="wait through quota windows and write reports every 15 minutes")
    run.add_argument("--max-new-per-model", type=positive_int, help="limit pending jobs scheduled per model")
    args = parser.parse_args()

    try:
        output_directory(args.db, args.run_id)
        if args.command == "report":
            with RunStore(args.db) as store:
                write_report(store, args.db, args.run_id)
            return
        if args.command == "status":
            with RunStore(args.db) as store:
                show_progress(store, args.run_id, args.watch)
            return
        if args.command == "batch-check":
            check_batch(batch_directory(args.db, args.run_id))
            return
        if args.command in ("batch-submit", "batch-collect", "batch-import"):
            directory = batch_directory(args.db, args.run_id)
            with exclusive_runner(args.db), RunStore(args.db) as store:
                if args.command == "batch-submit":
                    submit_batches(directory, args.run_id)
                    return
                if args.command == "batch-import":
                    count = import_batch_file(store, directory, args.run_id, args.model, args.output)
                    print(f"{args.model}: imported {count} new records")
                else:
                    collect_batches(store, directory, args.run_id)
                write_report(store, args.db, args.run_id)
            return
        if len(set(args.models)) != len(args.models):
            parser.error("--models must be unique")
        endpoints = [models[name] for name in args.models]
        if args.command == "batch-prepare":
            with exclusive_runner(args.db), RunStore(args.db) as store:
                prepare_batches(store, batch_directory(args.db, args.run_id), args.run_id, args.dataset, args.split,
                                endpoints, args.seed, args.per_group)
            return
        for endpoint in endpoints:
            client_for(endpoint).routes(endpoint)
        with exclusive_runner(args.db), RunStore(args.db) as store:
            directory = batch_directory(args.db, args.run_id)
            if (directory / "plan.json").exists():
                jobs = load_batch_plan(directory)["jobs"]
                if any(job.get("remote") and job["remote"].get("status") not in TERMINAL
                       for job in jobs.values()):
                    raise ValueError("A Bedrock batch is still active for this run. Collect it first or use a new run ID.")
            try:
                asyncio.run(execute(store, args.run_id, args.dataset, args.split,
                                    endpoints, args.seed,
                                    args.workers, args.max_new_per_model,
                                    follow=args.keep_waiting, per_group=args.per_group, serial=args.serial,
                                    snapshot=lambda: write_report(store, args.db, args.run_id)))
            except KeyboardInterrupt:
                print("\nInterrupted. Completed responses are saved and can be resumed.")
            write_report(store, args.db, args.run_id)
    except (ValueError, OSError, HTTPError, BotoCoreError, ClientError) as error:
        parser.exit(2, f"beygir: {error}\n")


if __name__ == "__main__":
    main()
