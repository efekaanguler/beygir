"""Run BEYGIR and regenerate reports from saved responses."""

import argparse
import asyncio
from pathlib import Path

from dotenv import load_dotenv

from .dataset import DEFAULT_ITEMS, SEED
from .endpoints import DEFAULT_MODELS, catalog, configured_routes
from .report import write_report
from .runner import execute
from .storage import RunStore, exclusive_runner, output_directory

DEFAULT_DB = Path("runs/beygir.sqlite3")


def positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def main() -> None:
    load_dotenv(".env")
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="start or resume a benchmark run")
    report = commands.add_parser("report", help="rebuild CSV and JSONL exports without API calls")
    for command in (run, report):
        command.add_argument("run_id")
        command.add_argument("--db", type=Path, default=DEFAULT_DB)
    run.add_argument("--dataset", type=Path, default=DEFAULT_ITEMS)
    run.add_argument("--split", choices=("pilot", "main"), default="pilot")
    models = catalog()
    run.add_argument("--models", nargs="+", choices=sorted(models), default=list(DEFAULT_MODELS))
    run.add_argument("--seed", type=int, default=SEED)
    run.add_argument("--workers", type=positive_int, default=3, help="concurrent requests per model")
    run.add_argument("--max-new-per-model", type=positive_int, help="limit pending jobs scheduled per model")
    args = parser.parse_args()

    try:
        output_directory(args.db, args.run_id)
        if args.command == "report":
            with RunStore(args.db) as store:
                write_report(store, args.db, args.run_id)
            return
        if len(set(args.models)) != len(args.models):
            parser.error("--models must be unique")
        endpoints = [models[name] for name in args.models]
        configured_routes()
        with exclusive_runner(args.db), RunStore(args.db) as store:
            try:
                asyncio.run(execute(store, args.run_id, args.dataset, args.split,
                                    endpoints, args.seed,
                                    args.workers, args.max_new_per_model))
            except KeyboardInterrupt:
                print("\nInterrupted. Completed responses are saved and can be resumed.")
            write_report(store, args.db, args.run_id)
    except ValueError as error:
        parser.exit(2, f"beygir: {error}\n")


if __name__ == "__main__":
    main()
