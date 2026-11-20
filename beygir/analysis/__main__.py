"""Reproduce the final paper's analyses offline from a saved run."""

import argparse
import sqlite3
from pathlib import Path

from ..report import score_responses
from ..storage import RunStore, output_directory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis", choices=("paper", "play5-check"))
    parser.add_argument("--db", type=Path, default=Path("runs/beygir.sqlite3"))
    parser.add_argument("--run-id", default="stockholm-extended-main")
    parser.add_argument("--output", type=Path, help="destination directory for tables and figures")
    args = parser.parse_args()
    try:
        default = output_directory(args.db, args.run_id) / "analysis"
        output = args.output if args.output is not None else default
        with RunStore(args.db, read_only=True) as store:
            if args.analysis == "paper":
                from .paper import write_paper_results
                from .results import PaperResults

                with store.db:
                    store.db.execute("BEGIN")
                    spec, items = store.load(args.run_id)
                    rows = list(score_responses(spec, items, store.completed(args.run_id)))
                write_paper_results(output, PaperResults(spec, items, rows))
            else:
                from .quality import write_generation_audit

                write_generation_audit(store, args.run_id, output)
        print(f"Wrote {args.analysis} outputs to {output}")
    except (OSError, ValueError, sqlite3.Error) as error:
        parser.exit(2, f"beygir analysis: {error}\n")


if __name__ == "__main__":
    main()
