"""Export the paper's plots and table values from scored responses."""

import json
from collections import Counter
from itertools import product
from pathlib import Path

from ..report import write_csv
from .plots import plot_generation, plot_length, plot_names, plot_rules
from .results import MODELS, NAMES, RULES, TASKS, PaperResults


def write_tex_rows(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8") as target:
        for row in rows:
            target.write(" & ".join(row) + r" \\" + "\n")
        target.write(r"\bottomrule" + "\n")


def name_table(records):
    totals = {}
    for record in records:
        cell = totals.setdefault((record["task"], record["rule"]), Counter())
        cell.update({key: record[key] for key in ("pairs", "chess_correct", "okuz_correct")})
    for task, label in TASKS.items():
        cells = [label]
        for rule in RULES:
            n, chess, okuz = (totals[task, rule][k] for k in ("pairs", "chess_correct", "okuz_correct"))
            cells.extend((f"{100 * chess / n:.1f}", f"{100 * okuz / n:.1f}", f"{100 * (chess - okuz) / n:+.2f}"))
        yield (*cells, str(n))


def audit_table(records):
    by_model_length = {(r["model"], r["plies"]): r for r in records}
    for model, label in MODELS.items():
        cells = [label]
        for length in (5, 20, 50):
            row = by_model_length[model, length]
            cells.extend((f"{100 * row['balanced_accuracy']:.1f}",
                          f"{100 * row['index_correct'] / row['illegal_n']:.1f}"))
        yield cells


def generation_table(records):
    totals = {}
    for row in records:
        for model in (row["model"], "all"):
            cell = totals.setdefault((model, row["rule"], row["name"]), Counter())
            cell.update({key: row[key] for key in ("all_legal", "responses")})
    for model in (*MODELS, "all"):
        cells = [MODELS.get(model, "All models")]
        for rule, name in product(RULES, NAMES):
            cell = totals[model, rule, name]
            cells.append(f"{cell['all_legal']}/{cell['responses']}")
        yield cells


def write_paper_results(output: Path, results: PaperResults) -> None:
    output.mkdir(parents=True, exist_ok=True)
    names = results.name_comparisons()
    rules = results.rule_comparisons()
    audit = results.audit_comparisons()
    generation = results.generation_comparisons()
    for stem, records, plot, figure in (
        ("name_comparisons", names, plot_names, "name_effect.pdf"),
        ("rule_comparisons", rules, plot_rules, "rule_comparison.pdf"),
        ("audit_length", audit, plot_length, "audit_length.pdf"),
        ("generation_comparisons", generation, plot_generation, "play5_comparison.pdf"),
    ):
        write_csv(output / f"{stem}.csv", records)
        plot(records, output / figure)
    write_tex_rows(output / "name_rows.tex", name_table(names))
    write_tex_rows(output / "audit_rows.tex", audit_table(audit))
    write_tex_rows(output / "play_rows.tex", generation_table(generation))
    n = sum(row["pairs"] for row in names)
    chess = sum(row["chess_only"] for row in names)
    okuz = sum(row["okuz_only"] for row in names)
    summary = {"name_pairs": n, "chess_only": chess, "okuz_only": okuz,
               "name_gap_pp": 100 * (chess - okuz) / n,
               "all_legal_generation": sum(row["all_legal"] for row in generation),
               "generation_responses": sum(row["responses"] for row in generation)}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
