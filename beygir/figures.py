"""Generate publication-ready summary figures from derived records."""

from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

LABELS = {"standard": "Standard", "mandatory_capture": "Mandatory capture",
          "bishop_knight_swap": "Bishop–knight swap"}
COLORS = {"Chess": "#23539f", "Okuz": "#b65422"}


def new_figure(panels: int, models: list[str]):
    figure = Figure(figsize=(3.7 * len(models), 2.8 * panels + 0.8), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(panels, len(models), squeeze=False, sharey=True)
    return figure, axes


def style_axis(axis, rules: list[str], heading: str):
    axis.set_title(heading, fontsize=10)
    axis.set_xticks(range(len(rules)), [LABELS[rule] for rule in rules], rotation=17, ha="right")
    axis.set_ylim(0, 1.03)
    axis.yaxis.set_major_formatter(PercentFormatter(1))
    axis.grid(axis="y", alpha=0.22)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)


def legend(figure):
    handles = [Line2D([], [], marker="o", linestyle="none", color=color, label=name)
               for name, color in COLORS.items()]
    figure.legend(handles=handles, loc="outside upper center", ncols=2, frameon=False)


def write_figures(directory: Path, metrics: list[dict], pairs: list[dict], spec: dict) -> None:
    model_names = {endpoint["model_id"]: endpoint["alias"] for endpoint in spec["endpoints"]}
    rules = spec["rules"]
    panels = (
        ("mate", "mate_move_correct", "common_mate", "Correct mating move"),
        ("history", "classification_correct", "all", "Correct legality classification"),
    )
    lookup = {(row["model"], row["task"], row["metric"], row["scope"], row["left_rule"]): row
              for row in pairs
              if row["left_name"] == "Chess" and row["right_name"] == "Okuz"
              and row["left_rule"] == row["right_rule"]}
    if any(row["observed_pair_n"] for row in lookup.values()):
        figure, axes = new_figure(len(panels), list(model_names))
        for i, (task, metric, scope, label) in enumerate(panels):
            for column, (model, alias) in enumerate(model_names.items()):
                axis = axes[i][column]
                style_axis(axis, rules, alias if i == 0 else "")
                if column == 0:
                    axis.set_ylabel(label)
                for j, rule in enumerate(rules):
                    pair = lookup.get((model, task, metric, scope, rule))
                    if not pair or not pair["observed_pair_n"]:
                        continue
                    n = pair["observed_pair_n"]
                    left = (pair["both_correct"] + pair["left_only_correct"]) / n
                    right = (pair["both_correct"] + pair["right_only_correct"]) / n
                    axis.plot((j - 0.11, j + 0.11), (left, right), color="#9b9b9b", linewidth=1)
                    for x, rate, name in ((j - 0.11, left, "Chess"), (j + 0.11, right, "Okuz")):
                        axis.scatter(x, rate, color=COLORS[name], s=32, zorder=3)
        legend(figure)
        figure.savefig(directory / "matched_accuracy.png", dpi=170)

    variants = [rule for rule in rules if rule != "standard"]
    values = {(row["model"], row["task"], row["rule"], row["name"]): row["standard_only_rate"]
              for row in metrics if row["scope"] == "all" and row["rule"] in variants}
    if any(value is not None for value in values.values()):
        figure, axes = new_figure(2, list(model_names))
        for i, (task, label) in enumerate((("mate", "Standard-only mate move"), ("history", "Standard audit answer"))):
            for column, (model, alias) in enumerate(model_names.items()):
                axis = axes[i][column]
                style_axis(axis, variants, alias if i == 0 else "")
                if column == 0:
                    axis.set_ylabel(label)
                for j, rule in enumerate(variants):
                    for offset, name in ((-0.18, "Chess"), (0.18, "Okuz")):
                        value = values.get((model, task, rule, name))
                        if value is not None:
                            axis.bar(j + offset, value, width=0.34, color=COLORS[name])
                            axis.scatter(j + offset, value, s=14, color=COLORS[name], zorder=3)
        legend(figure)
        figure.savefig(directory / "standard_rule_errors.png", dpi=170)
