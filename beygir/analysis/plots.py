"""Vector figures corresponding to the four result sections of the report."""

from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .results import MODELS, NAMES, RULES, TASKS, VARIANTS

COLORS = ("#0072B2", "#E69F00", "#009E73", "#CC79A7",
          "#56B4E9", "#D55E00", "#6D5DB3", "#595959")
PLOT_STYLE = {"font.size": 9, "pdf.fonttype": 42, "figure.dpi": 160}


def style_axis(axis, label):
    axis.spines[['top', 'right']].set_visible(False)
    axis.set_ylabel(label)
    axis.set_axisbelow(True)
    axis.grid(axis='y', color='#dddddd', linewidth=0.5)


def bars(axis, categories, values, *, show_mean=False):
    series = [(m, label, color) for (m, label), color in zip(MODELS.items(), COLORS)]
    if show_mean:
        series.append(('mean', 'All-model mean', '#191919'))
    width = 0.82 / len(series)
    for index, (model, label, color) in enumerate(series):
        offsets = [x + (index - (len(series) - 1) / 2) * width for x in range(len(categories))]
        heights = [values[model, c] for c in categories] if model != 'mean' else [
            mean(values[m, c] for m in MODELS) for c in categories]
        rectangles = axis.bar(offsets, heights, width=width * 0.92, color=color, label=label)
        if model == 'mean':
            axis.bar_label(rectangles, labels=[f'{height:.1f}%' for height in heights],
                           padding=3, fontsize=8, fontweight='bold')
    axis.set_xticks(range(len(categories)), list(categories))


def save(figure, path):
    handles, labels = figure.axes[0].get_legend_handles_labels()
    if handles:
        figure.legend(handles, labels, ncol=4, loc='lower center', frameon=False, fontsize=8,
                      bbox_to_anchor=(0.5, 0.015))
    figure.savefig(path)
    plt.close(figure)

@plt.rc_context(PLOT_STYLE)
def plot_names(records, path):
    fig, axes = plt.subplots(3, 1, figsize=(6.9, 4.55), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.14, right=0.99, top=0.96, bottom=0.19, hspace=0.35)
    for rule, axis in zip(RULES, axes):
        values = {(r["model"], TASKS[r["task"]]): r["gap_pp"] for r in records if r["rule"] == rule}
        bars(axis, list(TASKS.values()), values)
        axis.set_title(RULES[rule].replace('--','–'), loc='left', fontsize=9, pad=3)
        style_axis(axis, 'Chess − Okuz (pp)')
        axis.axhline(0, color='#3a3a3a', linewidth=0.65)
        axis.set_ylim(-21,21)
        axis.set_yticks((-20,0,20))
    axes[-1].set_xlabel('Task')
    save(fig, path)


@plt.rc_context(PLOT_STYLE)
def plot_rules(records, path):
    pooled = {r["rule"]: r for r in records if r["model"] == "all"}
    fig, (fixed, changed) = plt.subplots(1, 2, figsize=(7.0, 2.85), sharey=True)
    fig.subplots_adjust(left=0.085, right=0.98, top=0.84, bottom=0.31, wspace=0.16)
    rule_names = ('Standard', 'Mandatory\ncapture', 'Bishop–knight\nswap')
    fixed.set_title('A. Same legal sequences', loc='left', fontsize=10, fontweight='bold', pad=10)
    rects = fixed.bar(range(3), [100 * pooled[rule]["control_correct"] / pooled[rule]["control_n"] for rule in RULES],
                      width=0.58, color=['#555c65', '#0072B2', '#D55E00'])
    fixed.bar_label(rects, fmt='%.1f%%', padding=3, fontsize=9, fontweight='bold')
    fixed.set_xticks(range(3), rule_names)
    style_axis(fixed, 'Correct (%)')

    changed.set_title('B. Sequences that change legality', loc='left', fontsize=10, fontweight='bold', pad=10)
    for offset, key, label, color in (
        (-0.17, "changed_both_correct", 'Both decisions correct', '#0072B2'),
        (0.17, "changed_both_and_index_correct", 'Both + exact index', '#c5dfea'),
    ):
        pairs = [pooled[rule] for rule in VARIANTS]
        rects = changed.bar([x + offset for x in range(2)], [100 * c[key] / c["changed_n"] for c in pairs],
                            width=0.31, label=label, color=color)
        changed.bar_label(rects, fmt='%.1f%%', padding=3, fontsize=8)
    changed.set_xticks(range(2), rule_names[1:])
    changed.legend(loc='upper center', bbox_to_anchor=(0.5, -0.27), frameon=False, fontsize=8)
    style_axis(changed, '')
    fixed.set_ylim(0, 32)
    fixed.set_yticks((0, 10, 20, 30))
    save(fig, path)


@plt.rc_context(PLOT_STYLE)
def plot_length(records, path):
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 4.65), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.10, right=0.98, top=0.93, bottom=0.23, hspace=0.40)
    lengths = ('5', '20', '50')
    for axis, metric, label in zip(axes, ("index_correct", "balanced_accuracy"),
                                   ("First illegal move found", "Legality balanced accuracy")):
        values = {(r["model"], str(r["plies"])): 100 * r[metric] / (r["illegal_n"] if metric == "index_correct" else 1)
                  for r in records}
        bars(axis, lengths, values, show_mean=True)
        axis.set_title(label, fontsize=10, loc='left')
        style_axis(axis, 'Accuracy (%)')
        axis.set_ylim(0, 100)
        axis.set_yticks((0, 25, 50, 75, 100))
    axes[1].axhline(50, color='#666666', linestyle='--', linewidth=0.75)
    axes[1].set_xlabel('Sequence length (plies)')
    save(fig, path)


@plt.rc_context(PLOT_STYLE)
def plot_generation(records, path):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.65), sharey=True)
    fig.subplots_adjust(left=0.09, right=0.99, top=0.89, bottom=0.30, wspace=0.10)
    rule_labels = dict(zip(RULES, ('Standard', 'Mandatory\ncapture', 'Bishop–knight\nswap')))
    for name, axis in zip(NAMES, axes):
        values = {(r["model"], rule_labels[r["rule"]]): 100 * r["all_legal"] / r["responses"]
                  for r in records if r["name"] == name}
        bars(axis, list(rule_labels.values()), values)
        axis.set_title(name, fontsize=10)
        style_axis(axis, 'All five moves legal (%)' if name == 'Chess' else '')
        axis.set_ylim(0, 82)
        axis.set_yticks((0, 20, 40, 60, 80))
    save(fig, path)
