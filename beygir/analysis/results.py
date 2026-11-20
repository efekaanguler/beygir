"""Matched summaries for the final paper's four figures."""

from collections import Counter, defaultdict
from itertools import product
from statistics import mean

from ..extended_dataset import AUDIT_LENGTHS
from ..prompts import NAMES
from ..rules import Rule
from ..scoring import summarize

MODELS = {
    "openai.gpt-oss-120b-1:0": "GPT-OSS 120B",
    "openai.gpt-oss-20b-1:0": "GPT-OSS 20B",
    "qwen.qwen3-32b-v1:0": "Qwen3 32B",
    "zai.glm-4.7": "GLM 4.7",
    "zai.glm-4.7-flash": "GLM 4.7 Flash",
    "deepseek.v3.2": "DeepSeek V3.2",
    "mistral.devstral-2-123b": "Devstral 2 123B",
    "eu.amazon.nova-lite-v1:0": "Nova Lite v1",
}
RULES = {Rule.STANDARD.value: "Standard", Rule.CAPTURE.value: "Mandatory capture",
         Rule.SWAP.value: "Bishop–knight swap"}
VARIANTS = (Rule.CAPTURE.value, Rule.SWAP.value)
TASKS = {"mate": "Mate", "history": "State", "play5": "Generate 5",
         **{task: f"Audit {n}" for task, n in AUDIT_LENGTHS.items()}}
EXPECTED_ITEMS = {"mate": 60, "history": 60, "play5": 50, **dict.fromkeys(AUDIT_LENGTHS, 50)}


def primary_metric(task: str) -> str:
    return {"history": "state_exact", "play5": "line_legal"}.get(task, "classification_correct")


class PaperResults:
    def __init__(self, spec: dict, items: list[dict], rows: list[dict]):
        if (spec["split"] != "main" or Counter(item["task"] for item in items) != EXPECTED_ITEMS
                or {e["model_id"] for e in spec["endpoints"]} != set(MODELS)
                or set(spec["names"]) != set(NAMES) or set(spec["rules"]) != set(RULES)):
            raise ValueError("Paper figures require the final eight-model, six-condition main protocol.")
        if len(rows) != 15_360 or not all(row["received"] for row in rows):
            raise ValueError("Paper figures require the complete main run (15,360 saved responses).")
        self.cells = defaultdict(list)
        self.pairs = defaultdict(dict)
        for row in rows:
            task, rule, name, model = (row[key] for key in ("task", "rule", "name", "model"))
            self.cells[task, rule, name, model].append(row)
            self.pairs[task, name, model, row["item_id"]][rule] = row

    def name_comparisons(self) -> list[dict]:
        records = []
        for task, rule, model in product(TASKS, RULES, MODELS):
            metric = primary_metric(task)
            chess = {r["item_id"]: r[metric] for r in self.cells[task, rule, NAMES[0], model]}
            okuz = {r["item_id"]: r[metric] for r in self.cells[task, rule, NAMES[1], model]}
            pairs = [(chess[item], okuz[item]) for item in chess]
            records.append({
                "task": task, "rule": rule, "model": model, "metric": metric, "pairs": len(pairs),
                "chess_correct": sum(chess.values()), "okuz_correct": sum(okuz.values()),
                "chess_only": sum(a and not b for a, b in pairs),
                "okuz_only": sum(b and not a for a, b in pairs),
                "gap_pp": 100 * (sum(chess.values()) - sum(okuz.values())) / len(pairs),
            })
        return records

    def rule_comparisons(self) -> list[dict]:
        records = []
        for rule, model in product(RULES, (*MODELS, "all")):
            counts = Counter()
            for (task, _, item_model, _), answers in self.pairs.items():
                if task not in AUDIT_LENGTHS or (model != "all" and model != item_model):
                    continue
                standard, variant = answers[Rule.STANDARD.value], answers[rule]
                left, right = standard["classification_correct"], variant["classification_correct"]
                if standard["group"] == "shared_positive":
                    counts["control_n"] += 1
                    counts["control_correct"] += right
                    counts["control_correct_to_wrong"] += left and not right
                    counts["control_wrong_to_correct"] += right and not left
                if variant["standard_only_eligible"]:
                    counts["changed_n"] += 1
                    counts["changed_standard_correct"] += left
                    counts["changed_variant_correct"] += right
                    counts["changed_both_correct"] += left and right
                    counts["changed_both_and_index_correct"] += left and right and variant["first_index_correct"]
            records.append({"rule": rule, "model": model, **{
                key: counts[key] for key in ("control_n", "control_correct",
                    "control_correct_to_wrong", "control_wrong_to_correct", "changed_n",
                    "changed_standard_correct", "changed_variant_correct", "changed_both_correct",
                    "changed_both_and_index_correct")
            }})
        return records

    def audit_comparisons(self) -> list[dict]:
        records = []
        for model, (task, length) in product(MODELS, AUDIT_LENGTHS.items()):
            cells = [self.cells[task, rule, name, model] for rule, name in product(RULES, NAMES)]
            illegal = [row for cell in cells for row in cell if row["gold_class"]]
            records.append({"model": model, "plies": length,
                            "balanced_accuracy": mean(summarize(cell)["balanced_accuracy"] for cell in cells),
                            "illegal_n": len(illegal),
                            "index_correct": sum(row["first_index_correct"] for row in illegal)})
        return records

    def generation_comparisons(self) -> list[dict]:
        records = []
        for model, rule, name in product(MODELS, RULES, NAMES):
            cell = self.cells["play5", rule, name, model]
            records.append({"model": model, "rule": rule, "name": name, "responses": len(cell),
                            "format_ok": sum(row["format_valid"] for row in cell),
                            "first_legal": sum(row["legal_prefix"] >= 1 for row in cell),
                            "all_legal": sum(row["line_legal"] for row in cell)})
        return records
