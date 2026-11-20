"""Score structured answers against the manifest without contacting an API."""

import json
from itertools import combinations

import chess

STANDARD = "standard"
TASK_METRICS = {
    "mate": ("classification_correct", "mate_move_correct"),
    "history": ("classification_correct", "state_exact", "first_index_correct"),
}


def message(body):
    try:
        return body["choices"][0]["message"]["content"]
    except (TypeError, KeyError, IndexError):
        return None


def parse_object(body) -> dict:
    content = message(body)
    if not isinstance(content, str):
        return {}
    try:
        parsed = json.loads(content)
    except ValueError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def scored_item(item, model, name, rule, received, body) -> dict:
    row = {
        "model": model,
        "item_id": item["id"],
        "task": item["task"],
        "group": item["group"],
        "name": name,
        "rule": rule,
        "received": received,
    }
    answer = parse_object(body)
    if item["task"] == "mate":
        gold = item["gold"][rule]
        predicted = answer.get("mate_exists")
        if type(predicted) is not bool:
            predicted = None
        move = answer.get("move")
        try:
            move_valid = type(move) is str and chess.Move.from_uci(move).uci() == move and len(move) == 4
        except ValueError:
            move_valid = False
        consistent = ((predicted is True and move_valid)
                      or (predicted is False and "move" in answer and move is None))
        standard_only = set(item["gold"][STANDARD]) - set(gold)
        row.update(
            gold_class=bool(gold), prediction=predicted, gold_moves=json.dumps(gold),
            returned_move=move if type(move) is str else None,
            classification_correct=predicted == bool(gold) if received else None,
            format_valid=consistent if received else None,
            mate_move_correct=(predicted is True and move_valid and move in gold)
            if gold and received else None,
            mate_eligible=bool(gold),
            common_mate=all(item["gold"].values()),
            standard_only_eligible=bool(standard_only),
            standard_only_error=move_valid and move in standard_only if received else None,
        )
        return row

    expected_index = item["gold"]["first_illegal"][rule]
    expected_state = item["gold"]["occupants"]
    occupants = answer.get("occupants")
    index = answer.get("first_illegal")
    index_valid = ("first_illegal" in answer
                   and (index is None or type(index) is int and 1 <= index <= len(item["audit_line"])))
    prediction = index is not None if index_valid else None
    state_valid = (isinstance(occupants, dict) and set(occupants) == set(expected_state)
                   and all(isinstance(value, str) and len(value) == 1
                           and value in ".KQRBNPkqrbnp" for value in occupants.values()))
    squares = occupants if isinstance(occupants, dict) else {}
    square_hits = sum(squares.get(key) == value for key, value in expected_state.items())
    same_as_standard = index == item["gold"]["first_illegal"][STANDARD]
    differs_from_standard = expected_index != item["gold"]["first_illegal"][STANDARD]
    row.update(
        gold_class=expected_index is not None, prediction=prediction,
        gold_index=expected_index, returned_index=index if index_valid else None,
        gold_occupants=json.dumps(expected_state, sort_keys=True),
        returned_occupants=json.dumps(occupants, sort_keys=True),
        classification_correct=prediction == (expected_index is not None) if received else None,
        format_valid=index_valid and state_valid if received else None,
        state_exact=square_hits == len(expected_state) if received else None,
        square_hits=square_hits if received else None,
        square_total=len(expected_state),
        first_index_correct=index_valid and index == expected_index if received else None,
        standard_only_eligible=differs_from_standard,
        standard_only_error=index_valid and same_as_standard if received else None,
    )
    return row


def ratio(numerator: int, denominator: int):
    return numerator / denominator if denominator else None


def summarize(rows: list[dict]) -> dict:
    responses = [row for row in rows if row["received"]]
    valid = [row for row in responses if row["prediction"] is not None]
    tp = sum(row["prediction"] and row["gold_class"] for row in valid)
    fp = sum(row["prediction"] and not row["gold_class"] for row in valid)
    fn = sum(not row["prediction"] and row["gold_class"] for row in valid)
    tn = sum(not row["prediction"] and not row["gold_class"] for row in valid)
    recall = ratio(tp, tp + fn)
    total_positive = sum(row["gold_class"] for row in responses)
    total_negative = len(responses) - total_positive
    sensitivity_with_errors = ratio(tp, total_positive)
    specificity_with_errors = ratio(tn, total_negative)
    standard_only = [row for row in responses if row["standard_only_eligible"]]
    result = {
        "planned": len(rows), "responses": len(responses),
        "missing": len(rows) - len(responses),
        "format_errors": sum(not row["format_valid"] for row in responses),
        "scorable_classifications": len(valid),
        "actual_positive": total_positive, "actual_negative": total_negative,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "accuracy": ratio(sum(row["classification_correct"] for row in responses), len(responses)),
        "precision": ratio(tp, tp + fp), "recall": recall, "f1": ratio(2 * tp, 2 * tp + fp + fn),
        "balanced_accuracy": (sensitivity_with_errors + specificity_with_errors) / 2
        if sensitivity_with_errors is not None and specificity_with_errors is not None else None,
        "standard_only_n": len(standard_only),
        "standard_only_rate": ratio(sum(row["standard_only_error"] for row in standard_only), len(standard_only)),
    }
    if rows[0]["task"] == "mate":
        eligible = [row for row in responses if row["mate_eligible"]]
        common = [row for row in eligible if row["common_mate"]]
        result.update(
            mate_move_n=len(eligible),
            mate_move_accuracy=ratio(sum(row["mate_move_correct"] for row in eligible), len(eligible)),
            common_mate_n=len(common),
            common_mate_accuracy=ratio(sum(row["mate_move_correct"] for row in common), len(common)),
        )
    else:
        illegal = [row for row in responses if row["gold_class"]]
        result.update(
            state_exact_accuracy=ratio(sum(row["state_exact"] for row in responses), len(responses)),
            square_accuracy=ratio(sum(row["square_hits"] for row in responses),
                                  sum(row["square_total"] for row in responses)),
            first_index_accuracy=ratio(sum(row["first_index_correct"] for row in responses), len(responses)),
            first_index_illegal_n=len(illegal),
            first_index_illegal_accuracy=ratio(sum(row["first_index_correct"] for row in illegal), len(illegal)),
        )
    return result


def paired(rows: list[dict], model: str, task: str, scope: str, conditions: list[tuple]):
    by_condition = {(name, rule): {} for name, rule in conditions}
    for row in rows:
        by_condition[row["name"], row["rule"]][row["item_id"]] = row
    for left, right in combinations(conditions, 2):
        left_rows, right_rows = by_condition[left], by_condition[right]
        shared = sorted(left_rows.keys() & right_rows.keys())
        for metric in TASK_METRICS[task]:
            candidates = [(left_rows[item], right_rows[item]) for item in shared]
            if metric == "mate_move_correct":
                candidates = [(a, b) for a, b in candidates if a["mate_eligible"] and b["mate_eligible"]]
            both = [(a[metric], b[metric]) for a, b in candidates if a["received"] and b["received"]]
            left_only = sum(a and not b for a, b in both)
            right_only = sum(b and not a for a, b in both)
            yield {
                "model": model, "task": task, "scope": scope, "metric": metric,
                "left_name": left[0], "left_rule": left[1],
                "right_name": right[0], "right_rule": right[1],
                "planned_pair_n": len(candidates), "observed_pair_n": len(both),
                "both_correct": sum(a and b for a, b in both),
                "left_only_correct": left_only, "right_only_correct": right_only,
                "neither_correct": sum(not a and not b for a, b in both),
                "left_minus_right": ratio(left_only - right_only, len(both)),
            }


def name_effects(rows: list[dict], model: str, task: str, scope: str, rules: list[str]):
    indexed = {(row["item_id"], row["name"], row["rule"]): row for row in rows}
    item_ids = sorted({row["item_id"] for row in rows})
    for variant in (rule for rule in rules if rule != STANDARD):
        for metric in TASK_METRICS[task]:
            eligible = []
            for item_id in item_ids:
                quartet = [indexed[item_id, name, rule]
                           for rule in (STANDARD, variant) for name in ("Chess", "Okuz")]
                if metric != "mate_move_correct" or all(row["mate_eligible"] for row in quartet):
                    eligible.append(quartet)
            complete = [[row[metric] for row in quartet] for quartet in eligible
                        if all(row["received"] for row in quartet)]
            n = len(complete)
            standard_gap = sum(chess - okuz for chess, okuz, _, _ in complete)
            variant_gap = sum(chess - okuz for _, _, chess, okuz in complete)
            yield {
                "model": model, "task": task, "scope": scope, "metric": metric, "variant": variant,
                "planned_n": len(eligible), "matched_n": n,
                "standard_chess_minus_okuz_pp": ratio(100 * standard_gap, n),
                "variant_chess_minus_okuz_pp": ratio(100 * variant_gap, n),
                "difference_of_gaps_pp": ratio(100 * (variant_gap - standard_gap), n),
            }
