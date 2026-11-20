"""Check prompt parity and separate formatting failures from illegal moves."""

import itertools
from collections import Counter, defaultdict
from pathlib import Path

from ..report import write_csv
from ..scoring import message, parse_object, scored_item
from ..storage import RunStore
from .replay import UCI, independent_replay, tolerant_moves


def write_generation_audit(store: RunStore, run_id: str, output: Path) -> Path:
    with store.db:
        store.db.execute("BEGIN")
        spec, saved_items = store.load(run_id)
        attempts = [a for a in store.attempts(run_id, task="play5") if a["status"] == "response"]
    items = {item["id"]: item for item in saved_items if item["task"] == "play5"}
    models = [model["model_id"] for model in spec["endpoints"]]
    names, rules = spec["names"], spec["rules"]
    expected = set(itertools.product(items, names, rules))
    if not expected or len(attempts) != len(expected) * len(models):
        raise ValueError("The five-ply audit requires a complete generation task in the saved run.")
    for item, rule in itertools.product(items.values(), rules):
        line = item["gold"]["example_legal_line"]
        if len(line) != 5 or independent_replay(item["fen"], rule, line)[0] is not None:
            raise ValueError(f"Invalid test witness for {item['id']} under {rule}.")

    prompts = defaultdict(set)
    model_cases = defaultdict(set)
    openings = defaultdict(Counter)
    problems = defaultdict(Counter)
    totals = defaultdict(Counter)
    by_rule = defaultdict(Counter)
    for attempt in attempts:
        model, item_id, name, rule = (attempt[key] for key in ("model", "item_id", "name", "rule"))
        item = items[item_id]
        request, response = attempt["request"], attempt["response"]
        content = request["messages"][0]["content"]
        prompt = content[0]["text"] if isinstance(content, list) else content
        prompts[item_id, name, rule].add(prompt)
        model_cases[model].add((item_id, name, rule))

        proposed = parse_object(response).get("moves")
        strict = isinstance(proposed, list) and len(proposed) == 5 and all(
            isinstance(move, str) and UCI.fullmatch(move) for move in proposed)
        index, reason = independent_replay(item["fen"], rule, proposed[:5]) if isinstance(proposed, list) else (None, None)
        official = scored_item(item, model, name, rule, True, response)
        strict_success = strict and index is None
        if strict != official["format_valid"] or strict_success != official["line_legal"]:
            raise ValueError(f"Independent replay disagrees with official scoring for {model}, {item_id}, {name}, {rule}.")
        found = tolerant_moves(message(response), proposed)
        tolerant = len(found) == 5 and independent_replay(item["fen"], rule, found)[0] is None
        san = (isinstance(proposed, list) and len(proposed) == 5
               and independent_replay(item["fen"], rule, proposed, relaxed=True)[0] is None)
        stop_reason = (response.get("bedrock") or {}).get("stopReason") if isinstance(response, dict) else None

        stats = totals[model]
        stats["responses"] += 1
        stats["format_ok"] += strict
        stats["first_legal"] += official["legal_prefix"] >= 1
        stats["format_ok_first_illegal"] += strict and index == 1
        stats["strict_success"] += strict_success
        stats["tolerant_success"] += tolerant
        stats["san_or_tolerant_success"] += san or tolerant
        stats["max_tokens"] += stop_reason == "max_tokens"
        by_rule[model][rule] += strict_success
        if strict and index == 1:
            problems[model][reason] += 1
        if isinstance(proposed, list) and proposed and isinstance(proposed[0], str):
            openings[model][proposed[0]] += 1

    if set(model_cases) != set(models) or any(cases != expected for cases in model_cases.values()):
        raise ValueError("Models received different sets of five-ply cases.")
    if any(len(texts) != 1 for texts in prompts.values()) or set(prompts) != expected:
        raise ValueError("Models received different prompt text for a matched case.")
    for item_id, rule in itertools.product(items, rules):
        chess, okuz = (next(iter(prompts[item_id, name, rule])) for name in ("Chess", "Okuz"))
        if chess.replace("Game: Chess", "Game: Okuz", 1) != okuz:
            raise ValueError(f"Game-name prompts differ beyond the title for {item_id}, {rule}.")

    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "play5_quality_audit.csv", [
        {"model": model, **totals[model], **by_rule[model],
         **{f"illegal_first_{reason}": count for reason, count in sorted(problems[model].items())}}
        for model in models
    ])
    write_csv(output / "play5_opening_moves.csv", [
        {"model": model, "first_move": move, "count": count}
        for model in models for move, count in openings[model].most_common()
    ])
    print(f"Verified {len(attempts):,} responses, {len(expected)} shared prompts, and {len(items)} legal witnesses.")
    return output
