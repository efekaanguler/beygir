"""Freeze the original tasks and the four extended legality tasks."""

import argparse
import hashlib
import json
import random
from pathlib import Path

import chess

from .dataset import DEFAULT_ITEMS, GROUPS, SEED, audit_candidate, position_key, synthetic_board
from .rules import Position, Rule, UnsupportedPosition, playable_positions

EXTENDED_ITEMS = Path("benchmark/extended_items.jsonl")
AUDIT_LENGTHS = {"audit5": 5, "audit20": 20, "audit50": 50}
NEW_TASKS = ("play5", *AUDIT_LENGTHS)
AUDIT_GROUP_COUNTS = dict(zip(GROUPS, (20, 10, 10, 10)))


def shared_trace(positions: dict[Rule, Position], length: int, rng: random.Random) -> list[str] | None:
    boards = [position.copy() for position in positions.values()]
    moves = []
    for _ in range(length):
        common = set.intersection(*(set(board.legal_moves()) for board in boards))
        if not common:
            return None
        move = rng.choice(sorted(common, key=chess.Move.uci))
        moves.append(move.uci())
        for board in boards:
            board.board.push(move)
    return moves


def values_match_group(gold: dict[str, int | None], group: str) -> bool:
    standard, capture, swap = (gold[rule.value] for rule in Rule)
    if group == "shared_positive":
        return standard is None and capture is None and swap is None
    if group == "shared_negative":
        return standard is not None and standard == capture == swap
    if group == "capture_sensitive":
        return standard is None and capture is not None and swap is None
    return standard is None and swap is not None and capture is None


def create_item(board: chess.Board, task: str, group: str, rng: random.Random) -> dict | None:
    positions = playable_positions(board, exclude_special_moves=True)
    length = AUDIT_LENGTHS.get(task, 5)
    clean = shared_trace(positions, length, rng)
    if clean is None:
        return None
    item = {"task": task, "group": group, "fen": board.fen(),
            "source": {"type": "synthetic"}, "protocol": "extended", "length": length}

    if task == "play5":
        return {**item, "gold": {"example_legal_line": clean}}

    # Choose a shared legal line, then introduce one counterexample for the requested stratum.
    audit = clean if group == "shared_positive" else audit_candidate(positions, clean, group, length, rng)
    if not audit:
        return None
    gold = {rule.value: position.replay(audit)[0] for rule, position in positions.items()}
    if not values_match_group(gold, group):
        return None
    return {**item, "audit_line": audit, "gold": {"first_illegal": gold}}


def load_original(source: Path, seed: int) -> list[dict]:
    groups = {}
    output = []
    for line in source.read_text(encoding="utf-8").splitlines():
        row = {**json.loads(line), "protocol": "extended"}
        if row["split"] == "pilot":
            output.append(row)
        else:
            groups.setdefault((row["task"], row["group"]), []).append(row)
    rng = random.Random(seed)
    for _, rows in sorted(groups.items()):
        output.extend(rng.sample(sorted(rows, key=lambda row: row["id"]), 15))
    return output


def build_extended(source: Path = DEFAULT_ITEMS, seed: int = SEED) -> list[dict]:
    output = load_original(source, seed)
    used = {position_key(chess.Board(row["fen"])) for row in output}
    rng = random.Random(seed + 1)
    for task in NEW_TASKS:
        target_groups = {"generated": 50} if task == "play5" else AUDIT_GROUP_COUNTS
        for group, size in target_groups.items():
            accepted = 0
            for _ in range(12_000):
                board = synthetic_board(rng)
                key = position_key(board)
                if key in used:
                    continue
                try:
                    row = create_item(board, task, group, rng)
                except UnsupportedPosition:
                    continue
                if row is None:
                    continue
                row["split"] = "pilot" if accepted == 0 else "main"
                row["seed"] = seed
                row["id"] = hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()[:20]
                output.append(row)
                used.add(key)
                accepted += 1
                if accepted == size + 1:
                    break
            if accepted != size + 1:
                raise RuntimeError(f"Could not produce {task}/{group} with seed {seed}")
    return sorted(output, key=lambda row: (row["split"], row["task"], row["group"], row["id"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_ITEMS)
    parser.add_argument("--output", type=Path, default=EXTENDED_ITEMS)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    rows = build_extended(args.source, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as result:
        for row in rows:
            result.write(json.dumps(row, sort_keys=True) + "\n")
    print(f"Saved {sum(row['split'] == 'main' for row in rows)} main items and "
          f"{sum(row['split'] == 'pilot' for row in rows)} separate pilot items to {args.output}")


if __name__ == "__main__":
    main()
