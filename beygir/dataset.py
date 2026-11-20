"""Build a seeded benchmark and preserve every rule-specific answer."""

import argparse
import csv
import hashlib
import io
import json
import random
from collections import Counter
from pathlib import Path
from urllib.request import urlopen

import chess
import zstandard

from .rules import Position, Rule, UnsupportedPosition, playable_positions

SEED = 20261120
GROUPS = ("shared_positive", "shared_negative", "capture_sensitive", "swap_sensitive")
TASKS = ("mate", "history")
SOURCE_URL = "https://database.lichess.org/lichess_db_puzzle.csv.zst"
DEFAULT_SOURCE = Path("benchmark/lichess.csv")
DEFAULT_ITEMS = Path("benchmark/items.jsonl")


def fetch_sample(destination: Path, count: int = 1200) -> None:
    """Freeze a small CC0 input slice from the public Lichess export."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(".tmp")
    fields = ["PuzzleId", "FEN", "Moves", "Themes", "GameUrl"]
    with urlopen(SOURCE_URL, timeout=60) as response:
        with zstandard.ZstdDecompressor().stream_reader(response) as reader:
            with io.TextIOWrapper(io.BufferedReader(reader), encoding="utf-8") as stream:
                eligible = (row for row in csv.DictReader(stream)
                            if set(row["Themes"].split()) & {"mateIn1", "mateIn2"})
                with tmp.open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
                    writer.writeheader()
                    for _ in range(count):
                        writer.writerow(next(eligible))
    tmp.replace(destination)


def lichess_boards(source: Path):
    with source.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            board = chess.Board(row["FEN"])
            board.push_uci(row["Moves"].split()[0])
            yield board, {"type": "lichess", "puzzle_id": row["PuzzleId"], "game_url": row["GameUrl"]}


def synthetic_board(rng: random.Random) -> chess.Board:
    """Draw a sparse board, rejected later if invalid in any ruleset."""
    board = chess.Board(None)
    squares = rng.sample(list(chess.SQUARES), rng.randint(5, 9))
    board.set_piece_at(squares[0], chess.Piece(chess.KING, chess.WHITE))
    board.set_piece_at(squares[1], chess.Piece(chess.KING, chess.BLACK))
    pieces = (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.PAWN)
    board.turn = rng.choice((chess.WHITE, chess.BLACK))
    for index, square in enumerate(squares[2:]):
        kind = rng.choice(pieces)
        if kind == chess.PAWN and chess.square_rank(square) in (0, 1, 6, 7):
            kind = rng.choice(pieces[:-1])
        color = board.turn if index < 2 else rng.choice((chess.WHITE, chess.BLACK))
        board.set_piece_at(square, chess.Piece(kind, color))
    return board


def position_key(board: chess.Board) -> str:
    return " ".join(board.fen().split()[:4])


def mate_item(board: chess.Board, source: dict, rng: random.Random, needed: set[str]):
    positions = playable_positions(board)
    answers = {rule.value: position.mate_moves() for rule, position in positions.items()}
    standard, capture, swap = (set(answers[rule.value]) for rule in Rule)
    groups = []
    if standard and capture and swap:
        groups.append("shared_positive")
    if not (standard or capture or swap):
        groups.append("shared_negative")
    if standard - capture:
        groups.append("capture_sensitive")
    if standard - swap:
        groups.append("swap_sensitive")
    eligible = [group for group in groups if group in needed]
    if not eligible:
        return None
    # Prefer a rule-sensitive use for scarce mating positions.
    group = rng.choice([g for g in eligible if g.endswith("sensitive")] or eligible)
    return {"task": "mate", "group": group, "fen": board.fen(), "source": source, "gold": answers}


def common_line(positions: dict[Rule, Position], length: int, rng: random.Random):
    boards = {rule: position.copy() for rule, position in positions.items()}
    line = []
    changed = set()
    for _ in range(length):
        common = set.intersection(*(set(position.legal_moves()) for position in boards.values()))
        if not common:
            return None
        move = rng.choice(sorted(common, key=chess.Move.uci))
        line.append(move.uci())
        changed.update((move.from_square, move.to_square))
        for position in boards.values():
            position.board.push(move)
    final = boards[Rule.STANDARD]
    empties = sorted(square for square in changed if not final.board.piece_at(square))
    occupied = sorted(square for square in changed if final.board.piece_at(square))
    if not empties or not occupied:
        return None
    squares = [chess.square_name(rng.choice(empties)), chess.square_name(rng.choice(occupied))]
    rng.shuffle(squares)
    return line, squares, {square: final.piece_at(square) for square in squares}


def audit_candidate(positions: dict[Rule, Position], prefix: list[str], group: str,
                    length: int, rng: random.Random):
    if group == "shared_positive":
        independent = common_line(positions, length, rng)
        return independent[0] if independent else None

    target = {"capture_sensitive": Rule.CAPTURE, "swap_sensitive": Rule.SWAP}.get(group)
    indices = list(range(length))
    rng.shuffle(indices)
    for index in indices:
        start = prefix[:index]
        states = [position.replay(start)[1] for position in positions.values()]
        standard, capture, swap = (set(state.legal_moves()) for state in states)
        if target is None:
            board = states[0].board
            candidates = set(board.generate_pseudo_legal_moves()) - standard - capture - swap
            candidates = {move for move in candidates if not move.promotion}
            if not candidates:
                king = board.king(board.turn)
                candidates = {chess.Move(king, square) for square in chess.SQUARES
                              if chess.square_distance(king, square) == 2}
        else:
            candidates = (standard & swap) - capture if target is Rule.CAPTURE else (standard & capture) - swap
        if not candidates:
            continue
        wrong = rng.choice(sorted(candidates, key=chess.Move.uci))
        suffix = prefix[index + 1:]
        if target is not None:
            # Keep both the standard and the non-target variant legal.
            other = states[2] if target is Rule.CAPTURE else states[1]
            base = states[0]
            for position in (base, other):
                position.board.push(wrong)
            suffix = []
            for _ in range(length - index - 1):
                moves = set(base.legal_moves()) & set(other.legal_moves())
                if not moves:
                    break
                move = rng.choice(sorted(moves, key=chess.Move.uci))
                suffix.append(move.uci())
                for position in (base, other):
                    position.board.push(move)
            if len(suffix) != length - index - 1:
                continue
        return start + [wrong.uci()] + suffix
    return None


def history_item(board: chess.Board, group: str, rng: random.Random):
    positions = playable_positions(board)
    length = rng.randint(3, 5)
    shared = common_line(positions, length, rng)
    if not shared:
        return None
    state_line, squares, occupants = shared
    audit = audit_candidate(positions, state_line, group, length, rng)
    if not audit or audit == state_line:
        return None
    answers = {rule.value: position.replay(audit)[0] for rule, position in positions.items()}
    standard, capture, swap = (answers[rule.value] for rule in Rule)
    accepted = {
        "shared_positive": standard is None and capture is None and swap is None,
        "shared_negative": all(index is not None for index in answers.values()),
        "capture_sensitive": standard is None and capture is not None and swap is None,
        "swap_sensitive": standard is None and swap is not None and capture is None,
    }
    if not accepted[group]:
        return None
    return {
        "task": "history", "group": group, "fen": board.fen(), "source": {"type": "synthetic"},
        "state_line": state_line, "squares": squares, "audit_line": audit,
        "gold": {"occupants": occupants, "first_illegal": answers},
    }


def build(source: Path, seed: int, per_group: int, max_draws: int = 150_000) -> list[dict]:
    rng = random.Random(seed)
    counts: Counter = Counter()
    rows = []
    used_boards = set()
    used_games = set()
    # One extra per task/group for the disjoint pilot.
    size = per_group + 1

    def add(item: dict) -> None:
        key = (item["task"], item["group"])
        item["split"] = "pilot" if counts[key] == 0 else "main"
        item["seed"] = seed
        item["id"] = hashlib.sha256(json.dumps(item, sort_keys=True).encode()).hexdigest()[:20]
        counts[key] += 1
        used_boards.add(position_key(chess.Board(item["fen"])))
        rows.append(item)

    if source.exists():
        inputs = list(lichess_boards(source))
        rng.shuffle(inputs)
        for board, provenance in inputs:
            needed = {group for group in GROUPS if counts["mate", group] < size}
            if not needed:
                break
            game = provenance["game_url"].split("#", 1)[0]
            if game in used_games or position_key(board) in used_boards:
                continue
            try:
                item = mate_item(board, provenance, rng, needed)
            except UnsupportedPosition:
                continue
            if item:
                add(item)
                used_games.add(game)

    for task in TASKS:
        for _ in range(max_draws):
            needed = [group for group in GROUPS if counts[task, group] < size]
            if not needed:
                break
            board = synthetic_board(rng)
            if position_key(board) in used_boards:
                continue
            try:
                item = (mate_item(board, {"type": "synthetic"}, rng, set(needed)) if task == "mate"
                        else history_item(board, rng.choice(needed), rng))
            except UnsupportedPosition:
                continue
            if item:
                add(item)
        if any(counts[task, group] != size for group in GROUPS):
            raise RuntimeError(f"Could not fill {task} in {max_draws} draws: {counts}")
    return sorted(rows, key=lambda item: (item["split"], item["task"], item["group"], item["id"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_ITEMS)
    parser.add_argument("--fetch-lichess", action="store_true")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--per-group", type=int, default=50)
    args = parser.parse_args()
    if args.per_group < 1:
        parser.error("--per-group must be positive")
    if args.fetch_lichess:
        fetch_sample(args.source)
    rows = build(args.source, args.seed, args.per_group)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for item in rows:
            handle.write(json.dumps(item, sort_keys=True) + "\n")
    print(f"Wrote {len(rows)} items to {args.output}")


if __name__ == "__main__":
    main()
