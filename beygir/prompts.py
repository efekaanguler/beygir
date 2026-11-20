"""Render and fingerprint the English evaluation prompts."""

import hashlib
from pathlib import Path

import chess

from .rules import Rule

NAMES = ("Chess", "Okuz")
PROMPT_DIR = Path(__file__).resolve().parents[1] / "prompts"


def load_templates() -> dict[str, str]:
    return {path.stem: path.read_text(encoding="utf-8").strip()
            for path in sorted(PROMPT_DIR.glob("*.txt"))}


def fingerprint(templates: dict[str, str]) -> str:
    content = "\n".join(f"{name}\n{text}" for name, text in sorted(templates.items()))
    return hashlib.sha256(content.encode()).hexdigest()


def render(item: dict, name: str, rule: Rule, templates: dict[str, str]) -> str:
    board = chess.Board(item["fen"])
    rows = [f"{rank + 1} " + " ".join(
        piece.symbol() if (piece := board.piece_at(chess.square(file, rank))) else "."
        for file in range(8)
    ) for rank in reversed(range(8))]
    rows.append("  a b c d e f g h")
    side = "White" if board.turn == chess.WHITE else "Black"
    parts = [
        f"Game: {name}",
        templates["rules"],
        templates[rule.value],
        "\n".join(rows),
        f"Side to move: {side}",
        templates[item["task"]],
    ]
    if item["task"] == "history":
        parts.extend((
            f"Line A: {' '.join(item['state_line'])}",
            f"Requested squares: {' '.join(item['squares'])}",
            f"Line B: {' '.join(item['audit_line'])}",
        ))
    return "\n\n".join(parts)


def response_format(item: dict) -> dict:
    """Describe the output shape without disclosing candidate answers."""

    def object_schema(properties: dict) -> dict:
        return {"type": "object", "properties": properties, "required": list(properties),
                "additionalProperties": False}

    if item["task"] == "mate":
        fields = {
            "mate_exists": {"type": "boolean"},
            "move": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        }
    else:
        fields = {
            "occupants": object_schema({square: {"type": "string"} for square in item["squares"]}),
            "first_illegal": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
        }
    return {"type": "json_schema", "json_schema": {
        "name": item["task"], "strict": True, "schema": object_schema(fields),
    }}
