"""Executable rules for the three BEYGIR conditions."""

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Optional

import chess


class Rule(str, Enum):
    STANDARD = "standard"
    CAPTURE = "mandatory_capture"
    SWAP = "bishop_knight_swap"


class UnsupportedPosition(ValueError):
    """A special chess move or invalid board falls outside the benchmark."""


def movement_board(board: chess.Board, rule: Rule) -> chess.Board:
    """Encode displayed pieces by their movement types without changing squares."""
    if rule is not Rule.SWAP:
        return board.copy(stack=False)
    encoded = board.copy(stack=False)
    for square, piece in board.piece_map().items():
        if piece.piece_type in (chess.BISHOP, chess.KNIGHT):
            swapped = chess.KNIGHT if piece.piece_type == chess.BISHOP else chess.BISHOP
            encoded.set_piece_at(square, chess.Piece(swapped, piece.color))
    return encoded


@dataclass
class Position:
    rule: Rule
    board: chess.Board

    @classmethod
    def from_board(cls, board: chess.Board, rule: Rule) -> "Position":
        if board.castling_rights or board.ep_square is not None:
            raise UnsupportedPosition("Castling rights and en passant targets are excluded")
        encoded = movement_board(board, rule)
        if not encoded.is_valid():
            raise UnsupportedPosition(f"Invalid starting board under {rule.value}")
        return cls(rule, encoded)

    def legal_moves(self) -> tuple[chess.Move, ...]:
        moves = tuple(self.board.legal_moves)
        if any(move.promotion or self.board.is_en_passant(move) for move in moves):
            raise UnsupportedPosition("Promotion and en passant are excluded")
        if self.rule is Rule.CAPTURE:
            captures = tuple(move for move in moves if self.board.is_capture(move))
            return captures or moves
        return moves

    def copy(self) -> "Position":
        return Position(self.rule, self.board.copy(stack=False))

    def piece_at(self, square: str) -> str:
        piece = self.board.piece_at(chess.parse_square(square))
        if piece is None:
            return "."
        if self.rule is Rule.SWAP and piece.piece_type in (chess.BISHOP, chess.KNIGHT):
            original = chess.KNIGHT if piece.piece_type == chess.BISHOP else chess.BISHOP
            return chess.Piece(original, piece.color).symbol()
        return piece.symbol()

    def mate_moves(self) -> list[str]:
        mates = []
        for move in self.legal_moves():
            after = self.copy()
            after.board.push(move)
            if after.board.is_check() and not after.legal_moves():
                mates.append(move.uci())
        return sorted(mates)

    def replay(self, moves: Iterable[str]) -> tuple[Optional[int], "Position"]:
        """Return the one-based first illegal index and the last valid state."""
        current = self.copy()
        for index, text in enumerate(moves, start=1):
            move = chess.Move.from_uci(text)
            if move not in current.legal_moves():
                return index, current
            current.board.push(move)
        return None, current


def playable_positions(board: chess.Board) -> dict[Rule, Position]:
    positions = {rule: Position.from_board(board, rule) for rule in Rule}
    if any(not position.legal_moves() for position in positions.values()):
        raise UnsupportedPosition("An initial board must allow play under all rules")
    return positions
