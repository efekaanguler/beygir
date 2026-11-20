import collections
import json
import unittest
from pathlib import Path

import chess

from beygir.dataset import GROUPS, TASKS, build, position_key
from beygir.prompts import load_templates, render
from beygir.rules import Position, Rule, playable_positions


class RuleTests(unittest.TestCase):
    def test_capture_required_and_king_safe(self):
        board = chess.Board("4r1k1/8/8/8/8/8/3pR3/4K3 w - - 0 1")
        state = Position.from_board(board, Rule.CAPTURE)
        moves = {move.uci() for move in state.legal_moves()}
        self.assertNotIn("e2d2", moves)
        self.assertIn("e1d2", moves)
        self.assertIn("e2e8", moves)
        self.assertTrue(all(state.board.is_capture(move) for move in state.legal_moves()))

        board = chess.Board("6k1/8/8/3p4/4P3/8/8/4K1N1 w - - 0 1")
        state = Position.from_board(board, Rule.CAPTURE)
        self.assertEqual(["e4d5"], [move.uci() for move in state.legal_moves()])

    def test_swap_checks_and_preserves_identity(self):
        board = chess.Board("k7/8/8/8/8/5b2/8/4K3 w - - 0 1")
        self.assertFalse(Position.from_board(board, Rule.STANDARD).board.is_check())
        self.assertTrue(Position.from_board(board, Rule.SWAP).board.is_check())

        board = chess.Board("k7/8/8/8/8/8/8/4KBN1 w - - 0 1")
        state = Position.from_board(board, Rule.SWAP)
        index, after = state.replay(["f1g3"])
        self.assertIsNone(index)
        self.assertEqual("B", after.piece_at("g3"))
        self.assertEqual(".", after.piece_at("f1"))
        self.assertEqual(1, state.replay(["f1g2"])[0])
        self.assertIsNone(state.replay(["g1h2"])[0])

    def test_verified_mate(self):
        board = chess.Board("7k/8/5KQ1/8/8/8/8/8 w - - 0 1")
        for position in playable_positions(board).values():
            self.assertEqual(["g6g7"], position.mate_moves())


class ManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.items = [json.loads(line) for line in Path("benchmark/items.jsonl").read_text().splitlines()]

    def test_frozen_manifest_is_reproducible_and_disjoint(self):
        self.assertEqual(self.items, build(Path("benchmark/lichess.csv"), 20261120, 50))
        self.assertEqual(len(self.items), len({position_key(chess.Board(row["fen"])) for row in self.items}))
        counts = collections.Counter((item["split"], item["task"], item["group"]) for item in self.items)
        for task in TASKS:
            for group in GROUPS:
                self.assertEqual(50, counts["main", task, group])
                self.assertEqual(1, counts["pilot", task, group])

    def test_all_labels_and_common_state(self):
        for item in self.items:
            with self.subTest(id=item["id"]):
                for rule, position in playable_positions(chess.Board(item["fen"])).items():
                    if item["task"] == "mate":
                        self.assertEqual(item["gold"][rule.value], position.mate_moves())
                    else:
                        index, board = position.replay(item["state_line"])
                        self.assertIsNone(index)
                        self.assertEqual(item["gold"]["occupants"],
                                         {s: board.piece_at(s) for s in item["squares"]})
                        self.assertEqual(item["gold"]["first_illegal"][rule.value],
                                         position.replay(item["audit_line"])[0])

    def test_names_only_change_game_header(self):
        templates = load_templates()
        for item in self.items[:5]:
            for rule in Rule:
                chess_prompt = render(item, "Chess", rule, templates)
                okuz_prompt = render(item, "Okuz", rule, templates)
                self.assertEqual(okuz_prompt, chess_prompt.replace("Game: Chess", "Game: Okuz", 1))
