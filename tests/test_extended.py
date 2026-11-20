import itertools
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import chess

from beygir.endpoints import catalog
from beygir.extended_dataset import AUDIT_GROUP_COUNTS, AUDIT_LENGTHS, EXTENDED_ITEMS, build_extended
from beygir.prompts import NAMES, render, response_format, templates_for
from beygir.report import write_report
from beygir.rules import Position, Rule, playable_positions
from beygir.scoring import scored_item, summarize
from beygir.storage import RunStore


def mock_reply(payload):
    return {"choices": [{"message": {"content": json.dumps(payload)}}]}


class ExtendedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = [json.loads(line) for line in EXTENDED_ITEMS.read_text().splitlines()]

    def test_extended_manifest_reproduces_and_is_balanced(self):
        self.assertEqual(self.rows, build_extended())
        self.assertEqual(len(self.rows), len({row["fen"].rsplit(" ", 2)[0] for row in self.rows}))
        counts = Counter(row["task"] for row in self.rows if row["split"] == "main")
        self.assertEqual({"mate": 60, "history": 60, "play5": 50,
                          "audit5": 50, "audit20": 50, "audit50": 50}, counts)
        for task in AUDIT_LENGTHS:
            self.assertEqual(AUDIT_GROUP_COUNTS, Counter(
                row["group"] for row in self.rows if row["split"] == "main" and row["task"] == task
            ))

    def test_legal_witnesses_and_full_replay_labels(self):
        for row in self.rows:
            if row["task"] not in ("play5", *AUDIT_LENGTHS):
                continue
            positions = playable_positions(chess.Board(row["fen"]), exclude_special_moves=True)
            if row["task"] == "play5":
                witness = row["gold"]["example_legal_line"]
                self.assertEqual(5, len(witness))
                for position in positions.values():
                    self.assertIsNone(position.replay(witness)[0])
            else:
                self.assertEqual(AUDIT_LENGTHS[row["task"]], len(row["audit_line"]))
                for rule, position in positions.items():
                    index = position.replay(row["audit_line"])[0]
                    self.assertEqual(row["gold"]["first_illegal"][rule.value], index)

    def test_prompts_only_change_name_and_do_not_disclose_gold(self):
        items = [next(row for row in self.rows if row["task"] == task)
                 for task in ("mate", "history", "play5", *AUDIT_LENGTHS)]
        templates = templates_for(items)
        for row, rule in itertools.product(items, Rule):
            chess_prompt = render(row, "Chess", rule, templates)
            okuz_prompt = render(row, "Okuz", rule, templates)
            self.assertEqual(chess_prompt.replace("Game: Chess", "Game: Okuz", 1), okuz_prompt)
            if row["task"] in AUDIT_LENGTHS:
                self.assertIn(f"{row['length']}. {row['audit_line'][-1]}", chess_prompt)
                fields = response_format(row)["json_schema"]["schema"]["required"]
                self.assertEqual(["all_legal", "first_illegal"], fields)
            if row["task"] == "play5":
                self.assertNotIn(" ".join(row["gold"]["example_legal_line"]), chess_prompt)

    def test_generation_scores_length_and_first_illegal(self):
        row = next(row for row in self.rows if row["task"] == "play5")
        valid = row["gold"]["example_legal_line"]
        for rule in Rule:
            full = scored_item(row, "model", "Chess", rule.value, True, mock_reply({"moves": valid}))
            invalid = scored_item(row, "model", "Chess", rule.value, True,
                                  mock_reply({"moves": [valid[0], "bad", *valid[2:]]}))
            short = scored_item(row, "model", "Chess", rule.value, True, mock_reply({"moves": valid[:4]}))
            self.assertTrue(full["line_legal"])
            self.assertEqual(5, full["legal_prefix"])
            self.assertFalse(invalid["line_legal"])
            self.assertEqual(2, invalid["first_index_in_generated_line"])
            self.assertFalse(short["line_legal"])
            self.assertEqual(0.5, summarize([full, short])["all_moves_legal_accuracy"])
        board = chess.Board("7k/P7/8/8/8/8/8/7K w - - 0 1")
        position = Position.from_board(board, Rule.STANDARD, exclude_special_moves=True)
        self.assertEqual(1, position.replay(["a7a8q"])[0])

    def test_offline_reports_include_new_task_metrics_and_figure(self):
        audited = [next(row for row in self.rows
                        if row["task"] == task and row["group"] == group)
                   for task, group in itertools.product(AUDIT_LENGTHS, ("shared_positive", "shared_negative"))]
        played = next(row for row in self.rows if row["task"] == "play5")
        items = [*audited, played]
        model = catalog()["aws-qwen32"]
        run = "extended-test"
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "run.sqlite3"
            with RunStore(db_path) as store:
                store.start(run, {"endpoints": [model.signature()], "names": list(NAMES),
                                  "rules": [rule.value for rule in Rule]}, items)
                for row, name, rule in itertools.product(items, NAMES, Rule):
                    if row["task"] == "play5":
                        payload = {"moves": row["gold"]["example_legal_line"]}
                    else:
                        index = row["gold"]["first_illegal"][rule.value]
                        payload = {"all_legal": index is None, "first_illegal": index}
                    store.record({"run_id": run, "item_id": row["id"], "name": name, "rule": rule.value,
                                  "model": model.model_id, "route": "mock", "quota_group": "mock",
                                  "started": 0.0, "duration": 0.0, "status": "response", "http_status": 200,
                                  "charged_tokens": 0, "request": {}, "response": mock_reply(payload)})
                output = write_report(store, db_path, run)
                self.assertTrue((output / "audit_by_length.png").is_file())
                self.assertIn("all_moves_legal_accuracy", (output / "metrics.csv").read_text())
                for task in store.task_progress(run):
                    self.assertEqual(task["planned"], task["saved"])
