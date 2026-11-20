import csv
import json
import os
import tempfile
import unittest
from collections import Counter
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import httpx

from beygir.dataset import DEFAULT_ITEMS, SEED
from beygir.endpoints import Limits, Route, catalog, configured_routes
from beygir.quota import Quota
from beygir.report import write_report
from beygir.runner import RoutePool, api_attempt, execute
from beygir.scoring import name_effects, scored_item, summarize
from beygir.rules import Rule
from beygir.storage import RunStore


class ScoreTests(unittest.TestCase):
    def test_openrouter_pins_supported_model_provider(self):
        environment = {
            "OPENROUTER_API_KEY": "mock-openrouter",
            "OPENROUTER_PROVIDER": "deepinfra/turbo",
            "OPENROUTER_MINIMAX_PROVIDER": "gmicloud/fp8",
        }
        with patch.dict(os.environ, environment):
            for alias, model_id, provider in (
                ("llama", "meta-llama/llama-3.3-70b-instruct", "deepinfra/turbo"),
                ("minimax", "minimax/minimax-m2.7", "gmicloud/fp8"),
            ):
                endpoint = catalog()[alias]
                request = endpoint.payload("Test", {})
                self.assertEqual(model_id, request["model"])
                self.assertEqual({"only": [provider], "allow_fallbacks": False}, request["provider"])
                self.assertEqual("openrouter", configured_routes(endpoint)[0].label)

    def test_false_positive_and_malformed_have_distinct_effects(self):
        good = {"id": "m1", "task": "mate", "group": "x", "gold": {"standard": ["g6g7"]}}
        bad = {"id": "m2", "task": "mate", "group": "x", "gold": {"standard": []}}
        correct = {"choices": [{"message": {"content": '{"mate_exists":true,"move":"g6g7"}'}}]}
        rows = [
            scored_item(good, "model", "Chess", "standard", True, correct),
            scored_item(bad, "model", "Chess", "standard", True, correct),
            scored_item(good, "model", "Chess", "standard", True, {}),
            scored_item(bad, "model", "Chess", "standard", False, None),
        ]
        result = summarize(rows)
        self.assertEqual(1, result["missing"])
        self.assertEqual(1, result["format_errors"])
        self.assertEqual(0.5, result["precision"])
        self.assertEqual(1, result["recall"])
        self.assertAlmostEqual(1 / 3, result["accuracy"])
        self.assertAlmostEqual(0.25, result["balanced_accuracy"])
        self.assertEqual(0.5, result["mate_move_accuracy"])

    def test_name_effect_requires_complete_quartet(self):
        rows = [
            {"item_id": item, "name": name, "rule": rule,
             "received": not (item == "incomplete" and name == "Chess" and rule == "standard"),
             "classification_correct": name == "Chess" or rule == "standard",
             "mate_eligible": True, "mate_move_correct": True}
            for item in ("complete", "incomplete")
            for name in ("Chess", "Okuz")
            for rule in ("standard", "swap")
        ]
        effect, _ = name_effects(rows, "m", "mate", "all", ["standard", "swap"])
        self.assertEqual(2, effect["planned_n"])
        self.assertEqual(1, effect["matched_n"])
        self.assertEqual(100, effect["difference_of_gaps_pp"])

    def test_quota_accounts_for_minute_and_day(self):
        quota = Quota(Limits(rpm=2, rpd=3, tpm=1500, tpd=5000), [])
        quota.reserve(700, 100)
        quota.reserve(700, 101)
        self.assertAlmostEqual(59.01, quota.wait(700, 101))
        quota.reserve(700, 200)
        self.assertEqual(float("inf"), quota.wait(700, 201))


class EndToEndTests(unittest.IsolatedAsyncioTestCase):
    async def test_transient_error_is_logged_and_next_attempt_uses_healthy_route(self):
        keys_seen = []

        def server(request):
            token = request.headers["Authorization"]
            keys_seen.append(token)
            if token == "Bearer test-one":
                return httpx.Response(503, headers={"Retry-After": "20"}, json={"error": "busy"})
            return httpx.Response(200, json={"choices": [], "usage": {"total_tokens": 25}})

        with tempfile.TemporaryDirectory() as directory:
            with RunStore(Path(directory) / "run.sqlite3") as store:
                item = {"id": "sample"}
                store.start("retry", {}, [item])
                endpoint = catalog()["qwen"]
                pool = RoutePool([Route("one", "g1", "test-one"), Route("two", "g2", "test-two")],
                                 endpoint, store)
                async with httpx.AsyncClient(transport=httpx.MockTransport(server)) as client:
                    for name, expected in (("Chess", "transport_error"), ("Chess", "response"),
                                           ("Okuz", "response")):
                        status = await api_attempt(client, store, endpoint, pool, "retry",
                                                   item, name, Rule.STANDARD, {})
                        self.assertEqual(expected, status)
                self.assertEqual(["Bearer test-one", "Bearer test-two", "Bearer test-two"], keys_seen)
                attempts = list(store.attempts("retry"))
                self.assertEqual(["transport_error", "response", "response"],
                                 [row["status"] for row in attempts])
                self.assertEqual([503, 200, 200], [row["http_status"] for row in attempts])

    async def test_blocked_quota_group_uses_only_independent_route(self):
        with tempfile.TemporaryDirectory() as directory:
            with RunStore(Path(directory) / "run.sqlite3") as store:
                pool = RoutePool(
                    [Route("one", "same", "a"), Route("two", "same", "b"), Route("three", "other", "c")],
                    catalog()["gpt20"], store,
                )
                pool.groups["same"].server_daily_remaining = 0
                route, _, _ = await pool.acquire()
                self.assertEqual("three", route.label)

    async def test_cached_run_can_add_an_endpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            with RunStore(Path(directory) / "run.sqlite3") as store:
                first = catalog()["gpt20"].signature()
                second = catalog()["qwen"].signature()
                item = {"id": "test"}
                store.start("extend", {"seed": 1, "endpoints": [first]}, [item])
                store.start("extend", {"seed": 1, "endpoints": [first, second]}, [item])
                spec, cached = store.load("extend")
                self.assertEqual([first, second], spec["endpoints"])
                self.assertEqual([item], cached)
                with self.assertRaises(ValueError):
                    store.start("extend", {"seed": 2, "endpoints": [first, second]}, [item])

    async def test_resume_and_offline_exports(self):
        requests = []
        authenticators = Counter()
        client_type = httpx.AsyncClient

        def server(request: httpx.Request):
            if request.method == "GET":
                return httpx.Response(200, json={"data": [{"id": "openai/gpt-oss-20b"}]})
            requests.append(request.url.path)
            authenticators[request.headers["Authorization"]] += 1
            payload = json.loads(request.content)
            schema = payload["response_format"]["json_schema"]
            if schema["name"] == "mate":
                content = '{"mate_exists":false,"move":null}'
            else:
                squares = schema["schema"]["properties"]["occupants"]["required"]
                content = json.dumps({"occupants": {s: "." for s in squares}, "first_illegal": None})
            return httpx.Response(200, json={"choices": [{"message": {"content": content}}],
                                             "usage": {"total_tokens": 1}})

        def new_client(**kwargs):
            return client_type(transport=httpx.MockTransport(server), **kwargs)

        environment = {
            "GROQ_API_KEY_1": "mock-key-one",
            "GROQ_API_KEY_2": "mock-key-two",
            "GROQ_QUOTA_GROUP_1": "same",
            "GROQ_QUOTA_GROUP_2": "same",
        }
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "run.sqlite3"
            endpoint = replace(catalog()["gpt20"], limits=Limits(rpm=60), token_estimate=1)
            with patch.dict(os.environ, environment), patch("beygir.runner.httpx.AsyncClient", new_client):
                with RunStore(database) as store:
                    await execute(store, "mock", DEFAULT_ITEMS, "pilot", [endpoint], SEED, 2, 2)
                    self.assertEqual(len(requests), 2)
                    self.assertEqual(len(authenticators), 2)
                    # Resume only pending jobs and verify the original two survive.
                    await execute(store, "mock", DEFAULT_ITEMS, "pilot", [endpoint], SEED, 2, 1)
                    self.assertEqual(len(requests), 3)
                    self.assertEqual(len(store.completed("mock")), 3)
                    self.assertEqual(len(store.recent_usage(endpoint.model_id, "same", 0)), 3)
                    await execute(store, "mock", DEFAULT_ITEMS, "pilot", [endpoint], SEED, 3, None)
                    await execute(store, "mock", DEFAULT_ITEMS, "pilot", [endpoint], SEED, 3, None)
                    self.assertEqual(len(requests), 48)
            with RunStore(database) as store:
                exports = write_report(store, database, "mock")
                self.assertEqual(len(requests), 48)
                with (exports / "item_scores.csv").open() as file:
                    rows = list(csv.DictReader(file))
                self.assertEqual(len(rows), 48)
                self.assertEqual(sum(row["received"] == "True" for row in rows), 48)
                with (exports / "matched_pairs.csv").open() as file:
                    pairs = list(csv.DictReader(file))
                overall = [row for row in pairs if row["task"] == "mate" and row["scope"] == "all"
                           and row["metric"] == "classification_correct"]
                self.assertEqual(len(overall), 15)
                cached = (exports / "raw_attempts.jsonl").read_text()
                self.assertNotIn("mock-key", cached)
                self.assertEqual(len(cached.splitlines()), 48)
                self.assertTrue((exports / "matched_accuracy.png").is_file())
                self.assertTrue((exports / "standard_rule_errors.png").is_file())
