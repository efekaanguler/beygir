"""Execute a reproducible factorial run and resume from durable responses."""

import asyncio
import hashlib
import itertools
import json
import random
import time
from collections import deque
from pathlib import Path

import httpx

from .endpoints import Endpoint, Route, available_models, configured_routes
from .prompts import NAMES, fingerprint, load_templates, render, response_format
from .quota import Quota
from .rules import Rule
from .storage import RunStore

MAX_QUOTA_WAIT = 90
RETRIES = 3


def load_items(path: Path, split: str) -> list[dict]:
    items = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    selected = [item for item in items if item["split"] == split]
    if not selected or len({item["id"] for item in selected}) != len(selected):
        raise ValueError(f"{path} has no unique set of {split} items")
    return selected


class RoutePool:
    def __init__(self, routes: list[Route], endpoint: Endpoint, store: RunStore):
        self.routes = deque(routes)
        self.disabled = set()
        self.token_estimate = endpoint.token_estimate
        self.groups = {
            group: Quota(endpoint.limits, store.recent_usage(endpoint.model_id, group, time.time() - 86_400))
            for group in {route.quota_group for route in routes}
        }

    async def acquire(self):
        deadline = time.monotonic() + MAX_QUOTA_WAIT
        while True:
            now = time.time()
            delays = []
            for _ in range(len(self.routes)):
                route = self.routes[0]
                self.routes.rotate(-1)
                if route.label in self.disabled:
                    continue
                quota = self.groups[route.quota_group]
                delay = quota.wait(self.token_estimate, now)
                if delay <= 0:
                    return route, quota, quota.reserve(self.token_estimate, now)
                delays.append(delay)
            earliest = min(delays, default=float("inf"))
            if time.monotonic() + earliest > deadline:
                return None
            await asyncio.sleep(earliest)


def response_tokens(body, estimate: int) -> int:
    if not isinstance(body, dict) or not isinstance(body.get("usage"), dict):
        return estimate
    total = body["usage"].get("total_tokens")
    return total if type(total) is int and total >= 0 else estimate


def redact(value, keys: list[str]):
    """Keep credentials out of attempt logs even if an upstream error echoes one."""
    if isinstance(value, str):
        for key in keys:
            value = value.replace(key, "[redacted]")
    elif isinstance(value, list):
        return [redact(item, keys) for item in value]
    elif isinstance(value, dict):
        return {redact(key, keys): redact(item, keys) for key, item in value.items()}
    return value


async def api_attempt(client, store, endpoint, pool, run_id, item, name, rule, payload):
    reservation = await pool.acquire()
    if reservation is None:
        return "unavailable"
    route, quota, event = reservation
    started = event[0]
    status = "transport_error"
    http_status = None
    charged = event[1]
    body = None
    try:
        response = await client.post(f"{endpoint.base_url}/chat/completions",
                                     headers=route.headers(), json=payload)
        http_status = response.status_code
        try:
            body = response.json()
        except ValueError:
            body = response.text
        quota.observe(response.headers, http_status)
        error = body.get("error") if isinstance(body, dict) else None
        validation_failed = isinstance(error, dict) and error.get("code") == "json_validate_failed"
        if (response.is_success and not error) or validation_failed:
            status = "response"
            charged = response_tokens(body, charged)
        elif http_status == 429:
            status = "rate_limited"
        elif http_status >= 500 or http_status in (408, 409):
            status = "transport_error"
        else:
            status = "endpoint_error"
            if http_status in (401, 403):
                pool.disabled.add(route.label)
                status = "route_error"
    except asyncio.CancelledError:
        status = "interrupted"
        raise
    except httpx.RequestError as error:
        body = {"error": type(error).__name__}
    finally:
        event[1] = charged
        store.record({
            "run_id": run_id, "item_id": item["id"], "name": name, "rule": rule.value,
            "model": endpoint.model_id, "route": route.label, "quota_group": route.quota_group,
            "started": started, "duration": round(time.time() - started, 3), "status": status,
            "http_status": http_status, "charged_tokens": charged, "request": payload,
            "response": redact(body, [candidate.key for candidate in pool.routes]),
        })
    return status


async def pick_routes(client: httpx.AsyncClient, endpoints: list[Endpoint]):
    cached = {}
    result = {}
    for endpoint in endpoints:
        active = []
        for route in configured_routes(endpoint):
            key = endpoint.base_url, route.label
            if key not in cached:
                try:
                    cached[key] = await available_models(client, endpoint, route)
                except (ValueError, httpx.HTTPError):
                    cached[key] = set()
                    print(f"Model catalog unavailable for {route.label}.")
            if endpoint.model_id in cached[key]:
                active.append(route)
        if active:
            result[endpoint.alias] = active
        else:
            print(f"{endpoint.alias}: {endpoint.model_id} unavailable on configured routes.")
    return result


async def execute(store: RunStore, run_id: str, dataset: Path, split: str, endpoints: list[Endpoint],
                  seed: int, workers: int, max_new: int | None) -> None:
    items = load_items(dataset, split)
    templates = load_templates()
    spec = {
        "format": 2, "split": split, "seed": seed,
        "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
        "prompt_sha256": fingerprint(templates),
        "endpoints": [endpoint.signature() for endpoint in endpoints],
        "names": list(NAMES), "rules": [rule.value for rule in Rule],
    }
    store.start(run_id, spec, items)
    completed = store.completed(run_id)
    all_jobs = list(itertools.product(items, NAMES, Rule))
    random.Random(seed).shuffle(all_jobs)
    pending = {}
    for endpoint in endpoints:
        jobs = [(item, name, rule) for item, name, rule in all_jobs
                if (endpoint.model_id, item["id"], name, rule.value) not in completed]
        pending[endpoint.alias] = deque(jobs if max_new is None else jobs[:max_new])
    needed = [endpoint for endpoint in endpoints if pending[endpoint.alias]]
    if not needed:
        print(f"{run_id}: all requested results are already saved.")
        return

    async with httpx.AsyncClient(timeout=60, trust_env=False) as client:
        routes = await pick_routes(client, needed)

        async def run_endpoint(endpoint: Endpoint) -> None:
            pool = RoutePool(routes[endpoint.alias], endpoint, store)
            jobs = pending[endpoint.alias]
            planned = len(jobs)
            finished = 0
            paused = False

            async def worker() -> None:
                nonlocal finished, paused
                while jobs and not paused:
                    item, name, rule = jobs.popleft()
                    payload = endpoint.payload(render(item, name, rule, templates), response_format(item))
                    for attempt in range(RETRIES):
                        status = await api_attempt(client, store, endpoint, pool, run_id,
                                                   item, name, rule, payload)
                        if status == "response":
                            finished += 1
                            if finished == planned or finished % 25 == 0:
                                print(f"{endpoint.alias}: {finished}/{planned} responses saved", flush=True)
                            break
                        if status in ("unavailable", "endpoint_error"):
                            paused = True
                            print(f"{endpoint.alias}: paused ({status}); pending items can be resumed.", flush=True)
                            break
                        if status == "transport_error" and attempt + 1 < RETRIES:
                            await asyncio.sleep(2 ** attempt)

            async with asyncio.TaskGroup() as group:
                for _ in range(min(workers, planned)):
                    group.create_task(worker())

        async with asyncio.TaskGroup() as group:
            for endpoint in needed:
                if endpoint.alias in routes:
                    group.create_task(run_endpoint(endpoint))
