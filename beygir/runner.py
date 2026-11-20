"""Execute a reproducible factorial run and resume from durable responses."""

import asyncio
import hashlib
import itertools
import json
import random
import time
from collections import deque
from collections.abc import Callable
from contextlib import nullcontext, suppress
from pathlib import Path

import httpx

from .clients import client_for
from .endpoints import Endpoint, Route
from .prompts import NAMES, fingerprint, load_templates, render, response_format
from .quota import Quota, retry_delay
from .rules import Rule
from .storage import RunStore

MAX_QUOTA_WAIT = 90
RETRIES = 3
BATCH_COOLDOWN = 300
SNAPSHOT_INTERVAL = 900


def load_items(path: Path, split: str) -> list[dict]:
    items = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    selected = [item for item in items if item["split"] == split]
    if not selected or len({item["id"] for item in selected}) != len(selected):
        raise ValueError(f"{path} has no unique set of {split} items")
    return selected


class RoutePool:
    def __init__(self, routes: list[Route], endpoint: Endpoint, store: RunStore, follow: bool = False):
        self.routes = deque(routes)
        self.alias = endpoint.alias
        self.follow = follow
        self.next_notice = 0.0
        self.disabled = set()
        self.retry_at = {}
        self.transient_failures = {}
        self.token_estimate = endpoint.token_estimate
        self.groups = {
            group: Quota(endpoint.limits, store.recent_usage(
                endpoint.model_id, group, time.time() - 86_400,
                [route.label for route in routes if route.quota_group == group],
            ))
            for group in {route.quota_group for route in routes}
        }

    def mark_transient_failure(self, route: Route, headers) -> float:
        label = route.label
        failures = self.transient_failures.get(label, 0) + 1
        self.transient_failures[label] = failures
        delay = retry_delay(headers, min(30, 2 ** min(failures, 5)))
        self.retry_at[label] = max(self.retry_at.get(label, 0.0), time.monotonic() + delay)
        return delay

    def mark_success(self, route: Route) -> None:
        if self.retry_at.get(route.label, 0.0) <= time.monotonic():
            self.transient_failures.pop(route.label, None)

    async def acquire(self):
        deadline = time.monotonic() + MAX_QUOTA_WAIT
        while True:
            now = time.time()
            monotonic_now = time.monotonic()
            delays = []
            for _ in range(len(self.routes)):
                route = self.routes[0]
                self.routes.rotate(-1)
                if route.label in self.disabled:
                    continue
                quota = self.groups[route.quota_group]
                route_wait = self.retry_at.get(route.label, 0.0) - monotonic_now
                delay = max(route_wait, quota.wait(self.token_estimate, now))
                if delay <= 0:
                    return route, quota, quota.reserve(self.token_estimate, now)
                delays.append(delay)
            earliest = min(delays, default=float("inf"))
            if earliest == float("inf") or (not self.follow and monotonic_now + earliest > deadline):
                return None
            if self.follow and earliest > MAX_QUOTA_WAIT and monotonic_now >= self.next_notice:
                print(f"{self.alias}: all routes waiting; next eligible in about {earliest / 60:.0f} min.",
                      flush=True)
                self.next_notice = monotonic_now + SNAPSHOT_INTERVAL
            await asyncio.sleep(min(earliest, SNAPSHOT_INTERVAL) if self.follow else earliest)


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


async def api_attempt(client, store, endpoint, pool, run_id, item, name, rule, payload,
                      request_gate: asyncio.Lock | None = None):
    reservation = await pool.acquire()
    if reservation is None:
        return "unavailable"
    route, quota, event = reservation
    async with request_gate if request_gate is not None else nullcontext():
        started = event[0] = time.time()
        status = "transport_error"
        http_status = None
        charged = event[1]
        body = None
        response_headers = {}
        try:
            adapter = client_for(endpoint)
            response = await adapter.complete(client, endpoint, route, payload)
            http_status = response.status_code
            response_headers = response.headers
            try:
                body = adapter.normalize(response.json())
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
            if status == "transport_error":
                delay = pool.mark_transient_failure(route, response_headers)
                print(f"{endpoint.alias}: {route.label} cooling down for {delay:g}s after a temporary error.",
                      flush=True)
            elif status == "response":
                pool.mark_success(route)
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
        adapter = client_for(endpoint)
        for route in adapter.routes(endpoint):
            key = (endpoint.base_url, route.label, endpoint.model_id if endpoint.region else "")
            if key not in cached:
                try:
                    cached[key] = await adapter.models(client, endpoint, route)
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


def prepare_run(store: RunStore, run_id: str, dataset: Path, split: str, endpoints: list[Endpoint],
                seed: int, per_group: int | None = None):
    items = load_items(dataset, split)
    templates = load_templates()
    if per_group is not None:
        groups = {}
        for item in items:
            groups.setdefault((item["task"], item["group"]), []).append(item)
        rng = random.Random(seed)
        items = []
        for key, group in sorted(groups.items()):
            if len(group) < per_group:
                raise ValueError(f"Only {len(group)} items available for {key}")
            items.extend(rng.sample(sorted(group, key=lambda row: row["id"]), per_group))
    spec = {
        "format": 2, "split": split, "seed": seed,
        "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
        "prompt_sha256": fingerprint(templates),
        "endpoints": [endpoint.signature() for endpoint in endpoints],
        "names": list(NAMES), "rules": [rule.value for rule in Rule],
    }
    if per_group is not None:
        spec["per_group"] = per_group
    store.start(run_id, spec, items)
    return items, templates


async def execute(store: RunStore, run_id: str, dataset: Path, split: str, endpoints: list[Endpoint],
                  seed: int, workers: int, max_new: int | None, *,
                  follow: bool = False, snapshot: Callable[[], object] | None = None,
                  per_group: int | None = None, serial: bool = False) -> None:
    items, templates = prepare_run(store, run_id, dataset, split, endpoints, seed, per_group)
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
        request_gate = asyncio.Lock() if serial else None

        async def run_endpoint(endpoint: Endpoint) -> None:
            pool = RoutePool(routes[endpoint.alias], endpoint, store, follow)
            jobs = pending[endpoint.alias]
            planned = len(jobs)
            finished = 0
            paused = False

            async def worker() -> None:
                nonlocal finished, paused
                while jobs and not paused:
                    job = jobs.popleft()
                    item, name, rule = job
                    payload = client_for(endpoint).payload(endpoint, render(item, name, rule, templates), response_format(item))
                    for _ in range(RETRIES):
                        status = await api_attempt(client, store, endpoint, pool, run_id,
                                                   item, name, rule, payload, request_gate)
                        if status == "response":
                            finished += 1
                            if finished == planned or finished % 25 == 0:
                                print(f"{endpoint.alias}: {finished}/{planned} responses saved", flush=True)
                            break
                        if status in ("unavailable", "endpoint_error"):
                            paused = True
                            print(f"{endpoint.alias}: paused ({status}); pending items can be resumed.", flush=True)
                            break
                    else:
                        if follow:
                            print(f"{endpoint.alias}: retry batch exhausted; retrying the item after a 5 min cooldown.",
                                  flush=True)
                            await asyncio.sleep(BATCH_COOLDOWN)
                            jobs.append(job)

            async with asyncio.TaskGroup() as group:
                for _ in range(min(1 if serial else workers, planned)):
                    group.create_task(worker())

        async def snapshots() -> None:
            while True:
                await asyncio.sleep(SNAPSHOT_INTERVAL)
                try:
                    snapshot()
                except (OSError, ValueError) as error:
                    print(f"{run_id}: snapshot postponed ({error}). Raw replies remain saved.", flush=True)

        reporter = asyncio.create_task(snapshots()) if follow and snapshot else None
        try:
            async with asyncio.TaskGroup() as group:
                for endpoint in needed:
                    if endpoint.alias in routes:
                        group.create_task(run_endpoint(endpoint))
        finally:
            if reporter is not None:
                reporter.cancel()
                with suppress(asyncio.CancelledError):
                    await reporter
