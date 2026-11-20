"""Per-model quota accounting shared by credentials in the same organization."""

import re
import time
from collections import deque
from email.utils import parsedate_to_datetime

from .endpoints import Limits


def seconds(raw: str) -> float:
    try:
        return float(raw)
    except ValueError:
        units = {"h": 3600, "m": 60, "s": 1, "ms": 0.001}
        return sum(float(value) * units[unit]
                   for value, unit in re.findall(r"(\d+(?:\.\d+)?)(ms|h|m|s)", raw))


def retry_delay(headers, default: float) -> float:
    """Read Retry-After as a duration or an HTTP date."""
    value = headers.get("retry-after", "")
    if value:
        try:
            return max(0.0, float(value))
        except ValueError:
            pass
        delay = seconds(value)
        if delay:
            return delay
        try:
            return max(0.0, parsedate_to_datetime(value).timestamp() - time.time())
        except (ValueError, TypeError, OverflowError):
            pass
    return default


def window_wait(events, now: float, period: int, request_limit: int,
                token_limit: int, reservation: int) -> float:
    count = len(events)
    consumed = sum(event[1] for event in events)
    if reservation > token_limit:
        return float("inf")
    if count < request_limit and consumed + reservation <= token_limit:
        return 0.0
    for started, tokens in events:
        count -= 1
        consumed -= tokens
        if count < request_limit and consumed + reservation <= token_limit:
            return max(0.0, started + period - now + 0.05)
    return float("inf")


class Quota:
    def __init__(self, limits: Limits, history: list[tuple[float, int]]):
        self.limits = limits
        self.events = deque([started, tokens] for started, tokens in history)
        self.blocked_until = 0.0
        self.server_daily_remaining = None
        self.server_daily_ready = 0.0
        self.server_reset_hint = 900.0

    def wait(self, tokens: int, now: float) -> float:
        while self.events and self.events[0][0] <= now - 86_400:
            self.events.popleft()
        minute = []
        for event in reversed(self.events):
            if event[0] <= now - 60:
                break
            minute.append(event)
        if self.server_daily_remaining == 0 and now >= self.server_daily_ready:
            self.server_daily_remaining = None
        return max(
            0.0, self.blocked_until - now,
            self.server_daily_ready - now if self.server_daily_remaining == 0 else 0.0,
            window_wait(list(reversed(minute)), now, 60, self.limits.rpm, self.limits.tpm, tokens),
            window_wait(self.events, now, 86_400, self.limits.rpd, self.limits.tpd, tokens),
        )

    def reserve(self, tokens: int, now: float) -> list:
        event = [now, tokens]
        self.events.append(event)
        if self.server_daily_remaining:
            self.server_daily_remaining -= 1
            if self.server_daily_remaining == 0:
                self.server_daily_ready = now + self.server_reset_hint + 0.1
        return event

    def observe(self, headers, status: int) -> None:
        now = time.time()
        remaining = headers.get("x-ratelimit-remaining-requests", "")
        reset = seconds(headers.get("x-ratelimit-reset-requests", ""))
        if reset > 0:
            self.server_reset_hint = reset
        if remaining.isdigit():
            value = int(remaining)
            self.server_daily_remaining = value
            if value == 0:
                self.server_daily_ready = now + retry_delay(headers, self.server_reset_hint) + 0.1
        if status == 429:
            token_reset = seconds(headers.get("x-ratelimit-reset-tokens", "")) or 10
            delay = retry_delay(headers, token_reset)
            self.blocked_until = max(self.blocked_until, now + delay + 0.1)
