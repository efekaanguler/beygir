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


class Quota:
    def __init__(self, limits: Limits, history: list[tuple[float, int]]):
        self.limits = limits
        self.events = deque([started, tokens] for started, tokens in history)
        self.blocked_until = 0.0
        self.server_daily_remaining = None

    def wait(self, tokens: int, now: float) -> float:
        while self.events and self.events[0][0] <= now - 86_400:
            self.events.popleft()
        minute = [event for event in reversed(self.events) if event[0] > now - 60]
        daily_exhausted = (len(self.events) >= self.limits.rpd
                           or sum(event[1] for event in self.events) + tokens > self.limits.tpd
                           or self.server_daily_remaining == 0)
        if daily_exhausted:
            return float("inf")
        delay = max(0.0, self.blocked_until - now)
        if len(minute) >= self.limits.rpm or sum(event[1] for event in minute) + tokens > self.limits.tpm:
            return max(delay, minute[-1][0] + 60 - now + 0.01) if minute else float("inf")
        return delay

    def reserve(self, tokens: int, now: float) -> list:
        event = [now, tokens]
        self.events.append(event)
        if self.server_daily_remaining:
            self.server_daily_remaining -= 1
        return event

    def observe(self, headers, status: int) -> None:
        now = time.time()
        remaining = headers.get("x-ratelimit-remaining-requests", "")
        if remaining.isdigit():
            value = int(remaining)
            self.server_daily_remaining = (value if self.server_daily_remaining is None
                                           else min(value, self.server_daily_remaining))
        if status == 429:
            token_reset = seconds(headers.get("x-ratelimit-reset-tokens", "")) or 10
            delay = retry_delay(headers, token_reset)
            self.blocked_until = max(self.blocked_until, now + delay + 0.1)
