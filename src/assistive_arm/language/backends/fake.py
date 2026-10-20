"""FakeBackend for tests and offline demos: returns canned replies, can be told to fail or stall."""

from __future__ import annotations

import time

from .base import BackendError


class FakeBackend:
    def __init__(self, name: str = "fake", replies: dict[str, str] | None = None, default: str | None = None,
                 fail: bool = False, delay: float = 0.0, available: bool = True):
        self.name = name
        self.replies = replies or {}
        self.default = default
        self.fail, self.delay, self._available = fail, delay, available
        self.calls: list[str] = []

    def available(self) -> bool:
        return self._available

    def complete(self, system: str, user: str, json_schema: dict, timeout: float) -> str:
        self.calls.append(user)
        if self.delay:
            time.sleep(min(self.delay, timeout))
            if self.delay >= timeout:
                raise BackendError(f"{self.name}: timeout")
        if self.fail:
            raise BackendError(f"{self.name}: simulated failure")
        reply = self.replies.get(user, self.default)
        if reply is None:
            raise BackendError(f"{self.name}: no canned reply")
        return reply
