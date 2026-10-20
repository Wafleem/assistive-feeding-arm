"""The one interface every LLM backend implements.

A backend only turns (system prompt, user text) into a raw string, which should be
the intent JSON. Validation, fallback and safety words live in the router, not here,
so a backend can never bypass them.
"""

from __future__ import annotations

from typing import Protocol


class BackendError(RuntimeError):
    """Any failure the router should treat as "try the next backend": timeout,
    network error, auth error, rate limit, refusal, empty output."""


class LLMBackend(Protocol):
    name: str

    def available(self) -> bool:
        """True if this backend can be used right now (server reachable / key present).
        Must be cheap and must never raise."""
        ...

    def complete(self, system: str, user: str, json_schema: dict, timeout: float) -> str:
        """Return the model's raw text (expected: one JSON object matching json_schema).
        Raise BackendError on any failure. Never log or return API keys."""
        ...
