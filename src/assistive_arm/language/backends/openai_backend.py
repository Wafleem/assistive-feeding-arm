"""OpenAIBackend: OpenAI models through the OpenAI API (optional, needs network + an API key).

NOT a ChatGPT Plus/Pro subscription: consumer subscriptions are not API access and
must not be driven programmatically. Use an API key from the OpenAI platform.

TODO(language): implement with the official ``openai`` package from the
``llm-cloud`` extra (import lazily); key from ``OPENAI_API_KEY``; model from
ASSIST_OPENAI_MODEL; constrain the reply to ``json_schema`` with the API's
structured-output / JSON-schema response format; short timeout, no retries
(the router falls back); map every error to BackendError. Check the current
OpenAI docs for the exact parameters. ``available()``: key present, no network call.
"""

from __future__ import annotations

import os

from .base import BackendError


class OpenAIBackend:
    name = "openai"

    def __init__(self, model: str | None = None):
        self.model = model or os.environ.get("ASSIST_OPENAI_MODEL", "")  # TODO: pick and pin a model

    def available(self) -> bool:
        return bool(os.environ.get("OPENAI_API_KEY"))

    def complete(self, system: str, user: str, json_schema: dict, timeout: float) -> str:
        raise BackendError("TODO(language): OpenAIBackend.complete not implemented")
