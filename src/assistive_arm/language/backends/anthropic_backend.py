"""AnthropicBackend: Claude through the Claude API (optional, needs network + an API key).

NOT a Claude.ai / Claude Pro / Max subscription: consumer subscriptions are not API
access and must not be driven programmatically. Use an API key from the Claude
Console (ask the organisers for credits).

TODO(language) - implementation notes (verify against the current SDK docs):
* Install the optional extra: ``pip install -e ".[llm-cloud]"`` (adds ``anthropic``).
  Import lazily inside the methods so offline runs and tests never need it.
* Client: ``anthropic.Anthropic()`` reads ``ANTHROPIC_API_KEY`` from the environment
  (put it in ``.env``, never in code, never log it). Pass ``timeout=`` and
  ``max_retries=0`` or 1: the router does the falling back, so keep each attempt short.
* Model: from ASSIST_ANTHROPIC_MODEL, default ``claude-opus-5-5``. For a short
  JSON-routing call the team may prefer a faster, cheaper model such as
  ``claude-haiku-5-5``; measure accuracy and latency on the recorded utterances
  and decide. Use exact model IDs (no date suffixes).
* Structured output: constrain the response to ``json_schema`` with
  ``output_config={"format": {...}}`` on ``client.messages.create`` (or
  ``client.messages.parse`` with a schema), so the reply is one valid JSON object.
  Don't use assistant prefill (rejected on current models).
* Keep it short: a small ``max_tokens`` (a few hundred), a fixed system prompt
  (cacheable), and a low effort setting for this simple classification task.
* Check ``response.stop_reason`` before reading content; treat ``refusal``,
  ``max_tokens`` and any SDK exception (``anthropic.APIConnectionError``,
  ``RateLimitError``, ``APIStatusError``, ``APITimeoutError``) as BackendError.
* ``available()``: True if ANTHROPIC_API_KEY is set (do not make a network call).
"""

from __future__ import annotations

import os

from .base import BackendError

DEFAULT_MODEL = "claude-opus-5-5"


class AnthropicBackend:
    name = "anthropic"

    def __init__(self, model: str | None = None):
        self.model = model or os.environ.get("ASSIST_ANTHROPIC_MODEL", DEFAULT_MODEL)

    def available(self) -> bool:
        return bool(os.environ.get("ANTHROPIC_API_KEY"))

    def complete(self, system: str, user: str, json_schema: dict, timeout: float) -> str:
        raise BackendError("TODO(language): AnthropicBackend.complete not implemented")
