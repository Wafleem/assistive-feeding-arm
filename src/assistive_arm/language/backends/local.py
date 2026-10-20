"""LocalLLMBackend: the DEFAULT. A small quantised model on the AGX Orin.

"Nothing you say leaves the table": speech and intents stay on our hardware, it
works without venue internet, and latency is predictable.

TODO(language):
* Serve a small instruct model (~1-4B, 4-bit) with Ollama or llama.cpp's server on
  the AGX Orin; both expose an OpenAI-compatible ``POST {base_url}/chat/completions``.
* Request JSON only: llama.cpp supports a JSON-schema/grammar constraint
  (``response_format`` with a schema); Ollama supports ``format`` = a JSON schema.
  Use whichever the chosen server supports, so the output always parses.
* Use the standard library (urllib) or httpx; no cloud SDK needed.
* ``available()``: a quick GET on the server's health/models endpoint, cached for a few seconds.
* Measure latency per utterance on the AGX (target < 1 s) and record it in docs/COMPUTE.md.

Config (env): ASSIST_LOCAL_LLM_URL (default http://localhost:11434/v1), ASSIST_LOCAL_LLM_MODEL.
"""

from __future__ import annotations

import os

from .base import BackendError


class LocalLLMBackend:
    name = "local"

    def __init__(self, base_url: str | None = None, model: str | None = None):
        self.base_url = base_url or os.environ.get("ASSIST_LOCAL_LLM_URL", "http://localhost:11434/v1")
        self.model = model or os.environ.get("ASSIST_LOCAL_LLM_MODEL", "")  # TODO: pick and pin a model

    def available(self) -> bool:
        return False  # TODO(language): health check against the local server

    def complete(self, system: str, user: str, json_schema: dict, timeout: float) -> str:
        raise BackendError("TODO(language): LocalLLMBackend.complete not implemented")
