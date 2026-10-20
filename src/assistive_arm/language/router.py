"""LLM router: free text (+ dialogue state + what's on the plate) -> validated FeedIntent.

PLAN
* Backend: a small local model on the AGX Orin via Ollama / llama.cpp (OpenAI-
  compatible HTTP), with a hosted-API fallback when the venue network allows.
* Prompt = system rules + FeedIntent.json_schema() + current plate items + dialogue
  state; request JSON-only output (structured output / grammar if the server supports it).
* Validate with FeedIntent.from_dict; on timeout (> 1.5 s) or invalid JSON fall back
  to rules.parse. Safety words never go through here (rules first).
"""

from __future__ import annotations

from typing import Callable, Protocol

from .dialogue import DialogueState
from .schema import FeedIntent


class LLMBackend(Protocol):
    def complete(self, system: str, user: str, timeout: float) -> str: ...


class OpenAICompatibleBackend:
    """Ollama, llama.cpp server, vLLM and most hosted APIs speak this protocol."""

    def __init__(self, base_url: str = "http://localhost:11434/v1", model: str = "TODO-pick-a-3B-model"):
        self.base_url, self.model = base_url, model

    def complete(self, system: str, user: str, timeout: float) -> str:
        raise NotImplementedError("TODO(language): POST /chat/completions")


class Router:
    def __init__(self, backend: LLMBackend | None, fallback: Callable[[str], FeedIntent]):
        self.backend, self.fallback = backend, fallback

    def route(self, text: str, state: DialogueState) -> FeedIntent:
        raise NotImplementedError("TODO(language): Router.route")
