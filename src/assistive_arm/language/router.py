"""IntentRouter: utterance (+ dialogue state + plate contents) -> validated FeedIntent.

Order of authority, never reordered:

1. **Safety words** (``rules.SAFETY_WORDS``: "stop", "wait", "no", "I'm done", ...)
   are matched by the rule layer FIRST and returned immediately. No model is called,
   and no model output can override them.
2. **LLM backends** in the configured order (``backends_from_env``; default: the
   local model on the AGX). Each gets the system prompt + ``FeedIntent.json_schema()``
   + the plate contents + the dialogue state, and a per-backend timeout. Its text must
   parse as JSON and pass ``FeedIntent.from_dict``. Any BackendError, timeout,
   invalid JSON or failed validation => try the next backend.
3. **Rule parser** (``rules.parse``) as the final fallback: always available, offline.

TODO(language):
* implement ``route`` exactly in that order, recording which layer answered
  (``RouteResult.source``) and the latency, for the dashboard and the eval;
* the system prompt (keep it fixed so cloud backends can cache it): the schema,
  the menu (food_perception.vocabulary), examples, "answer with one JSON object";
* never log raw API keys; log utterances only if the user agreed (privacy).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .backends.base import LLMBackend
from .dialogue import DialogueState
from .schema import FeedIntent


@dataclass
class RouteResult:
    intent: FeedIntent
    source: str  # "safety_rules", "local", "anthropic", "openai", "rules"
    latency: float = 0.0
    errors: list[str] = field(default_factory=list)  # why earlier backends were skipped


class IntentRouter:
    def __init__(self, backends: list[LLMBackend], timeout_per_backend: float = 1.5):
        self.backends = list(backends)
        self.timeout = timeout_per_backend

    def route(self, utterance: str, state: DialogueState) -> RouteResult:
        raise NotImplementedError("TODO(language): IntentRouter.route (safety rules -> backends -> rules)")
