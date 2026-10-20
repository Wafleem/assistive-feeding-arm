"""Section 5: language layer (speech, provider-agnostic LLM router, rules, dialogue). See README.md here."""

from .backends import (
    AnthropicBackend, BackendError, FakeBackend, LLMBackend, LocalLLMBackend, OpenAIBackend, backends_from_env,
)
from .dialogue import DialogueState
from .router import IntentRouter, RouteResult
from .rules import SAFETY_WORDS, parse
from .schema import Action, FeedIntent, Intent, IntentError

__all__ = [
    "Action", "AnthropicBackend", "BackendError", "DialogueState", "FakeBackend", "FeedIntent", "Intent",
    "IntentError", "IntentRouter", "LLMBackend", "LocalLLMBackend", "OpenAIBackend", "RouteResult", "SAFETY_WORDS",
    "backends_from_env", "parse",
]
