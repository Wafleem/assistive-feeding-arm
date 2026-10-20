"""Section 5: language layer (speech, LLM router, rules, dialogue). See README.md in this folder."""

from .dialogue import DialogueState
from .router import LLMBackend, OpenAICompatibleBackend, Router
from .rules import SAFETY_WORDS, parse
from .schema import Action, FeedIntent, IntentError

__all__ = ["Action", "DialogueState", "FeedIntent", "IntentError", "LLMBackend", "OpenAICompatibleBackend",
           "Router", "SAFETY_WORDS", "parse"]
