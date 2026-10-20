"""Rule-based parser: the offline fallback AND the safety path.

PLAN
* Safety words first, always, before any LLM: "stop", "wait", "no", "I'm done", "too close"
  -> STOP / PAUSE / CONFIRM(no) / DONE in < 50 ms. ``IntentRouter`` calls this before any backend.
* Then patterns: feed (+ item via food_perception.vocabulary.canonical, + count),
  drink, slower/faster, smaller/bigger bite, I'm done, what's on the plate, yes/no.
* Anything unsure -> UNKNOWN with a clarifying question; never guess.
The previous project's intent parser (git tag v0-wheelchair-skills,
src/assistive_arm/intent/parser.py) is a working example of the pattern.
"""

from __future__ import annotations

from .schema import FeedIntent

# Matched before ANY model is called; nothing downstream can override them.
SAFETY_WORDS = ("stop", "wait", "no", "i'm done", "im done", "ow", "ouch", "too close", "back off", "hold on")


def parse(text: str) -> FeedIntent:
    raise NotImplementedError("TODO(language): rules.parse")
