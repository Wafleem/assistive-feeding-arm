"""Meal-level conversation state: what's left, preferences, pace, pending questions.

TODO(language): apply(intent) updates the state and returns the sentence to speak
("Okay, strawberries, smaller bites."); persist preferences per user between meals.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DialogueState:
    plate: dict[str, int] = field(default_factory=dict)  # item -> count seen
    preferences: dict = field(default_factory=lambda: {"pace": 1.0, "bite_size": "normal"})
    last_item: str | None = None
    pending_question: str | None = None
    history: list[str] = field(default_factory=list)

    def apply(self, intent) -> str:
        raise NotImplementedError("TODO(language): DialogueState.apply")
