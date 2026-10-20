"""The validated command schema. The LLM router and the rule fallback both produce this,
and nothing un-validated ever reaches orchestration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum


class Action(str, Enum):
    FEED = "feed"  # "feed me the strawberries", "next bite", "another grape"
    DRINK = "drink"  # "a sip of water"
    ADJUST = "adjust"  # "slower", "smaller bite", "a bit higher"
    PAUSE = "pause"  # "wait", "hold on"
    RESUME = "resume"
    STOP = "stop"  # safety word: always handled by rules, never waits on an LLM
    DONE = "done"  # "I'm done", "I'm full"
    QUERY = "query"  # "what's on the plate?"
    CONFIRM = "confirm"  # yes / no to a pending question
    UNKNOWN = "unknown"


class IntentError(ValueError):
    pass


@dataclass
class FeedIntent:
    action: Action
    item: str | None = None  # canonical menu name (food_perception.vocabulary), e.g. "strawberry"
    count: int | None = None
    adjust: dict = field(default_factory=dict)  # {"pace": "slower"} / {"bite_size": "smaller"}
    answer: bool | None = None  # for CONFIRM
    question: str | None = None  # for UNKNOWN: what to ask the user
    raw: str = ""

    def __post_init__(self):
        self.action = Action(self.action)
        self.validate()

    def validate(self) -> None:
        """TODO(language): enforce per-action fields, e.g. FEED needs an item on the menu
        or None ("next bite" = planner's choice); ADJUST keys in {pace, bite_size};
        CONFIRM needs answer; count in 1..5."""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["action"] = self.action.value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "FeedIntent":
        raise NotImplementedError("TODO(language): FeedIntent.from_dict (reject unknown fields)")

    @staticmethod
    def json_schema() -> dict:
        raise NotImplementedError("TODO(language): JSON schema for the LLM's structured output")
