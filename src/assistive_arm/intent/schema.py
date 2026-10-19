"""Validated command schema. Both the rule parser and an LLM produce this.

An LLM is asked to emit JSON matching ``Intent.json_schema()``; whatever it
returns goes through ``Intent.from_dict`` so a hallucinated task or a
malformed target can never reach the planner.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from enum import Enum


class Task(str, Enum):
    PRESS_BUTTON = "press_button"
    GRASP = "grasp"
    OPEN_DOOR = "open_door"
    RELEASE = "release"
    STOP = "stop"
    UNKNOWN = "unknown"


# Labels a button may carry. Floors are 1-99 or B1-B9; plus the usual extras.
_SPECIAL_BUTTONS = {"L", "G", "M", "P", "up", "down", "open", "close", "alarm", "call"}
_BUTTON_RE = re.compile(r"^(?:[1-9][0-9]?|B[1-9])$")
# Pressing these has consequences beyond the user's own trip: ask twice.
SENSITIVE_BUTTONS = {"alarm", "call"}


class IntentError(ValueError):
    pass


@dataclass
class Intent:
    task: Task
    target: str | None = None  # "7", "shopping bag", "door handle"
    part: str | None = None  # "handle" for a bag
    params: dict = field(default_factory=dict)  # e.g. {"direction": "pull", "handle_type": "lever"}
    context: str | None = None  # "elevator"
    needs_extra_confirmation: bool = False
    raw: str = ""

    def __post_init__(self):
        self.task = Task(self.task)
        self.validate()

    def validate(self) -> None:
        if self.task is Task.PRESS_BUTTON:
            if not self.target:
                raise IntentError("press_button needs a button label")
            if not (_BUTTON_RE.match(self.target) or self.target in _SPECIAL_BUTTONS):
                raise IntentError(f"unrecognised button label {self.target!r}")
            self.needs_extra_confirmation = self.needs_extra_confirmation or self.target in SENSITIVE_BUTTONS
        elif self.task is Task.GRASP:
            if not self.target or len(self.target) > 60:
                raise IntentError("grasp needs a short object description")
        elif self.task is Task.OPEN_DOOR:
            d = self.params.get("direction", "auto")
            h = self.params.get("handle_type", "auto")
            if d not in ("auto", "pull", "push"):
                raise IntentError(f"bad door direction {d!r}")
            if h not in ("auto", "lever", "knob", "push_bar"):
                raise IntentError(f"bad handle type {h!r}")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["task"] = self.task.value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Intent":
        allowed = {"task", "target", "part", "params", "context", "needs_extra_confirmation", "raw"}
        extra = set(d) - allowed
        if extra:
            raise IntentError(f"unexpected fields {sorted(extra)}")
        try:
            return cls(**d)
        except ValueError as e:  # bad Task value
            raise IntentError(str(e)) from e

    @staticmethod
    def json_schema() -> dict:
        return {
            "type": "object",
            "required": ["task"],
            "additionalProperties": False,
            "properties": {
                "task": {"enum": [t.value for t in Task]},
                "target": {"type": ["string", "null"]},
                "part": {"type": ["string", "null"]},
                "context": {"type": ["string", "null"]},
                "params": {
                    "type": "object",
                    "properties": {
                        "direction": {"enum": ["auto", "pull", "push"]},
                        "handle_type": {"enum": ["auto", "lever", "knob", "push_bar"]},
                    },
                },
            },
        }
