"""Intent -> an explicit, inspectable plan of perceive / move / skill steps.

The plan is data so it can be read back to the user ("I'll look for button 7,
move closer, then press it") and logged next to the outcome.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from ..intent.schema import Intent, Task

StepKind = Literal["clear_view", "locate", "observe", "skill", "stop", "ask"]


@dataclass
class Step:
    kind: StepKind
    query: str | None = None  # perception query for locate
    cameras: tuple[str, ...] = ()
    skill: str | None = None
    params: dict = field(default_factory=dict)
    say: str = ""


@dataclass
class TaskPlan:
    intent: Intent
    steps: list[Step]

    def describe(self) -> str:
        return " then ".join(s.say for s in self.steps if s.say)


def perception_query(intent: Intent) -> str | None:
    if intent.task is Task.PRESS_BUTTON:
        return f"elevator button {intent.target}" if intent.context == "elevator" or intent.context is None else f"{intent.context} button {intent.target}"
    if intent.task is Task.GRASP:
        return f"{intent.target} {intent.part}" if intent.part else intent.target
    if intent.task is Task.OPEN_DOOR:
        return "door handle"
    return None


def support_query(intent: Intent) -> str | None:
    """The surface the target is mounted on; its plane gives the approach direction."""
    if intent.task is Task.PRESS_BUTTON:
        return "elevator panel"
    if intent.task is Task.OPEN_DOOR:
        return "door"
    return None  # bag handles are grasped top-down


def plan(intent: Intent, coarse=("scene",), fine=("wrist",)) -> TaskPlan:
    t = intent.task
    if t is Task.STOP:
        return TaskPlan(intent, [Step("stop", say="stopping")])
    if t is Task.UNKNOWN:
        return TaskPlan(intent, [Step("ask", say=intent.params.get("question", "Sorry?"))])
    if t is Task.RELEASE:
        return TaskPlan(intent, [Step("skill", skill="release", say="letting go")])

    q = perception_query(intent)
    sup = {"support": support_query(intent)}
    view = "top" if t is Task.GRASP else "front"
    skill = {Task.PRESS_BUTTON: "press_button", Task.GRASP: "grasp_handle", Task.OPEN_DOOR: "open_door"}[t]
    return TaskPlan(
        intent,
        [
            # The arm is the biggest occluder of the wheelchair camera: park it first.
            Step("clear_view"),
            Step("locate", query=q, cameras=tuple(coarse) + tuple(fine), params=sup, say=f"look for the {q}"),
            Step("observe", params={"view": view, "distance": 0.25}, say="move closer"),
            Step("locate", query=q, cameras=tuple(fine) + tuple(coarse), params=sup, say="check again up close"),
            Step("skill", skill=skill, params=dict(intent.params), say=skill.replace("_", " ")),
        ],
    )
