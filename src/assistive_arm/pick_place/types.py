"""Pick-and-place data types (contract)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

StepKind = Literal["move", "gripper_open", "gripper_close", "verify_grip"]
StepStatus = Literal["ok", "failed", "skipped"]


@dataclass
class TaughtPose:
    """One taught follower pose. Joints in DEGREES, in the arm's joint order
    (shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll)."""

    name: str
    joints_deg: list[float]
    gripper_pct: float | None = None  # 0 = closed .. 100 = open, if recorded
    note: str = ""


@dataclass
class PoseLibrary:
    task: str
    robot: str  # e.g. "pi-arm2/follower-arm": poses are only valid on the arm they were taught on
    poses: dict[str, TaughtPose] = field(default_factory=dict)


@dataclass(frozen=True)
class Step:
    kind: StepKind
    pose: str | None = None  # TaughtPose name for "move"


@dataclass
class StepResult:
    step: Step
    status: StepStatus
    reason: str = ""
    duration_s: float = 0.0


@dataclass
class RunResult:
    ok: bool
    steps: list[StepResult] = field(default_factory=list)
    reason: str = ""
