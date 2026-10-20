"""Runs the state machine with injected section components (so it is testable with fakes).

PLAN
* ``FeedingMachine(robot, food, grasp, tracker, mouth_detector, feeding_cfg, ctx, say)``.
* ``handle(intent)`` from the language layer: FEED starts a bite cycle; STOP aborts from
  ANY state immediately (robot.stop(), then retreat if near the face); PAUSE/RESUME;
  ADJUST updates feeding speeds/bite size; DONE parks the arm.
* ``step()`` advances one state, enforcing TIMEOUTS; every transition is logged with
  its reason and timestamp (the dashboard shows it; the eval reads it).
* Abort paths: a failure in any state ends in RETREAT or IDLE with a spoken reason,
  never in an unknown pose near the face.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .states import S


@dataclass
class Transition:
    t: float
    src: S
    dst: S
    event: str
    reason: str = ""


@dataclass
class FeedingMachine:
    state: S = S.IDLE
    log: list[Transition] = field(default_factory=list)

    def handle(self, intent) -> None:
        raise NotImplementedError("TODO(orchestration): FeedingMachine.handle")

    def step(self) -> S:
        raise NotImplementedError("TODO(orchestration): FeedingMachine.step")
