"""When may the arm come close, and when must it leave.

ReadyGate (PLAN): ready only when the mouth is open AND the head is still
(speed < max_speed) AND tracking is confident and fresh, continuously for
``hold`` seconds. Any failure resets the timer. Expose a human-readable
``reason`` for the dashboard ("mouth closed", "head moving 0.06 m/s", ...).

HeadMotionMonitor (PLAN): call ``start`` when the final approach begins, then
``check`` every tick. Retreat (ok=False, with a reason) if: the head moves fast,
drifts more than max_drift from the anchor, turns more than max_turn_deg,
tracking is lost or stale, or (optionally) the mouth closes before the bite.
"""

from __future__ import annotations

from dataclasses import dataclass

from .types import MouthState


@dataclass
class ReadyParams:
    hold: float = 0.5  # s the mouth must be open and still
    max_speed: float = 0.04  # m/s "still"; must exceed the KF velocity noise
    min_confidence: float = 0.6
    max_age: float = 0.25  # s without a fresh measurement => not ready


class ReadyGate:
    def __init__(self, params: ReadyParams | None = None):
        self.p = params or ReadyParams()
        self.since: float | None = None
        self.reason = "no data"

    def update(self, s: MouthState | None, now: float) -> bool:
        raise NotImplementedError("TODO(mouth_perception): ReadyGate.update")


@dataclass
class RetreatParams:
    max_speed: float = 0.12  # m/s: a quick head movement
    max_drift: float = 0.04  # m from where the approach started
    max_turn_deg: float = 25.0
    max_age: float = 0.30  # s without a fresh measurement
    retreat_if_closed: bool = True


@dataclass
class MotionVerdict:
    ok: bool
    reason: str = ""


class HeadMotionMonitor:
    def __init__(self, params: RetreatParams | None = None):
        self.p = params or RetreatParams()
        self.anchor: MouthState | None = None

    def start(self, s: MouthState) -> None:
        self.anchor = s

    def check(self, s: MouthState | None) -> MotionVerdict:
        raise NotImplementedError("TODO(mouth_perception): HeadMotionMonitor.check")
