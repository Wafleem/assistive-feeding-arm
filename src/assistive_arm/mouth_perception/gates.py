"""When may the arm come close, and when must it leave.

ReadyGate: ready only when the mouth is open AND the head is still
(speed < max_speed) AND tracking is confident and fresh, continuously for
``hold`` seconds. Any failure resets the timer. ``reason`` is human-readable
for the dashboard ("mouth closed", "head moving 0.06 m/s", ...).

HeadMotionMonitor: call ``start`` when the final approach begins, then
``check`` every tick. Retreat (ok=False, with a reason) if: tracking is lost or
stale, the head moves fast, drifts more than max_drift from the anchor, turns
more than max_turn_deg, or the mouth closes before the bite.
TODO(mouth_perception): tune the numbers on recorded real-face data with the
safety lead (every number here is a safety number, see AGENTS.md section 4).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

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

    def _blocker(self, s: MouthState | None) -> str:
        if s is None:
            return "no data"
        if s.age > self.p.max_age:
            return "tracking stale"
        if s.confidence < self.p.min_confidence:
            return f"low confidence {s.confidence:.2f}"
        if not s.is_open:
            return "mouth closed"
        if s.speed > self.p.max_speed:
            return f"head moving {s.speed:.2f} m/s"
        return ""

    def update(self, s: MouthState | None, now: float) -> bool:
        blocker = self._blocker(s)
        if blocker:
            self.since, self.reason = None, blocker
            return False
        if self.since is None:
            self.since = now
        held = now - self.since
        if held >= self.p.hold:
            self.reason = "ready"
            return True
        self.reason = f"hold still ({held:.1f}/{self.p.hold:.1f} s)"
        return False


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
        if s is None:
            return MotionVerdict(False, "tracking lost")
        if s.age > self.p.max_age:
            return MotionVerdict(False, "tracking stale")
        if s.speed > self.p.max_speed:
            return MotionVerdict(False, f"head moving fast ({s.speed:.2f} m/s)")
        if self.anchor is not None:
            drift = float(np.linalg.norm(s.position - self.anchor.position))
            if drift > self.p.max_drift:
                return MotionVerdict(False, f"head moved {drift * 100:.1f} cm")
            cos = float(np.clip(np.dot(_unit(s.normal), _unit(self.anchor.normal)), -1.0, 1.0))
            turn = float(np.degrees(np.arccos(cos)))
            if turn > self.p.max_turn_deg:
                return MotionVerdict(False, f"head turned {turn:.0f} deg")
        if self.p.retreat_if_closed and not s.is_open:
            return MotionVerdict(False, "mouth closed")
        return MotionVerdict(True)


def _unit(v) -> np.ndarray:
    v = np.asarray(v, float)
    return v / (np.linalg.norm(v) or 1.0)
