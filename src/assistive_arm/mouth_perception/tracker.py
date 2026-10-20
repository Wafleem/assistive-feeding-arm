"""MouthTracker: detector observations in, filtered + latency-compensated MouthState out.

PLAN
* Feed each MouthObservation (ignore below min_confidence) into the KF.
* Smooth the face normal (exponential average, renormalise).
* Open/closed with hysteresis on open_ratio (no flicker at the threshold).
* ``state(now)`` returns the KF prediction at now + latency, plus ``age`` =
  time since the last accepted measurement (stale => feeding must not advance).
* Measure ``latency`` end to end on the real system (COMPUTE.md latency table).
"""

from __future__ import annotations

from dataclasses import dataclass

from .kalman import ConstantVelocityKF, KalmanParams
from .types import MouthObservation, MouthState


@dataclass
class TrackerParams:
    latency: float = 0.10  # s, capture -> arm motion. TODO: measure
    open_threshold: float = 0.35  # open_ratio above this = open
    open_hysteresis: float = 0.10  # must drop below threshold - hysteresis to count as closed
    normal_alpha: float = 0.3  # smoothing of the face normal
    min_confidence: float = 0.5
    kalman: KalmanParams | None = None


class MouthTracker:
    def __init__(self, params: TrackerParams | None = None):
        self.p = params or TrackerParams()
        self.kf = ConstantVelocityKF(self.p.kalman)

    def update(self, obs: MouthObservation | None) -> bool:
        """Fuse one observation. Return True if it was accepted."""
        raise NotImplementedError("TODO(mouth_perception): MouthTracker.update")

    def state(self, now: float, compensate_latency: bool = True) -> MouthState | None:
        """Predicted mouth state for a command issued at ``now`` (None before the first fix)."""
        raise NotImplementedError("TODO(mouth_perception): MouthTracker.state")
