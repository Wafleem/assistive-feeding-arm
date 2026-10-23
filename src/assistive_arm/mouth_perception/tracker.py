"""MouthTracker: detector observations in, filtered + latency-compensated MouthState out.

* Each MouthObservation (ignored below ``min_confidence``) goes into the KF.
* The face normal is smoothed (exponential average, renormalised).
* Open/closed uses hysteresis on ``open_ratio`` (no flicker at the threshold).
* ``state(now)`` returns the KF prediction at ``now + latency``, plus ``age`` =
  time since the last accepted measurement (stale => feeding must not advance).
* If the person moves while tracking is lost, the gate would keep rejecting the
  new position; after ``max_consecutive_rejects`` rejections (or ``reset_after``
  seconds without an accepted fix) the filter restarts at the new measurement.
* TODO(mouth_perception): measure ``latency`` end to end on the real system
  (COMPUTE.md latency table) and fit ``KalmanParams`` from recorded head motion.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .kalman import ConstantVelocityKF, KalmanParams
from .types import MouthObservation, MouthState


@dataclass
class TrackerParams:
    latency: float = 0.10  # s, capture -> arm motion. TODO: measure
    open_threshold: float = 0.35  # open_ratio above this = open
    open_hysteresis: float = 0.10  # must drop below threshold - hysteresis to count as closed
    normal_alpha: float = 0.3  # smoothing of the face normal
    min_confidence: float = 0.5
    max_consecutive_rejects: int = 5  # then re-initialise at the new position
    reset_after: float = 0.5  # s without an accepted fix: re-initialise on the next one
    kalman: KalmanParams | None = None


class MouthTracker:
    def __init__(self, params: TrackerParams | None = None):
        self.p = params or TrackerParams()
        self.kf = ConstantVelocityKF(self.p.kalman)
        self.normal: np.ndarray | None = None
        self.open_ratio = 0.0
        self.is_open = False
        self.confidence = 0.0
        self.last_accept: float | None = None
        self._rejects = 0

    def reset(self) -> None:
        self.__init__(self.p)

    def update(self, obs: MouthObservation | None) -> bool:
        """Fuse one observation. Return True if it was accepted."""
        if obs is None or obs.confidence < self.p.min_confidence:
            return False
        stale = self.last_accept is not None and obs.t - self.last_accept > self.p.reset_after
        if stale or self._rejects >= self.p.max_consecutive_rejects:
            self.kf = ConstantVelocityKF(self.p.kalman)  # lost and found again: start over here
            self._rejects = 0
        if not self.kf.update(obs.position, obs.t):
            self._rejects += 1
            return False
        self._rejects = 0
        n = np.asarray(obs.normal, float)
        n = n / (np.linalg.norm(n) or 1.0)
        if self.normal is None:
            self.normal = n
        else:
            m = (1 - self.p.normal_alpha) * self.normal + self.p.normal_alpha * n
            self.normal = m / (np.linalg.norm(m) or 1.0)
        self.open_ratio = float(obs.open_ratio)
        if self.open_ratio > self.p.open_threshold:
            self.is_open = True
        elif self.open_ratio < self.p.open_threshold - self.p.open_hysteresis:
            self.is_open = False
        self.confidence = float(obs.confidence)
        self.last_accept = float(obs.t)
        return True

    def state(self, now: float, compensate_latency: bool = True) -> MouthState | None:
        """Predicted mouth state for a command issued at ``now`` (None before the first fix)."""
        if not self.kf.initialized or self.last_accept is None:
            return None
        ahead = self.p.latency if compensate_latency else 0.0
        pos, vel = self.kf.predict_ahead(now + ahead - self.kf.t)
        return MouthState(
            t=now + ahead,
            position=pos,
            velocity=vel,
            normal=self.normal.copy(),
            open_ratio=self.open_ratio,
            is_open=self.is_open,
            confidence=self.confidence,
            age=max(0.0, now - self.last_accept),
        )
