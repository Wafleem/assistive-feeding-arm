"""Constant-velocity Kalman filter for the mouth position, with latency compensation.

The RoboMaster playbook, pointed at a mouth instead of an armour plate.

PLAN
* State x = [px, py, pz, vx, vy, vz] in the arm base frame; white-noise
  acceleration process model (sigma_accel); position measurements (sigma_meas).
* predict/update at the camera rate using the frame CAPTURE timestamp, not the
  processing time. Drop out-of-order frames.
* Gate outliers by Mahalanobis distance (chi-square, 3 dof): a mis-detected chin
  must never yank the spoon.
* ``predict_ahead(dt)`` returns where the mouth will be when a command issued now
  takes effect (exposure + transport + inference + planning + controller latency).

TUNING NOTES (to verify with recorded data)
* sigma_meas ~3-5 mm for RealSense depth at 40-60 cm.
* sigma_accel trades lag against velocity noise; the ReadyGate's "still"
  threshold must sit above the resulting velocity noise.
* Consider constant-acceleration or IMM if head motion is jerky.

DONE WHEN: tests/test_mouth_perception.py passes (convergence, outlier gating,
latency compensation beats the raw estimate).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class KalmanParams:
    sigma_accel: float = 0.3  # m/s^2, white-noise acceleration (process noise). TODO: fit from data
    sigma_meas: float = 0.004  # m, measurement noise per axis. TODO: measure on the RealSense
    gate_chi2: float = 16.27  # 3-dof chi-square at p=0.001
    init_pos_var: float = 0.01**2
    init_vel_var: float = 0.2**2


class ConstantVelocityKF:
    """State x = [px, py, pz, vx, vy, vz]."""

    def __init__(self, params: KalmanParams | None = None):
        self.p = params or KalmanParams()
        self.x: np.ndarray | None = None  # (6,)
        self.P = np.eye(6)
        self.t: float | None = None
        self.rejected = 0  # count of gated-out measurements

    @property
    def initialized(self) -> bool:
        return self.x is not None

    def reset(self, z: np.ndarray, t: float) -> None:
        """Initialise at position z (zero velocity) at time t."""
        raise NotImplementedError("TODO(mouth_perception): ConstantVelocityKF.reset")

    def predict(self, t: float) -> None:
        """Propagate state and covariance to time t (no-op if t <= self.t)."""
        raise NotImplementedError("TODO(mouth_perception): ConstantVelocityKF.predict")

    def update(self, z, t: float) -> bool:
        """Predict to t, then fuse measurement z. Return False if gated out or out of order."""
        raise NotImplementedError("TODO(mouth_perception): ConstantVelocityKF.update")

    def predict_ahead(self, dt: float) -> tuple[np.ndarray, np.ndarray]:
        """(position, velocity) dt seconds after the filter time, without changing the filter."""
        raise NotImplementedError("TODO(mouth_perception): ConstantVelocityKF.predict_ahead")
