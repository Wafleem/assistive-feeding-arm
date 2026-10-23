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
    gate_chi2: float = 16.27  # 3-dof chi-square at p=0.001: beyond this, down-weight (not reject)
    hard_gate_chi2: float = 400.0  # beyond this (~20 sigma), reject outright
    max_extrapolate_s: float = 0.25  # predict_ahead uses the velocity for at most this long
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
        self.x = np.concatenate([np.asarray(z, float).reshape(3), np.zeros(3)])
        self.P = np.diag([self.p.init_pos_var] * 3 + [self.p.init_vel_var] * 3)
        self.t = float(t)

    def _F_Q(self, dt: float) -> tuple[np.ndarray, np.ndarray]:
        F = np.eye(6)
        F[:3, 3:] = dt * np.eye(3)
        q = self.p.sigma_accel**2
        Q = np.zeros((6, 6))
        Q[:3, :3] = dt**4 / 4 * q * np.eye(3)
        Q[:3, 3:] = Q[3:, :3] = dt**3 / 2 * q * np.eye(3)
        Q[3:, 3:] = dt**2 * q * np.eye(3)
        return F, Q

    def predict(self, t: float) -> None:
        """Propagate state and covariance to time t (no-op if t <= self.t)."""
        if self.x is None or self.t is None or t <= self.t:
            return
        F, Q = self._F_Q(t - self.t)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        self.t = float(t)

    def update(self, z, t: float, R: np.ndarray | None = None) -> bool:
        """Predict to t, then fuse measurement z (covariance R, default sigma_meas^2 I).
        Mildly surprising measurements are down-weighted (Huber), wild ones rejected.
        Return False if rejected or out of order."""
        z = np.asarray(z, float).reshape(3)
        R = self.p.sigma_meas**2 * np.eye(3) if R is None else np.asarray(R, float)
        if self.x is None:
            self.reset(z, t)
            return True
        if t < self.t:
            return False  # out of order: a newer frame was already fused
        self.predict(t)
        H = np.hstack([np.eye(3), np.zeros((3, 3))])
        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        d2 = float(y @ np.linalg.solve(S, y))
        if d2 > self.p.hard_gate_chi2:
            self.rejected += 1
            return False
        if d2 > self.p.gate_chi2:  # Huber: trust it less the more it surprises us
            R = R * (d2 / self.p.gate_chi2)
            S = H @ self.P @ H.T + R
        S_inv = np.linalg.inv(S)
        K = self.P @ H.T @ S_inv
        self.x = self.x + K @ y
        I_KH = np.eye(6) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T  # Joseph form
        return True

    def predict_ahead(self, dt: float) -> tuple[np.ndarray, np.ndarray]:
        """(position, velocity) dt seconds after the filter time, without changing the filter."""
        if self.x is None:
            raise RuntimeError("filter not initialised")
        dt_v = float(np.clip(dt, -self.p.max_extrapolate_s, self.p.max_extrapolate_s))  # no drifting off when stale
        return self.x[:3] + dt_v * self.x[3:], self.x[3:].copy()
