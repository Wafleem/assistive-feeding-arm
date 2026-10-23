"""Continuous, smooth face tracking for the SO-101 with the spoon kept level.

Replaces the step-and-plan pointing servo for centring: instead of asking a planner for a
pose every second (stop, plan, accelerate, stop), a 20 Hz loop streams small joint setpoints.

Each tick, from local forward kinematics of the commanded joints:
  task errors  e = [lateral miss, vertical miss]  of the lips from the spoon's line (m)
                   [spoon pitch]                  (rad, scaled by ``pitch_scale``; level = 0)
                   [reach - reach at start]       (m; centring doesn't move toward the face)
  joint rate   dq = -gain * DLS(J) e    over pan, shoulder lift, elbow, wrist flex
               (wrist roll is held: the bowl stays up)
  smoothing    a soft deadband on the miss (no hunting), per-joint velocity limits and an
               acceleration limit, so the arm glides and settles instead of stepping.
If the face is lost, the arm decelerates to a stop (still smoothly) and waits.

Pure: the session streams the returned setpoints (``SO101ViamRobot.stream_joints``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..robot_iface.so101_kinematics import fk as so101_fk
from ..robot_iface.so101_kinematics import joint_limits

ROLL = 4


@dataclass(frozen=True)
class TrackConfig:
    rate_hz: float = 20.0
    gain: float = 1.6  # 1/s: errors decay with a ~0.6 s time constant
    max_vel_deg: tuple = (30.0, 20.0, 20.0, 30.0)  # deg/s: pan, lift, elbow, flex
    max_acc_deg: float = 80.0  # deg/s^2 (smoothness)
    deadband_m: float = 0.006  # miss below this: no correction
    taper_m: float = 0.02  # correction ramps in over this much more miss (soft deadband)
    pitch_scale: float = 0.25  # m per rad: weight of spoon level vs. aim
    pitch_target_deg: float = 0.0  # level
    damping: float = 0.001  # damped least squares (J is ~0.01 m/deg; keep this well below it)
    reach_weight: float = 0.3  # holding the reach is a soft preference: aim and level come first
    spoon_dz: float = 0.0  # m: spoon line above the tool axis. TODO: measure
    limit_margin_deg: float = 3.0  # stay this far inside the joint limits


def _unit(v) -> np.ndarray:
    v = np.asarray(v, float)
    return v / (np.linalg.norm(v) or 1.0)


class LevelTracker:
    def __init__(self, cfg: TrackConfig | None = None, fk=so101_fk, limits=None):
        self.cfg = cfg or TrackConfig()
        self.fk = fk
        lim = joint_limits() if limits is None else np.asarray(limits, float)
        self.lo, self.hi = lim[:, 0] + self.cfg.limit_margin_deg, lim[:, 1] - self.cfg.limit_margin_deg
        self.q = None
        self.v = np.zeros(5)
        self.reach0 = None
        self.last = {}

    def reset(self, q_meas) -> None:
        self.q = np.asarray(q_meas, float).copy()
        self.v = np.zeros(5)
        p = self.fk(self.q)[:3, 3]
        self.reach0 = float(np.linalg.norm(p[:2]))

    def task_error(self, q, face) -> np.ndarray:
        """[lateral miss, vertical miss, scaled pitch error, reach error] at joints q."""
        T = self.fk(q)
        p, d = T[:3, 3] + np.array([0.0, 0.0, self.cfg.spoon_dz]), T[:3, 2]
        v = np.asarray(face, float) - p
        v_perp = v - np.dot(v, d) * d
        h = _unit(np.cross(d, [0.0, 0.0, 1.0]))  # sideways, horizontal
        u = np.cross(h, d)  # up, perpendicular to the spoon
        pitch = np.arcsin(np.clip(d[2], -1, 1)) - np.radians(self.cfg.pitch_target_deg)
        reach = np.linalg.norm(T[:3, 3][:2]) - self.reach0
        return np.array([np.dot(v_perp, h), np.dot(v_perp, u), self.cfg.pitch_scale * pitch,
                         self.cfg.reach_weight * reach])

    def _soft(self, e: np.ndarray) -> np.ndarray:
        """Soft deadband on the aim part (lateral, vertical): 0 inside, ramps to full outside."""
        c = self.cfg
        miss = float(np.linalg.norm(e[:2]))
        w = np.clip((miss - c.deadband_m) / c.taper_m, 0.0, 1.0)
        out = e.copy()
        out[:2] *= w * (miss - c.deadband_m) / miss if miss > c.deadband_m else 0.0
        return out

    def step(self, face, dt: float | None = None) -> np.ndarray:
        """Next joint setpoint (deg). ``face`` = lips in the base frame (m), or None if lost."""
        c = self.cfg
        dt = dt or 1.0 / c.rate_hz
        v_des = np.zeros(5)
        if face is not None:
            e = self.task_error(self.q, face)
            J = np.zeros((4, 4))
            for i in range(4):  # numeric Jacobian over pan, lift, elbow, flex (m or scaled rad per deg)
                dq = np.zeros(5)
                dq[i] = 0.05
                J[:, i] = (self.task_error(self.q + dq, face) - e) / 0.05
            es = self._soft(e)
            rate = -c.gain * J.T @ np.linalg.solve(J @ J.T + c.damping**2 * np.eye(4), es)
            v_des[:4] = np.clip(rate, -np.asarray(c.max_vel_deg), np.asarray(c.max_vel_deg))
            self.last = {"lateral_cm": round(e[0] * 100, 1), "vertical_cm": round(e[1] * 100, 1),
                         "pitch_deg": round(float(np.degrees(e[2] / c.pitch_scale)), 1),
                         "reach_cm": round(e[3] / c.reach_weight * 100, 1), "miss_cm": round(float(np.linalg.norm(e[:2])) * 100, 1)}
        dv = np.clip(v_des - self.v, -c.max_acc_deg * dt, c.max_acc_deg * dt)
        self.v = self.v + dv
        self.v[ROLL] = 0.0
        q_new = np.clip(self.q + self.v * dt, self.lo, self.hi)
        self.v = (q_new - self.q) / dt  # a joint at its limit stops
        self.q = q_new
        return self.q.copy()
