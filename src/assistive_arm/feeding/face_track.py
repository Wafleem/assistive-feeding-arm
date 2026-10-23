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
    max_vel_deg: tuple = (30.0, 20.0, 20.0, 15.0)  # deg/s: pan, lift, elbow, flex (flex gentle: it swings the camera)
    max_acc_deg: float = 80.0  # deg/s^2 (smoothness)
    deadband_m: float = 0.006  # miss below this: no correction
    taper_m: float = 0.02  # correction ramps in over this much more miss (soft deadband)
    pitch_scale: float = 0.5  # m per rad: weight of spoon level vs. aim (level wins)
    pitch_target_deg: float = 0.0  # tool pitch at which the SPOON is level (dashboard: Save spoon level)
    damping: float = 0.001  # damped least squares (J is ~0.01 m/deg; keep this well below it)
    reach_weight: float = 0.3  # centring: holding the reach is a soft preference (aim and level first)
    reach_weight_extend: float = 1.0  # extend/retreat: the reach IS the commanded motion
    spoon_dz: float = 0.0  # m: spoon line above the tool axis. TODO: measure
    limit_margin_deg: float = 3.0  # stay this far inside the joint limits
    # Where a LEVEL spoon can comfortably go (SO-101 FK: a level spoon tops out at ~0.37 m above
    # the arm base, and only in folded postures). Lips above z_cap: the spoon stays level at the
    # cap and the dashboard reports how far above reach they are.
    z_cap: float = 0.34  # m, base frame (raised from 0.32 on request: ~2 cm under the level-spoon maximum)
    # Lips above z_cap: the spoon may tip UP (food slides back toward the handle, no spill),
    # never down, by at most this much to reach higher lips and keep them in view.
    max_tip_up_deg: float = 12.0
    z_floor: float = 0.04  # m
    # Comfortable posture (deg): pan, lift, elbow, flex (roll is held). Keeps the arm out of the
    # stretched/folded extremes the unconstrained solution drifts toward.
    posture_lo: tuple = (-100.0, -97.0, -60.0, -60.0)
    posture_hi: tuple = (100.0, 10.0, 93.0, 80.0)


@dataclass(frozen=True)
class ExtendConfig:
    """Moving the spoon in toward the lips with the tracker (every number: safety lead review)."""

    approach_speed: float = 0.015  # m/s the reach target moves in
    retreat_speed: float = 0.025  # m/s back out (gentle: a fast back-out after a slow approach looked jumpy)
    stop_tip_distance: float = 0.20  # m spoon tip to lips: stop here. FIRST RUNS: 0.20
    min_tip_distance_floor: float = 0.03  # hard floor (person mode, AGENTS.md section 4)
    spoon_tip_from_tool: float = 0.21  # m from the tool frame to the spoon tip. TODO: measure
    aim_ok_cm: float = 1.5  # only move in while the lips are this close to the spoon line
    level_ok_deg: float = 2.0  # ... and the spoon is this level
    max_head_speed: float = 0.05  # m/s: faster => pause
    retreat_head_speed: float = 0.12  # m/s: faster => back out
    max_face_age: float = 0.4  # s without a face fix => back out
    hold_s: float = 8.0  # s at the stop distance, then back out

    def __post_init__(self):
        if self.stop_tip_distance < self.min_tip_distance_floor:
            raise ValueError(f"stop_tip_distance {self.stop_tip_distance} m is below the "
                             f"{self.min_tip_distance_floor} m floor")


def extend_decision(c: ExtendConfig, face_seen: bool, age: float, head_speed: float, miss_cm: float,
                    tilt_deg: float, tip: float, travelled: float, held_s: float) -> tuple[int, str]:
    """+1 move in, 0 hold, -1 back out; with a reason for the dashboard."""
    out = -1 if travelled > 1e-4 else 0
    if not face_seen or age > c.max_face_age:
        return out, "face lost: backing out" if out else "face lost: waiting"
    if head_speed > c.retreat_head_speed:
        return out, f"head moving fast ({head_speed:.2f} m/s): backing out"
    if tip < c.stop_tip_distance - 0.02:
        return -1, f"spoon tip {tip * 100:.0f} cm from the lips: too close, backing out"
    if tip <= c.stop_tip_distance + 0.005:
        if held_s < c.hold_s:
            return 0, f"at the stop distance ({tip * 100:.0f} cm), holding {held_s:.0f}/{c.hold_s:.0f} s"
        return -1, "hold time over: backing out"
    if miss_cm > c.aim_ok_cm or abs(tilt_deg) > c.level_ok_deg:
        return 0, f"re-centring before moving in (miss {miss_cm:.1f} cm, tilt {tilt_deg:+.1f} deg)"
    if head_speed > c.max_head_speed:
        return 0, f"waiting for a still head ({head_speed:.2f} m/s)"
    return 1, f"moving in: spoon tip {tip * 100:.0f} cm -> stop at {c.stop_tip_distance * 100:.0f} cm"


def _unit(v) -> np.ndarray:
    v = np.asarray(v, float)
    return v / (np.linalg.norm(v) or 1.0)


class LevelTracker:
    def __init__(self, cfg: TrackConfig | None = None, fk=so101_fk, limits=None):
        self.cfg = cfg or TrackConfig()
        self.fk = fk
        lim = joint_limits() if limits is None else np.asarray(limits, float)
        self._lim_lo, self._lim_hi = lim[:, 0] + self.cfg.limit_margin_deg, lim[:, 1] - self.cfg.limit_margin_deg
        self.lo, self.hi = self._lim_lo.copy(), self._lim_hi.copy()
        self.q = None
        self.v = np.zeros(5)
        self.reach0 = None
        self.reach_w = self.cfg.reach_weight
        self.last = {}

    def reset(self, q_meas) -> None:
        self.q = np.asarray(q_meas, float).copy()
        self.v = np.zeros(5)
        p = self.fk(self.q)[:3, 3]
        self.reach0 = float(np.linalg.norm(p[:2]))  # the reach target (extend moves it)
        self.reach_start = self.reach0
        # Comfort limits, but never a jump: a start outside them only stops motion further out.
        lo = np.maximum(self._lim_lo, np.r_[self.cfg.posture_lo, -1e9])
        hi = np.minimum(self._lim_hi, np.r_[self.cfg.posture_hi, 1e9])
        self.lo, self.hi = np.minimum(lo, self.q), np.maximum(hi, self.q)

    def _high(self, face) -> bool:
        return float(face[2]) > self.cfg.z_cap + self.cfg.spoon_dz

    def reachable_face(self, face, tool_pos=None) -> np.ndarray:
        """The aim point: the lips, with the height clamped to what the spoon can reach: a
        level spoon up to z_cap, plus the allowed tip-up tilt over the distance to the lips."""
        c = self.cfg
        f = np.asarray(face, float).copy()
        top = c.z_cap
        if tool_pos is not None:
            horiz = float(np.linalg.norm(f[:2] - np.asarray(tool_pos, float)[:2]))
            top += horiz * np.tan(np.radians(c.max_tip_up_deg))
        f[2] = float(np.clip(f[2], c.z_floor + c.spoon_dz, top + c.spoon_dz))
        return f

    def task_error(self, q, face) -> np.ndarray:
        """[lateral miss, vertical miss, scaled pitch error, reach error] at joints q.
        ``face`` None: no aim terms (used to back out level when the face is lost)."""
        T = self.fk(q)
        p, d = T[:3, 3] + np.array([0.0, 0.0, self.cfg.spoon_dz]), T[:3, 2]
        if face is None:
            pitch = np.arcsin(np.clip(d[2], -1, 1)) - np.radians(self.cfg.pitch_target_deg)
            reach = np.linalg.norm(T[:3, 3][:2]) - self.reach0
            return np.array([0.0, 0.0, self.cfg.pitch_scale * pitch, self.reach_w * reach])
        v = self.reachable_face(face, T[:3, 3]) - p
        v_perp = v - np.dot(v, d) * d
        h = _unit(np.cross(d, [0.0, 0.0, 1.0]))  # sideways, horizontal
        u = np.cross(h, d)  # up, perpendicular to the spoon
        pitch = np.arcsin(np.clip(d[2], -1, 1)) - np.radians(self.cfg.pitch_target_deg)
        if self._high(face):  # lips above a level spoon's reach: tip-up up to the limit is free
            up = np.radians(self.cfg.max_tip_up_deg)
            pitch = min(pitch, 0.0) + max(pitch - up, 0.0)  # penalise only tip-down or too much tip-up
        reach = np.linalg.norm(T[:3, 3][:2]) - self.reach0
        return np.array([np.dot(v_perp, h), np.dot(v_perp, u), self.cfg.pitch_scale * pitch, self.reach_w * reach])

    def _soft(self, e: np.ndarray) -> np.ndarray:
        """Soft deadband on the aim part (lateral, vertical): 0 inside, ramps to full outside."""
        c = self.cfg
        miss = float(np.linalg.norm(e[:2]))
        w = np.clip((miss - c.deadband_m) / c.taper_m, 0.0, 1.0)
        out = e.copy()
        out[:2] *= w * (miss - c.deadband_m) / miss if miss > c.deadband_m else 0.0
        return out

    @property
    def travelled(self) -> float:
        return self.reach0 - self.reach_start

    def tip_distance(self, face, spoon_tip_from_tool: float) -> float:
        """Spoon tip to lips (m) at the current setpoint."""
        p = self.fk(self.q)[:3, 3]
        return float(np.linalg.norm(np.asarray(face, float) - p)) - spoon_tip_from_tool

    def set_extending(self, on: bool) -> None:
        self.reach_w = self.cfg.reach_weight_extend if on else self.cfg.reach_weight

    def actual_travelled(self) -> float:
        """How far the setpoint really is from where tracking started (m), not the target."""
        return float(np.linalg.norm(self.fk(self.q)[:3, 3][:2])) - self.reach_start

    def advance(self, direction: int, dt: float, c: ExtendConfig) -> None:
        """Move the reach target in (+1) or back toward where tracking started (-1)."""
        if direction > 0:
            self.reach0 += c.approach_speed * dt
        elif direction < 0:
            self.reach0 = max(self.reach_start, self.reach0 - c.retreat_speed * dt)

    def step(self, face, dt: float | None = None, blind: bool = False) -> np.ndarray:
        """Next joint setpoint (deg). ``face`` = lips in the base frame (m), or None if lost
        (then hold still, unless ``blind``: keep levelling and following the reach target,
        which is how a retreat backs out without a face)."""
        c = self.cfg
        dt = dt or 1.0 / c.rate_hz
        v_des = np.zeros(5)
        if face is not None or blind:
            e = self.task_error(self.q, face)
            J = np.zeros((4, 4))
            for i in range(4):  # numeric Jacobian over pan, lift, elbow, flex (m or scaled rad per deg)
                dq = np.zeros(5)
                dq[i] = 0.05
                J[:, i] = (self.task_error(self.q + dq, face) - e) / 0.05
            es = self._soft(e)
            rate = -c.gain * J.T @ np.linalg.solve(J @ J.T + c.damping**2 * np.eye(4), es)
            v_des[:4] = np.clip(rate, -np.asarray(c.max_vel_deg), np.asarray(c.max_vel_deg))
            if face is None:
                face = self.fk(self.q)[:3, 3]  # for the report only
            self.last = {"lateral_cm": round(e[0] * 100, 1), "vertical_cm": round(e[1] * 100, 1),
                         "pitch_deg": round(float(np.degrees(e[2] / c.pitch_scale)), 1),  # tilt ERROR (0 inside an allowed band)
                         "tilt_deg": round(float(np.degrees(np.arcsin(np.clip(self.fk(self.q)[2, 2], -1, 1))) - c.pitch_target_deg), 1),
                         "reach_cm": round(e[3] / self.reach_w * 100, 1), "miss_cm": round(float(np.linalg.norm(e[:2])) * 100, 1),
                         "lips_above_reach_cm": round(max(0.0, float(face[2]) - float(self.reachable_face(face, self.fk(self.q)[:3, 3])[2])) * 100, 1),
                         "tip_up_allowed": bool(self._high(face)),
                         "lips_below_reach_cm": round(max(0.0, (c.z_floor + c.spoon_dz) - float(face[2])) * 100, 1)}
        dv = np.clip(v_des - self.v, -c.max_acc_deg * dt, c.max_acc_deg * dt)
        self.v = self.v + dv
        self.v[ROLL] = 0.0
        q_new = np.clip(self.q + self.v * dt, self.lo, self.hi)
        self.v = (q_new - self.q) / dt  # a joint at its limit stops
        self.q = q_new
        return self.q.copy()


def solve_level_pose(q_deg, reach_m: float, height_m: float, level_pitch_deg: float = 0.0, fk=so101_fk,
                     limits=None, posture=None, iters: int = 200) -> tuple[np.ndarray, bool]:
    """Shoulder, elbow and wrist flex for a tool at (reach, height) with the spoon at level, base
    pan and wrist roll unchanged. Returns (joints, reached within 5 mm / 1 deg)."""
    q = np.asarray(q_deg, float).copy()
    lim = joint_limits() if limits is None else np.asarray(limits, float)
    lo, hi = lim[:, 0] + 3, lim[:, 1] - 3
    if posture is not None:
        lo[1:4] = np.maximum(lo[1:4], posture[0]); hi[1:4] = np.minimum(hi[1:4], posture[1])
    target = np.array([reach_m, height_m, np.radians(level_pitch_deg)])
    w = np.array([1.0, 1.0, 0.5])

    def task(qq):
        T = fk(qq)
        return np.array([np.linalg.norm(T[:3, 3][:2]), T[2, 3], np.arcsin(np.clip(T[2, 2], -1, 1))])

    for _ in range(iters):
        e = (target - task(q)) * w
        if np.max(np.abs(e)) < 5e-5:
            break
        J = np.zeros((3, 3))
        for i, j in enumerate((1, 2, 3)):
            dq = np.zeros(5); dq[j] = 0.05
            J[:, i] = (task(q + dq) - task(q)) * w / 0.05
        q[1:4] = np.clip(q[1:4] + np.clip(J.T @ np.linalg.solve(J @ J.T + 1e-6 * np.eye(3), e), -5, 5), lo[1:4], hi[1:4])
    err = target - task(q)
    return q, bool(abs(err[0]) < 0.005 and abs(err[1]) < 0.005 and abs(np.degrees(err[2])) < 1.0)


def max_level_reach(q_deg, level_pitch_deg: float = 0.0, cfg: "TrackConfig | None" = None, fk=so101_fk,
                    extra_m: float = 0.45) -> float:
    """Largest tool reach (m from the base axis) at the current height with a level spoon, inside
    the tracker's comfort posture. Binary search on ``solve_level_pose``."""
    c = cfg or TrackConfig()
    T = fk(q_deg)
    r0, z0 = float(np.linalg.norm(T[:3, 3][:2])), float(T[2, 3])
    posture = (np.asarray(c.posture_lo[1:]), np.asarray(c.posture_hi[1:]))
    lo, hi, q = r0, r0 + extra_m, np.asarray(q_deg, float)
    if not solve_level_pose(q, r0, z0, level_pitch_deg, fk, posture=posture)[1]:
        return r0
    for _ in range(14):
        mid = 0.5 * (lo + hi)
        qm, ok = solve_level_pose(q, mid, z0, level_pitch_deg, fk, posture=posture)
        if ok:
            lo, q = mid, qm
        else:
            hi = mid
    return lo


def jog_level(q_deg, d_yaw_deg: float = 0.0, d_z_m: float = 0.0, level_pitch_deg: float = 0.0,
              fk=so101_fk, limits=None, iters: int = 200) -> tuple[np.ndarray, dict]:
    """Joint target for a debug jog that keeps the spoon level: yaw = base pan only (a pure
    rotation about the base axis, so level and reach are untouched); height = shoulder, elbow
    and wrist flex solved together for (height + d_z, same reach, spoon pitch = level).
    Returns (joints, report); report["ok"] is False if the target is outside the arm's reach."""
    q = np.asarray(q_deg, float).copy()
    lim = joint_limits() if limits is None else np.asarray(limits, float)
    q[0] = float(np.clip(q[0] + d_yaw_deg, lim[0, 0] + 3, lim[0, 1] - 3))
    T0 = fk(q)
    if d_z_m == 0.0:  # yaw only: a pure base rotation; nothing else moves
        return q, {"ok": True, "height_cm": round(float(T0[2, 3]) * 100, 1),
                   "reach_cm": round(float(np.linalg.norm(T0[:3, 3][:2])) * 100, 1),
                   "tilt_deg": round(float(np.degrees(np.arcsin(np.clip(T0[2, 2], -1, 1)))) - level_pitch_deg, 1),
                   "miss_cm": 0.0}
    target = np.array([np.linalg.norm(T0[:3, 3][:2]), T0[2, 3] + d_z_m, np.radians(level_pitch_deg)])

    def task(qq):
        T = fk(qq)
        return np.array([np.linalg.norm(T[:3, 3][:2]), T[2, 3], np.arcsin(np.clip(T[2, 2], -1, 1))])

    w = np.array([1.0, 1.0, 0.5])  # m, m, rad (0.5 m per rad)
    for _ in range(iters):
        e = (target - task(q)) * w
        if np.max(np.abs(e)) < 5e-5:
            break
        J = np.zeros((3, 3))
        for i, j in enumerate((1, 2, 3)):
            dq = np.zeros(5); dq[j] = 0.05
            J[:, i] = (task(q + dq) - task(q)) * w / 0.05
        step = J.T @ np.linalg.solve(J @ J.T + 1e-6 * np.eye(3), e)
        q[1:4] = np.clip(q[1:4] + np.clip(step, -5, 5), lim[1:4, 0] + 3, lim[1:4, 1] - 3)
    got = task(q)
    err = (target - got) * np.array([100, 100, np.degrees(1)])
    ok = abs(err[0]) < 0.5 and abs(err[1]) < 0.5 and abs(err[2]) < 1.0
    return q, {"ok": bool(ok), "height_cm": round(float(got[1]) * 100, 1), "reach_cm": round(float(got[0]) * 100, 1),
               "tilt_deg": round(float(np.degrees(got[2])) - level_pitch_deg, 1),
               "miss_cm": round(float(np.hypot(err[0], err[1])), 1)}
