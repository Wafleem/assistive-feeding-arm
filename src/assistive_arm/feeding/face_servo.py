"""Visual servoing toward a person's face on the SO-101 bench rig (spoon fixed in the gripper,
webcam on top of the gripper).

Motion is deliberately constrained, because it happens at a face:
* Two poses are TAUGHT with the leader arm: ``look`` (turned toward the person, far
  enough away) and ``near`` (the end of the approach path). The arm only ever moves on
  the straight joint-space line between them (progress ``s`` in [0, 1]) plus small
  centring offsets on the base pan and the wrist flex. No inverse kinematics near the face.
* Centring: an image Jacobian (pixels per degree of pan / wrist flex), measured by
  small probe moves (``ImageJacobian.from_probes``), turns the pixel error between the
  lips and the spoon's line of sight into joint steps, clamped per step and in total.
* Approach: ``s`` grows only while the face is centred, tracked, still and facing the
  camera, and only until the spoon tip is ``stop_tip_distance`` from the lips.
* Anything else => retreat to ``look`` (s = 0, offsets kept) or hold.

Pure logic: the caller measures, this decides, the caller moves (scripts/so101_face_servo.py).
Every number in ``FaceServoConfig`` is a safety number: review with the safety lead.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..mouth_perception.types import MouthState

PAN, LIFT, ELBOW, FLEX, ROLL = range(5)  # SO-101 joint order (degrees)


@dataclass(frozen=True)
class FaceServoConfig:
    # Where the spoon points, relative to the camera (camera frame, m): the lens sits ~28 mm
    # above and ~8 mm right of the spoon, and the spoon tip is ~120 mm ahead of the lens.
    spoon_offset_cam: tuple[float, float] = (-0.008, 0.028)
    spoon_ahead_of_lens: float = 0.12
    # Centring
    center_tol_px: float = 25.0  # |error| below this = centred (approach allowed)
    deadband_px: float = 8.0  # no correction below this
    gain: float = 0.6  # fraction of the error corrected per step
    max_step_deg: float = 3.0  # per centring step, per joint
    max_offset_deg: tuple[float, float] = (30.0, 25.0)  # total |pan|, |flex| offset from the taught path
    # Approach
    stop_tip_distance: float = 0.20  # m spoon tip to lips: stop here. FIRST RUNS: 0.20. Never below the floor.
    min_tip_distance_floor: float = 0.03  # m, hard floor (person mode standoff, AGENTS.md section 4)
    approach_ds: float = 0.05  # progress per approach step along look -> near
    # Readiness / retreat
    min_frontalness: float = 0.6
    max_head_speed: float = 0.05  # m/s: faster => no approach; above retreat_head_speed => retreat
    retreat_head_speed: float = 0.12
    max_age: float = 0.4  # s since the last face fix; older during approach => retreat
    joint_speed_degs: float = 20.0  # arm speed cap while servoing (module floor is 12)

    def __post_init__(self):
        if self.stop_tip_distance < self.min_tip_distance_floor:
            raise ValueError(f"stop_tip_distance {self.stop_tip_distance} m is below the "
                             f"{self.min_tip_distance_floor} m floor")


@dataclass
class ImageJacobian:
    """[du, dv] (px) = J @ [d_pan, d_flex] (deg), measured at the look pose."""

    J: np.ndarray  # (2, 2)

    @classmethod
    def from_probes(cls, du_dv_pan, du_dv_flex, probe_deg: float) -> "ImageJacobian":
        """Pixel shift of the lips after +probe_deg on pan, and after +probe_deg on flex."""
        J = np.column_stack([np.asarray(du_dv_pan, float), np.asarray(du_dv_flex, float)]) / probe_deg
        if abs(np.linalg.det(J)) < 1e-3:
            raise ValueError(f"probe moves did not move the face in independent directions: J={J.tolist()}")
        return cls(J)

    def to_dict(self) -> dict:
        return {"J_px_per_deg": self.J.tolist()}

    @classmethod
    def from_dict(cls, d) -> "ImageJacobian":
        return cls(np.asarray(d["J_px_per_deg"], float))


@dataclass
class FaceView:
    """What the camera says about the face this tick (from mouth_perception)."""

    lips_px: tuple[float, float]  # image position of the lips centre
    camera_distance: float  # m, lens to lips
    frontalness: float  # 1 = facing the camera
    image_size: tuple[int, int]  # (width, height)
    fx: float
    fy: float


@dataclass
class ServoCommand:
    kind: str  # "hold" | "center" | "approach" | "retreat" | "done"
    d_offset: np.ndarray = field(default_factory=lambda: np.zeros(2))  # [d_pan, d_flex] degrees
    ds: float = 0.0
    reason: str = ""


def target_pixel(view: FaceView, cfg: FaceServoConfig) -> np.ndarray:
    """Where the lips must appear for the spoon to point at them (the spoon line seen from the lens)."""
    w, h = view.image_size
    d = max(view.camera_distance, 0.05)
    ox, oy = cfg.spoon_offset_cam
    return np.array([w / 2 + view.fx * ox / d, h / 2 + view.fy * oy / d])


def tip_distance(view: FaceView, cfg: FaceServoConfig) -> float:
    return view.camera_distance - cfg.spoon_ahead_of_lens


def joint_target(look, near, s: float, offset) -> np.ndarray:
    """Joints on the taught path at progress s, plus the centring offsets on pan and flex."""
    look, near = np.asarray(look, float), np.asarray(near, float)
    q = look + float(np.clip(s, 0.0, 1.0)) * (near - look)
    q[PAN] += offset[0]
    q[FLEX] += offset[1]
    return q


class FaceServoController:
    def __init__(self, jac: ImageJacobian, cfg: FaceServoConfig | None = None, approach: bool = True):
        self.jac, self.cfg, self.approach = jac, cfg or FaceServoConfig(), approach
        self.s = 0.0
        self.offset = np.zeros(2)
        self.approaching = False  # set once s > 0: from then on, failures mean retreat

    def _centring(self, err: np.ndarray) -> np.ndarray:
        d = -self.cfg.gain * np.linalg.solve(self.jac.J, err)
        d = np.clip(d, -self.cfg.max_step_deg, self.cfg.max_step_deg)
        lim = np.asarray(self.cfg.max_offset_deg)
        return np.clip(self.offset + d, -lim, lim) - self.offset

    def step(self, view: FaceView | None, state: MouthState | None) -> ServoCommand:
        c = self.cfg
        if view is None or state is None or state.age > c.max_age:
            return self._fail("face lost")
        if state.speed > c.retreat_head_speed:
            return self._fail(f"head moving fast ({state.speed:.2f} m/s)")
        if view.frontalness < c.min_frontalness:
            return self._fail(f"face turned away (facing {view.frontalness:.2f})")
        tip = tip_distance(view, c)
        if tip < c.stop_tip_distance - 0.02:
            return self._fail(f"spoon tip {tip * 100:.0f} cm from the lips: closer than the stop distance")

        err = np.asarray(view.lips_px, float) - target_pixel(view, c)
        if np.max(np.abs(err)) > c.deadband_px:
            d = self._centring(err)
            if np.any(np.abs(d) > 1e-3):
                self.offset = self.offset + d
                return ServoCommand("center", d, reason=f"error ({err[0]:+.0f}, {err[1]:+.0f}) px")
            if np.max(np.abs(err)) > c.center_tol_px:
                return ServoCommand("hold", reason="centring offset at its limit: re-teach the look pose")
        if not self.approach:
            return ServoCommand("hold", reason="centred (approach disabled)")
        if tip <= c.stop_tip_distance:
            return ServoCommand("done", reason=f"spoon tip {tip * 100:.0f} cm from the lips")
        if np.max(np.abs(err)) > c.center_tol_px:
            return ServoCommand("hold", reason="not centred yet")
        if state.speed > c.max_head_speed:
            return ServoCommand("hold", reason=f"waiting for a still head ({state.speed:.2f} m/s)")
        if self.s >= 1.0:
            return ServoCommand("hold", reason="end of the taught path (teach a nearer 'near' pose)")
        ds = min(c.approach_ds, 1.0 - self.s)
        self.s += ds
        self.approaching = True
        return ServoCommand("approach", ds=ds, reason=f"spoon tip {tip * 100:.0f} cm -> stop at "
                                                      f"{c.stop_tip_distance * 100:.0f} cm")

    def _fail(self, reason: str) -> ServoCommand:
        if self.approaching and self.s > 0:
            self.s, self.approaching = 0.0, False
            return ServoCommand("retreat", reason=reason)
        return ServoCommand("hold", reason=reason)


# --------------------------------------------------------------------------------------------
# Pointing servo: no calibration. The face's 3D position (camera + frame system) and the
# tool's pose give the direction from the gripper to the lips; the arm's own planner
# (MoveToPosition) turns the spoon toward it in small steps, and (extend) moves along it.
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PointingConfig:
    aligned_deg: float = 3.0  # spoon within this angle of the lips = centred
    max_turn_deg: float = 8.0  # largest re-aim per step
    approach_step: float = 0.03  # m per extend step
    spoon_tip_from_tool: float = 0.21  # m from the tool frame to the spoon tip along tool +Z. TODO: measure
    stop_tip_distance: float = 0.20  # m spoon tip to lips: stop here. FIRST RUNS: 0.20
    min_tip_distance_floor: float = 0.03  # hard floor (person mode, AGENTS.md section 4)
    max_head_speed: float = 0.05  # m/s: faster => no approach
    retreat_head_speed: float = 0.12  # m/s: faster during an approach => retreat
    max_age: float = 0.6  # s since the last face fix (a few dropped frames are fine)
    position_leeway_mm: float = 8.0  # planner leeway on the tool position while re-aiming
    orientation_leeway: float = 0.03  # planner leeway on each orientation-vector component (~2 deg)
    max_joint_jump_deg: float = 25.0  # a single planned step moving any joint more than this => stop
    joint_speed_degs: float = 20.0
    # Level spoon (don't spill): aim by base pan + raising/lowering the gripper, never by tilting.
    level_spoon: bool = True
    level_tol_deg: float = 2.0  # spoon pitch within this of horizontal = level
    aim_tol_m: float = 0.015  # level mode: the spoon line passes within this of the lips = centred
    max_dz_step: float = 0.03  # m height change per step
    spoon_dz: float = 0.0  # m: spoon line height above the tool axis (taped in the fixed jaw). TODO: measure
    roll_leeway_deg: float = 4.0  # wrist roll held at its starting value (bowl up) within this
    level_pitch_deg: float = 0.0  # tool pitch at which the SPOON is level (saved by "Save spoon level")

    def __post_init__(self):
        if self.stop_tip_distance < self.min_tip_distance_floor:
            raise ValueError(f"stop_tip_distance {self.stop_tip_distance} m is below the "
                             f"{self.min_tip_distance_floor} m floor")


@dataclass
class PointCommand:
    kind: str  # "hold" | "point" | "approach" | "retreat" | "done"
    position: np.ndarray | None = None  # tool position target (m, base frame)
    direction: np.ndarray | None = None  # tool +Z target (unit, base frame)
    reason: str = ""
    angle_deg: float = 0.0
    tip_distance: float = 0.0


def _unit(v) -> np.ndarray:
    v = np.asarray(v, float)
    return v / (np.linalg.norm(v) or 1.0)


def turn_toward(a, b, max_deg: float) -> np.ndarray:
    """Rotate unit vector a toward b by at most max_deg (in their common plane)."""
    a, b = _unit(a), _unit(b)
    ang = float(np.degrees(np.arccos(np.clip(np.dot(a, b), -1, 1))))
    if ang <= max_deg or ang < 1e-6:
        return b
    t = max_deg / ang
    w = np.radians(ang)
    return _unit((np.sin((1 - t) * w) * a + np.sin(t * w) * b) / np.sin(w))


def _rot_z(v, a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    v = np.asarray(v, float)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1], v[2]])


def _wrap(a: float) -> float:
    return float((a + np.pi) % (2 * np.pi) - np.pi)


def arm_plane_normal(tool_pos) -> np.ndarray:
    """Normal of the vertical plane through the base axis and the tool (SO-101: the tool axis
    always lies in this plane, because the arm has no wrist yaw)."""
    az = np.arctan2(tool_pos[1], tool_pos[0])
    return np.array([-np.sin(az), np.cos(az), 0.0])


def aim_pan_tilt(tool_pos, tool_z, face, max_pan_deg: float, max_tilt_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """Next aiming step that a 5-DOF pan + planar arm (no wrist yaw, like the SO-101) can
    actually reach: swing the whole tool pose about the base axis toward the face's azimuth
    (base pan, <= max_pan_deg), then tilt within the arm's vertical plane toward the face
    (<= max_tilt_deg). Turning sideways while holding the tool position still is
    impossible for this arm (the planner reports zero IK solutions)."""
    tool_pos, face = np.asarray(tool_pos, float), np.asarray(face, float)
    az_plane = np.arctan2(tool_pos[1], tool_pos[0])
    az_face = np.arctan2(face[1], face[0])
    d_az = float(np.clip(_wrap(az_face - az_plane), -np.radians(max_pan_deg), np.radians(max_pan_deg)))
    p_new, z_rot = _rot_z(tool_pos, d_az), _rot_z(_unit(tool_z), d_az)
    remaining = _wrap(az_face - (az_plane + d_az))
    face_in_plane = _rot_z(face, -remaining)  # same radius and height, in the new plane
    return p_new, turn_toward(z_rot, face_in_plane - p_new, max_tilt_deg)


def spoon_tilt_deg(tool_z) -> float:
    """Pitch of the spoon axis from horizontal (+ = tip up)."""
    return float(np.degrees(np.arcsin(np.clip(_unit(tool_z)[2], -1, 1))))


def line_miss(tool_pos, direction, face) -> float:
    """Distance from the lips to the spoon's line (m)."""
    v = np.asarray(face, float) - np.asarray(tool_pos, float)
    d = _unit(direction)
    return float(np.linalg.norm(v - np.dot(v, d) * d))


def aim_level(tool_pos, tool_z, face, max_pan_deg: float, max_dz: float, max_tilt_deg: float,
              spoon_dz: float = 0.0, level_pitch_deg: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Next aiming step that keeps the spoon level: swing about the base axis toward the face's
    azimuth (<= max_pan_deg), move the gripper up/down toward lip height (<= max_dz), and bring
    the spoon axis to horizontal (<= max_tilt_deg per step if it starts tilted). Reachable for
    the SO-101: the target stays in the arm's vertical plane."""
    tool_pos, face = np.asarray(tool_pos, float), np.asarray(face, float)
    az_plane = np.arctan2(tool_pos[1], tool_pos[0])
    az_face = np.arctan2(face[1], face[0])
    d_az = float(np.clip(_wrap(az_face - az_plane), -np.radians(max_pan_deg), np.radians(max_pan_deg)))
    p_new = _rot_z(tool_pos, d_az)
    p_new[2] += float(np.clip(face[2] - (tool_pos[2] + spoon_dz), -max_dz, max_dz))
    az_new = az_plane + d_az
    lp = np.radians(level_pitch_deg)  # the tool pitch at which the spoon itself is level
    level = np.array([np.cos(az_new) * np.cos(lp), np.sin(az_new) * np.cos(lp), np.sin(lp)])
    return p_new, turn_toward(_rot_z(_unit(tool_z), d_az), level, max_tilt_deg)


class PointingServo:
    def __init__(self, cfg: PointingConfig | None = None, approach: bool = False):
        self.cfg, self.approach = cfg or PointingConfig(), approach
        self.travelled = 0.0  # m moved toward the face (retreat undoes it)

    def step(self, tool_pos, tool_z, face, state: MouthState | None) -> PointCommand:
        c = self.cfg
        if face is None or state is None or state.age > c.max_age:
            return self._fail("face lost")
        if state.speed > c.retreat_head_speed:
            return self._fail(f"head moving fast ({state.speed:.2f} m/s)")
        tool_pos, tool_z, face = np.asarray(tool_pos, float), _unit(tool_z), np.asarray(face, float)
        to_face = face - tool_pos
        d = _unit(to_face)
        tip = float(np.linalg.norm(to_face)) - c.spoon_tip_from_tool
        ang = float(np.degrees(np.arccos(np.clip(np.dot(tool_z, d), -1, 1))))
        if tip < c.stop_tip_distance - 0.02:
            return self._fail(f"spoon tip {tip * 100:.0f} cm from the lips: inside the stop distance", tip, ang)
        if c.level_spoon:
            return self._step_level(tool_pos, tool_z, face, state, tip, ang)
        if ang > c.aligned_deg:
            p_new, z_new = aim_pan_tilt(tool_pos, tool_z, face, c.max_turn_deg, c.max_turn_deg)
            pan = np.degrees(_wrap(np.arctan2(p_new[1], p_new[0]) - np.arctan2(tool_pos[1], tool_pos[0])))
            return PointCommand("point", p_new, z_new, f"re-aim {ang:.1f} deg (pan {pan:+.1f})", ang, tip)
        if not self.approach:
            return PointCommand("hold", reason=f"centred ({ang:.1f} deg)", angle_deg=ang, tip_distance=tip)
        if tip <= c.stop_tip_distance + 0.005:  # within 5 mm: no zero-length steps
            return PointCommand("done", reason=f"spoon tip {tip * 100:.0f} cm from the lips", angle_deg=ang,
                                tip_distance=tip)
        if state.speed > c.max_head_speed:
            return PointCommand("hold", reason=f"waiting for a still head ({state.speed:.2f} m/s)", angle_deg=ang,
                                tip_distance=tip)
        step = min(c.approach_step, tip - c.stop_tip_distance)
        self.travelled += step
        n = arm_plane_normal(tool_pos)
        d = _unit(d - np.dot(d, n) * n)  # stay in the arm's plane (reachable)
        return PointCommand("approach", tool_pos + step * d, d,
                            f"extend {step * 100:.0f} cm (tip {tip * 100:.0f} -> stop at {c.stop_tip_distance * 100:.0f} cm)",
                            ang, tip)

    def _step_level(self, tool_pos, tool_z, face, state, tip, ang) -> PointCommand:
        c = self.cfg
        spoon_pos = tool_pos + np.array([0.0, 0.0, c.spoon_dz])
        miss, tilt = line_miss(spoon_pos, tool_z, face), spoon_tilt_deg(tool_z) - c.level_pitch_deg
        info = dict(angle_deg=ang, tip_distance=tip)
        if miss > c.aim_tol_m or abs(tilt) > c.level_tol_deg:
            p_new, z_new = aim_level(tool_pos, tool_z, face, c.max_turn_deg, c.max_dz_step, c.max_turn_deg, c.spoon_dz,
                                     c.level_pitch_deg)
            pan = np.degrees(_wrap(np.arctan2(p_new[1], p_new[0]) - np.arctan2(tool_pos[1], tool_pos[0])))
            return PointCommand("point", p_new, z_new, f"aim: miss {miss * 100:.1f} cm, tilt {tilt:+.1f} deg "
                                f"(pan {pan:+.1f} deg, height {(p_new[2] - tool_pos[2]) * 100:+.1f} cm)", **info)
        if not self.approach:
            return PointCommand("hold", reason=f"centred, level (miss {miss * 100:.1f} cm, tilt {tilt:+.1f} deg)", **info)
        if tip <= c.stop_tip_distance + 0.005:  # within 5 mm: no zero-length steps
            return PointCommand("done", reason=f"spoon tip {tip * 100:.0f} cm from the lips", **info)
        if state.speed > c.max_head_speed:
            return PointCommand("hold", reason=f"waiting for a still head ({state.speed:.2f} m/s)", **info)
        step = min(c.approach_step, tip - c.stop_tip_distance)
        self.travelled += step
        az = np.arctan2(tool_pos[1], tool_pos[0])
        d = np.array([np.cos(az), np.sin(az), 0.0])  # straight in, level
        return PointCommand("approach", tool_pos + step * d, d,
                            f"extend {step * 100:.0f} cm level (tip {tip * 100:.0f} -> stop at "
                            f"{c.stop_tip_distance * 100:.0f} cm)", **info)

    def _fail(self, reason: str, tip: float = 0.0, ang: float = 0.0) -> PointCommand:
        if self.approach and self.travelled > 0:  # the executor backs out by .travelled, then zeroes it
            return PointCommand("retreat", reason=reason, angle_deg=ang, tip_distance=tip)
        return PointCommand("hold", reason=reason, angle_deg=ang, tip_distance=tip)
