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
