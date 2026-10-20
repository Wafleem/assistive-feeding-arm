"""Static safety envelope: workspace, keep-out zones, speed and force ceilings.

This is the *second* layer. The first is hardware: the xArm's collision
detection at its most sensitive setting, reduced-mode speed limits set on the
controller, a hardware e-stop within the user's (or a carer's) reach, and a
gripper with a limited force. Everything here assumes those exist.

Everything near the face is handled by ``feeding.safety`` (a moving head
needs a moving keep-out zone and an approach corridor). This module holds
what does not move: the table, the user's body below the head, reach.

Frames: arm base frame, metres. Demo layout (table-mounted arm, see
docs/PROPS.md): +x across the table, +y toward the user's side, +z up,
z = 0 at the table top. Measure the real layout and override; these numbers
match the MuJoCo feeding scene, not your venue.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..geometry import Pose


@dataclass(frozen=True)
class Box:
    name: str
    lo: tuple[float, float, float]
    hi: tuple[float, float, float]

    def contains(self, p, margin: float = 0.0) -> bool:
        p = np.asarray(p)
        return bool(np.all(p >= np.asarray(self.lo) - margin) and np.all(p <= np.asarray(self.hi) + margin))

    def distance(self, p) -> float:
        p = np.asarray(p)
        d = np.maximum(np.asarray(self.lo) - p, 0) + np.maximum(p - np.asarray(self.hi), 0)
        return float(np.linalg.norm(d))

    @property
    def center(self) -> np.ndarray:
        return (np.asarray(self.lo) + np.asarray(self.hi)) / 2

    @property
    def size(self) -> np.ndarray:
        return np.asarray(self.hi) - np.asarray(self.lo)


@dataclass(frozen=True)
class Sphere:
    name: str
    center: tuple[float, float, float]
    radius: float

    def contains(self, p, margin: float = 0.0) -> bool:
        return self.distance(p) <= margin

    def distance(self, p) -> float:
        return max(0.0, float(np.linalg.norm(np.asarray(p) - np.asarray(self.center))) - self.radius)


Zone = Box | Sphere


def default_keep_out() -> tuple[Zone, ...]:
    # The user's torso and shoulders, below the head (the head itself is a
    # moving zone owned by feeding.safety). Matches sim/scenes/feeding.xml.
    return (Box("user_torso", (0.12, 0.42, -0.35), (0.48, 0.80, 0.22)),)


@dataclass(frozen=True)
class SafetyConfig:
    # Speeds, m/s. Conservative starting values, not certified ones
    # (ISO/TS 15066 is the reference for transient contact with the face: low).
    max_speed: float = 0.15
    near_user_speed: float = 0.05  # within near_user_margin of a user zone
    contact_speed: float = 0.02  # any guarded move into contact (food, plate, utensils)
    near_user_margin: float = 0.20
    # Forces, N.
    max_contact_force: float = 15.0  # ceiling for any guarded move
    grip_force: float = 20.0  # food is soft: skills ask for less (see grasping)
    # Reach: xArm6 nominal 0.70 m.
    max_reach: float = 0.68
    min_height: float = -0.005  # the table top is z = 0
    keep_out: tuple[Zone, ...] = field(default_factory=default_keep_out)
    confirm_before_contact: bool = True


@dataclass
class Violation:
    rule: str
    detail: str

    def __str__(self) -> str:
        return f"{self.rule}: {self.detail}"


class SafetyEnvelope:
    def __init__(self, config: SafetyConfig | None = None, extra_zones: tuple[Zone, ...] = ()):
        self.config = config or SafetyConfig()
        self.extra_zones = tuple(extra_zones)  # e.g. the tracked head, updated by feeding

    @property
    def zones(self) -> tuple[Zone, ...]:
        return tuple(self.config.keep_out) + self.extra_zones

    def with_zones(self, *zones: Zone) -> "SafetyEnvelope":
        return SafetyEnvelope(self.config, tuple(zones))

    def check_point(self, p) -> list[Violation]:
        c = self.config
        p = np.asarray(p, dtype=float)
        out = []
        r = float(np.linalg.norm(p))
        if r > c.max_reach:
            out.append(Violation("reach", f"{r:.3f} m > {c.max_reach} m"))
        if p[2] < c.min_height:
            out.append(Violation("min_height", f"z={p[2]:.3f} m < {c.min_height} m"))
        for z in self.zones:
            if z.contains(p):
                out.append(Violation("keep_out", f"inside {z.name}"))
        return out

    def check_pose(self, pose: Pose) -> list[Violation]:
        return self.check_point(pose.position)

    def check_path(self, a: Pose, b: Pose, step: float = 0.01) -> list[Violation]:
        """TCP straight-line check. Whole-arm collision checking is the motion
        planner's job (Viam motion service, fed the same zones as obstacles)."""
        n = max(1, int(np.ceil(np.linalg.norm(b.position - a.position) / step)))
        for i in range(n + 1):
            v = self.check_point(a.position + (b.position - a.position) * i / n)
            if v:
                return v
        return []

    def distance_to_user(self, p) -> float:
        user = [z for z in self.zones if z.name.startswith(("user", "head", "face"))]
        return min((z.distance(p) for z in user), default=float("inf"))

    def speed_limit(self, p, requested: float | None = None) -> float:
        c = self.config
        limit = c.near_user_speed if self.distance_to_user(p) < c.near_user_margin else c.max_speed
        return min(limit, requested) if requested is not None else limit

    def clamp_force(self, requested: float) -> float:
        return min(requested, self.config.max_contact_force)
