"""Software safety envelope for an arm that works next to a seated person.

This is a *second* layer. The first layers are hardware: the xArm's own
collision-sensitivity / reduced-mode limits, a hardware e-stop the user can
reach (and a wireless one for a carer), and a gripper with a limited force.
Everything here assumes those exist and are configured to match.

Frames: all boxes are axis-aligned in the arm base frame (see geometry.py).
The default numbers assume the arm on the *right* armrest, user to its +y.
Measure your chair and override ``SafetyConfig``; do not trust these numbers.
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
        """Euclidean distance from p to the box (0 inside)."""
        p = np.asarray(p)
        d = np.maximum(np.asarray(self.lo) - p, 0) + np.maximum(p - np.asarray(self.hi), 0)
        return float(np.linalg.norm(d))

    @property
    def center(self) -> np.ndarray:
        return (np.asarray(self.lo) + np.asarray(self.hi)) / 2

    @property
    def size(self) -> np.ndarray:
        return np.asarray(self.hi) - np.asarray(self.lo)


def default_keep_out() -> tuple[Box, ...]:
    return (
        # Seated user's torso and head, arm on the right armrest => user at +y.
        Box("user_torso_head", (-0.55, 0.10, -0.25), (0.10, 0.55, 0.80)),
        # Thighs, knees and feet on the footrest.
        Box("user_legs", (-0.20, 0.10, -0.75), (0.45, 0.50, -0.10)),
        # The chair itself: seat, armrest under the arm, wheels, batteries.
        Box("wheelchair", (-0.60, -0.15, -0.75), (0.15, 0.60, -0.08)),
    )


@dataclass(frozen=True)
class SafetyConfig:
    # Speeds, m/s. ISO/TS 15066 transient-contact limits are the reference;
    # these are conservative starting values, not certified ones.
    max_speed: float = 0.15
    near_user_speed: float = 0.05  # within near_user_margin of a keep-out box
    contact_speed: float = 0.02  # any guarded move toward a surface
    near_user_margin: float = 0.20
    # Forces, N.
    max_contact_force: float = 15.0  # hard ceiling for any guarded move
    button_press_force: float = 6.0  # stop pressing once the button pushes back this hard
    handle_turn_force: float = 25.0  # pushing a lever down / twisting a knob
    door_pull_force: float = 30.0  # swinging a door along its arc
    grip_force: float = 30.0
    # Reach: the xArm6 nominal reach is 0.70 m; keep a margin.
    max_reach: float = 0.68
    min_height: float = -0.30  # below the armrest is the chair and the user's legs
    keep_out: tuple[Box, ...] = field(default_factory=default_keep_out)
    # Every move into contact needs a "yes" from the user first.
    confirm_before_contact: bool = True


@dataclass
class Violation:
    rule: str
    detail: str

    def __str__(self) -> str:
        return f"{self.rule}: {self.detail}"


class SafetyEnvelope:
    def __init__(self, config: SafetyConfig | None = None):
        self.config = config or SafetyConfig()

    def check_point(self, p) -> list[Violation]:
        c = self.config
        p = np.asarray(p, dtype=float)
        out = []
        r = float(np.linalg.norm(p))
        if r > c.max_reach:
            out.append(Violation("reach", f"{r:.3f} m > {c.max_reach} m"))
        if p[2] < c.min_height:
            out.append(Violation("min_height", f"z={p[2]:.3f} m < {c.min_height} m"))
        for box in c.keep_out:
            if box.contains(p):
                out.append(Violation("keep_out", f"inside {box.name}"))
        return out

    def check_pose(self, pose: Pose) -> list[Violation]:
        return self.check_point(pose.position)

    def check_path(self, a: Pose, b: Pose, step: float = 0.01) -> list[Violation]:
        """Checks the straight segment a->b (TCP only; whole-arm collision
        checking is the motion planner's job, fed with the same boxes)."""
        n = max(1, int(np.ceil(np.linalg.norm(b.position - a.position) / step)))
        for i in range(n + 1):
            p = a.position + (b.position - a.position) * i / n
            v = self.check_point(p)
            if v:
                return v
        return []

    def distance_to_user(self, p) -> float:
        boxes = [b for b in self.config.keep_out if b.name.startswith("user")]
        return min((b.distance(p) for b in boxes), default=float("inf"))

    def speed_limit(self, p, requested: float | None = None) -> float:
        c = self.config
        limit = c.near_user_speed if self.distance_to_user(p) < c.near_user_margin else c.max_speed
        return min(limit, requested) if requested is not None else limit

    def clamp_force(self, requested: float) -> float:
        return min(requested, self.config.max_contact_force)
