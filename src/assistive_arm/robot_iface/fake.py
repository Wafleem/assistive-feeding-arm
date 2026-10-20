"""Kinematics-free robot for fast unit tests of skills, planner and executor.

The TCP teleports along straight lines in 1 mm steps; contact is modelled by
spring "surfaces" (a soft button travel then a rigid stop), and graspable
objects are points with a width. It has no cameras: tests that need vision
use ``SimRobot`` or inject a locator.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..geometry import Pose, interpolate_poses, normalize
from ..vision.types import CameraSpec, RGBDFrame
from .base import GripperState, MoveResult, RobotInterface, Wrench


@dataclass
class SpringSurface:
    """Contact patch: soft for ``travel`` metres (a button), rigid afterwards."""

    point: np.ndarray
    normal: np.ndarray  # out of the surface, toward free space
    radius: float = 0.03
    k_soft: float = 500.0  # N/m (a soft contact, e.g. a button or a piece of food)
    travel: float = 0.006
    k_hard: float = 3000.0  # bottomed-out button + fingertip compliance
    name: str = "surface"
    max_penetration: float = field(default=0.0, init=False)

    def force_at(self, p: np.ndarray) -> np.ndarray:
        n = normalize(self.normal)
        d = self.point - p
        pen = float(np.dot(d, n))
        lateral = np.linalg.norm(d - pen * n)
        if pen <= 0 or lateral > self.radius:
            return np.zeros(3)
        self.max_penetration = max(self.max_penetration, pen)
        f = self.k_soft * min(pen, self.travel) + self.k_hard * max(0.0, pen - self.travel)
        return f * n


@dataclass
class Graspable:
    point: np.ndarray
    width: float
    mass: float = 0.0
    name: str = "object"
    radius: float = 0.02


class FakeRobot(RobotInterface):
    collision_force = 60.0  # unguarded contact above this is reported as a collision
    _speed = 0.1

    def __init__(self, start: Pose | None = None, surfaces=None, graspables=None, reachable_radius=0.75):
        self.pose = start or Pose([0.30, 0.0, 0.30], [0.0, 1.0, 0.0, 0.0])  # tool pointing down
        self.surfaces: list[SpringSurface] = list(surfaces or [])
        self.graspables: list[Graspable] = list(graspables or [])
        self.reachable_radius = reachable_radius
        self.gripper = GripperState(self.gripper_max_width, False)
        self.held: Graspable | None = None
        self._tare = np.zeros(3)
        self.history: list[Pose] = [self.pose]
        self.stopped = False
        self.t = 0.0  # virtual clock: advanced by moves (1 mm per 0.05 s at 2 cm/s) and wait()

    # motion -------------------------------------------------------------------
    def get_tcp_pose(self) -> Pose:
        return self.pose

    def _raw_force(self, p) -> np.ndarray:
        f = sum((s.force_at(p) for s in self.surfaces), np.zeros(3))
        if self.held is not None:
            f = f + np.array([0.0, 0.0, -9.81 * self.held.mass])
        return f

    def _path(self, target: Pose, max_force: float | None) -> MoveResult:
        if self.stopped:
            return MoveResult(False, "stopped", self.pose)
        if np.linalg.norm(target.position) > self.reachable_radius:
            return MoveResult(False, "unreachable", self.pose)
        dist = float(np.linalg.norm(target.position - self.pose.position))
        steps = max(1, int(np.ceil(dist / 0.001)))
        start, peak, travelled = self.pose, 0.0, 0.0
        self.t += dist / max(self._speed, 1e-3)
        for p in interpolate_poses(start, target, steps):
            f = float(np.linalg.norm(self._raw_force(p.position) - self._tare))
            travelled += float(np.linalg.norm(p.position - self.pose.position))
            self.pose = p
            self.history.append(p)
            peak = max(peak, f)
            if max_force is not None and f > max_force:
                return MoveResult(False, "force_limit", p, peak, travelled)
            if f > self.collision_force:
                return MoveResult(False, "collision", p, peak, travelled)
        return MoveResult(True, "", self.pose, peak, travelled)

    def move_to_pose(self, pose: Pose, speed: float = 0.10) -> MoveResult:
        self._speed = speed
        return self._path(pose, None)

    def move_linear(self, pose: Pose, speed: float = 0.03, max_force: float | None = None) -> MoveResult:
        self._speed = speed
        return self._path(pose, max_force)

    def now(self) -> float:
        return self.t

    def wait(self, seconds: float) -> None:
        self.t += seconds

    # gripper ------------------------------------------------------------------
    def open_gripper(self, width: float | None = None) -> GripperState:
        self.held = None
        self.gripper = GripperState(width if width is not None else self.gripper_max_width, False)
        return self.gripper

    def close_gripper(self, force: float = 20.0) -> GripperState:
        for g in self.graspables:
            if np.linalg.norm(g.point - self.pose.position) <= g.radius:
                self.held = g
                self.gripper = GripperState(g.width, True, force)
                return self.gripper
        self.gripper = GripperState(self.gripper_min_width, False, force)
        return self.gripper

    def gripper_state(self) -> GripperState:
        return self.gripper

    # sensing ------------------------------------------------------------------
    def get_wrench(self) -> Wrench:
        return Wrench(self._raw_force(self.pose.position) - self._tare)

    def tare_wrench(self) -> None:
        self._tare = self._raw_force(self.pose.position)

    def cameras(self) -> dict[str, CameraSpec]:
        return {}

    def get_frames(self, names=None) -> dict[str, RGBDFrame]:
        return {}

    def camera_pose(self, name: str) -> Pose:
        raise KeyError(name)

    def stop(self) -> None:
        self.stopped = True
