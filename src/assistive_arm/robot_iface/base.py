"""The one robot API every skill, planner and policy talks to.

``SimRobot`` (MuJoCo), ``ViamRobot`` (a real xArm6 through Viam) and
``FakeRobot`` (pure Python, for unit tests) all implement it, so a skill that
works against one is exercised unchanged against the others.

The interface is deliberately small and *guarded*: the only way to make
contact with the world is ``move_linear(..., max_force=...)``, which stops on
a force threshold. Free-space moves (``move_to_pose``) are expected to be
collision-checked by the implementation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from ..geometry import Pose
from ..perception.types import CameraSpec, RGBDFrame


@dataclass
class GripperState:
    width: float  # metres between finger pads
    holding: bool  # closed-loop: fingers stopped before fully closing
    force: float = 0.0  # commanded / measured grip force, N


@dataclass
class Wrench:
    """External force/torque on the tool, base frame, after tare."""

    force: np.ndarray = field(default_factory=lambda: np.zeros(3))
    torque: np.ndarray = field(default_factory=lambda: np.zeros(3))

    @property
    def magnitude(self) -> float:
        return float(np.linalg.norm(self.force))


@dataclass
class MoveResult:
    ok: bool
    reason: str = ""  # "", "force_limit", "unreachable", "safety", "stopped"
    final_pose: Pose | None = None
    peak_force: float = 0.0
    travelled: float = 0.0  # metres along the commanded path


class RobotInterface(ABC):
    """Arm + gripper + cameras + (optional) force sensing, all in the base frame."""

    # Gripper geometry (xArm Gripper: 0-85 mm stroke). Implementations may override.
    gripper_max_width: float = 0.085
    gripper_min_width: float = 0.0

    @abstractmethod
    def get_tcp_pose(self) -> Pose: ...

    @abstractmethod
    def move_to_pose(self, pose: Pose, speed: float = 0.10) -> MoveResult:
        """Collision-checked free-space move (planner chooses the path)."""

    @abstractmethod
    def move_linear(self, pose: Pose, speed: float = 0.03, max_force: float | None = None) -> MoveResult:
        """Straight-line Cartesian move. If ``max_force`` is set this is a guarded
        move: it stops (ok=False, reason='force_limit') as soon as the tared
        external force exceeds it."""

    @abstractmethod
    def open_gripper(self, width: float | None = None) -> GripperState: ...

    @abstractmethod
    def close_gripper(self, force: float = 20.0) -> GripperState: ...

    @abstractmethod
    def get_wrench(self) -> Wrench: ...

    @abstractmethod
    def tare_wrench(self) -> None:
        """Zero the F/T reading (call in free space right before a contact phase)."""

    @abstractmethod
    def cameras(self) -> dict[str, CameraSpec]: ...

    @abstractmethod
    def get_frames(self, names: list[str] | None = None) -> dict[str, RGBDFrame]: ...

    @abstractmethod
    def camera_pose(self, name: str) -> Pose:
        """Current pose of camera ``name`` in the base frame (OpenCV optical axes)."""

    @abstractmethod
    def stop(self) -> None:
        """Software stop: halt motion now. Not a substitute for the hardware e-stop."""

    # Optional hooks -----------------------------------------------------------
    def gripper_state(self) -> GripperState | None:
        """Current jaw state without commanding anything (None if unknown)."""
        return None

    def joint_positions(self) -> np.ndarray | None:
        return None

    def oracle_detector(self):
        """Ground-truth detector, only available in simulation."""
        return None
