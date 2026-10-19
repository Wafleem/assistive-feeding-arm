"""Data types shared by cameras, detectors and the 3D locator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

# Where a camera lives. This decides how its pose is obtained:
#   wrist - rigidly on the gripper; pose = TCP pose x hand-eye calibration.
#           Best close-up view, but it moves (re-capture after every move) and
#           sees nothing while the gripper is against a surface.
#   base  - on the wheelchair (mast, headrest or armrest). Fixed in the arm
#           base frame, so its extrinsic is a one-off calibration and it keeps
#           watching while the arm works (contact checks, user keep-out zone).
#   world - fixed in the environment (e.g. a lab tripod). Needs the
#           wheelchair's pose in the world (AprilTag on the chair or mocap).
CameraMount = Literal["wrist", "base", "world"]


@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @classmethod
    def from_fovy(cls, fovy_deg: float, width: int, height: int) -> "CameraIntrinsics":
        """Pinhole intrinsics for a MuJoCo camera (square pixels, centred principal point)."""
        f = (height / 2.0) / np.tan(np.deg2rad(fovy_deg) / 2.0)
        return cls(f, f, width / 2.0, height / 2.0, width, height)

    @property
    def matrix(self) -> np.ndarray:
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1.0]])


@dataclass(frozen=True)
class CameraSpec:
    name: str
    mount: CameraMount
    intrinsics: CameraIntrinsics


@dataclass
class RGBDFrame:
    """One aligned colour + depth capture. ``depth`` is metres, 0 = invalid."""

    camera: str
    rgb: np.ndarray  # (H, W, 3) uint8
    depth: np.ndarray  # (H, W) float32 metres
    intrinsics: CameraIntrinsics
    # Simulator-only ground-truth segmentation, (H, W) int geom ids (-1 = none).
    # Real cameras leave this None; only the oracle detector reads it.
    segmentation: np.ndarray | None = None
    timestamp: float = 0.0


@dataclass
class Detection:
    label: str
    score: float
    bbox: tuple[float, float, float, float]  # x0, y0, x1, y1 pixels
    mask: np.ndarray | None = None  # (H, W) bool
    extras: dict = field(default_factory=dict)


@dataclass
class Target3D:
    """A located object/part in the arm base frame."""

    label: str
    position: np.ndarray  # (3,) metres, base frame
    normal: np.ndarray  # (3,) unit, base frame, points out of the surface toward the camera
    score: float
    camera: str
    n_points: int = 0
    extras: dict = field(default_factory=dict)
