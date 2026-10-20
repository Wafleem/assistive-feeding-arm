"""Grasping & acquisition types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from ..geometry import Pose


@dataclass
class GraspCandidate:
    pose: Pose  # TCP pose at the grasp (tool +z = approach direction), base frame
    width: float  # expected jaw opening on the object, m (85 mm max for the gripper)
    score: float  # higher is better
    source: str  # "pca_topdown", "anygrasp", "contact_graspnet", ...


@dataclass
class AcquisitionResult:
    ok: bool
    message: str = ""
    item: str = ""
    # Where the edible part is relative to the TCP, along tool +z (m). Feeding uses it
    # to put the BITE (not the fingertips) at the standoff point in front of the lips.
    bite_offset: float = 0.0
    evidence: dict = field(default_factory=dict)  # grip width, re-detection result, ...


class GraspModel(Protocol):
    """Anything that proposes grasps for one object (its 3D points / mask / RGB-D crop)."""

    def propose(self, points: np.ndarray, rgbd=None, mask=None) -> list[GraspCandidate]: ...
