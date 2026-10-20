"""Heuristic top-down grasp from an object's 3D points (the fallback that always exists).

PLAN
1. Project the object's points to the table plane; PCA on x-y.
2. Fingers close along the MINOR axis; width = extent along it (+ margin).
   Reject if width > max_width (85 mm stroke minus margin).
3. Tool +z points down; tool +x along the major axis (look_rotation).
4. TCP height: fingertips just above the support (plate top) + max(4 mm, 30% of
   object height), so the pads straddle the object's equator, never the plate.
5. Score: narrower and better isolated = higher.

DONE WHEN: tests/test_grasping.py PCA tests pass (a 22 mm grape -> ~22 mm width,
a 70 mm cup -> rejected/flagged for the cup strategy, yaw follows an elongated item).
"""

from __future__ import annotations

import numpy as np

from .types import GraspCandidate


def pca_topdown_grasp(points: np.ndarray, support_z: float | None = None,
                      max_width: float = 0.075, clearance: float = 0.004) -> GraspCandidate | None:
    raise NotImplementedError("TODO(grasping): pca_topdown_grasp")
