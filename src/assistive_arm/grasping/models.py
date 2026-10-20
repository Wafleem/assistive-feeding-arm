"""Open grasp models behind ``GraspModel``. All stubs; they run on the Linux GPU box or AGX.

| Model | Notes |
|---|---|
| AnyGrasp (graspnet) | strong on cluttered tabletops; SDK licence needed; CUDA |
| Contact-GraspNet | open weights; point cloud in, 6-DoF grasps out; CUDA |
| PCA top-down (pca.py) | heuristic fallback, CPU, always available |

TODO(grasping): wrap each as a Viam vision/generic service so it can move between
the AGX and the Linux PC by config (docs/COMPUTE.md); filter proposals to
near-top-down for small food on a plate, and by gripper width.
"""

from __future__ import annotations

import numpy as np

from .types import GraspCandidate


class AnyGraspModel:
    def propose(self, points: np.ndarray, rgbd=None, mask=None) -> list[GraspCandidate]:
        raise NotImplementedError("TODO(grasping): AnyGrasp wrapper (Linux GPU / AGX)")


class ContactGraspNetModel:
    def propose(self, points: np.ndarray, rgbd=None, mask=None) -> list[GraspCandidate]:
        raise NotImplementedError("TODO(grasping): Contact-GraspNet wrapper (Linux GPU / AGX)")
