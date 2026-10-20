"""Find food on the plate: open-vocabulary detection + masks + depth -> 3D items.

Pipeline (each stage swappable, see README.md in this folder):
    RGB-D frame --detector(query)--> boxes (+ masks) --depth--> points --> FoodItem
    detector: sim oracle | OWLv2 | Grounding DINO | NanoOWL (Jetson)   [vision/detectors.py]
    masks:    oracle | SAM 2 | NanoSAM                                    [TODO: MaskRefiner]
    attributes / state ("is the spoon loaded?"): VLM on the crop           [vlm.py, TODO]
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..vision.locate import Locator
from .vocabulary import MenuItem


@dataclass
class FoodItem:
    name: str  # canonical menu name
    position: np.ndarray  # (3,) centre of the visible surface, base frame
    points: np.ndarray  # (N, 3) its 3D points, base frame (for grasp planning)
    score: float
    camera: str
    menu: MenuItem | None = None
    attributes: dict = field(default_factory=dict)  # from a VLM: {"ripe": True, ...}

    @property
    def top_z(self) -> float:
        return float(np.max(self.points[:, 2])) if len(self.points) else float(self.position[2])

    @property
    def extent_xy(self) -> float:
        if len(self.points) < 3:
            return 0.0
        xy = self.points[:, :2] - self.points[:, :2].mean(0)
        return float(2 * np.max(np.linalg.norm(xy, axis=1)))


class FoodPerception:
    """PLAN
    * ``find(robot, name)``: canonicalise the name (vocabulary.py), run
      ``Locator.locate_all`` for that query, convert each Target3D to a FoodItem
      (its 3D points are in ``target.extras["points"]``), sort easiest-first:
      isolated from neighbours (finger clearance), near the plate centre.
    * ``scan(robot)``: find every menu item; feeds "what's on the plate?" and the meal plan.
    * Later: SAM 2 / NanoSAM masks when the detector only gives boxes; VLM attributes.
    """

    def __init__(self, locator: Locator, cameras=("scene", "wrist"), plate_center=None):
        self.locator = locator
        self.cameras = list(cameras)
        self.plate_center = None if plate_center is None else np.asarray(plate_center, float)

    def find(self, robot, name: str, cameras=None) -> list[FoodItem]:
        raise NotImplementedError("TODO(food_perception): FoodPerception.find")

    def scan(self, robot, names=None, cameras=None) -> dict[str, list[FoodItem]]:
        raise NotImplementedError("TODO(food_perception): FoodPerception.scan")
