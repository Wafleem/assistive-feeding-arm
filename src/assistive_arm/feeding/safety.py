"""The moving keep-out zone around the face.

PLAN
* ``face_zone(mouth_state, cfg)`` -> a Sphere around the estimated head centre
  (mouth - 0.10 m * normal), radius cfg.head_radius; push it to the motion planner
  every tick with ``robot.update_world([...])`` (Viam obstacle) and add it to the
  SafetyEnvelope via ``with_zones`` for TCP checks.
* ``check_near_face(p, mouth_state, cfg)`` -> violations: inside the head sphere
  but outside the approach corridor; closer to the lips than cfg.standoff.
* ``speed_cap(p, mouth_state, cfg)`` -> cfg.near_face_speed_cap within near_face_radius.
"""

from __future__ import annotations

import numpy as np

from ..motion.safety import Sphere, Violation
from ..mouth_perception.types import MouthState
from .config import FeedingConfig


def face_zone(s: MouthState, cfg: FeedingConfig) -> Sphere:
    raise NotImplementedError("TODO(feeding): face_zone")


def check_near_face(p: np.ndarray, s: MouthState, cfg: FeedingConfig) -> list[Violation]:
    raise NotImplementedError("TODO(feeding): check_near_face")


def speed_cap(p: np.ndarray, s: MouthState, cfg: FeedingConfig) -> float:
    raise NotImplementedError("TODO(feeding): speed_cap")
