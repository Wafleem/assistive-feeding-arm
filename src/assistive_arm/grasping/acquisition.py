"""Bite acquisition strategies, chosen by ``food_perception.vocabulary`` (MenuItem.strategy).

| Strategy | Foods | Plan | Status |
|---|---|---|---|
| pinch | grape, strawberry, melon/cheese cube | top-down approach, guarded descent (<= 5 N, plate contact), close gently (MenuItem.max_grip_force), check width, lift, re-detect the plate | TODO (MVP) |
| utensil_scoop / skewer | yogurt, pudding, cut fruit | grasp the spoon/fork from a FIXED rest (known pose), scoop/skewer in a weighted bowl, keep the bowl level, check "spoon loaded" with the VLM | TODO (stretch) |
| cup | water, juice | side grasp a 60-80 mm cup, lift upright, present the STRAW tip to the mouth; never tilt near the face | TODO (stretch) |

Closed-loop checks after every grasp: gripper width within the expected range
(not fully closed = holding), gripper current/position from Viam, re-detect the
item gone from its spot on the plate.
"""

from __future__ import annotations

from ..motion.primitives import SkillContext
from ..robot_iface.base import RobotInterface
from .types import AcquisitionResult, GraspCandidate


def acquire_pinch(robot: RobotInterface, ctx: SkillContext, item, grasp: GraspCandidate,
                  grip_force: float = 8.0) -> AcquisitionResult:
    raise NotImplementedError("TODO(grasping): acquire_pinch")


def acquire_with_utensil(robot: RobotInterface, ctx: SkillContext, item, utensil_pose) -> AcquisitionResult:
    raise NotImplementedError("TODO(grasping): utensil acquisition (stretch)")


def acquire_cup(robot: RobotInterface, ctx: SkillContext, cup) -> AcquisitionResult:
    raise NotImplementedError("TODO(grasping): cup + straw (stretch)")
