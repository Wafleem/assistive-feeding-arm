"""Plan and execute the transfer from the plate to just in front of the mouth.

PLAN (deliver_bite)
1. Move to the staging pose: bite point at mouth + pre_approach * normal, tool +z = -normal
   (Viam motion service, with the table, plate, head sphere and the carried item
   attached to the gripper as obstacles).
2. If cfg.confirm_each_bite: ask ("Ready for a grape?"); "no" or timeout => declined.
3. Wait for ReadyGate (open + still for N ms) up to wait_ready_timeout => else timeout.
4. HeadMotionMonitor.start(state). Final approach at servo_rate_hz: each tick take the
   LATENCY-COMPENSATED mouth state, target = mouth + standoff * normal (minus the bite
   offset), step toward it at <= final_speed and <= speed_cap; guarded at
   max_force_near_face. Any monitor failure, "stop", or force => retreat immediately.
5. Hold until bite taken (mouth closes after being open near the bite, pull detected by
   F/T or gripper current, or "got it"), up to bite_timeout. Finger food: open the
   gripper when the bite is taken.
6. Retreat along +normal at retreat_speed to staging, then to the park pose.
Log DeliveryResult metrics every time.
"""

from __future__ import annotations

from ..geometry import Pose
from ..motion.primitives import SkillContext
from ..mouth_perception.types import MouthState
from ..robot_iface.base import RobotInterface
from .config import FeedingConfig
from .types import DeliveryResult


def staging_pose(s: MouthState, cfg: FeedingConfig, bite_offset: float = 0.0) -> Pose:
    """TCP pose with the bite point cfg.pre_approach out along the face normal."""
    raise NotImplementedError("TODO(feeding): staging_pose")


def final_pose(s: MouthState, cfg: FeedingConfig, bite_offset: float = 0.0) -> Pose:
    """TCP pose with the bite point cfg.standoff in front of the lips (NEVER closer)."""
    raise NotImplementedError("TODO(feeding): final_pose")


def deliver_bite(robot: RobotInterface, ctx: SkillContext, tracker, detector, cfg: FeedingConfig,
                 bite_offset: float = 0.0, label: str = "bite") -> DeliveryResult:
    raise NotImplementedError("TODO(feeding): deliver_bite")


def retreat(robot: RobotInterface, ctx: SkillContext, s: MouthState | None, cfg: FeedingConfig) -> None:
    raise NotImplementedError("TODO(feeding): retreat")
