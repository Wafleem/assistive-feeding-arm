"""Skill contract shared by hand-written and learned (VLA) skills."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal

from ..geometry import Pose
from .safety import SafetyEnvelope
from ..robot_iface.base import MoveResult, RobotInterface

Status = Literal["success", "partial", "failed", "aborted", "refused"]


@dataclass
class SkillContext:
    safety: SafetyEnvelope = field(default_factory=SafetyEnvelope)
    # Ask the user a yes/no question (voice, a big touchscreen button, or a nod).
    # Must return False on timeout. Tests pass ``lambda msg: True``.
    confirm: Callable[[str], bool] = lambda msg: False
    log: Callable[[str], None] = lambda msg: None


@dataclass
class SkillResult:
    status: Status
    message: str = ""
    evidence: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status in ("success", "partial")


class SafetyRefusal(Exception):
    pass


def safe_move(robot: RobotInterface, ctx: SkillContext, pose: Pose, speed: float = 0.10) -> MoveResult:
    """Free-space move after checking the target and the straight-line TCP path."""
    violations = ctx.safety.check_pose(pose) or ctx.safety.check_path(robot.get_tcp_pose(), pose)
    if violations:
        raise SafetyRefusal("; ".join(map(str, violations)))
    speed = ctx.safety.speed_limit(pose.position, speed)
    ctx.log(f"move_to {pose} @ {speed:.3f} m/s")
    return robot.move_to_pose(pose, speed)


def guarded_linear(
    robot: RobotInterface, ctx: SkillContext, pose: Pose, max_force: float, speed: float | None = None
) -> MoveResult:
    """Contact move: checked, slow, and force-limited (never above the global ceiling)."""
    violations = ctx.safety.check_path(robot.get_tcp_pose(), pose)
    if violations:
        raise SafetyRefusal("; ".join(map(str, violations)))
    c = ctx.safety.config
    speed = min(speed or c.contact_speed, c.contact_speed, ctx.safety.speed_limit(pose.position))
    limit = min(max_force, c.max_contact_force)
    ctx.log(f"guarded_linear {pose} @ {speed:.3f} m/s, max {limit:.1f} N")
    return robot.move_linear(pose, speed, max_force=limit)


def guarded_waypoints(
    robot: RobotInterface, ctx: SkillContext, waypoints: list[Pose], max_force: float
) -> tuple[int, MoveResult]:
    """Follow waypoints with guarded moves; returns (waypoints completed, last result)."""
    res = MoveResult(True, "", robot.get_tcp_pose())
    for i, wp in enumerate(waypoints):
        res = guarded_linear(robot, ctx, wp, max_force)
        if not res.ok:
            return i, res
    return len(waypoints), res


def safe_move_first(robot: RobotInterface, ctx: SkillContext, poses: list[Pose], speed: float = 0.10) -> tuple[MoveResult, Pose]:
    """Try equivalent poses in order (e.g. the tool rolled 180 deg about its axis,
    which a symmetric gripper does not care about); first reachable one wins."""
    res, last_err = MoveResult(False, "no candidates"), None
    for pose in poses:
        try:
            res = safe_move(robot, ctx, pose, speed)
        except SafetyRefusal as e:
            last_err = e
            continue
        if res.ok or res.reason in ("collision", "stopped"):
            return res, pose
    if last_err is not None and not res.ok and res.reason == "no candidates":
        raise last_err
    return res, poses[-1]
