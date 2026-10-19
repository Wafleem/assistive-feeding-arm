"""Grasp a handle (shopping bag, mug, basket) and bring it to the user.

Top-down approach onto the handle bar with the fingers closing across it,
closed-loop gripper check, a lift that *weighs* the load with the F/T
reading, then a slow move to a hand-over pose just outside the user's
keep-out zone. The arm keeps holding until the user says "let go".
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..geometry import Pose, look_rotation, normalize
from ..perception.types import Target3D
from ..robot_iface.base import RobotInterface
from .base import SafetyRefusal, SkillContext, SkillResult, guarded_linear, safe_move


@dataclass
class GraspParams:
    pregrasp_height: float = 0.10
    grasp_depth: float = 0.02  # perception sees the top of the bar; put the fingertips below it
    lift_height: float = 0.12
    max_payload_kg: float = 3.0  # xArm6 payload is 5 kg including the ~0.8 kg gripper
    # Where to hold things for the user: in front of them, above knee height,
    # outside the keep-out boxes, high enough that a hanging bag clears the
    # legs. Measure on the real chair. None = stay at the lift pose.
    handover: Pose | None = field(default_factory=lambda: Pose([0.30, 0.0, 0.25], [0.0, 1.0, 0.0, 0.0]))


def grasp_handle(
    robot: RobotInterface,
    ctx: SkillContext,
    target: Target3D,
    handle_axis=(0.0, 1.0, 0.0),
    params: GraspParams | None = None,
) -> SkillResult:
    p = params or GraspParams()
    c = ctx.safety.config
    axis = np.asarray(handle_axis, dtype=float)
    axis[2] = 0.0  # top-down grasp: only the handle's horizontal direction matters
    axis = normalize(axis) if np.linalg.norm(axis) > 1e-6 else np.array([0.0, 1.0, 0.0])
    # Tool +z down, tool +x along the bar => fingers (tool y) close across it.
    rot = look_rotation([0, 0, -1], x_hint=axis)
    grasp = Pose.from_matrix(target.position - [0, 0, p.grasp_depth], rot)
    pre = grasp.translated([0, 0, p.pregrasp_height])
    lifted = grasp.translated([0, 0, p.lift_height])
    ev: dict = {}
    try:
        robot.open_gripper()
        if not safe_move(robot, ctx, pre).ok:
            return SkillResult("failed", "pre-grasp pose unreachable")
        if c.confirm_before_contact and not ctx.confirm(f"Grab the {target.label}?"):
            return SkillResult("aborted", "user declined")
        res = guarded_linear(robot, ctx, grasp, max_force=c.max_contact_force)
        if not res.ok:
            safe_move(robot, ctx, pre)
            return SkillResult("failed", f"bumped into something on the way down ({res.peak_force:.1f} N)")
        robot.tare_wrench()
        g = robot.close_gripper(force=c.grip_force)
        ev["grip_width"] = g.width
        if not g.holding:
            robot.open_gripper()
            safe_move(robot, ctx, pre)
            return SkillResult("failed", "gripper closed on nothing", ev)
        res = guarded_linear(robot, ctx, lifted, max_force=c.door_pull_force)
        mass = max(0.0, -robot.get_wrench().force[2] / 9.81)
        ev["mass_kg"] = round(mass, 3)
        if not res.ok or mass > p.max_payload_kg:
            guarded_linear(robot, ctx, grasp, max_force=c.max_contact_force)
            robot.open_gripper()
            safe_move(robot, ctx, pre)
            why = "too heavy" if mass > p.max_payload_kg else f"lift stopped ({res.reason})"
            return SkillResult("failed", f"put it back: {why}", ev)
        g = robot.gripper_state() or robot.close_gripper(force=c.grip_force)
        if not g.holding:  # slipped during the lift?
            return SkillResult("failed", "lost the object during lift", ev)
        if p.handover is not None:
            # carry at lift height, then go to the hand-over pose
            carry = Pose(p.handover.position.copy(), lifted.quat)
            carry.position[2] = max(carry.position[2], lifted.position[2])
            res = safe_move(robot, ctx, carry, speed=c.near_user_speed)
            if res.ok:
                res = safe_move(robot, ctx, p.handover, speed=c.near_user_speed)
            ev["at_handover"] = res.ok
        g = robot.gripper_state()
        if g is not None and not g.holding:
            return SkillResult("failed", f"dropped the {target.label} on the way", ev)
    except SafetyRefusal as e:
        return SkillResult("refused", str(e), ev)
    return SkillResult("success" if res.ok else "partial", f"holding the {target.label}", ev)


def release(robot: RobotInterface, ctx: SkillContext) -> SkillResult:
    if ctx.safety.config.confirm_before_contact and not ctx.confirm("Let go now?"):
        return SkillResult("aborted", "user declined")
    robot.open_gripper()
    up = robot.get_tcp_pose().translated([0, 0, 0.05])
    try:
        safe_move(robot, ctx, up, speed=ctx.safety.config.near_user_speed)
    except SafetyRefusal as e:
        return SkillResult("partial", f"released, did not retract: {e}")
    return SkillResult("success", "released")
