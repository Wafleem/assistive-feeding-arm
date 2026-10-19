"""Runs a TaskPlan on any RobotInterface: speech -> intent -> plan -> perception -> skill."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from .geometry import Pose, look_rotation, normalize
from .intent.schema import Intent, Task
from .perception.locate import Locator
from .perception.types import Target3D
from .planning.task_planner import TaskPlan, plan
from .robot_iface.base import RobotInterface
from .skills.base import SafetyRefusal, SkillContext, SkillResult, safe_move, safe_move_first
from .skills.door import DoorParams, open_door
from .skills.grasp import grasp_handle, release
from .skills.press_button import press_button


@dataclass
class ExecutionLog:
    plan: TaskPlan
    targets: list[Target3D] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    result: SkillResult | None = None


# Arm parked low and to the right, tool forward: out of the wheelchair camera's
# view of the workspace and away from the user. Measure on the real chair.
CLEAR_VIEW_POSE = Pose.from_matrix([0.20, -0.35, 0.20], look_rotation([1, 0, 0], [0, 0, 1]))


class Executor:
    def __init__(self, robot: RobotInterface, locator: Locator | None, ctx: SkillContext | None = None,
                 coarse_cameras=("scene",), fine_cameras=("wrist",), clear_view_pose: Pose | None = CLEAR_VIEW_POSE,
                 door_defaults: DoorParams | None = None):
        self.robot, self.locator = robot, locator
        self.ctx = ctx or SkillContext()
        self.coarse, self.fine = tuple(coarse_cameras), tuple(fine_cameras)
        self.clear_view_pose = clear_view_pose
        self.door_defaults = door_defaults or DoorParams()

    def _cams(self, wanted) -> list[str]:
        have = set(self.robot.cameras())
        return [c for c in wanted if c in have]

    def observe_poses(self, target: Target3D, view: str, distance: float) -> list[Pose]:
        """Wrist-camera viewpoints looking at the target, in order of preference."""
        if view == "top":
            p = target.position + [0, 0, distance]
            return [Pose.from_matrix(p, look_rotation([0, 0, -1], h)) for h in ([1, 0, 0], [-1, 0, 0])]
        n = normalize(target.normal)
        return [Pose.from_matrix(target.position + distance * n, look_rotation(-n, h)) for h in ([0, 0, 1], [0, 0, -1])]

    def _observe(self, target: Target3D, view: str, distance: float, log: ExecutionLog) -> bool:
        """Move the wrist camera close to the target; back off the distance if out of reach."""
        for d in (distance, distance - 0.05, distance - 0.10):
            try:
                res, _ = safe_move_first(self.robot, self.ctx, self.observe_poses(target, view, d))
            except SafetyRefusal as e:
                log.notes.append(f"observe at {d:.2f} m refused ({e})")
                continue
            if res.ok:
                return True
            log.notes.append(f"observe at {d:.2f} m failed ({res.reason})")
            if res.reason in ("collision", "stopped"):
                break
        log.notes.append("skipping close-up look")
        return False

    def run(self, intent: Intent) -> ExecutionLog:
        log = ExecutionLog(plan(intent, self.coarse, self.fine))
        if intent.needs_extra_confirmation and not self.ctx.confirm(
            f"'{intent.target}' affects other people. Are you sure?"
        ):
            log.result = SkillResult("aborted", "user declined sensitive action")
            return log
        target: Target3D | None = None
        for step in log.plan.steps:
            if step.kind == "stop":
                self.robot.stop()
                log.result = SkillResult("success", "stopped")
                return log
            if step.kind == "ask":
                log.result = SkillResult("refused", step.say)
                return log
            if step.kind == "clear_view":
                if self.clear_view_pose is not None and self._cams(self.coarse):
                    try:
                        safe_move(self.robot, self.ctx, self.clear_view_pose)
                    except SafetyRefusal as e:
                        log.notes.append(f"clear-view pose refused ({e})")
                continue
            if step.kind == "locate":
                found = (
                    self.locator.locate(self.robot, step.query, self._cams(step.cameras), step.params.get("support"))
                    if self.locator else None
                )
                if found is not None:
                    target = found
                    log.targets.append(found)
                elif target is None:
                    log.result = SkillResult("failed", f"could not find the {step.query}")
                    return log
                else:
                    log.notes.append(f"re-locate failed; keeping the {target.camera} estimate")
            elif step.kind == "observe":
                self._observe(target, step.params["view"], step.params["distance"], log)
            elif step.kind == "skill":
                log.result = self._run_skill(step.skill, target, intent, step.params)
                return log
        return log

    def _run_skill(self, name: str, target: Target3D | None, intent: Intent, params: dict) -> SkillResult:
        if name == "release":
            return release(self.robot, self.ctx)
        assert target is not None
        if name == "press_button":
            return press_button(self.robot, self.ctx, target)
        if name == "grasp_handle":
            axis = target.extras.get("axis", (0.0, 1.0, 0.0))
            return grasp_handle(self.robot, self.ctx, target, handle_axis=axis)
        if name == "open_door":
            p = replace(self.door_defaults)
            if params.get("handle_type") in ("lever", "knob"):
                p.handle_type = params["handle_type"]
            if params.get("direction") in ("pull", "push"):
                p.direction = params["direction"]
            for k in ("hinge_point", "hinge_side", "door_width"):  # perception may know better
                if k in target.extras:
                    setattr(p, k, np.asarray(target.extras[k]) if k == "hinge_point" else target.extras[k])
            return open_door(self.robot, self.ctx, target, p)
        raise ValueError(name)


def run_utterance(robot: RobotInterface, locator: Locator | None, text: str, ctx: SkillContext | None = None) -> ExecutionLog:
    from .intent.parser import parse

    return Executor(robot, locator, ctx).run(parse(text))


__all__ = ["Executor", "ExecutionLog", "Task", "run_utterance"]
