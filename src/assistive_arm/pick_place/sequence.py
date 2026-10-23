"""Replay a taught pick-and-place.

PLAN for ``run_pick_and_place(robot, library, cfg)``:
1. Refuse if teleop is running, or if a pose the sequence needs is missing from the library.
2. Set speed/acceleration (first run on a task: ``cfg.first_run_speed_deg_s``).
3. Walk ``PICK_PLACE_STEPS``: "move" → ``move_to_joint_positions(pose)``; gripper steps →
   open/close; "verify_grip" → the gripper reports holding (else fail with "empty grasp").
4. On any failure: if holding nothing, retreat → home; if holding, stop where it is and
   report (a human decides). Return a RunResult with every StepResult and its timing.
"""

from __future__ import annotations

from .config import PickPlaceConfig
from .types import PoseLibrary, RunResult, Step

PICK_PLACE_STEPS: tuple[Step, ...] = (
    Step("move", "home"),
    Step("gripper_open"),
    Step("move", "pre_pick"),
    Step("move", "pick"),
    Step("gripper_close"),
    Step("verify_grip"),
    Step("move", "lift"),
    Step("move", "pre_place"),
    Step("move", "place"),
    Step("gripper_open"),
    Step("move", "retreat"),
    Step("move", "home"),
)


def required_poses(steps: tuple[Step, ...] = PICK_PLACE_STEPS) -> set[str]:
    return {s.pose for s in steps if s.pose}


def run_pick_and_place(robot, library: PoseLibrary, cfg: PickPlaceConfig | None = None) -> RunResult:
    raise NotImplementedError("TODO(pick_place): run_pick_and_place")
