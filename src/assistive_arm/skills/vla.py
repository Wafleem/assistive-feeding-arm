"""Run a vision-language-action policy (OpenVLA, Octo, pi0, ...) inside the safety envelope.

A VLA outputs small end-effector deltas at 3-15 Hz. We never send those to
the arm raw: every step is clamped in size, checked against the keep-out
boxes, executed as a guarded move, and the episode is time/step bounded.
The user confirms once before the policy is allowed to move.

Policies plug in through ``VLAPolicy``. ``ScriptedPolicy`` is a stand-in used
by tests and for checking the plumbing in sim before a real model is wired.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from ..geometry import Pose, axis_angle_to_mat
from ..robot_iface.base import RobotInterface
from .base import SafetyRefusal, SkillContext, SkillResult, guarded_linear


@dataclass
class VLAAction:
    delta_pos: np.ndarray  # (3,) metres, base frame
    delta_rot: np.ndarray = field(default_factory=lambda: np.zeros(3))  # axis-angle, base frame
    gripper_open: float = 1.0  # 1 = open, 0 = closed
    terminate: bool = False


class VLAPolicy(Protocol):
    def reset(self, instruction: str) -> None: ...

    def act(self, obs: dict) -> VLAAction: ...


@dataclass
class VLARunParams:
    max_steps: int = 120
    max_step_m: float = 0.02
    max_step_rad: float = 0.15
    max_force: float = 15.0


def run_vla(
    robot: RobotInterface,
    ctx: SkillContext,
    policy: VLAPolicy,
    instruction: str,
    params: VLARunParams | None = None,
    cameras: list[str] | None = None,
) -> SkillResult:
    p = params or VLARunParams()
    if ctx.safety.config.confirm_before_contact and not ctx.confirm(f"Let the learned policy try: '{instruction}'?"):
        return SkillResult("aborted", "user declined")
    policy.reset(instruction)
    ev = {"steps": 0, "clamped": 0, "peak_force": 0.0}
    gripper_closed = False
    for step in range(p.max_steps):
        frames = robot.get_frames(cameras)
        tcp = robot.get_tcp_pose()
        obs = {
            "instruction": instruction,
            "images": {k: f.rgb for k, f in frames.items()},
            "tcp_pose": tcp,
            "gripper_width": None,
        }
        a = policy.act(obs)
        ev["steps"] = step + 1
        if a.terminate:
            return SkillResult("success", "policy finished", ev)
        dp = np.asarray(a.delta_pos, dtype=float)
        dr = np.asarray(a.delta_rot, dtype=float)
        if np.linalg.norm(dp) > p.max_step_m:
            dp = dp / np.linalg.norm(dp) * p.max_step_m
            ev["clamped"] += 1
        if np.linalg.norm(dr) > p.max_step_rad:
            dr = dr / np.linalg.norm(dr) * p.max_step_rad
            ev["clamped"] += 1
        rot = axis_angle_to_mat(dr, np.linalg.norm(dr)) if np.linalg.norm(dr) > 1e-9 else np.eye(3)
        target = Pose.from_matrix(tcp.position + dp, rot @ tcp.rot)
        try:
            res = guarded_linear(robot, ctx, target, max_force=p.max_force)
        except SafetyRefusal as e:
            return SkillResult("refused", f"policy step rejected: {e}", ev)
        ev["peak_force"] = max(ev["peak_force"], res.peak_force)
        if not res.ok and res.reason != "force_limit":
            return SkillResult("failed", f"step failed: {res.reason}", ev)
        want_closed = a.gripper_open < 0.5
        if want_closed != gripper_closed:
            if want_closed:
                robot.close_gripper(force=ctx.safety.config.grip_force)
            else:
                robot.open_gripper()
            gripper_closed = want_closed
    return SkillResult("failed", "step budget exhausted", ev)


class ScriptedPolicy:
    """Moves straight to ``goal``, closes the gripper, lifts, terminates."""

    def __init__(self, goal, lift: float = 0.08, tol: float = 0.005):
        self.goal = np.asarray(goal, dtype=float)
        self.lift, self.tol = lift, tol

    def reset(self, instruction: str) -> None:
        self.phase = "reach"

    def act(self, obs: dict) -> VLAAction:
        pos = obs["tcp_pose"].position
        if self.phase == "reach":
            d = self.goal - pos
            if np.linalg.norm(d) < self.tol:
                self.phase = "lift"
                return VLAAction(np.zeros(3), gripper_open=0.0)
            return VLAAction(d, gripper_open=1.0)
        if pos[2] < self.goal[2] + self.lift - self.tol:
            return VLAAction(np.array([0, 0, self.goal[2] + self.lift - pos[2]]), gripper_open=0.0)
        return VLAAction(np.zeros(3), gripper_open=0.0, terminate=True)


class OpenVLAPolicy:
    """Adapter sketch for OpenVLA-7B (https://github.com/openvla/openvla). UNTESTED.

    Needs a CUDA GPU (~16 GB in bf16) or a remote inference server. OpenVLA
    predicts 7-DoF deltas in the *training robot's* frame and scale (e.g.
    ``unnorm_key='bridge_orig'`` = WidowX), so ``frame_rot`` and ``scale`` must
    map them into our base frame; fine-tune on sim + teleop data from this
    arm before trusting it. Images should come from a camera placed like the
    training data's (third-person for Bridge, i.e. the off-arm camera).
    """

    def __init__(self, model_id="openvla/openvla-7b", unnorm_key="bridge_orig", camera="scene",
                 frame_rot=np.eye(3), scale=1.0, device="cuda"):
        import torch  # noqa: F401  (lazy: heavy optional dependency)
        from transformers import AutoModelForVision2Seq, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_id, torch_dtype="auto", trust_remote_code=True
        ).to(device)
        self.unnorm_key, self.camera, self.device = unnorm_key, camera, device
        self.frame_rot, self.scale = np.asarray(frame_rot), scale

    def reset(self, instruction: str) -> None:
        self.prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"

    def act(self, obs: dict) -> VLAAction:
        from PIL import Image

        img = Image.fromarray(obs["images"][self.camera])
        inputs = self.processor(self.prompt, img).to(self.device)
        a = self.model.predict_action(**inputs, unnorm_key=self.unnorm_key, do_sample=False)
        a = np.asarray(a, dtype=float)
        return VLAAction(self.frame_rot @ a[:3] * self.scale, self.frame_rot @ a[3:6], gripper_open=float(a[6]))
