"""Press a labelled button (elevator, door opener, light switch).

Approach along the surface normal with the fingers closed into a single
pusher, ask the user, then press with a guarded move that stops when the
button pushes back with ``button_press_force``. Success needs evidence: the
force threshold must be reached roughly where the surface was expected,
not centimetres early (frame, wrong button) and not never (missed).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..geometry import Pose, normalize, tool_pose_facing
from ..perception.types import Target3D
from ..robot_iface.base import RobotInterface
from .base import SafetyRefusal, SkillContext, SkillResult, guarded_linear, safe_move, safe_move_first


@dataclass
class PressButtonParams:
    standoff: float = 0.08  # pre-press distance in front of the button face
    overtravel: float = 0.02  # how far past the expected face we allow the press (sway + depth error)
    early_contact_tolerance: float = 0.012  # contact earlier than this before the face => wrong surface
    dwell_retract: float = 0.0


def press_button(
    robot: RobotInterface, ctx: SkillContext, target: Target3D, params: PressButtonParams | None = None
) -> SkillResult:
    p = params or PressButtonParams()
    c = ctx.safety.config
    approach = -normalize(target.normal)
    # Tool +x up keeps the wrist camera above the fingers on vertical panels; the
    # closed fingers are symmetric, so rolled 180 deg is an equally good fallback
    # that avoids a wrist flip for buttons low and toward the user.
    rolls = [(0, 0, 1), (0, 0, -1)]
    try:
        robot.close_gripper(force=5.0)  # fingers together = one blunt pusher
        res, pre = safe_move_first(
            robot, ctx, [tool_pose_facing(target.position, approach, x_hint=h, standoff=p.standoff) for h in rolls])
        end = Pose(pre.position + (p.standoff + p.overtravel) * approach, pre.quat)
        if not res.ok:
            return SkillResult("failed", f"could not reach pre-press pose: {res.reason}")
        if c.confirm_before_contact and not ctx.confirm(f"Press the {target.label}?"):
            return SkillResult("aborted", "user declined")
        robot.tare_wrench()
        res = guarded_linear(robot, ctx, end, max_force=c.button_press_force)
    except SafetyRefusal as e:
        return SkillResult("refused", str(e))

    evidence = {"travelled": res.travelled, "peak_force": res.peak_force, "reason": res.reason}
    retract = safe_move(robot, ctx, pre, speed=c.contact_speed * 2)
    evidence["retracted"] = retract.ok

    if res.reason != "force_limit":
        return SkillResult("failed", "no resistance felt: button not where expected", evidence)
    if res.travelled < p.standoff - p.early_contact_tolerance:
        return SkillResult(
            "failed",
            f"contact {p.standoff - res.travelled:.3f} m before the expected face: wrong surface?",
            evidence,
        )
    evidence["press_depth"] = float(res.travelled - p.standoff)
    return SkillResult("success", f"pressed {target.label}", evidence)


def button_pressed_by_depth(evidence: dict, min_depth: float = 0.002) -> bool:
    """Optional stricter check: the button moved in before stiffening."""
    return bool(evidence.get("press_depth", -np.inf) >= min_depth)
