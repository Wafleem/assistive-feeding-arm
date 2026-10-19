"""Grasp a door handle, turn it, and swing the door open along its arc.

Both the handle turn and the door swing are the same primitive: follow an arc
about an axis with guarded moves. For a lever the axis is the lever spindle
(the grip point is ``lever_length`` from it); for a knob the grip is on the
spindle so the arc degenerates to a wrist twist; for the door it is the hinge
line. A real door's hinge is estimated (handle side + door width, or the
detected door edge), so the arc is wrong by a few centimetres. The guarded
moves plus the force limit are what keep that from turning into a fight
with the door; a true impedance controller on the arm is the upgrade path.

Scope for the MVP: crack the door to ``open_angle`` and keep holding it.
A door with a closer swings back as soon as it is released, so getting the
chair through needs the user to wedge the door (footrest) or a second skill.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from ..geometry import Pose, arc_waypoints, look_rotation, normalize
from ..perception.types import Target3D
from ..robot_iface.base import RobotInterface
from .base import SafetyRefusal, SkillContext, SkillResult, guarded_linear, guarded_waypoints, safe_move


@dataclass
class DoorParams:
    handle_type: Literal["lever", "knob"] = "lever"
    direction: Literal["pull", "push"] = "pull"
    hinge_side: Literal["left", "right"] = "left"  # as seen from the robot
    door_width: float = 0.85  # handle-to-hinge distance if no hinge is detected
    handle_projection: float = 0.065  # how far the grip stands off the door face (ADA levers ~6-7 cm)
    hinge_point: np.ndarray | None = None  # any point on the hinge line, base frame
    # Grip point (mid-lever, where perception centres) to spindle. Err short: in sim
    # (true 0.06 m) 0.04-0.06 m all work, >= 0.065 m jams against handle_turn_force.
    lever_length: float = 0.055
    turn_angle: float = np.deg2rad(40)  # lever down / knob twist
    open_angle: float = np.deg2rad(25)
    min_success_angle: float = np.deg2rad(8)
    standoff: float = 0.10
    grasp_depth: float = 0.015  # perception sees the front of the handle; reach past it
    arc_step: float = np.deg2rad(2)
    release_at_end: bool = False


def _horizontal_in_plane(normal) -> np.ndarray:
    n = normalize(normal)
    return normalize(np.cross(n, [0.0, 0.0, 1.0]))  # horizontal, in the door plane


def estimate_hinge(handle: Target3D, p: DoorParams) -> np.ndarray:
    """Point on the hinge line from the handle, the door plane and which side the hinge is."""
    if p.hinge_point is not None:
        return np.asarray(p.hinge_point, dtype=float)
    n = normalize(handle.normal)
    h = _horizontal_in_plane(n)  # points to the robot's left when facing the door
    side = 1.0 if p.hinge_side == "left" else -1.0
    # The hinge line lies in the door plane, behind the handle grip.
    return handle.position - p.handle_projection * n + side * p.door_width * h


def _still_holding(robot: RobotInterface) -> bool:
    g = robot.gripper_state()
    return True if g is None else g.holding


def _pick_sign(start: Pose, center, axis, angle, prefer) -> float:
    """Rotation sign whose first waypoint best matches ``prefer(delta) -> score``."""
    scores = {}
    for s in (1.0, -1.0):
        wp = arc_waypoints(start, center, axis, s * angle, 1)[0]
        scores[s] = prefer(wp.position - start.position)
    return max(scores, key=scores.get)


def open_door(robot: RobotInterface, ctx: SkillContext, handle: Target3D, params: DoorParams | None = None) -> SkillResult:
    p = params or DoorParams()
    c = ctx.safety.config
    n = normalize(handle.normal)  # out of the door, toward the robot
    along = _horizontal_in_plane(n)
    # Fingers close vertically across a horizontal lever bar: tool +x horizontal.
    rot = look_rotation(-n, x_hint=along)
    grasp = Pose.from_matrix(handle.position - p.grasp_depth * n, rot)
    pre = Pose.from_matrix(handle.position + p.standoff * n, rot)
    ev: dict = {"hinge": estimate_hinge(handle, p).round(3).tolist()}
    try:
        robot.open_gripper()
        if not safe_move(robot, ctx, pre).ok:
            return SkillResult("failed", "pre-grasp pose unreachable", ev)
        if c.confirm_before_contact and not ctx.confirm(f"Grab the door handle and {p.direction} the door open?"):
            return SkillResult("aborted", "user declined", ev)
        res = guarded_linear(robot, ctx, grasp, max_force=c.max_contact_force)
        if not res.ok:
            safe_move(robot, ctx, pre)
            return SkillResult("failed", f"hit something reaching the handle ({res.peak_force:.1f} N)", ev)
        robot.tare_wrench()
        if not robot.close_gripper(force=c.grip_force).holding:
            robot.open_gripper()
            safe_move(robot, ctx, pre)
            return SkillResult("failed", "missed the handle", ev)

        # 1) turn the handle
        if p.handle_type == "lever":
            # Levers point from the spindle (latch edge) toward the hinge, so the
            # spindle is lever_length from the grip point, away from the hinge.
            side = 1.0 if p.hinge_side == "left" else -1.0
            spindle = handle.position - side * p.lever_length * along
            sign = _pick_sign(grasp, spindle, n, p.turn_angle, prefer=lambda d: -d[2])  # lever goes down
        else:
            spindle = handle.position
            sign = 1.0  # TODO: try the other way on force_limit; most knobs turn either way
        turn = arc_waypoints(grasp, spindle, n, sign * p.turn_angle, max(2, int(p.turn_angle / p.arc_step)))
        done, res = guarded_waypoints(robot, ctx, turn, max_force=c.handle_turn_force)
        ev["turned_deg"] = float(np.rad2deg(p.turn_angle * done / len(turn)))
        if done == len(turn) and not _still_holding(robot):
            safe_move(robot, ctx, pre)
            return SkillResult("failed", "the handle slipped out while turning", ev)
        if done < len(turn):
            robot.open_gripper()
            safe_move(robot, ctx, pre)
            return SkillResult("failed", f"handle would not turn ({res.peak_force:.1f} N)", ev)

        # 2) swing the door about the hinge
        hinge = estimate_hinge(handle, p)
        start = robot.get_tcp_pose()
        want = 1.0 if p.direction == "pull" else -1.0
        sign = _pick_sign(start, hinge, [0, 0, 1], p.open_angle, prefer=lambda d: want * float(np.dot(d, n)))
        steps = max(2, int(p.open_angle / p.arc_step))
        swing = arc_waypoints(start, hinge, [0, 0, 1], sign * p.open_angle, steps)
        done, res = guarded_waypoints(robot, ctx, swing, max_force=c.door_pull_force)
        angle = p.open_angle * done / steps
        ev["opened_deg"] = float(np.rad2deg(angle))
        ev["peak_force"] = res.peak_force
        # Moving the TCP along the arc proves nothing unless the handle came with it.
        ev["holding_after_swing"] = _still_holding(robot)
        if not ev["holding_after_swing"]:
            safe_move(robot, ctx, robot.get_tcp_pose().translated(0.08 * n))
            return SkillResult("failed", "lost the handle during the swing; door state unknown", ev)
        if p.release_at_end:
            robot.open_gripper()
            safe_move(robot, ctx, robot.get_tcp_pose().translated(0.08 * n))
    except SafetyRefusal as e:
        return SkillResult("refused", str(e), ev)
    if done == steps:
        return SkillResult("success", f"door open {ev['opened_deg']:.0f} deg", ev)
    if angle >= p.min_success_angle:
        return SkillResult("partial", f"door open {ev['opened_deg']:.0f} deg, then {res.reason}", ev)
    robot.open_gripper()
    return SkillResult("failed", f"door did not open ({res.reason}, {res.peak_force:.1f} N)", ev)
