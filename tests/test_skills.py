"""Skills against FakeRobot: fast, deterministic, no physics."""

import numpy as np

from assistive_arm.geometry import Pose
from assistive_arm.perception.types import Target3D
from assistive_arm.planning.safety import SafetyEnvelope
from assistive_arm.robot_iface import FakeRobot, Graspable, SpringSurface
from assistive_arm.skills import (
    DoorParams, ScriptedPolicy, SkillContext, grasp_handle, open_door, press_button, release, run_vla,
)

YES = SkillContext(confirm=lambda m: True)
NO = SkillContext(confirm=lambda m: False)
BUTTON = np.array([0.50, 0.05, 0.35])
FACING_ROBOT = np.array([-1.0, 0.0, 0.0])


def button_robot(**surface_kw):
    return FakeRobot(surfaces=[SpringSurface(BUTTON, FACING_ROBOT, **surface_kw)])


def target(pos=BUTTON, normal=FACING_ROBOT, label="elevator button 7"):
    return Target3D(label, np.asarray(pos, float), np.asarray(normal, float), 1.0, "test")


def test_press_button_success_stops_on_force():
    r = button_robot()
    res = press_button(r, YES, target())
    assert res.status == "success", res
    assert YES.safety.config.button_press_force < res.evidence["peak_force"] < YES.safety.config.max_contact_force
    assert r.surfaces[0].max_penetration > 0.004  # the button actually travelled
    assert res.evidence["retracted"]


def test_press_button_misses_when_button_is_not_there():
    r = FakeRobot()  # no surface at all
    res = press_button(r, YES, target())
    assert res.status == "failed" and "no resistance" in res.message


def test_press_button_flags_early_contact():
    # The real surface is 3 cm in front of where perception put it (e.g. a bezel).
    r = FakeRobot(surfaces=[SpringSurface(BUTTON + [-0.03, 0, 0], FACING_ROBOT, travel=0.0)])
    res = press_button(r, YES, target())
    assert res.status == "failed" and "before the expected face" in res.message


def test_no_contact_without_user_confirmation():
    r = button_robot()
    res = press_button(r, NO, target())
    assert res.status == "aborted"
    assert r.surfaces[0].max_penetration == 0.0


def test_press_refused_inside_keep_out():
    res = press_button(FakeRobot(), YES, target(pos=[-0.10, 0.30, 0.30]))  # the user's chest
    assert res.status == "refused"


def test_grasp_bag_weighs_and_hands_over():
    handle = np.array([0.42, -0.25, 0.24])
    r = FakeRobot(graspables=[Graspable(handle - [0, 0, 0.02], width=0.016, mass=1.2)])
    res = grasp_handle(r, YES, target(handle, [0, 0, 1], "shopping bag handle"))
    assert res.status == "success", res
    assert abs(res.evidence["mass_kg"] - 1.2) < 0.05
    assert np.allclose(r.get_tcp_pose().position, [0.30, 0.0, 0.25])
    assert release(r, YES).status == "success" and r.held is None


def test_grasp_refuses_overweight_and_puts_it_back():
    handle = np.array([0.42, -0.25, 0.24])
    r = FakeRobot(graspables=[Graspable(handle - [0, 0, 0.02], width=0.016, mass=4.0)])
    res = grasp_handle(r, YES, target(handle, [0, 0, 1], "shopping bag handle"))
    assert res.status == "failed" and "too heavy" in res.message
    assert r.held is None


def test_grasp_detects_empty_gripper():
    res = grasp_handle(FakeRobot(), YES, target([0.42, -0.25, 0.24], [0, 0, 1], "bag handle"))
    assert res.status == "failed" and "nothing" in res.message


def test_door_arc_geometry_and_success():
    handle = np.array([0.50, -0.08, 0.25])
    hinge = np.array([0.58, -0.80, 0.25])
    r = FakeRobot(graspables=[Graspable(handle - 0.015 * FACING_ROBOT, width=0.02)])
    p = DoorParams(hinge_side="right", hinge_point=hinge, open_angle=np.deg2rad(20))
    res = open_door(r, YES, target(handle, FACING_ROBOT, "door handle"), p)
    assert res.status == "success", res
    end = r.get_tcp_pose().position
    # the hand ended on the door's arc, swung toward the robot (pull)
    r0 = np.linalg.norm((handle - 0.015 * FACING_ROBOT - hinge)[:2])
    assert abs(np.linalg.norm((end - hinge)[:2]) - r0) < 0.03
    assert end[0] < handle[0]


def test_door_stops_on_force_when_it_will_not_move():
    handle = np.array([0.50, -0.08, 0.25])
    # A rigid "door" just behind the robot's pulling direction: the swing meets resistance at once.
    r = FakeRobot(graspables=[Graspable(handle - 0.015 * FACING_ROBOT, width=0.02)],
                  surfaces=[SpringSurface(handle + [-0.02, 0, 0], [1, 0, 0], radius=0.5, travel=0.0)])
    p = DoorParams(hinge_side="right", hinge_point=[0.58, -0.80, 0.25], handle_type="knob")
    res = open_door(r, YES, target(handle, FACING_ROBOT, "door handle"), p)
    assert res.status in ("failed", "partial")
    assert res.evidence.get("peak_force", 0) <= YES.safety.config.door_pull_force + 25  # bounded


def test_vla_runner_clamps_and_finishes():
    goal = np.array([0.40, -0.20, 0.20])
    r = FakeRobot(start=Pose([0.30, -0.30, 0.30], [0, 1, 0, 0]))
    res = run_vla(r, YES, ScriptedPolicy(goal), "pick up the cup")
    assert res.status == "success", res
    assert res.evidence["clamped"] > 0  # big first steps were limited
    steps = [np.linalg.norm(b.position - a.position) for a, b in zip(r.history, r.history[1:])]
    assert max(steps) <= 0.0011  # FakeRobot moves in 1 mm sub-steps of <=2 cm commands


def test_vla_runner_refuses_steps_into_the_user():
    r = FakeRobot(start=Pose([0.15, 0.05, 0.30], [0, 1, 0, 0]))
    res = run_vla(r, YES, ScriptedPolicy([-0.20, 0.30, 0.30]), "hand me the cup")
    assert res.status == "refused"
    env = SafetyEnvelope()
    assert all(not env.check_point(p.position) for p in r.history)
