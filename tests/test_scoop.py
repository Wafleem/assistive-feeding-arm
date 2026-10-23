"""Scripted spoon scoop on the real SO-101 kinematics (grasping/scoop.py): reach the bowl, tip the
spoon down to its bottom, turn it back up through the food, lift, come back. Every streamed
setpoint is re-checked here against the bowl, the joint limits and the speed caps."""

import numpy as np
import pytest

from assistive_arm.grasping.scoop import (
    BowlTarget, ScoopConfig, _FastFK, _Spoon, pan_axis_xy, plan_scoop, spoon_collision,
)
from assistive_arm.robot_iface.so101_kinematics import fk, joint_limits

LOOK = np.array([12.04, -98.2, 70.07, 16.88, -7.3])  # poses/so101_face.json "look" on the rig
LEVEL_PITCH = -4.83  # the rig's saved spoon level (tool pitch, deg)
CFG = ScoopConfig(dip_deg=50.0)  # deep enough to scoop; quick to plan


def bowl_at(r, az_deg=0.0, rim=0.06, depth=0.04, z=0.008):
    a = np.radians(az_deg)
    return BowlTarget(np.r_[pan_axis_xy() + r * np.array([np.cos(a), np.sin(a)]), z], rim, depth)


BOWL = bowl_at(0.36, -25.0)


@pytest.fixture(scope="module")
def plan():
    p = plan_scoop(LOOK, BOWL, CFG, LEVEL_PITCH)
    assert p.ok, p.reason
    return p


def phase(plan, name):
    a, b = next((s, e) for n, s, e in plan.phases if n == name)
    return (plan.t >= a - 1e-9) & (plan.t <= b + 1e-9)


def test_fast_fk_is_the_same_kinematics():
    f = _FastFK()
    for q in np.random.default_rng(0).uniform(-90, 90, (50, 5)):
        assert np.allclose(f(q), fk(q), atol=1e-12)


def test_pan_axis_is_in_front_of_the_base_origin():
    assert np.allclose(pan_axis_xy(), [0.0388, 0.0], atol=0.001)


def test_scoop_reaches_the_bottom_tipped_down_then_comes_back_level_to_the_start(plan):
    assert [n for n, _, _ in plan.phases] == ["over", "dip", "scoop", "lift", "back"]
    bottom = BOWL.bottom[2]
    assert plan.tip[:, 2].min() - bottom == pytest.approx(CFG.bottom_clearance, abs=0.001)  # never presses on it
    dip = phase(plan, "dip")
    assert plan.tilt_deg[dip][-1] == pytest.approx(-plan.report["dip_deg"], abs=1.0)  # tipped down at the bottom
    assert plan.report["dip_deg"] >= 40.0
    sc = phase(plan, "scoop")
    assert plan.tilt_deg[sc][-1] == pytest.approx(CFG.end_tip_up_deg, abs=1.0)  # turned back up: food held
    lift = phase(plan, "lift")
    assert np.all(np.abs(plan.tilt_deg[lift] - CFG.end_tip_up_deg) < 1.0)  # carried out (nearly) level
    assert plan.tip[lift][-1, 2] - bottom > BOWL.depth + CFG.rim_clearance  # out of the bowl
    assert np.allclose(plan.q[-1], LOOK)  # back where it started


def test_the_tip_sweeps_along_the_bottom_toward_the_far_wall(plan):
    u = np.r_[(BOWL.bottom[:2] - pan_axis_xy()) / np.linalg.norm(BOWL.bottom[:2] - pan_axis_xy()), 0]
    sc = phase(plan, "scoop")
    x = (plan.tip[sc] - BOWL.bottom) @ u
    assert np.all(np.diff(x) > -1e-4)  # outward, never back
    assert x[-1] - x[0] >= CFG.min_scoop_length
    low = plan.tip[sc][:, 2] - BOWL.bottom[2] < CFG.bottom_clearance + 0.005
    assert low.mean() > 0.3  # a good part of the sweep hugs the bottom


def test_every_streamed_setpoint_is_clear_of_the_bowl_and_the_table(plan):
    u = np.r_[(BOWL.bottom[:2] - pan_axis_xy()) / np.linalg.norm(BOWL.bottom[:2] - pan_axis_xy()), 0]
    sp = _Spoon(CFG, u)
    for q in plan.q:
        assert spoon_collision(*sp.points(q), BOWL, CFG) is None


def test_joints_stay_in_limits_on_one_elbow_side_with_the_roll_held(plan):
    lim = joint_limits()
    assert np.all(plan.q[:, :4] >= np.minimum(lim[:4, 0] + CFG.limit_margin_deg, LOOK[:4]) - 1e-6)
    assert np.all(plan.q[:, :4] <= lim[:4, 1] - CFG.limit_margin_deg + 1e-6)
    assert plan.q[:, 2].min() >= CFG.elbow_min_deg - 1e-6  # no flip through a straight elbow
    assert np.allclose(plan.q[:, 4], LOOK[4])  # the spoon's hollow stays the side that was levelled


def test_speeds_are_capped_and_every_phase_starts_and_ends_at_rest(plan):
    dt = 1.0 / CFG.rate_hz
    v = np.abs(np.diff(plan.q, axis=0)) / dt
    assert v.max() <= CFG.max_joint_speed_deg * 1.05
    tip_v = np.linalg.norm(np.diff(plan.tip, axis=0), axis=1) / dt
    sc = phase(plan, "scoop")[1:]
    assert tip_v[sc].max() <= CFG.scoop_speed * 1.05
    for _, a, b in plan.phases:
        for t in (a, b):
            i = int(np.argmin(np.abs(plan.t - t)))
            near = v[max(0, i - 1):i + 1]
            assert near.max() < 0.1 * CFG.max_joint_speed_deg  # at rest between phases


def test_the_deepest_reachable_dip_is_used_when_90_deg_is_out_of_reach():
    p = plan_scoop(LOOK, BOWL, ScoopConfig(dip_deg=90.0), LEVEL_PITCH)
    assert p.ok and p.report["dip_deg"] < 90.0 and "not 90" in p.reason
    assert p.report["dip_deg"] >= 40.0


@pytest.mark.parametrize("bowl, why", [
    (bowl_at(0.20), "out of the arm's reach"),  # too close for a 21 cm spoon
    (bowl_at(0.36, rim=0.025), "too small"),
    (bowl_at(0.36, depth=0.10), "rim"),  # too deep: the handle can't get under the rim
])
def test_impossible_bowls_are_refused_before_anything_moves(bowl, why):
    p = plan_scoop(LOOK, bowl, ScoopConfig(dip_deg=50.0, min_dip_deg=40.0), LEVEL_PITCH)
    assert not p.ok and why in p.reason and len(p.q) == 0
