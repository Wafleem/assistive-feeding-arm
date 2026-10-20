"""Section 3 (grasping). The VLA runner works; the grasp planner is a spec."""

import numpy as np

from assistive_arm.geometry import Pose
from assistive_arm.grasping import pca_topdown_grasp
from assistive_arm.grasping.vla import ScriptedPolicy, run_vla
from assistive_arm.motion.primitives import SkillContext
from assistive_arm.robot_iface import FakeRobot
from conftest import todo

YES = SkillContext(confirm=lambda m: True)


def sphere_points(c, r, n=300, seed=0):
    v = np.random.default_rng(seed).normal(size=(n, 3))
    p = np.asarray(c) + r * v / np.linalg.norm(v, axis=1, keepdims=True)
    return p[p[:, 2] > c[2] - 0.3 * r]  # the camera sees the top


@todo
def test_pca_grasp_on_a_grape():
    g = pca_topdown_grasp(sphere_points([0.32, -0.04, 0.021], 0.011), support_z=0.01)
    assert g is not None and 0.015 < g.width < 0.03
    assert np.allclose(g.pose.z_axis, [0, 0, -1], atol=1e-6)  # top-down
    assert 0.012 < g.pose.position[2] < 0.025  # fingertips above the plate, around the equator


@todo
def test_pca_grasp_rejects_too_wide():
    assert pca_topdown_grasp(sphere_points([0.4, 0, 0.05], 0.06), support_z=0.0) is None


def test_vla_runner_clamps_steps_and_finishes():
    r = FakeRobot(start=Pose([0.30, -0.30, 0.30], [0, 1, 0, 0]))
    res = run_vla(r, YES, ScriptedPolicy([0.40, -0.20, 0.20]), "pick up the grape")
    assert res.status == "success", res
    assert res.evidence["clamped"] > 0


def test_vla_runner_refuses_steps_into_the_user():
    r = FakeRobot(start=Pose([0.30, 0.30, 0.10], [0, 1, 0, 0]))
    res = run_vla(r, YES, ScriptedPolicy([0.30, 0.60, 0.10]), "hand it to me")
    assert res.status == "refused"
