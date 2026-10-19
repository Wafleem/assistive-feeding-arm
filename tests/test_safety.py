import numpy as np

from assistive_arm.geometry import Pose
from assistive_arm.planning.safety import SafetyConfig, SafetyEnvelope


def test_user_keep_out_and_reach():
    env = SafetyEnvelope()
    assert env.check_point([0.4, 0.0, 0.3]) == []
    assert any(v.rule == "keep_out" for v in env.check_point([-0.2, 0.3, 0.3]))  # user's chest
    assert any(v.rule == "reach" for v in env.check_point([0.8, 0.0, 0.0]))
    assert any(v.rule == "min_height" for v in env.check_point([0.4, -0.3, -0.5]))


def test_path_through_user_is_rejected_even_if_endpoints_are_fine():
    env = SafetyEnvelope()
    a, b = Pose([-0.05, -0.30, 0.3]), Pose([-0.05, 0.60, 0.3])  # either side of the user's chest
    assert env.check_pose(a) == []
    assert env.check_pose(b) == []
    assert env.check_path(a, b)


def test_speed_drops_near_the_user():
    env = SafetyEnvelope()
    far = env.speed_limit([0.45, -0.30, 0.3])
    near = env.speed_limit([0.15, 0.05, 0.3])
    assert far == SafetyConfig().max_speed and near == SafetyConfig().near_user_speed
    assert env.speed_limit([0.45, -0.3, 0.3], requested=0.02) == 0.02


def test_force_ceiling():
    assert SafetyEnvelope().clamp_force(100.0) == SafetyConfig().max_contact_force
    assert np.isclose(SafetyEnvelope().clamp_force(3.0), 3.0)
