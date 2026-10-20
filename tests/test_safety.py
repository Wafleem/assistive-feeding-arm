import numpy as np

from assistive_arm.geometry import Pose
from assistive_arm.motion.safety import SafetyConfig, SafetyEnvelope, Sphere


def test_torso_keep_out_reach_and_table():
    env = SafetyEnvelope()
    assert env.check_point([0.35, 0.0, 0.10]) == []  # above the plate
    assert any(v.rule == "keep_out" for v in env.check_point([0.30, 0.60, 0.0]))  # the user's chest
    assert any(v.rule == "reach" for v in env.check_point([0.8, 0.0, 0.1]))
    assert any(v.rule == "min_height" for v in env.check_point([0.4, 0.0, -0.05]))  # into the table


def test_moving_zones_add_to_static_ones():
    head = Sphere("head", (0.30, 0.40, 0.35), 0.12)
    env = SafetyEnvelope().with_zones(head)
    assert any("head" in str(v) for v in env.check_point([0.30, 0.35, 0.35]))
    assert env.check_point([0.30, 0.20, 0.35]) == []


def test_path_through_a_zone_is_rejected():
    env = SafetyEnvelope().with_zones(Sphere("head", (0.30, 0.40, 0.35), 0.12))
    assert env.check_path(Pose([0.15, 0.40, 0.35]), Pose([0.45, 0.40, 0.35]))


def test_speed_drops_near_the_user():
    env = SafetyEnvelope().with_zones(Sphere("head", (0.30, 0.40, 0.35), 0.12))
    assert env.speed_limit([0.40, -0.20, 0.10]) == SafetyConfig().max_speed
    assert env.speed_limit([0.30, 0.22, 0.35]) == SafetyConfig().near_user_speed
    assert np.isclose(env.clamp_force(100.0), SafetyConfig().max_contact_force)
