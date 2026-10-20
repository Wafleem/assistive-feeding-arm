"""Spec for section 4 (feeding motion and safety). Remove @todo as you implement."""

import numpy as np

from assistive_arm.feeding import FeedingConfig, check_near_face, final_pose, staging_pose
from assistive_arm.mouth_perception import MouthState
from conftest import todo

N = np.array([0.0, -1.0, 0.0])
MOUTH = MouthState(0.0, np.array([0.30, 0.30, 0.32]), np.zeros(3), N, 1.0, True, 1.0, 0.0)
CFG = FeedingConfig()


@todo
def test_final_pose_stops_in_front_of_the_lips():
    p = final_pose(MOUTH, CFG)
    assert np.isclose(np.linalg.norm(p.position - MOUTH.position), CFG.standoff, atol=1e-6)
    assert np.dot(p.position - MOUTH.position, N) > 0  # on the outside of the face
    assert np.allclose(p.z_axis, -N, atol=1e-6)  # approaching along the face normal


@todo
def test_staging_pose_is_further_out_on_the_same_line():
    s, f = staging_pose(MOUTH, CFG), final_pose(MOUTH, CFG)
    d = s.position - f.position
    assert np.isclose(np.linalg.norm(d), CFG.pre_approach - CFG.standoff, atol=1e-6)
    assert np.allclose(d / np.linalg.norm(d), N, atol=1e-6)


@todo
def test_nothing_may_get_closer_than_the_standoff_or_enter_the_head():
    assert check_near_face(MOUTH.position + 0.01 * N, MOUTH, CFG)  # 1 cm from the lips: too close
    assert check_near_face(MOUTH.position + np.array([0.08, 0.06, 0.0]), MOUTH, CFG)  # beside the cheek
    assert not check_near_face(MOUTH.position + 0.05 * N, MOUTH, CFG)  # in the corridor, 5 cm out
