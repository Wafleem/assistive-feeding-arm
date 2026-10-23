"""Local SO-101 FK matches poses the real arm reported (orin-dev-kit, 2026-10-23)."""

import numpy as np
import pytest

from assistive_arm.robot_iface.so101_kinematics import fk, joint_limits

SAMPLES = [  # joints (deg), end position (mm), orientation vector (tool +Z)
    ([21.978, -89.143, 89.319, -12.659, -7.560], [143.40, -42.39, 164.27], [0.9054, -0.3654, 0.2162]),
    ([-1.582, -98.374, 74.022, 32.264, -2.198], [135.03, 2.48, 179.86], [0.9901, 0.0274, -0.1376]),
]


@pytest.mark.parametrize("q,pos,ov", SAMPLES)
def test_fk_matches_the_arm(q, pos, ov):
    T = fk(q)
    assert np.linalg.norm(T[:3, 3] * 1000 - pos) < 1.0  # mm
    assert np.degrees(np.arccos(np.clip(np.dot(T[:3, 2], np.asarray(ov) / np.linalg.norm(ov)), -1, 1))) < 0.5


def test_limits_are_the_modules():
    lim = joint_limits()
    assert lim.shape == (5, 2) and lim[0, 1] == pytest.approx(110.0, abs=0.01)
