"""Additive RGB frame contract specs; no SDK or camera implementation required."""

import numpy as np
import pytest

from assistive_arm.robot_iface.fake import FakeRobot
from assistive_arm.vision.types import CameraIntrinsics, RGBDFrame, RGBFrame


def test_rgb_frame_does_not_imply_depth_or_calibration():
    frame = RGBFrame("wrist", np.zeros((2, 3, 3), dtype=np.uint8), 10.0)
    assert frame.captured_at_s is None and frame.capture_clock is None
    assert not hasattr(frame, "depth") and not hasattr(frame, "intrinsics")


@pytest.mark.parametrize("timestamp, expected", [(12.5, 12.5), (0.0, None), (float("nan"), None)])
def test_rgbd_projection_preserves_pixels_without_assuming_clock(timestamp, expected):
    rgb = np.zeros((2, 3, 3), dtype=np.uint8)
    depth = np.full((2, 3), 0.4, dtype=np.float32)
    original = RGBDFrame("wrist", rgb, depth, CameraIntrinsics(2, 2, 1, 1, 3, 2), timestamp=timestamp)

    class RGBDRobot(FakeRobot):
        def get_frames(self, names=None):
            assert names == ["wrist"]
            return {"wrist": original}

        def now(self):
            return 20.0

    robot = RGBDRobot()
    projected = robot.get_rgb_frames(["wrist"])["wrist"]
    assert projected.rgb is rgb and original.depth is depth
    assert projected.received_at_s == 20.0
    assert projected.captured_at_s == expected
    assert projected.capture_clock is None
    assert robot.get_frames(["wrist"])["wrist"] is original
