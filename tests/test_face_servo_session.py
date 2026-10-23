"""Session behaviour that doesn't need a face: refusals and explanations.
Centring and extend run on the continuous tracker: tests/test_face_track_session.py."""

import time

import numpy as np

from assistive_arm.orchestration.face_servo_session import FaceServoSession
from test_face_track_session import LOOK, KinArm, SeenFace, wait


def test_auto_center_without_a_face_or_a_look_pose_explains_itself(tmp_path):
    arm = KinArm(np.array([90.0, -89.1, 89.3, -12.7, -7.6]))  # facing away: no face
    s = FaceServoSession(arm, SeenFace(arm), poses_path=tmp_path / "p.json")
    s.start()
    try:
        s.command("auto_center")
        assert wait(lambda: s.mode == "idle" and "teach the look pose" in s.reason), s.reason
        assert arm.streamed == []
    finally:
        s.close()


def test_extend_is_refused_when_disabled(tmp_path):
    arm = KinArm(LOOK)
    s = FaceServoSession(arm, SeenFace(arm), poses_path=tmp_path / "p.json", allow_extend=False)
    try:
        r = None
        try:
            s.command("extend")
        except RuntimeError as e:
            r = str(e)
        assert r and "disabled" in r and arm.streamed == []
    finally:
        s.close()
