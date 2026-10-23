"""Auto center end to end with the continuous tracker, on a fake arm that uses the real
SO-101 kinematics (streamed setpoints are followed exactly) and a face that is only seen
when the camera points within 25 deg of it."""

import time
from types import SimpleNamespace

import numpy as np
import pytest

from assistive_arm.geometry import Pose
from assistive_arm.mouth_perception.types import MouthObservation
from assistive_arm.orchestration.face_servo_session import FaceServoSession
from assistive_arm.robot_iface.so101_kinematics import fk
from assistive_arm.vision.types import CameraIntrinsics, RGBDFrame

K = CameraIntrinsics(780.0, 780.0, 320.0, 240.0, 640, 480)
LOOK = np.array([-30.0, -89.1, 89.3, -12.7, -7.6])


def face_for(q, dist=0.6, up=0.06):
    T = fk(q)
    d = T[:3, 2].copy(); d[2] = 0; d /= np.linalg.norm(d)
    return T[:3, 3] + dist * d + np.array([0, 0, up])


FACE = face_for(LOOK)


class KinArm:
    def __init__(self, q):
        self.q = np.asarray(q, float).copy()
        self.streamed = []
        self.teleop = True

    def now(self):
        return time.monotonic()

    def joints(self, timeout=None):
        return self.q.copy()

    def move_joints(self, q):
        self.q = np.asarray(q, float).copy()

    def stream_joints(self, q):
        q = np.asarray(q, float)
        self.streamed.append(q.copy())
        self.q = q.copy()

    def tool_pose(self, timeout=None):
        T = fk(self.q)
        return T[:3, 3], T[:3, 2], 0.0

    def camera_pose(self, name):
        T = fk(self.q)
        return Pose.from_matrix(T[:3, 3], T[:3, :3])

    def arm_command(self, cmd):
        self.commands = getattr(self, "commands", []) + [cmd]
        return {"mode": "manual" if cmd.get("command") == "enter_manual_mode" else "auto"}

    def set_speed(self, v):
        return v

    def stop(self):
        pass

    def stop_teleop(self):
        self.teleop = False

    def start_teleop(self):
        self.teleop = True

    def get_frames(self, names=None):
        time.sleep(0.02)
        return {"webcam-2": RGBDFrame("webcam-2", np.zeros((480, 640, 3), np.uint8), np.zeros((480, 640), np.float32),
                                      K, timestamp=self.now())}


class SeenFace:
    def __init__(self, arm):
        self.arm, self.n = arm, 0
        self.last = self.last_landmarks = None

    def observe(self, robot):
        self.n += 1
        T = fk(self.arm.q)
        off = np.degrees(np.arccos(np.clip(np.dot(T[:3, 2], (FACE - T[:3, 3]) / np.linalg.norm(FACE - T[:3, 3])), -1, 1)))
        if off > 25 or self.n % 5 == 0:  # out of view, or a dropped frame
            self.last = self.last_landmarks = None
            return None
        lm = np.zeros((478, 3)); lm[13, :2] = lm[14, :2] = 0.5
        self.last_landmarks = lm
        self.last = SimpleNamespace(distance=float(np.linalg.norm(FACE - T[:3, 3])), frontalness=0.95)
        return MouthObservation(robot.now(), FACE.copy(), np.array([-1.0, 0, 0]), 0.6, 0.95)


def wait(until, timeout=15.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if until():
            return True
        time.sleep(0.05)
    return False


def test_auto_center_searches_then_tracks_smoothly_onto_the_lips_with_a_level_spoon(tmp_path):
    arm = KinArm(LOOK)
    s = FaceServoSession(arm, SeenFace(arm), poses_path=tmp_path / "p.json", settle=0.0, log_dir=tmp_path)
    s.start()
    try:
        s.command("teach_look")
        arm.q = np.array([0.0, -89.1, 89.3, -12.7, -7.6])  # straight ahead: the face is out of view
        s.command("auto_center")
        ok = wait(lambda: s.track is not None and s.track.last and s.track.last["miss_cm"] <= 1.2
                  and abs(s.track.last["pitch_deg"]) <= 1.0)
        assert ok, (s.reason, s.track.last if s.track else None)
        assert s.mode == "center" and not arm.teleop
        q = np.array(arm.streamed)
        step = np.abs(np.diff(q, axis=0)).max(axis=0)  # per 50 ms tick
        assert step[:4].max() <= 30.0 / 20 + 1e-6  # never faster than the velocity limit
        assert np.allclose(q[:, 4], q[0, 4])  # wrist roll never moved
        s.command("stop")
        assert s.mode == "idle"
    finally:
        s.close()


def test_spoon_level_calibration_nudges_saves_and_becomes_the_trackers_level(tmp_path):
    from assistive_arm.orchestration.face_servo_session import _tool_pitch
    arm = KinArm(LOOK)
    s = FaceServoSession(arm, SeenFace(arm), poses_path=tmp_path / "p.json", settle=0.0)
    s.start()
    try:
        s.command("hand_guide_on")
        assert arm.commands[-1] == {"command": "enter_manual_mode"} and s.hand_guide and not arm.teleop
        p0 = _tool_pitch(arm.q)
        s.command("nudge_spoon", deg=5)
        assert _tool_pitch(arm.q) - p0 == pytest.approx(5.0, abs=0.2)  # tip up = tool pitch up
        s.command("nudge_spoon", deg=-1)
        saved = s.command("save_level")
        lvl = s.taught.spoon_level["tool_pitch_deg"]
        assert lvl == pytest.approx(_tool_pitch(arm.q), abs=0.01) and "saved" in saved["msg"]
        assert s.track_cfg.pitch_target_deg == lvl and s.pcfg.level_pitch_deg == lvl
        assert abs(s.state()["level"]["spoon_tilt_now_deg"]) < 0.1
        s2 = FaceServoSession(arm, SeenFace(arm), poses_path=tmp_path / "p.json")  # survives a restart
        assert s2.track_cfg.pitch_target_deg == lvl
    finally:
        s.close()
