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

    def joints_fast(self, timeout=None):
        return self.q.copy() + getattr(self, "sag", 0.0)

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
    def __init__(self, arm, face=None):
        self.arm, self.n = arm, 0
        self.face = FACE if face is None else face
        self.last = self.last_landmarks = None

    def observe(self, robot):
        self.n += 1
        T = fk(self.arm.q)
        F = self.face
        off = np.degrees(np.arccos(np.clip(np.dot(T[:3, 2], (F - T[:3, 3]) / np.linalg.norm(F - T[:3, 3])), -1, 1)))
        if off > 25 or self.n % 5 == 0:  # out of view, or a dropped frame
            self.last = self.last_landmarks = None
            return None
        lm = np.zeros((478, 3)); lm[13, :2] = lm[14, :2] = 0.5
        self.last_landmarks = lm
        self.last = SimpleNamespace(distance=float(np.linalg.norm(F - T[:3, 3])), frontalness=0.95)
        return MouthObservation(robot.now(), F.copy(), np.array([-1.0, 0, 0]), 0.6, 0.95)


def level_first(s):
    s.command("level_spoon")
    assert wait(lambda: s.mode == "idle" and "spoon level" in s.reason), s.reason


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
        level_first(s)
        s.command("auto_center")
        ok = wait(lambda: s.track is not None and s.track.last and s.track.last["miss_cm"] <= 1.6
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


def test_extend_glides_in_to_the_stop_distance_holds_then_backs_out_level(tmp_path):
    from assistive_arm.feeding.face_track import ExtendConfig
    arm = KinArm(LOOK)
    near = face_for(LOOK, dist=0.48, up=0.0)  # within the arm's reach for a 20 cm stop
    s = FaceServoSession(arm, SeenFace(arm, near), poses_path=tmp_path / "p.json", settle=0.0, log_dir=tmp_path)
    s.ext_cfg = ExtendConfig(stop_tip_distance=0.20, hold_s=1.0, approach_speed=0.05)  # faster for the test
    s.start()
    try:
        level_first(s)
        start = np.linalg.norm(fk(arm.q)[:3, 3][:2])
        s.command("extend")
        assert wait(lambda: s.track is not None and s.track.travelled > 0.03, 20), s.reason
        # backs out, then keeps tracking (no STOP needed before the next Extend)
        assert wait(lambda: s.mode == "center" and s.track.travelled <= 1e-4, 30), s.reason
        time.sleep(0.5)
        # back where it began (1.5 cm: the start was measured sagged, the sag feedback lifts it)
        assert abs(np.linalg.norm(fk(arm.q)[:3, 3][:2]) - start) < 0.015
        s.command("stop")
        q = np.array(arm.streamed)
        assert np.allclose(q[:, 4], q[0, 4])  # roll held
    finally:
        s.close()


def test_retreat_after_a_stopped_extend_glides_back_out(tmp_path):
    from assistive_arm.feeding.face_track import ExtendConfig
    arm = KinArm(LOOK)
    s = FaceServoSession(arm, SeenFace(arm, face_for(LOOK, dist=0.48, up=0.0)), poses_path=tmp_path / "p.json", settle=0.0)
    s.ext_cfg = ExtendConfig(stop_tip_distance=0.20, approach_speed=0.05)
    s.start()
    try:
        level_first(s)
        start = np.linalg.norm(fk(arm.q)[:3, 3][:2])
        s.command("extend")
        assert wait(lambda: s.track is not None and s.track.actual_travelled() > 0.04, 20)
        s.command("stop")
        assert np.linalg.norm(fk(arm.q)[:3, 3][:2]) - start > 0.03  # still extended
        s.command("retreat")
        assert wait(lambda: s.mode == "done" and "backed out" in s.reason, 20), s.reason
        assert abs(np.linalg.norm(fk(arm.q)[:3, 3][:2]) - start) < 0.015  # sag feedback lifts the sagged start
    finally:
        s.close()


def test_extend_toward_an_out_of_reach_face_goes_as_far_as_it_can_and_says_how_short(tmp_path):
    from assistive_arm.feeding.face_track import ExtendConfig
    arm = KinArm(LOOK)
    s = FaceServoSession(arm, SeenFace(arm, face_for(LOOK, dist=0.75, up=0.0)), poses_path=tmp_path / "p.json", settle=0.0)
    s.ext_cfg = ExtendConfig(stop_tip_distance=0.20, approach_speed=0.08)
    s.start()
    try:
        level_first(s)
        s.command("extend")
        assert wait(lambda: any(("beyond the arm's reach" in e["msg"] or "reach limit" in e["msg"] or "holding here" in e["msg"]) for e in s.log), 20), list(s.log)[:4]
        assert wait(lambda: s.track is not None and s.track.travelled > 0.0, 20), s.reason
        s.command("stop")
    finally:
        s.close()


def test_level_spoon_button_then_centring(tmp_path):
    from assistive_arm.orchestration.face_servo_session import _tool_pitch
    tilted = LOOK.copy(); tilted[3] -= 20.0  # wrist flexed: spoon ~20 deg off level
    arm = KinArm(tilted)
    s = FaceServoSession(arm, SeenFace(arm), poses_path=tmp_path / "p.json", settle=0.0)
    s.start()
    try:
        assert abs(s._spoon_tilt()) > 10
        s.command("level_spoon")
        assert wait(lambda: s.mode == "idle" and "spoon level" in s.reason), s.reason
        assert abs(s._spoon_tilt()) < 0.5
        assert np.allclose(arm.q[[0, 1, 2, 4]], tilted[[0, 1, 2, 4]])  # only the wrist flex moved
        s.command("center")
        assert wait(lambda: s.mode == "center"), s.reason
        s.command("stop")
    finally:
        s.close()



def test_jog_buttons_keep_the_spoon_level(tmp_path):
    arm = KinArm(LOOK)
    s = FaceServoSession(arm, SeenFace(arm), poses_path=tmp_path / "p.json", settle=0.0)
    s.start()
    try:
        level_first(s)
        T0 = fk(arm.q)
        s.command("jog", dz_cm=2)
        assert wait(lambda: s.mode == "idle" and "jogged" in s.reason), s.reason
        T1 = fk(arm.q)
        assert T1[2, 3] - T0[2, 3] == pytest.approx(0.02, abs=0.003) and abs(s._spoon_tilt()) < 0.6
        heading = np.arctan2(T1[1, 2], T1[0, 2])  # the spoon's heading (the pan axis is 39 mm off the origin)
        s.command("jog", yaw_deg=5)  # to the right
        assert wait(lambda: s.mode == "idle" and "yaw right" in s.reason), s.reason
        T2 = fk(arm.q)
        assert np.degrees(heading - np.arctan2(T2[1, 2], T2[0, 2])) == pytest.approx(5.0, abs=0.3)  # turned right 5 deg
        assert abs(s._spoon_tilt()) < 0.6
    finally:
        s.close()



def test_tuning_applies_live_and_the_camera_pose_uses_measured_joints(tmp_path):
    arm = KinArm(LOOK)
    s = FaceServoSession(arm, SeenFace(arm), poses_path=tmp_path / "p.json", settle=0.0)
    s.start()
    try:
        msg = s.command("set_tuning", gain=1.0, speed_pct=50, accel=40, deadband_cm=1.5, smoothing_cm=2.0, rate_hz=40)
        t = s.tuning()
        assert t == {"gain": 1.0, "speed_pct": 50, "accel": 40.0, "deadband_cm": 1.5, "smoothing_cm": 2.0, "rate_hz": 40.0}, t
        assert s.tracker.kf.p.sigma_meas == 0.02 and "tuning" in msg["kind"]
        level_first(s)
        arm.sag = np.array([0.0, 2.0, 0.0, 0.0, 0.0])  # the real shoulder sags 2 deg below the command
        s.command("center")
        assert wait(lambda: len(s._qm_hist) > 5)
        q_seen = s._qm_hist[-1][1]
        assert q_seen[1] - s._q_hist[-1][1][1] == pytest.approx(2.0, abs=0.5)  # pose uses the sagging arm
        s.command("stop")
    finally:
        s.close()


def test_feed_button_goes_home_sweeps_to_the_person_bites_and_comes_back_home(tmp_path):
    from assistive_arm.feeding.face_track import ExtendConfig
    home = LOOK.copy(); home[0] += 50.0  # the person is 50 deg of pan away from home: out of view
    arm = KinArm(home)
    s = FaceServoSession(arm, SeenFace(arm, face_for(LOOK, dist=0.48, up=0.0)), poses_path=tmp_path / "p.json",
                         settle=0.0, log_dir=tmp_path)
    s.taught.look, s.taught.home = LOOK.tolist(), home.tolist()
    s.ext_cfg = ExtendConfig(stop_tip_distance=0.20, hold_s=1.0, spoon_tip_from_tool=0.21)
    s.start()
    try:
        s.command("feed")
        assert wait(lambda: s.mode == "extend", 30), s.reason  # swept over and found the face
        assert wait(lambda: s.track is not None and s.track.actual_travelled() > 0.03, 30), s.reason
        assert wait(lambda: s.mode == "idle" and "feed done" in s.reason, 60), s.reason
        assert np.max(np.abs(arm.q - home)) < 3.0  # back home
        assert any("bite done" in e["msg"] for e in s.log)
    finally:
        s.close()


def test_extend_ranges_with_fresh_depth_and_falls_back_to_the_eye_span(tmp_path):
    arm = KinArm(LOOK)
    s = FaceServoSession(arm, SeenFace(arm), poses_path=tmp_path / "p.json", settle=0.0)
    fast = np.array([0.5, -0.1, 0.3])
    assert s._stop_face(fast)[1] == "eye span"  # no depth yet
    s._depth_lips = (time.monotonic(), np.array([0.45, -0.1, 0.3]), 0.4)
    face, src = s._stop_face(fast)
    assert src == "depth" and np.allclose(face, [0.45, -0.1, 0.3])
    s._depth_lips = (time.monotonic() - 5.0, np.array([0.45, -0.1, 0.3]), 0.4)  # stale
    assert s._stop_face(fast)[1] == "eye span"
