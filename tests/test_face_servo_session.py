"""The dashboard's face-servo sequence end to end on a fake SO-101 and a fake face (offline)."""

import time
from types import SimpleNamespace

import numpy as np

from assistive_arm.feeding.face_servo import FaceServoConfig
from assistive_arm.geometry import Pose
from assistive_arm.mouth_perception.types import MouthObservation
from assistive_arm.orchestration.face_servo_session import FaceServoSession
from assistive_arm.vision.types import CameraIntrinsics, RGBDFrame

K = CameraIntrinsics(780.0, 780.0, 320.0, 240.0, 640, 480)
LOOK = np.array([60.0, -90.0, 80.0, 20.0, 0.0])
NEAR = np.array([60.0, -50.0, 40.0, 15.0, 0.0])
TRUE_J = np.array([[-12.0, 1.0], [0.5, 10.0]])  # px per degree (pan, flex)


class FakeSO101:
    def __init__(self):
        self.q = np.array([0.0, -100.0, 90.0, 30.0, 0.0])
        self.moves = 0
        self.teleop = True

    def now(self):
        return time.monotonic()

    def joints(self):
        return self.q.copy()

    def move_joints(self, q):
        self.q = np.asarray(q, float).copy()
        self.moves += 1

    def set_speed(self, v):
        return v

    def stop(self):
        pass

    def stop_teleop(self):
        self.teleop = False

    def start_teleop(self):
        self.teleop = True

    def get_frames(self, names=None):
        time.sleep(0.01)  # a camera delivers frames at a finite rate
        return {"webcam-2": RGBDFrame("webcam-2", np.zeros((480, 640, 3), np.uint8), np.zeros((480, 640), np.float32),
                                      K, timestamp=self.now())}

    def camera_pose(self, name):
        return Pose()


class FakeFace:
    """The face sits off to the side; the image responds to pan/flex with TRUE_J, and the
    distance shrinks as the arm moves from LOOK toward NEAR (0.70 m -> 0.30 m)."""

    def __init__(self, robot):
        self.robot = robot
        self.last = None
        self.last_landmarks = None

    def observe(self, robot):
        q = self.robot.q
        if abs(q[0] - LOOK[0]) > 40:  # facing the other way: no face
            self.last = self.last_landmarks = None
            return None
        px = np.array([420.0, 300.0]) + TRUE_J @ np.array([q[0] - LOOK[0], q[3] - LOOK[3]])
        s = np.clip((q[1] - LOOK[1]) / (NEAR[1] - LOOK[1]), 0, 1)
        dist = 0.70 - 0.40 * s
        lm = np.zeros((478, 3))
        lm[13, :2] = lm[14, :2] = px / [640, 480]
        self.last_landmarks = lm
        self.last = SimpleNamespace(distance=dist, frontalness=0.95)
        # The person sits still: in the base frame the mouth does not move while the arm does.
        return MouthObservation(robot.now(), np.array([0.8, -0.3, 0.3]), np.array([-1.0, 0, 0]), 0.6, 0.95)


def wait_idle(session, timeout=10.0):
    deadline = time.monotonic() + timeout
    time.sleep(0.05)
    while session.mode not in ("idle", "done") and time.monotonic() < deadline:
        time.sleep(0.02)
    return session.mode


def test_teach_sweep_calibrate_center_extend(tmp_path):
    robot = FakeSO101()
    s = FaceServoSession(robot, FakeFace(robot), poses_path=tmp_path / "p.json",
                         cfg=FaceServoConfig(stop_tip_distance=0.20), settle=0.0)
    s.start()
    try:
        robot.q = LOOK.copy(); s.command("teach_look")
        robot.q = NEAR.copy(); s.command("teach_near")
        robot.q = np.array([0.0, -100.0, 90.0, 30.0, 0.0])

        s.command("home"); assert wait_idle(s) == "idle"
        assert robot.q[0] == 0.0 and np.allclose(robot.q[1:], LOOK[1:]) and robot.teleop is False

        s.command("search"); assert wait_idle(s) == "idle"
        assert "face found" in s.reason and abs(robot.q[0] - LOOK[0]) <= 40  # stopped on the way, face in view

        s.command("sweep"); assert wait_idle(s) == "idle"
        assert np.allclose(robot.q, LOOK)

        s.command("calibrate"); assert wait_idle(s) == "idle", s.reason
        J = np.asarray(s.taught.jacobian["J_px_per_deg"])
        assert np.allclose(J, TRUE_J, atol=0.5), J

        s.command("center"); time.sleep(1.0); s.command("stop")
        st = s.state()
        assert max(abs(e) for e in st["face"]["error_px"]) <= s.cfg.center_tol_px, st

        s.command("extend"); assert wait_idle(s) == "done", s.reason
        tip = s.state()["face"]["tip_distance_cm"]
        assert 17.0 <= tip <= 21.0, tip  # stopped at the 20 cm stop distance, not closer
    finally:
        s.close()


def test_extend_refuses_without_taught_poses(tmp_path):
    robot = FakeSO101()
    s = FaceServoSession(robot, FakeFace(robot), poses_path=tmp_path / "p.json")
    s.start()
    try:
        s.command("extend")
        assert wait_idle(s) == "idle" and "teach" in s.reason
        assert robot.moves == 0
    finally:
        s.close()


def test_auto_center_needs_only_the_look_pose(tmp_path):
    robot = FakeSO101()
    s = FaceServoSession(robot, FakeFace(robot), poses_path=tmp_path / "p.json", settle=0.0)
    s.start()
    try:
        robot.q = LOOK.copy(); s.command("teach_look")
        robot.q = np.array([0.0, -90.0, 80.0, 20.0, 0.0])  # facing away: no face
        s.command("auto_center")
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:  # searches, calibrates, then centres (runs until STOP)
            f = s.state()["face"]
            if s.taught.jacobian and f and "error_px" in f and max(map(abs, f["error_px"])) <= s.cfg.center_tol_px:
                break
            time.sleep(0.05)
        s.command("stop")
        assert s.taught.jacobian is not None and s.taught.near is None
        assert max(map(abs, s.state()["face"]["error_px"])) <= s.cfg.center_tol_px
    finally:
        s.close()
