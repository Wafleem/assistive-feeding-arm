"""The dashboard's face-servo sequence end to end on a fake SO-101 and a fake face (offline).

The fake arm aims its tool along (pan, elevation) from the joints; ``move_tool`` stands in for
the module's planned MoveToPosition. The fake face sits off to the right, is visible within
35 deg of the tool axis, and drops every 4th frame (MediaPipe misses happen)."""

import time
from types import SimpleNamespace

import numpy as np

from assistive_arm.feeding.face_servo import FaceServoConfig
from assistive_arm.geometry import Pose
from assistive_arm.mouth_perception.types import MouthObservation
from assistive_arm.orchestration.face_servo_session import FaceServoSession
from assistive_arm.vision.types import CameraIntrinsics, RGBDFrame

K = CameraIntrinsics(780.0, 780.0, 320.0, 240.0, 640, 480)
FACE = np.array([0.60, -0.60, 0.30])  # 45 deg to the right of straight ahead
LOOK = np.array([-40.0, -90.0, 80.0, 20.0, 0.0])


def direction(q):
    pan, el = np.radians(q[0]), np.radians(q[3] - 20.0)
    return np.array([np.cos(el) * np.cos(pan), np.cos(el) * np.sin(pan), np.sin(el)])


def angle(a, b):
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b), -1, 1))))


class FakeSO101:
    """SO-101-like: the tool sits at (reach r, height h) in the vertical plane at the base pan,
    and its axis can only point within that plane (no wrist yaw). Targets outside that plane
    fail like the real module's planner does."""

    def __init__(self):
        self.q = np.array([0.0, -90.0, 80.0, 20.0, 0.0])
        self.r, self.h = 0.15, 0.20
        self.moves = 0
        self.teleop = True
        self.flip_next = False
        self.fail_next = 0  # make the next N plans fail (planner trouble)
        self.plans = []

    @property
    def pos(self):
        pan = np.radians(self.q[0])
        return np.array([self.r * np.cos(pan), self.r * np.sin(pan), self.h])

    @pos.setter
    def pos(self, p):
        p = np.asarray(p, float)
        self.r, self.h = float(np.linalg.norm(p[:2])), float(p[2])
        self.q[0] = np.degrees(np.arctan2(p[1], p[0]))

    def now(self):
        return time.monotonic()

    def joints(self, timeout=None):
        return self.q.copy()

    def move_joints(self, q):
        self.q = np.asarray(q, float).copy()
        self.moves += 1

    def tool_pose(self, timeout=None):
        return self.pos.copy(), direction(self.q), 0.0

    def move_tool(self, position, d, theta=0.0, leeway_mm=8.0, leeway_orient=0.03):
        p, d = np.asarray(position, float), np.asarray(d, float) / np.linalg.norm(d)
        az = np.arctan2(p[1], p[0])
        out_of_plane = abs(np.dot(d, [-np.sin(az), np.cos(az), 0.0]))
        self.plans.append((leeway_mm, leeway_orient))
        if self.fail_next > 0 or out_of_plane > 1.5 * leeway_orient:
            self.fail_next = max(0, self.fail_next - 1)
            raise RuntimeError("zero IK solutions produced, goal positions appears to be physically unreachable")
        self.pos = p
        self.q[0] += 60.0 if self.flip_next else 0.0
        self.q[3] = np.degrees(np.arcsin(np.clip(d[2], -1, 1))) + 20.0
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
    def __init__(self, robot):
        self.robot, self.n, self.present = robot, 0, True
        self.last = self.last_landmarks = None

    def observe(self, robot):
        self.n += 1
        r = self.robot
        off = angle(direction(r.q), FACE - r.pos)
        if not self.present or off > 35 or self.n % 4 == 0:
            self.last = self.last_landmarks = None
            return None
        lm = np.zeros((478, 3))
        lm[13, :2] = lm[14, :2] = (np.array([320.0, 240.0]) + 13.6 * off) / [640, 480]
        self.last_landmarks = lm
        self.last = SimpleNamespace(distance=float(np.linalg.norm(FACE - r.pos)), frontalness=0.95)
        return MouthObservation(robot.now(), FACE.copy(), np.array([-1.0, 0, 0]), 0.6, 0.95)


def wait(until, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if until():
            return True
        time.sleep(0.02)
    return False


def make(tmp_path, **cfg):
    robot = FakeSO101()
    s = FaceServoSession(robot, FakeFace(robot), poses_path=tmp_path / "p.json",
                         cfg=FaceServoConfig(**cfg), settle=0.0, log_dir=tmp_path / "logs")
    s.start()
    return robot, s


def test_auto_center_finds_the_face_and_points_at_it_without_calibration(tmp_path):
    robot, s = make(tmp_path)
    try:
        robot.q = LOOK.copy(); s.command("teach_look")
        robot.q = np.array([0.0, -90.0, 80.0, 20.0, 0.0])  # straight ahead: face out of view
        s.command("auto_center")
        assert wait(lambda: angle(direction(robot.q), FACE - robot.pos) <= 3.0), s.reason
        assert s.mode == "center" and robot.teleop is False
        s.command("stop")
        assert s.mode == "idle"
    finally:
        s.close()


def test_extend_stops_at_the_stop_distance_and_retreats_when_the_face_goes(tmp_path):
    robot, s = make(tmp_path, stop_tip_distance=0.20)
    try:
        robot.q = LOOK.copy(); s.command("teach_look")
        s.command("extend")
        assert wait(lambda: s.mode == "done"), s.reason
        tip = np.linalg.norm(FACE - robot.pos) - s.pcfg.spoon_tip_from_tool
        assert 0.19 <= tip <= 0.215, tip

        s.command("retreat")
        assert wait(lambda: s.mode == "idle")
        assert np.allclose(robot.q, LOOK)

        robot.r = 0.15
        s.command("extend")
        assert wait(lambda: s.pointer is not None and s.pointer.travelled > 0.05)
        s.det.present = False  # the person leaves mid-approach
        assert wait(lambda: s.mode == "idle" and "retreat" in s.reason), s.reason
        assert np.allclose(robot.q, LOOK)
    finally:
        s.close()


def test_a_planned_step_that_swings_a_joint_is_stopped(tmp_path):
    robot, s = make(tmp_path)
    try:
        robot.q = np.array([-30.0, -90.0, 80.0, 20.0, 0.0])
        robot.flip_next = True  # the "planner" swings the base 60 deg
        s.command("center")
        assert wait(lambda: s.mode == "idle" and "planner moved a joint" in s.reason), s.reason
    finally:
        s.close()


def test_auto_center_without_a_face_or_a_look_pose_explains_itself(tmp_path):
    robot, s = make(tmp_path)
    try:
        s.command("auto_center")
        assert wait(lambda: s.mode == "idle" and "teach the look pose" in s.reason), s.reason
        assert robot.moves == 0
    finally:
        s.close()


def test_a_failed_plan_is_retried_wider_logged_and_only_stops_after_three(tmp_path):
    import json
    robot, s = make(tmp_path)
    try:
        robot.q = np.array([-30.0, -90.0, 80.0, 20.0, 0.0])
        robot.fail_next = 1  # first attempt fails, the wider retry succeeds
        s.command("center")
        assert wait(lambda: angle(direction(robot.q), FACE - robot.pos) <= 3.0), s.reason
        s.command("stop")
        recs = [json.loads(l) for l in s.plan_log.read_text().splitlines()]
        assert recs[0]["ok"] is False and "zero IK" in recs[0]["error"] and recs[1]["wide"] is True and recs[1]["ok"]
        assert all(r["target_pos_mm"] and r["joints_before"] for r in recs)

        robot.q = np.array([-30.0, -90.0, 80.0, 20.0, 0.0])
        robot.fail_next = 99  # the planner keeps failing
        s.command("center")
        assert wait(lambda: s.mode == "idle" and "3 times in a row" in s.reason), s.reason
    finally:
        s.close()
