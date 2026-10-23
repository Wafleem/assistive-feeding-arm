"""Bowl finding and tracking with a fake detector and a known camera pose (offline)."""

import numpy as np

from assistive_arm.food_perception import BowlFinder, BowlParams, BowlTracker
from assistive_arm.geometry import Pose
from assistive_arm.vision.types import CameraIntrinsics, Detection, RGBDFrame

K = CameraIntrinsics(fx=820.0, fy=820.0, cx=320.0, cy=240.0, width=640, height=480)


def look_down_pose(height=0.5, x=0.25):
    """Camera above the table looking straight down: optical z = -base z, image x = base -y."""
    R = np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]], float)  # columns: cam x, y, z in the base frame
    return Pose.from_matrix([x, 0.0, height], R)


def box_for(point_base, radius, cam_pose):
    p = cam_pose.inverse().transform_point(point_base)
    u, v = K.fx * p[0] / p[2] + K.cx, K.fy * p[1] / p[2] + K.cy
    r_px = K.fx * radius / np.linalg.norm(point_base - cam_pose.position)
    return (u - r_px, v - r_px, u + r_px, v + r_px)


class FakeDetector:
    def __init__(self, boxes):
        self.boxes = boxes  # {label: [(bbox, score), ...]}

    def detect(self, frame, query):
        return [Detection(query, s, b) for b, s in self.boxes.get(query, [])]


class Rig:
    def __init__(self, pose, depth=None):
        self.pose, self.depth = pose, depth

    def now(self):
        return 0.0

    def get_frames(self, names=None):
        d = self.depth if self.depth is not None else np.zeros((480, 640), np.float32)
        return {"webcam-2": RGBDFrame("webcam-2", np.zeros((480, 640, 3), np.uint8), d, K)}

    def camera_pose(self, name):
        return self.pose


def test_monocular_bowls_land_on_the_table_plane():
    pose = look_down_pose()
    p = BowlParams(rim_height=0.04)
    truth = [np.array([0.30, 0.08, 0.04]), np.array([0.22, -0.10, 0.04])]
    boxes = {"bowl": [(box_for(t, 0.05, pose), 0.9) for t in truth]}
    found = BowlFinder(FakeDetector(boxes), params=p).find(Rig(pose))
    assert len(found) == 2 and all(b.source == "table_plane" for b in found)
    for b in found:
        err = min(np.linalg.norm(b.position - t) for t in truth)
        assert err < 0.005 and abs(b.radius - 0.05) < 0.005


def test_bowl_and_cup_on_the_same_object_are_merged():
    pose = look_down_pose()
    box = box_for(np.array([0.3, 0.0, 0.04]), 0.04, pose)
    finder = BowlFinder(FakeDetector({"bowl": [(box, 0.8)], "cup": [(box, 0.6)]}))
    found = finder.find(Rig(pose))
    assert len(found) == 1 and found[0].label == "bowl"


def test_depth_is_used_when_the_camera_has_it():
    pose = look_down_pose(height=0.5)
    depth = np.full((480, 640), 0.42, np.float32)  # rim 8 cm above the table: not the assumed 4 cm
    box = box_for(np.array([0.25, 0.0, 0.08]), 0.05, pose)
    b = BowlFinder(FakeDetector({"bowl": [(box, 0.9)]})).find(Rig(pose, depth))[0]
    assert b.source == "depth" and abs(b.position[2] - 0.08) < 0.005


def test_tracker_keeps_ids_while_a_bowl_slides_and_drops_vanished_ones():
    pose = look_down_pose()
    finder = BowlFinder(FakeDetector({}))
    tr = BowlTracker()
    a, b = np.array([0.30, 0.08, 0.04]), np.array([0.22, -0.10, 0.04])
    ids = None
    for k in range(10):  # bowl a slides 1 cm per frame toward -y
        a_k = a + [0, -0.01 * k, 0]
        finder.detector = FakeDetector({"bowl": [(box_for(a_k, 0.05, pose), 0.9), (box_for(b, 0.05, pose), 0.9)]})
        bowls = tr.update(finder.find(Rig(pose)))
        now = {round(x.position[0], 2): x.id for x in bowls}
        ids = ids or now
        assert sorted(now.values()) == sorted(ids.values())  # no new ids while sliding
    for _ in range(tr.c.max_missed + 1):
        tr.update([])
    assert tr.bowls() == []
