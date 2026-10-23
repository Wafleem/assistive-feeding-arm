"""Real-face mouth perception on synthetic landmarks (no MediaPipe, no camera needed)."""

import numpy as np

from assistive_arm.geometry import Pose
from assistive_arm.mouth_perception import FaceGeometryParams, MediaPipeMouthDetector, calibrate_eye_span, measure_face
from assistive_arm.mouth_perception import face_geometry as fg
from assistive_arm.vision.types import CameraIntrinsics, RGBDFrame

K = CameraIntrinsics(fx=820.0, fy=820.0, cx=320.0, cy=240.0, width=640, height=480)  # ~C270 at 640x480


def rot_y(deg):
    a = np.radians(deg)
    return np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])


def synthetic_face(dist=0.45, yaw=0.0, gap=0.004, offset=(0.0, 0.0)):
    """A flat-ish face (metres, face frame: x right, y down, -z toward the camera), placed at
    ``dist`` in front of the camera, turned by ``yaw`` degrees. Returns MediaPipe-style
    normalised landmarks and the true lips centre / normal in the camera frame."""
    pts = np.zeros((478, 3))
    pts[:, 2] = 0.0
    for i in fg.FACE_PLANE:
        pts[i] = np.random.default_rng(i).uniform([-0.06, -0.08, 0], [0.06, 0.09, 0])
    pts[fg.EYE_OUTER_R] = [-0.045, -0.03, 0.0]
    pts[fg.EYE_OUTER_L] = [0.045, -0.03, 0.0]
    pts[fg.MOUTH_CORNER_L] = [0.025, 0.05, 0.0]
    pts[fg.MOUTH_CORNER_R] = [-0.025, 0.05, 0.0]
    pts[fg.UPPER_INNER_LIP] = [0.0, 0.05 - gap / 2, 0.0]
    pts[fg.LOWER_INNER_LIP] = [0.0, 0.05 + gap / 2, 0.0]
    R = rot_y(yaw)
    centre = np.array([offset[0], offset[1], dist])
    cam = pts @ R.T + centre
    lips = np.array([0.0, 0.05, 0.0]) @ R.T + centre
    normal = R @ np.array([0.0, 0.0, -1.0])
    u = K.fx * cam[:, 0] / cam[:, 2] + K.cx
    v = K.fy * cam[:, 1] / cam[:, 2] + K.cy
    ref = cam[list(fg.DEPTH_POINTS), 2].mean()
    z_norm = (cam[:, 2] - ref) * K.fx / (dist * K.width)  # MediaPipe-like relative z (image-width units)
    return np.column_stack([u / K.width, v / K.height, z_norm]), lips, normal, cam


def test_frontal_face_range_position_and_normal_from_eye_span():
    lm, lips, normal, _ = synthetic_face(dist=0.45)
    m = measure_face(lm, K)
    assert m.distance_source == "eye_span"
    assert abs(m.distance - 0.45) < 0.01
    assert np.linalg.norm(m.position - lips) < 0.008
    assert np.dot(m.normal, normal) > 0.99 and m.frontalness > 0.95


def test_turned_face_keeps_range_and_reports_the_turn():
    lm, lips, normal, _ = synthetic_face(dist=0.5, yaw=30)
    m = measure_face(lm, K)
    assert abs(m.distance - 0.5) < 0.02  # eye span uses the 3D landmarks: turning does not shrink it
    assert np.degrees(np.arccos(np.clip(np.dot(m.normal, normal), -1, 1))) < 5
    assert m.frontalness < 0.9


def test_depth_image_overrides_the_eye_span_prior():
    lm, lips, _, cam = synthetic_face(dist=0.40)
    depth = np.full((K.height, K.width), 0.40, np.float32)
    small_eyes = FaceGeometryParams(eye_span_m=0.080)  # a wrong prior: depth must win
    m = measure_face(lm, K, depth, small_eyes)
    assert m.distance_source == "depth" and abs(m.distance - 0.40) < 0.005
    assert np.linalg.norm(m.position - lips) < 0.008


def test_mouth_opening_is_normalised():
    closed = measure_face(synthetic_face(gap=0.002)[0], K)
    wide = measure_face(synthetic_face(gap=0.03)[0], K)
    assert closed.open_ratio < 0.1 and wide.open_ratio > 0.9


def test_calibrate_eye_span_recovers_this_persons_span():
    lm, *_ = synthetic_face(dist=0.45)
    assert abs(calibrate_eye_span(lm, K, 0.45) - 0.090) < 0.001


class _Landmark:
    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z


class _Category:
    def __init__(self, name, score):
        self.category_name, self.score = name, score


class FakeLandmarker:
    """Stands in for mediapipe's FaceLandmarker: returns two faces (one far, one near)."""

    def __init__(self, faces, jaw_open=0.8):
        self.faces, self.jaw = faces, jaw_open

    def detect_for_video(self, image, ms):
        class R:
            pass

        r = R()
        r.face_landmarks = [[_Landmark(*p) for p in f] for f in self.faces]
        r.face_blendshapes = [[_Category("jawOpen", self.jaw)] for _ in self.faces]
        return r


class CameraOnlyRobot:
    """Minimal robot: one RGB camera (no depth) at a known pose."""

    def __init__(self, cam_pose):
        self.pose = cam_pose

    def now(self):
        return 12.0

    def get_frames(self, names=None):
        return {"webcam-2": RGBDFrame("webcam-2", np.zeros((480, 640, 3), np.uint8),
                                      np.zeros((480, 640), np.float32), K, timestamp=11.9)}

    def camera_pose(self, name):
        return self.pose


def test_mediapipe_detector_picks_the_nearest_face_and_maps_to_the_base_frame():
    near, lips, normal, _ = synthetic_face(dist=0.45)
    far, *_ = synthetic_face(dist=1.2, offset=(0.2, 0.0))
    cam_pose = Pose([0.1, 0.0, 0.3], [1, 0, 0, 0])
    det = MediaPipeMouthDetector(landmarker=FakeLandmarker([far, near]))
    obs = det.observe(CameraOnlyRobot(cam_pose))
    assert obs is not None and obs.source == "mediapipe" and obs.t == 11.9
    assert np.linalg.norm(obs.position - cam_pose.transform_point(lips)) < 0.01
    assert np.dot(obs.normal, normal) > 0.99
    assert obs.open_ratio == 0.8  # jawOpen blendshape wins over the lip-gap ratio
    assert obs.confidence > 0.9


def test_mediapipe_detector_rejects_a_side_on_face():
    side, *_ = synthetic_face(dist=0.45, yaw=80)
    det = MediaPipeMouthDetector(landmarker=FakeLandmarker([side]))
    assert det.observe(CameraOnlyRobot(Pose())) is None
