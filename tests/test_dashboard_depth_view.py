"""Dashboard depth view: depth and the mouth in one frame (skipped without OpenCV)."""

import importlib.util
import types
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from assistive_arm.mouth_perception import face_geometry as fg  # noqa: E402
from assistive_arm.vision.types import CameraIntrinsics  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("dashboard_server", ROOT / "dashboard" / "server.py")
server = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(server)

K = CameraIntrinsics(500.0, 500.0, 320.0, 240.0, 640, 480)


def face_landmarks() -> np.ndarray:
    """478 normalised landmarks: everything at the image centre, the lips on a small ring
    around (0.5, 0.6) and the mouth opening at its centre."""
    lm = np.tile([0.5, 0.5, 0.0], (478, 1))
    ring = server.LIPS_OUTER + server.LIPS_INNER + list(server.LIP_SURFACE)
    for n, i in enumerate(ring):
        a = 2 * np.pi * n / len(ring)
        lm[i, :2] = [0.5 + 0.05 * np.cos(a), 0.6 + 0.02 * np.sin(a)]
    lm[[fg.UPPER_INNER_LIP, fg.LOWER_INNER_LIP], :2] = [0.5, 0.6]
    return lm


def test_lips_depth_reads_the_lips_not_the_open_mouth():
    lm = face_landmarks()
    px = fg.landmarks_to_pixels(lm, K)
    depth = np.full((480, 640), 0.20, np.float32)
    depth[285:291, 317:324] = 0.9  # the opening: the far side of the mouth, around (320, 288)
    assert server.lips_depth(px, depth) == pytest.approx(0.20)
    assert server.lips_depth(px, np.zeros_like(depth)) is None  # no depth there: no number


def test_depth_view_renders_one_aligned_frame():
    rgb = np.full((480, 640, 3), 120, np.uint8)
    depth = np.full((480, 640), 0.25, np.float32)
    depth[:, :100] = 0.0  # no depth on the left: plain colour there
    frame = types.SimpleNamespace(rgb=rgb, depth=depth, intrinsics=K)
    session = types.SimpleNamespace(depth_status="ok", det=types.SimpleNamespace(params=fg.FaceGeometryParams()))
    render = server.make_depth_render()
    for lm in (face_landmarks(), None):  # with and without a face
        jpg = render(frame, lm, session)
        img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
        assert img.shape == (480, 640, 3)
        assert abs(int(img[100, 50, 0]) - 120) < 12  # no depth: the picture shows through unchanged
        assert int(img[100, 400, 2]) > int(img[100, 400, 0]) + 20  # 25 cm is near: tinted red (BGR)
