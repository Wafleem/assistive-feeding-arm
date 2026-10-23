"""Offline MediaPipe boundary tests: no weights, downloads, camera or GPU."""

import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from assistive_arm.mouth_perception.landmarks_2d import (
    MediaPipeMouthLandmarker, MouthLandmarkerConfig,
)


@pytest.fixture
def fake_mediapipe(monkeypatch, tmp_path):
    state = SimpleNamespace(faces=[], options=None, images=[], close_count=0)
    model = tmp_path / "test.task"
    model.write_bytes(b"offline test placeholder, never read by a model")
    state.model_path = str(model)

    class BaseOptions:
        Delegate = SimpleNamespace(CPU="cpu")

        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class FaceLandmarker:
        @classmethod
        def create_from_options(cls, options):
            state.options = options
            return cls()

        def detect(self, image):
            state.images.append(image)
            return SimpleNamespace(face_landmarks=state.faces)

        def close(self):
            state.close_count += 1

    mp = ModuleType("mediapipe")
    mp.Image = lambda **kwargs: SimpleNamespace(**kwargs)
    mp.ImageFormat = SimpleNamespace(SRGB="srgb")
    tasks = ModuleType("mediapipe.tasks")
    python = ModuleType("mediapipe.tasks.python")
    vision = ModuleType("mediapipe.tasks.python.vision")
    python.BaseOptions = BaseOptions
    python.vision = vision
    tasks.python = python
    mp.tasks = tasks
    vision.FaceLandmarker = FaceLandmarker
    vision.FaceLandmarkerOptions = lambda **kwargs: SimpleNamespace(**kwargs)
    vision.RunningMode = SimpleNamespace(IMAGE="image")
    for module in (mp, tasks, python, vision):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    return state


def face(*, x_offset=0.0, closed=False):
    landmarks = [SimpleNamespace(x=0.5 + x_offset, y=0.5, z=999.0) for _ in range(478)]
    for index, x, y in ((61, 0.25, 0.5), (291, 0.75, 0.5),
                        (0, 0.5, 0.2), (17, 0.5, 0.8),
                        (13, 0.5, 0.5 if closed else 0.4),
                        (14, 0.5, 0.5 if closed else 0.6)):
        landmarks[index].x, landmarks[index].y = x + x_offset, y
    return landmarks


def test_pixel_geometry_uses_image_aspect_ratio_and_capture_time(fake_mediapipe):
    fake_mediapipe.faces = [face()]
    with MediaPipeMouthLandmarker(MouthLandmarkerConfig(fake_mediapipe.model_path)) as detector:
        (mouth,) = detector.detect_rgb(np.zeros((100, 200, 3), dtype=np.uint8), timestamp_s=12.5)
    assert mouth.bbox_px == pytest.approx((50, 20, 150, 80))
    assert mouth.center_px == pytest.approx((100, 50))
    assert mouth.aperture_ratio == pytest.approx(0.2)
    assert mouth.timestamp_s == 12.5
    assert mouth.face_index == 0
    assert len(mouth.outer_lip_px) == len(mouth.inner_lip_px) == 20
    assert fake_mediapipe.close_count == 1
    assert fake_mediapipe.options.base_options.delegate == "cpu"
    assert fake_mediapipe.options.running_mode == "image"
    assert not fake_mediapipe.options.output_face_blendshapes


def test_rgb_channels_preserved_and_strided_input_made_contiguous(fake_mediapipe):
    rgb = np.zeros((4, 6, 3), dtype=np.uint8)
    rgb[:] = (250, 100, 10)
    rgb = rgb[:, ::2]
    with MediaPipeMouthLandmarker(MouthLandmarkerConfig(fake_mediapipe.model_path)) as detector:
        assert detector.detect_rgb(rgb) == ()
    image = fake_mediapipe.images[0]
    assert image.image_format == "srgb"
    assert image.data.flags.c_contiguous
    np.testing.assert_array_equal(image.data, rgb)
    assert tuple(image.data[0, 0]) == (250, 100, 10)


def test_multiple_faces_and_closed_lips_are_observations(fake_mediapipe):
    fake_mediapipe.faces = [face(), face(x_offset=0.1, closed=True)]
    with MediaPipeMouthLandmarker(MouthLandmarkerConfig(fake_mediapipe.model_path, num_faces=2)) as detector:
        mouths = detector.detect_rgb(np.zeros((100, 200, 3), dtype=np.uint8))
    assert [m.face_index for m in mouths] == [0, 1]
    assert mouths[1].center_px == pytest.approx((120, 50))
    assert mouths[1].aperture_ratio == 0.0
    assert mouths[1].timestamp_s is None
    assert fake_mediapipe.options.num_faces == 2


def test_incomplete_nonfinite_and_zero_width_results_are_omitted(fake_mediapipe):
    degenerate = face()
    degenerate[291].x = degenerate[61].x
    nonfinite = face()
    nonfinite[13].x = float("nan")
    fake_mediapipe.faces = [face()[:100], degenerate, nonfinite, face()]
    with MediaPipeMouthLandmarker(MouthLandmarkerConfig(fake_mediapipe.model_path, num_faces=4)) as detector:
        mouths = detector.detect_rgb(np.zeros((100, 200, 3), dtype=np.uint8))
    assert [m.face_index for m in mouths] == [3]


@pytest.mark.parametrize("rgb", [np.zeros((2, 2, 3)), np.zeros((2, 2), dtype=np.uint8),
                                    np.zeros((0, 2, 3), dtype=np.uint8)])
def test_invalid_frames_do_not_reach_model(fake_mediapipe, rgb):
    with MediaPipeMouthLandmarker(MouthLandmarkerConfig(fake_mediapipe.model_path)) as detector:
        with pytest.raises(ValueError):
            detector.detect_rgb(rgb)
    assert fake_mediapipe.images == []


def test_close_is_idempotent_and_closed_backend_rejects_use(fake_mediapipe):
    detector = MediaPipeMouthLandmarker(MouthLandmarkerConfig(fake_mediapipe.model_path))
    with pytest.raises(ValueError, match="timestamp_s"):
        detector.detect_rgb(np.zeros((2, 2, 3), dtype=np.uint8), timestamp_s=float("nan"))
    detector.close()
    detector.close()
    assert fake_mediapipe.close_count == 1
    with pytest.raises(RuntimeError, match="closed"):
        detector.detect_rgb(np.zeros((2, 2, 3), dtype=np.uint8))


def test_model_must_be_a_local_file_before_importing_backend(tmp_path):
    with pytest.raises(FileNotFoundError, match="model"):
        MediaPipeMouthLandmarker(MouthLandmarkerConfig(str(tmp_path / "missing.task")))


@pytest.mark.parametrize("kwargs", [{"num_faces": 0}, {"num_faces": 1.5},
                                      {"min_face_detection_confidence": 2.0},
                                      {"min_face_presence_confidence": float("nan")},
                                      {"min_mouth_width_px": 0}])
def test_invalid_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        MouthLandmarkerConfig("unused.task", **kwargs)
