"""Preview freshness and failure semantics with in-memory camera/model fakes."""

from io import BytesIO
import json
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image

from assistive_arm.mouth_perception.landmarks_2d import MouthLandmarks2D
from assistive_arm.orchestration.vision_preview import (
    PreviewConfig, PreviewSnapshot, PreviewStore, PreviewWorker, main, render_preview,
)
from assistive_arm.vision.types import Detection


def mouth():
    return MouthLandmarks2D(
        face_index=0, timestamp_s=None,
        outer_lip_px=((32, 20), (48, 20), (48, 30), (32, 30)),
        inner_lip_px=((35, 23), (45, 23), (45, 27), (35, 27)),
        bbox_px=(32, 20, 48, 30), center_px=(40, 25), aperture_ratio=0.25,
    )


def test_freshness_expires_image_and_detections_without_new_frames():
    store = PreviewStore(PreviewConfig(max_received_age_s=3))
    store.publish(PreviewSnapshot(
        camera="arm-camera", state="live", received_at_s=10,
        captured_at_s=1700000000.0, capture_clock="unix",
        spoons=(Detection("spoon", 0.9, (1, 2, 3, 4)),), mouths=(mouth(),), jpeg=b"image",
    ))
    live = store.view(12)
    assert live.jpeg == b"image"
    assert live.expires_in_s == 1
    assert live.status()["captured_at_s"] == 1700000000.0
    expired = store.view(13)
    assert expired.state == "stale"
    assert expired.jpeg is None
    assert expired.status()["spoons"] == []
    assert expired.status()["mouths"] == []
    assert expired.received_age_s == 3


def test_future_receiver_timestamp_is_not_treated_as_fresh():
    store = PreviewStore(PreviewConfig())
    store.publish(PreviewSnapshot(camera="arm-camera", state="live", received_at_s=10, jpeg=b"image"))
    assert store.view(9).jpeg is None


def test_jpeg_preserves_rgb_and_renders_spoon_and_lips_without_mutating_input():
    rgb = np.full((48, 64, 3), (180, 20, 40), dtype=np.uint8)
    original = rgb.copy()
    jpeg = render_preview(rgb, (Detection("spoon", 0.9, (2, 18, 20, 35)),), (mouth(),), PreviewConfig())
    rendered = np.asarray(Image.open(BytesIO(jpeg)).convert("RGB"))
    assert rendered.shape == rgb.shape
    assert rendered[43, 55, 0] > rendered[43, 55, 2] + 100
    assert np.any(rendered[18:36, 2:21, 1] > 100)  # green spoon outline
    assert np.any(rendered[20:31, 32:49, 1] > 100)  # orange lips
    np.testing.assert_array_equal(rgb, original)


class Camera:
    """Only camera and clock methods exist: movement is not part of this fake."""

    def __init__(self):
        self.clock = 10.0
        self.fail = False
        self.calls = []

    def now(self):
        return self.clock

    def get_rgb_frames(self, names):
        self.calls.append(names)
        if self.fail:
            raise RuntimeError("private connection details must not enter status")
        return {"arm-camera": SimpleNamespace(
            camera="arm-camera", rgb=np.zeros((48, 64, 3), dtype=np.uint8),
            received_at_s=self.clock, captured_at_s=None, capture_clock=None,
        )}


class Spoon:
    def detect_rgb(self, _rgb):
        return [Detection("spoon", 0.8, (2, 10, 20, 30))]


class Mouth:
    def detect_rgb(self, _rgb, *, timestamp_s=None):
        return (mouth(),)


def test_camera_failure_clears_previous_overlay_and_recovers():
    camera = Camera()
    store = PreviewStore(PreviewConfig())
    worker = PreviewWorker(camera, Spoon(), Mouth(), store)
    first = worker.capture_once()
    assert first.state == "live"
    assert store.view(camera.now()).jpeg is not None
    camera.fail = True
    failed = worker.capture_once()
    assert failed.state == "error"
    assert failed.error == "camera failed (RuntimeError)"
    assert failed.spoons == failed.mouths == ()
    assert store.view(camera.now()).jpeg is None
    camera.fail = False
    assert worker.capture_once().revision == first.revision + 2
    assert store.view(camera.now()).state == "live"
    assert camera.calls == [["arm-camera"]] * 3


def test_model_failure_clears_previous_overlay():
    class FailedMouth:
        def detect_rgb(self, *_args, **_kwargs):
            raise ValueError("invalid image")

    camera = Camera()
    store = PreviewStore(PreviewConfig())
    worker = PreviewWorker(camera, Spoon(), Mouth(), store)
    worker.capture_once()
    worker.mouth = FailedMouth()
    failed = worker.capture_once()
    assert failed.error == "mouth detector failed (ValueError)"
    assert store.view(camera.now()).jpeg is None


def test_slow_inference_is_already_stale_when_published():
    camera = Camera()

    class SlowSpoon(Spoon):
        def detect_rgb(self, rgb):
            camera.clock += 4
            return super().detect_rgb(rgb)

    store = PreviewStore(PreviewConfig(max_received_age_s=3))
    snapshot = PreviewWorker(camera, SlowSpoon(), Mouth(), store).capture_once()
    assert snapshot.spoon_ms == 4000
    assert store.view(camera.now()).state == "stale"
    assert store.view(camera.now()).jpeg is None


def test_usb_smoke_does_not_connect_viam_and_reports_stale_warmup(monkeypatch, capsys):
    """A directly attached camera needs no Viam connection or credentials."""
    camera = Camera()
    camera.closed = False
    configs = []

    class UsbReader(Camera):
        def __init__(self, config):
            configs.append(config)

        def __enter__(self):
            return camera

        def __exit__(self, *_exc):
            camera.closed = True

    class LocalMouth(Mouth):
        def __init__(self, _config):
            self.closed = False

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            self.closed = True

    class WarmupSpoon(Spoon):
        def __init__(self, _config):
            self.first = True

        def detect_rgb(self, rgb):
            if self.first:
                camera.clock += 4
                self.first = False
            return super().detect_rgb(rgb)

    def reject_viam(*_args, **_kwargs):
        raise AssertionError("USB preview must not create a Viam connection")

    monkeypatch.setitem(sys.modules, "assistive_arm.robot_iface.usb_camera", SimpleNamespace(
        UsbCameraConfig=SimpleNamespace, UsbCameraReader=UsbReader,
    ))
    monkeypatch.setattr("assistive_arm.robot_iface.viam.ViamCameraReader", reject_viam)
    monkeypatch.setattr("assistive_arm.mouth_perception.landmarks_2d.MediaPipeMouthLandmarker", LocalMouth)
    monkeypatch.setattr("assistive_arm.vision.rfdetr.RFDETRSpoonDetector", WarmupSpoon)
    code = main([
        "--source", "usb", "--video-device", "/dev/video-test", "--camera", "arm-camera",
        "--spoon-weights", "offline.pth", "--mouth-model", "offline.task", "--smoke-frames", "2",
    ])
    summaries = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert code == 0
    assert configs[0].device_path == "/dev/video-test"
    assert configs[0].camera_name == "arm-camera"
    assert [summary["state"] for summary in summaries] == ["stale", "live"]
    assert [summary["received_age_s"] for summary in summaries] == [4, 0]
    assert summaries[0]["spoons"] == 1  # Processed counts remain useful in a smoke summary.
    assert camera.closed
