"""The Viam camera and range sensor with a fake source camera and a fake depth model (offline)."""

import asyncio
import io

import numpy as np
import pytest
from PIL import Image
from viam.components.camera import Camera
from viam.media.video import CameraMimeType, NamedImage
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import ResponseMetadata
from viam.utils import dict_to_struct

from da3_depth.camera import DepthAnything3Camera
from da3_depth.config import CameraConfig
from da3_depth.range_sensor import DepthRangeSensor

W, H = 64, 48


class FakeSource(Camera):
    def __init__(self, name="src"):
        super().__init__(name)
        buf = io.BytesIO()
        Image.fromarray(np.full((H, W, 3), 128, np.uint8)).save(buf, format="JPEG")
        self.jpeg = buf.getvalue()

    async def get_images(self, **kwargs):
        return [NamedImage("color", self.jpeg, CameraMimeType.JPEG)], ResponseMetadata()

    async def get_point_cloud(self, **kwargs):
        raise NotImplementedError

    async def get_properties(self, **kwargs):
        return Camera.Properties(supports_pcd=False)


class FakeEstimator:
    depth_m = 1.5
    estimates_focal = False

    def __init__(self, cfg):
        self.device = "unloaded"
        self.focal_px = None

    def load(self):
        self.device = "cpu"

    def predict(self, rgb, k):
        assert rgb.shape == (H, W, 3)
        if k is None:
            assert self.estimates_focal
            self.focal_px = 50.0
        else:
            assert (k.width, k.height) == (W, H)
        return np.full((H, W), self.depth_m, np.float32)


def config(name, model, **attrs):
    return ComponentConfig(name=name, model=str(model), attributes=dict_to_struct(attrs))


@pytest.fixture
def make_camera(monkeypatch):
    monkeypatch.setattr(DepthAnything3Camera, "estimator_factory", FakeEstimator)
    created = []

    async def make(**attrs):
        attrs = {k: v for k, v in {"source": "src", "hfov_deg": 90.0, "produce_hz": 50.0, **attrs}.items() if v is not None}
        cam = DepthAnything3Camera.new(config("depth", DepthAnything3Camera.MODEL, **attrs),
                                       {Camera.get_resource_name("src"): FakeSource()})
        created.append(cam)
        for _ in range(200):
            if cam._frame is not None or cam._last_error or "failed" in cam._status:
                break
            await asyncio.sleep(0.01)
        return cam

    yield make
    for cam in created:
        cam._stop()


def test_config_rejects_typos_and_bad_values():
    with pytest.raises(ValueError, match="unknown attributes"):
        CameraConfig.from_attributes({"source": "a", "hfov": 45})
    with pytest.raises(ValueError, match="source"):
        CameraConfig.from_attributes({})
    with pytest.raises(ValueError, match="fx and fy"):
        CameraConfig.from_attributes({"source": "a", "fx": 500})
    cfg = CameraConfig.from_attributes({"source": "a", "max_range_m": 3, "scan_beams": 32})
    assert cfg.ranges.max_range == 3.0 and cfg.ranges.scan_beams == 32
    assert cfg.model == "depth-pro" and cfg.weights is None  # Depth Pro by default, its default weights
    assert cfg.compute == "cloud" and cfg.cloud_fallback  # on the cloud GPU, the local model as fallback
    with pytest.raises(ValueError, match="depth-pro only"):
        CameraConfig.from_attributes({"source": "a", "model": "da3-metric"})
    assert CameraConfig.from_attributes({"source": "a", "model": "da3-metric", "compute": "local"}).compute == "local"
    with pytest.raises(ValueError, match="model must be"):
        CameraConfig.from_attributes({"source": "a", "model": "midas"})


def test_validate_config_declares_the_source_dependency():
    req, opt = DepthAnything3Camera.validate_config(config("depth", DepthAnything3Camera.MODEL, source="webcam-2"))
    assert list(req) == ["webcam-2"] and list(opt) == []


@pytest.mark.asyncio
async def test_serves_images_point_cloud_and_ranges(make_camera):
    cam = await make_camera()
    images, _ = await cam.get_images()
    assert [i.name for i in images] == ["color", "depth", "depth_color"]
    assert images[1].mime_type == CameraMimeType.VIAM_RAW_DEPTH
    assert images[1].bytes_to_depth_array()[H // 2][W // 2] == 1500

    pcd, mime = await cam.get_point_cloud()
    assert mime == CameraMimeType.PCD
    pts = np.frombuffer(pcd.split(b"DATA binary\n")[1], np.float32).reshape(-1, 3)
    assert len(pts) == (W // 2) * (H // 2) and np.allclose(pts[:, 2], 1500.0)

    props = await cam.get_properties()
    assert props.supports_pcd and props.intrinsic_parameters.focal_x_px == pytest.approx(32.0)

    r = await cam.do_command({"command": "ranges"})
    assert r["distance_m"] == pytest.approx(1.5, abs=0.01) and r["age_s"] < 1.0
    status = await cam.do_command({})
    assert status["status"] == "running" and status["device"] == "cpu" and status["frames"] >= 1


@pytest.mark.asyncio
async def test_range_sensor_forwards_the_camera_readings(make_camera):
    cam = await make_camera()
    sensor = DepthRangeSensor.new(config("range", DepthRangeSensor.MODEL, camera="depth"),
                                  {Camera.get_resource_name("depth"): cam})
    readings = await sensor.get_readings()
    assert readings["nearest_m"] == pytest.approx(1.5, abs=0.01) and len(readings["scan_ranges_m"]) == 64


@pytest.mark.asyncio
async def test_stale_depth_is_refused(make_camera):
    cam = await make_camera(stale_after_s=0.5)
    cam._stop()  # the producer stops: nothing new arrives
    object.__setattr__(cam._frame, "captured_at", cam._frame.captured_at - 10.0)
    with pytest.raises(RuntimeError, match="old"):
        await cam.get_point_cloud()
    with pytest.raises(RuntimeError, match="old"):
        await cam.do_command({"command": "ranges"})


@pytest.mark.asyncio
async def test_missing_intrinsics_is_an_error_not_a_guess(make_camera):
    cam = await make_camera(hfov_deg=None)
    assert "hfov_deg or fx/fy" in (cam._last_error or "")
    with pytest.raises(RuntimeError, match="no depth yet"):
        await cam.get_images()


@pytest.mark.asyncio
async def test_a_failed_model_load_is_retried(make_camera, monkeypatch):
    import da3_depth.camera as camera_mod

    class FlakyEstimator(FakeEstimator):
        loads = 0

        def load(self):
            FlakyEstimator.loads += 1
            if FlakyEstimator.loads == 1:
                raise RuntimeError("No CUDA GPUs are available")
            super().load()

    monkeypatch.setattr(camera_mod, "LOAD_RETRY_S", 0.05)
    monkeypatch.setattr(DepthAnything3Camera, "estimator_factory", FlakyEstimator)
    cam = DepthAnything3Camera.new(config("depth", DepthAnything3Camera.MODEL, source="src", hfov_deg=90.0, produce_hz=50.0),
                                   {Camera.get_resource_name("src"): FakeSource()})
    try:
        for _ in range(200):
            if cam._frame is not None:
                break
            await asyncio.sleep(0.01)
        assert FlakyEstimator.loads == 2 and cam._frame is not None and cam._status == "running"
    finally:
        cam._stop()


@pytest.mark.asyncio
async def test_a_model_that_estimates_focal_needs_no_intrinsics(make_camera, monkeypatch):
    class SelfCalibrating(FakeEstimator):
        estimates_focal = True

    monkeypatch.setattr(DepthAnything3Camera, "estimator_factory", SelfCalibrating)
    cam = await make_camera(hfov_deg=None)
    assert cam._last_error is None and cam._frame is not None
    assert cam._frame.intrinsics.fx == 50.0  # the model's own estimate, used for the point cloud too
    pcd, _ = await cam.get_point_cloud()
    assert len(pcd) > 0
