"""Viam camera: an RGB camera + Depth Anything 3 -> depth images, a point cloud, lidar-style ranges.

PLAN
1. A background task (``produce_hz``) grabs a frame from ``source``, runs DA3 in a worker
   thread and keeps the newest result: depth (m), intrinsics, range readings.
2. GetImages serves "color" (JPEG), "depth" (Viam raw depth, mm) and "depth_color" (JPEG).
   GetPointCloud serves a PCD in mm, camera frame (+z out of the lens, +x right, +y down);
   give this camera the same frame as ``source`` so the 3D scene places it correctly.
3. DoCommand {"command": "ranges"} returns the lidar-style readings (the range sensor
   model serves them as sensor readings); anything else returns status.
   With ``preview_port`` set, a read-only page on 127.0.0.1 shows the same frames (preview.py).
4. Nothing older than ``stale_after_s`` is served: a stuck camera or model is an error,
   never a silently old point cloud.
"""

from __future__ import annotations

import asyncio
import io
import logging
import time
from dataclasses import dataclass
from typing import Any, ClassVar, Mapping, Optional, Sequence

import numpy as np
from PIL import Image
from viam.components.camera import Camera
from viam.media.video import CameraMimeType, NamedImage, ViamImage
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import ResourceName, ResponseMetadata
from viam.proto.component.camera import IntrinsicParameters
from viam.resource.base import ResourceBase
from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model, ModelFamily
from viam.utils import struct_to_dict

from .config import CameraConfig
from .estimator import DA3Estimator, DepthEstimator
from .geometry import Intrinsics, RangeReadings, colorize, depth_to_points, encode_viam_depth, points_to_pcd, range_readings
from .preview import DepthPreview

LOGGER = logging.getLogger(__name__)
FAMILY = ModelFamily("assistive-arm", "depth-anything-3")
JPEG_QUALITY = 85


@dataclass(frozen=True)
class DepthFrame:
    color_jpeg: bytes
    depth: np.ndarray  # (H, W) metres, 0 = invalid
    intrinsics: Intrinsics
    readings: RangeReadings
    captured_at: float  # time.monotonic() when the source frame arrived
    infer_ms: float
    seq: int = 0  # increases by one per frame


def _decode(image: ViamImage) -> tuple[np.ndarray, bytes]:
    """(RGB array, JPEG bytes) from a source image; JPEG sources are passed through untouched."""
    pil = Image.open(io.BytesIO(image.data)).convert("RGB")
    if image.mime_type == CameraMimeType.JPEG:
        return np.asarray(pil), image.data
    return np.asarray(pil), _jpeg(np.asarray(pil))


def _jpeg(rgb: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=JPEG_QUALITY)
    return buf.getvalue()


class DepthAnything3Camera(Camera, EasyResource):
    MODEL: ClassVar[Model] = Model(FAMILY, "camera")
    estimator_factory: ClassVar = DA3Estimator  # tests substitute a fake

    @classmethod
    def new(cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]):
        cam = cls(config.name)
        cam.reconfigure(config, dependencies)
        return cam

    @classmethod
    def validate_config(cls, config: ComponentConfig) -> tuple[Sequence[str], Sequence[str]]:
        cfg = CameraConfig.from_attributes(struct_to_dict(config.attributes))
        return [cfg.source], []

    def __init__(self, name: str):
        super().__init__(name)
        self._cfg: Optional[CameraConfig] = None
        self._source: Optional[Camera] = None
        self._estimator: Optional[DepthEstimator] = None
        self._loaded = False
        self._source_k: Optional[Intrinsics] = None  # from the source's properties, if it reports any
        self._source_k_checked = False
        self._task: Optional[asyncio.Task] = None
        self._preview: Optional[DepthPreview] = None
        self._frame: Optional[DepthFrame] = None
        self._status = "starting"
        self._frames = 0
        self._errors = 0
        self._last_error: Optional[str] = None

    def reconfigure(self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]) -> None:
        cfg = CameraConfig.from_attributes(struct_to_dict(config.attributes))
        source = dependencies.get(Camera.get_resource_name(cfg.source))
        if source is None:
            raise ValueError(f"source camera {cfg.source!r} not found")
        self._stop()
        old = self._cfg
        if old is None or (old.weights, old.device, old.process_res) != (cfg.weights, cfg.device, cfg.process_res):
            self._estimator = self.estimator_factory(cfg.weights, cfg.device, cfg.process_res)
            self._loaded = False
        self._cfg, self._source = cfg, source  # type: ignore[assignment]
        self._source_k, self._source_k_checked = None, False
        self._frame, self._last_error = None, None
        if cfg.preview_port:
            self._preview = DepthPreview(cfg.preview_port, cfg.vision_status_url, lambda: self._frame,
                                         lambda: self._cfg, self._status_dict).start()
        self._start()

    # --- producer -------------------------------------------------------------------------

    def _start(self) -> None:
        if self._task is not None and not self._task.done():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:  # constructed outside the module's loop; the first API call starts it
            return
        self._task = loop.create_task(self._run(), name=f"{self.name}-da3")

    def _stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            self._task = None
        if self._preview is not None:
            self._preview.stop()
            self._preview = None

    async def _run(self) -> None:
        cfg = self._cfg
        assert cfg is not None and self._estimator is not None
        if not self._loaded:
            self._status = f"loading {cfg.weights} (the first run downloads ~1.3 GB)"
            try:
                await asyncio.to_thread(self._estimator.load)
            except Exception as e:  # noqa: BLE001
                self._status = f"model failed to load: {e!r}"
                LOGGER.error("%s: %s", self.name, self._status)
                return  # no retry loop around a multi-GB download; reconfigure to try again
            self._loaded = True
        self._status = f"running on {self._estimator.device}"
        period = 1.0 / cfg.produce_hz
        while True:
            t0 = time.monotonic()
            try:
                await self._cycle(cfg)
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001
                self._errors += 1
                self._last_error = repr(e)
                LOGGER.warning("%s: depth cycle failed: %s", self.name, e)
            await asyncio.sleep(max(0.0, period - (time.monotonic() - t0)))

    async def _cycle(self, cfg: CameraConfig) -> None:
        assert self._source is not None and self._estimator is not None
        images, _ = await self._source.get_images(timeout=cfg.source_timeout_s)
        color = [im for im in images if im.mime_type in (CameraMimeType.JPEG, CameraMimeType.PNG)]
        if not color:
            raise RuntimeError(f"{cfg.source} returned no JPEG/PNG image")
        captured_at = time.monotonic()
        rgb, jpeg = _decode(color[0])
        k = await self._intrinsics(cfg, rgb.shape[1], rgb.shape[0])
        depth = await asyncio.to_thread(self._estimator.predict, rgb, k)
        infer_ms = (time.monotonic() - captured_at) * 1000.0
        self._frames += 1
        self._frame = DepthFrame(jpeg, depth, k, range_readings(depth, k, cfg.ranges), captured_at, infer_ms, self._frames)
        self._last_error = None

    async def _intrinsics(self, cfg: CameraConfig, w: int, h: int) -> Intrinsics:
        if cfg.fx is not None and cfg.fy is not None:
            cx = cfg.cx if cfg.cx is not None else w / 2.0
            cy = cfg.cy if cfg.cy is not None else h / 2.0
            return Intrinsics(cfg.fx, cfg.fy, cx, cy, w, h)
        if not self._source_k_checked:
            self._source_k_checked = True
            try:
                p = (await self._source.get_properties(timeout=cfg.source_timeout_s)).intrinsic_parameters  # type: ignore[union-attr]
                if p.width_px > 0 and p.height_px > 0 and p.focal_x_px > 0 and p.focal_y_px > 0:
                    self._source_k = Intrinsics(p.focal_x_px, p.focal_y_px, p.center_x_px, p.center_y_px, p.width_px, p.height_px)
            except Exception as e:  # noqa: BLE001
                LOGGER.info("%s: source reports no intrinsics (%s)", self.name, e)
        if self._source_k is not None:
            return self._source_k.scaled(w, h)
        if cfg.hfov_deg is not None:
            return Intrinsics.from_hfov(cfg.hfov_deg, w, h)
        raise ValueError(f"{cfg.source} reports no intrinsics: set hfov_deg or fx/fy (metric depth needs the focal length)")

    def _fresh(self) -> DepthFrame:
        self._start()
        cfg, frame = self._cfg, self._frame
        assert cfg is not None
        if frame is None:
            detail = f"; last error: {self._last_error}" if self._last_error else ""
            raise RuntimeError(f"no depth yet ({self._status}{detail})")
        age = time.monotonic() - frame.captured_at
        if age > cfg.stale_after_s:
            raise RuntimeError(f"newest depth is {age:.1f} s old (stale_after_s={cfg.stale_after_s}); last error: {self._last_error}")
        return frame

    # --- camera API -----------------------------------------------------------------------

    async def get_images(
        self,
        *,
        filter_source_names: Optional[Sequence[str]] = None,
        extra: Optional[Mapping[str, Any]] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> tuple[Sequence[NamedImage], ResponseMetadata]:
        frame = self._fresh()
        cfg = self._cfg
        assert cfg is not None
        wanted = set(filter_source_names or ("color", "depth", "depth_color"))
        out = []
        if "color" in wanted:
            out.append(NamedImage("color", frame.color_jpeg, CameraMimeType.JPEG))
        if "depth" in wanted:
            out.append(NamedImage("depth", encode_viam_depth(frame.depth), CameraMimeType.VIAM_RAW_DEPTH))
        if "depth_color" in wanted:
            viz = colorize(frame.depth, cfg.ranges.min_range, cfg.ranges.max_range)
            out.append(NamedImage("depth_color", _jpeg(viz), CameraMimeType.JPEG))
        return out, ResponseMetadata()

    async def get_point_cloud(
        self, *, extra: Optional[Mapping[str, Any]] = None, timeout: Optional[float] = None, **kwargs
    ) -> tuple[bytes, str]:
        frame = self._fresh()
        cfg = self._cfg
        assert cfg is not None
        pts = depth_to_points(frame.depth, frame.intrinsics, stride=cfg.stride,
                              min_range=cfg.ranges.min_range, max_range=cfg.ranges.max_range)
        return points_to_pcd(pts), CameraMimeType.PCD

    async def get_properties(self, *, timeout: Optional[float] = None, **kwargs) -> Camera.Properties:
        frame = self._frame
        k = None
        if frame is not None:
            i = frame.intrinsics
            k = IntrinsicParameters(width_px=i.width, height_px=i.height, focal_x_px=i.fx, focal_y_px=i.fy,
                                    center_x_px=i.cx, center_y_px=i.cy)
        return Camera.Properties(
            supports_pcd=True,
            intrinsic_parameters=k,
            mime_types=[CameraMimeType.JPEG, CameraMimeType.VIAM_RAW_DEPTH, CameraMimeType.PCD],
        )

    async def do_command(self, command: Mapping[str, Any], *, timeout: Optional[float] = None, **kwargs) -> Mapping[str, Any]:
        if command.get("command") == "ranges":
            frame = self._fresh()
            return {**frame.readings.to_dict(), "age_s": round(time.monotonic() - frame.captured_at, 3)}
        return self._status_dict()

    def _status_dict(self) -> dict:
        frame = self._frame
        return {
            "status": self._status,
            "source": self._cfg.source if self._cfg else None,
            "weights": self._cfg.weights if self._cfg else None,
            "frames": self._frames,
            "errors": self._errors,
            "last_error": self._last_error,
            "last_infer_ms": round(frame.infer_ms, 1) if frame else None,
            "age_s": round(time.monotonic() - frame.captured_at, 3) if frame else None,
            "intrinsics": None if frame is None else {k: round(float(v), 2) for k, v in vars(frame.intrinsics).items()},
            "preview": f"http://127.0.0.1:{self._preview.port}" if self._preview else None,
        }

    async def close(self) -> None:
        self._stop()
