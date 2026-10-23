"""RGB-only Linux USB capture, independent of Viam and all arm controls."""

from __future__ import annotations

from dataclasses import dataclass
import threading

import numpy as np

from ..vision.types import RGBFrame
from .base import RobotInterface


@dataclass(frozen=True)
class UsbCameraConfig:
    """Requested capture settings and bounded waits for the RGB preview.

    V4L2 may negotiate different settings or ignore the buffer request. A worker
    continuously drains capture regardless of inference speed; it retains only
    its newest image. Read-completion time is not a measured exposure time.
    """

    device_path: str = "/dev/v4l/by-id/usb-icSpring_icspring_camera-video-index0"
    camera_name: str = "arm-camera"
    width_px: int = 640
    height_px: int = 480
    fps: float = 30.0
    buffer_size: int = 1
    frame_timeout_s: float = 2.0
    max_frame_age_s: float = 1.0
    close_timeout_s: float = 2.0
    reconnect_delay_s: float = 1.0

    def __post_init__(self):
        if not self.device_path or not self.camera_name:
            raise ValueError("device_path and camera_name must be nonempty")
        for field in ("width_px", "height_px", "buffer_size"):
            value = getattr(self, field)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{field} must be a positive integer")
        for field in ("fps", "frame_timeout_s", "max_frame_age_s", "close_timeout_s", "reconnect_delay_s"):
            value = getattr(self, field)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{field} must be positive and finite")


class UsbCameraReader(RobotInterface):
    """Continuously consume one USB stream, returning only fresh RGB frames.

    Each successful poll returns a frame newer than the previous successful
    poll. Inference never controls the capture rate, so a slow model does not
    create a queue of old images. Capture failure invalidates the cached frame.
    All robot, calibrated camera, and depth operations fail locally.
    """

    def __init__(self, config: UsbCameraConfig | None = None):
        import cv2

        self.config = config or UsbCameraConfig()
        self._cv2 = cv2
        self._condition = threading.Condition()
        self._halt_capture = threading.Event()
        self._closed = False
        self._latest: RGBFrame | None = None
        self._sequence = 0
        self._delivered_sequence = 0
        self._failure: str | None = None
        self._thread = threading.Thread(target=self._capture, name="usb-rgb-capture", daemon=True)
        self._thread.start()

    def _capture(self):
        """The capture worker alone owns, reads, and releases the native device."""
        cv2, config = self._cv2, self.config
        while not self._halt_capture.is_set():
            capture = None
            try:
                capture = cv2.VideoCapture(config.device_path, cv2.CAP_V4L2)
                if not capture.isOpened():
                    raise OSError(f"Cannot open USB camera {config.device_path!r}")
                for prop, value in (
                    (cv2.CAP_PROP_FRAME_WIDTH, config.width_px),
                    (cv2.CAP_PROP_FRAME_HEIGHT, config.height_px),
                    (cv2.CAP_PROP_FPS, config.fps),
                    (cv2.CAP_PROP_BUFFERSIZE, config.buffer_size),
                ):
                    capture.set(prop, value)
                while not self._halt_capture.is_set():
                    ok, bgr = capture.read()
                    received_at = self.now()
                    if self._halt_capture.is_set():
                        break
                    if not ok or bgr is None:
                        raise RuntimeError("USB camera frame read failed")
                    if bgr.dtype != np.uint8 or bgr.ndim != 3 or bgr.shape[2] != 3 or not bgr.size:
                        raise ValueError("USB camera did not return a nonempty uint8 BGR image")
                    rgb = np.ascontiguousarray(bgr[:, :, ::-1])
                    frame = RGBFrame(config.camera_name, rgb, received_at)
                    with self._condition:
                        if self._closed:
                            break
                        self._latest = frame
                        self._failure = None
                        self._sequence += 1
                        self._condition.notify_all()
            except Exception as error:
                with self._condition:
                    self._latest = None
                    if not self._closed:
                        self._failure = str(error)
                    self._condition.notify_all()
            finally:
                if capture is not None:
                    try:
                        capture.release()
                    except Exception as error:
                        with self._condition:
                            self._latest = None
                            if not self._closed:
                                self._failure = f"USB camera release failed: {error}"
                            self._condition.notify_all()
            self._halt_capture.wait(config.reconnect_delay_s)

    def get_rgb_frames(self, names: list[str] | None = None) -> dict[str, RGBFrame]:
        selected = [self.config.camera_name] if names is None else list(names)
        unknown = set(selected).difference({self.config.camera_name})
        if unknown:
            raise KeyError(f"Unknown camera aliases: {sorted(unknown)}")
        deadline = self.now() + self.config.frame_timeout_s
        with self._condition:
            while True:
                if self._closed:
                    raise RuntimeError("UsbCameraReader is closed")
                if not selected:
                    return {}
                if self._failure is not None:
                    raise RuntimeError(self._failure)
                now = self.now()
                frame = self._latest
                if frame is not None and now - frame.received_at_s > self.config.max_frame_age_s:
                    self._latest = frame = None
                if frame is not None and self._sequence > self._delivered_sequence:
                    self._delivered_sequence = self._sequence
                    return {self.config.camera_name: frame}
                remaining = deadline - now
                if remaining <= 0:
                    self._latest = None
                    raise TimeoutError("No new, fresh RGB frame arrived from the USB camera")
                self._condition.wait(timeout=remaining)

    def close(self) -> None:
        """Release only the camera; never issue a robot stop or other command.

        OpenCV/V4L2 cannot guarantee interruption of a native read. If the driver
        is stuck, report a bounded timeout and let the worker release the camera
        when it returns; do not release concurrently with that native read.
        """
        with self._condition:
            self._closed = True
            self._latest = None
            self._halt_capture.set()
            self._condition.notify_all()
        self._thread.join(timeout=self.config.close_timeout_s)
        if self._thread.is_alive():
            raise TimeoutError("USB camera driver is still reading; release is pending in the capture worker")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    @staticmethod
    def _unsupported(*args, **kwargs):
        raise NotImplementedError("UsbCameraReader supports RGB camera reads only")

    get_tcp_pose = _unsupported
    move_to_pose = _unsupported
    move_linear = _unsupported
    open_gripper = _unsupported
    close_gripper = _unsupported
    get_wrench = _unsupported
    tare_wrench = _unsupported
    cameras = _unsupported
    get_frames = _unsupported
    camera_pose = _unsupported
    stop = _unsupported
    gripper_state = _unsupported
    joint_positions = _unsupported
    update_world = _unsupported
