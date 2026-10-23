"""Local camera adapter tests using a fake OpenCV capture, with no hardware."""

from dataclasses import FrozenInstanceError
import queue
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from assistive_arm.robot_iface.usb_camera import UsbCameraConfig, UsbCameraReader


@pytest.fixture
def usb(monkeypatch):
    frames = queue.Queue()
    calls = []
    readers = []
    state = SimpleNamespace(opened=True, frames=frames, calls=calls)

    class Capture:
        def isOpened(self):
            return state.opened

        def set(self, prop, value):
            calls.append(("set", prop, value))
            return True

        def read(self):
            calls.append(("read", threading.get_ident()))
            result = frames.get()
            if isinstance(result, Exception):
                raise result
            return result

        def release(self):
            calls.append(("release", threading.get_ident()))

    def video_capture(path, backend):
        calls.append(("open", path, backend))
        return Capture()

    cv2 = SimpleNamespace(
        VideoCapture=video_capture, CAP_V4L2=200, CAP_PROP_FRAME_WIDTH=3,
        CAP_PROP_FRAME_HEIGHT=4, CAP_PROP_FPS=5, CAP_PROP_BUFFERSIZE=38,
    )
    monkeypatch.setitem(sys.modules, "cv2", cv2)

    def make_reader(**kwargs):
        settings = {"frame_timeout_s": 0.05, "close_timeout_s": 0.2}
        settings.update(kwargs)
        reader = UsbCameraReader(UsbCameraConfig(**settings))
        readers.append(reader)
        return reader

    state.make_reader = make_reader
    state.push = lambda bgr: frames.put((True, np.asarray([[bgr]], dtype=np.uint8)))
    yield state
    for reader in readers:
        frames.put((False, None))
        reader.close()


def wait_for_sequence(reader, sequence):
    with reader._condition:
        assert reader._condition.wait_for(lambda: reader._sequence >= sequence, timeout=1.0)


def test_config_is_frozen_and_defaults_to_actual_usb_path():
    config = UsbCameraConfig()
    assert config.device_path.endswith("usb-icSpring_icspring_camera-video-index0")
    assert (config.width_px, config.height_px, config.fps, config.buffer_size) == (640, 480, 30, 1)
    with pytest.raises(FrozenInstanceError):
        config.fps = 60
    for values in ({"width_px": 0}, {"buffer_size": 0}, {"fps": float("nan")}, {"frame_timeout_s": 0}):
        with pytest.raises(ValueError):
            UsbCameraConfig(**values)


def test_rgb_conversion_and_receive_time_are_set_when_capture_arrives(usb):
    reader = usb.make_reader()
    usb.push([255, 0, 10])  # BGR -> RGB
    wait_for_sequence(reader, 1)
    received_at = reader._latest.received_at_s
    before_poll = time.monotonic()
    frame = reader.get_rgb_frames(["arm-camera"])["arm-camera"]
    assert frame.rgb.tolist() == [[[10, 0, 255]]]
    assert frame.rgb.dtype == np.uint8 and frame.rgb.flags.c_contiguous
    assert frame.received_at_s == received_at <= before_poll
    assert frame.captured_at_s is None and frame.capture_clock is None
    assert [c for c in usb.calls if c[0] == "set"] == [
        ("set", 3, 640), ("set", 4, 480), ("set", 5, 30), ("set", 38, 1),
    ]


def test_slow_consumer_gets_newest_frame_instead_of_a_buffered_backlog(usb):
    reader = usb.make_reader()
    usb.push([0, 0, 1])
    assert reader.get_rgb_frames()["arm-camera"].rgb[0, 0, 0] == 1
    for value in (2, 3, 4):
        usb.push([0, 0, value])
    wait_for_sequence(reader, 4)
    newest = reader.get_rgb_frames()["arm-camera"]
    assert newest.rgb[0, 0, 0] == 4
    # Another poll cannot relabel this same frame as a fresh capture.
    with pytest.raises(TimeoutError, match="No new, fresh"):
        reader.get_rgb_frames()
    assert reader._latest is None


def test_missing_frames_have_a_bounded_wait_and_can_recover(usb):
    reader = usb.make_reader()
    with pytest.raises(TimeoutError, match="No new, fresh"):
        reader.get_rgb_frames()
    usb.push([0, 255, 0])
    assert reader.get_rgb_frames()["arm-camera"].rgb[0, 0, 1] == 255


def test_expired_capture_is_not_returned_or_given_a_new_timestamp(usb):
    reader = usb.make_reader(max_frame_age_s=0.01)
    usb.push([0, 0, 255])
    wait_for_sequence(reader, 1)
    time.sleep(0.02)
    with pytest.raises(TimeoutError, match="No new, fresh"):
        reader.get_rgb_frames()
    assert reader._latest is None


@pytest.mark.parametrize("failure", [(False, None), OSError("camera disconnected")])
def test_capture_failure_invalidates_previous_image(usb, failure):
    reader = usb.make_reader()
    usb.push([0, 0, 255])
    wait_for_sequence(reader, 1)
    usb.frames.put(failure)
    with reader._condition:
        assert reader._condition.wait_for(lambda: reader._failure is not None, timeout=1.0)
    with pytest.raises(RuntimeError, match="read failed|disconnected"):
        reader.get_rgb_frames()
    assert reader._latest is None


def test_replug_reopens_after_delay_and_recovers_only_with_new_capture(usb):
    reader = usb.make_reader(reconnect_delay_s=0.05)
    usb.push([0, 0, 1])
    original = reader.get_rgb_frames()["arm-camera"]
    usb.frames.put((False, None))
    with reader._condition:
        assert reader._condition.wait_for(lambda: reader._failure is not None, timeout=1.0)
    assert reader._latest is None
    assert len([c for c in usb.calls if c[0] == "open"]) == 1
    with pytest.raises(RuntimeError, match="read failed"):
        reader.get_rgb_frames()
    usb.push([0, 0, 2])
    wait_for_sequence(reader, 2)
    recovered = reader.get_rgb_frames()["arm-camera"]
    assert recovered.rgb[0, 0, 0] == 2
    assert recovered.received_at_s > original.received_at_s
    assert reader._failure is None
    assert len([c for c in usb.calls if c[0] == "open"]) == 2
    assert len([c for c in usb.calls if c[0] == "release"]) == 1


def test_device_open_failure_is_reported_and_released(usb):
    usb.opened = False
    reader = usb.make_reader()
    with pytest.raises(RuntimeError, match="Cannot open USB camera"):
        reader.get_rgb_frames()
    reader.close()
    assert len([c for c in usb.calls if c[0] == "release"]) == 1


def test_only_configured_alias_is_readable(usb):
    reader = usb.make_reader(camera_name="wrist")
    assert reader.get_rgb_frames([]) == {}
    with pytest.raises(KeyError, match="Unknown camera aliases"):
        reader.get_rgb_frames(["arm-camera"])
    usb.push([1, 2, 3])
    assert list(reader.get_rgb_frames(["wrist"])) == ["wrist"]


def test_every_robot_or_calibrated_camera_operation_fails_locally(usb):
    reader = usb.make_reader()
    for name in (
        "get_tcp_pose", "move_to_pose", "move_linear", "open_gripper", "close_gripper",
        "get_wrench", "tare_wrench", "cameras", "get_frames", "camera_pose", "stop",
        "gripper_state", "joint_positions", "update_world",
    ):
        with pytest.raises(NotImplementedError, match="RGB camera reads only"):
            getattr(reader, name)()


def test_blocked_read_has_bounded_close_and_release_stays_on_worker(usb):
    reader = usb.make_reader(close_timeout_s=0.01)
    usb.push([1, 2, 3])
    reader.get_rgb_frames()
    with pytest.raises(TimeoutError, match="release is pending"):
        reader.close()
    with pytest.raises(RuntimeError, match="closed"):
        reader.get_rgb_frames()
    assert not any(c[0] == "release" for c in usb.calls)
    usb.frames.put((False, None))
    reader._thread.join(timeout=1.0)
    reader.close()
    releases = [c for c in usb.calls if c[0] == "release"]
    assert len(releases) == 1
    assert releases[0][1] == reader._thread.ident != threading.get_ident()
