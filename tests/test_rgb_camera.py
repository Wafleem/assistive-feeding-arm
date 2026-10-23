"""RGB-only contracts and camera access, with no SDK, network, or image library."""

import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from assistive_arm.robot_iface.viam import ViamCameraReader


@pytest.fixture
def camera_sdk(monkeypatch):
    """The fake SDK exposes only connection, image read, and disconnect."""
    calls = []
    pixels = np.array([[[255, 0, 0], [0, 0, 255]]], dtype=np.uint8)
    state = {
        "images": [SimpleNamespace(mime_type="image/png", data=b"encoded-color")],
        "metadata": SimpleNamespace(
            HasField=lambda field: field == "captured_at",
            captured_at=SimpleNamespace(seconds=17, nanos=250_000_000),
        ),
    }

    class Client:
        async def close(self):
            calls.append(("disconnect",))

    client = Client()

    class Options:
        @staticmethod
        def with_api_key(**kwargs):
            assert kwargs["api_key"] == "test-secret"
            assert kwargs["api_key_id"] == "test-id"
            assert kwargs["disable_sessions"] is True
            options = SimpleNamespace(
                disable_sessions=True,
                dial_options=SimpleNamespace(timeout=20.0, initial_connection_attempt_timeout=None,
                                             initial_connection_attempts=3),
            )
            state["options"] = options
            return options

    class RobotClient:
        @staticmethod
        async def at_address(address, options):
            calls.append(("connect", address, options.disable_sessions))
            return client

    RobotClient.Options = Options

    class Camera:
        @staticmethod
        def from_robot(robot, name):
            assert robot is client
            calls.append(("bind_camera", name))
            if name == "missing-camera":
                raise KeyError(name)

            async def get_images(*, timeout):
                calls.append(("get_images", name, timeout))
                return state["images"], state["metadata"]

            return SimpleNamespace(get_images=get_images)

    class DecodedImage:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def convert(self, mode):
            assert mode == "RGB"
            return pixels

    def open_image(stream):
        assert stream.read() == b"encoded-color"
        return DecodedImage()

    modules = {
        "viam": {}, "viam.components": {}, "viam.robot": {},
        "viam.components.camera": {"Camera": Camera},
        "viam.robot.client": {"RobotClient": RobotClient},
        "PIL": {"Image": SimpleNamespace(open=open_image)},
    }
    for name, attrs in modules.items():
        module = ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setenv("VIAM_ADDRESS", "camera.example")
    monkeypatch.setenv("VIAM_API_KEY", "test-secret")
    monkeypatch.setenv("VIAM_API_KEY_ID", "test-id")
    return SimpleNamespace(calls=calls, state=state, pixels=pixels)


@pytest.mark.parametrize("mime_type", ["image/png", "image/jpeg", "image/jpeg;quality=80"])
def test_camera_reader_decodes_sdk_bytes_and_preserves_capture_clock(camera_sdk, mime_type):
    camera_sdk.state["images"][0].mime_type = mime_type
    with ViamCameraReader() as reader:
        reader.now = lambda: 100.0
        frame = reader.get_rgb_frames()["wrist"]
        assert frame.camera == "wrist"
        assert np.array_equal(frame.rgb, camera_sdk.pixels)
        assert frame.rgb.dtype == np.uint8
        assert frame.rgb is not camera_sdk.pixels
        assert frame.received_at_s == 100.0
        assert frame.captured_at_s == 17.25 and frame.capture_clock == "unix"
    reader.close()
    assert camera_sdk.calls == [
        ("connect", "camera.example", True), ("bind_camera", "arm-camera"),
        ("get_images", "arm-camera", 10.0), ("disconnect",),
    ]
    assert not reader._io._thread.is_alive()
    with pytest.raises(RuntimeError, match="closed"):
        reader.get_rgb_frames()


@pytest.mark.parametrize("metadata", [None, SimpleNamespace(HasField=lambda _: False)])
def test_missing_metadata_is_not_replaced_with_receive_time(camera_sdk, metadata):
    camera_sdk.state["metadata"] = metadata
    with ViamCameraReader() as reader:
        frame = reader.get_rgb_frames()["wrist"]
    assert frame.captured_at_s is None and frame.capture_clock is None


def test_explicit_unix_epoch_capture_is_preserved(camera_sdk):
    camera_sdk.state["metadata"].captured_at = SimpleNamespace(seconds=0, nanos=0)
    with ViamCameraReader() as reader:
        frame = reader.get_rgb_frames()["wrist"]
    assert frame.captured_at_s == 0.0 and frame.capture_clock == "unix"


def test_alias_selection_and_unknown_alias_fail_before_any_read(camera_sdk):
    with ViamCameraReader(cameras={"left": "left-camera", "right": "right-camera"}) as reader:
        assert reader.get_rgb_frames([]) == {}
        with pytest.raises(KeyError, match="Unknown camera aliases"):
            reader.get_rgb_frames(["left", "typo"])
        assert not any(call[0] == "get_images" for call in camera_sdk.calls)
        assert list(reader.get_rgb_frames(["right"])) == ["right"]
    assert [call[1] for call in camera_sdk.calls if call[0] == "get_images"] == ["right-camera"]


def test_missing_colour_fails_instead_of_fabricating_a_frame(camera_sdk):
    camera_sdk.state["images"] = [SimpleNamespace(mime_type="image/vnd.viam.dep", data=b"depth")]
    with ViamCameraReader() as reader:
        with pytest.raises(ValueError, match="no JPEG or PNG"):
            reader.get_rgb_frames()


def test_all_robot_operations_fail_locally_without_motion_or_stop(camera_sdk):
    with ViamCameraReader() as reader:
        before = list(camera_sdk.calls)
        for name in (
            "get_tcp_pose", "move_to_pose", "move_linear", "open_gripper", "close_gripper",
            "get_wrench", "tare_wrench", "cameras", "get_frames", "camera_pose", "stop",
            "gripper_state", "joint_positions", "update_world",
        ):
            with pytest.raises(NotImplementedError, match="RGB camera reads only"):
                getattr(reader, name)()
        assert camera_sdk.calls == before
    assert camera_sdk.calls[-1] == ("disconnect",)


def test_failed_camera_binding_disconnects_without_stop(camera_sdk):
    with pytest.raises(KeyError, match="missing-camera"):
        ViamCameraReader(cameras={"wrist": "missing-camera"})
    assert camera_sdk.calls == [
        ("connect", "camera.example", True), ("bind_camera", "missing-camera"), ("disconnect",),
    ]


@pytest.mark.parametrize("connect_timeout_s", [45.0, 75.0])
def test_connection_budget_is_longer_without_lengthening_camera_reads(camera_sdk, monkeypatch, connect_timeout_s):
    from assistive_arm.robot_iface import viam

    deadlines = []
    run = viam._Loop.run

    def record_deadline(loop, coroutine, timeout=60):
        deadlines.append((coroutine.cr_code.co_name, timeout))
        return run(loop, coroutine, timeout=timeout)

    monkeypatch.setattr(viam._Loop, "run", record_deadline)
    # Omit the argument in the default case to verify the deployed default too.
    kwargs = {} if connect_timeout_s == 45.0 else {"connect_timeout_s": connect_timeout_s}
    with ViamCameraReader(**kwargs) as reader:
        dial = camera_sdk.state["options"].dial_options
        assert dial.timeout == connect_timeout_s
        assert dial.initial_connection_attempt_timeout == connect_timeout_s
        assert dial.initial_connection_attempts == 1
        reader.get_rgb_frames()
    assert deadlines == [("at_address", connect_timeout_s + 1), ("get_images", 11.0), ("close", 10.0)]
    assert ("get_images", "arm-camera", 10.0) in camera_sdk.calls


@pytest.mark.parametrize("deadline", [0.0, -1.0, np.nan, np.inf, -np.inf])
def test_invalid_connection_timeout_fails_before_connecting(camera_sdk, deadline):
    with pytest.raises(ValueError, match="connect_timeout_s"):
        ViamCameraReader(connect_timeout_s=deadline)
    assert camera_sdk.calls == []
