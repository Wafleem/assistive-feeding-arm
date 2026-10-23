"""Real hardware through Viam: the venue's xArm6 + parallel gripper + RealSense-class cameras.

UNTESTED against hardware (none yet). Written against the Python viam-sdk
API (``RobotClient``, ``Arm``, ``Gripper``, ``Camera``, ``Sensor``,
``MotionClient``); re-check signatures for the SDK version you install.

Expected machine config (component names are constructor arguments):

* ``arm``     - ``viam:ufactory:xArm6`` (frame parent: ``world``; put the
                table mount offset in the frame system, not in code)
* ``gripper`` - xArm gripper module, frame parent ``arm``
* ``wrist-cam`` / ``scene-cam`` - ``viam:camera:realsense`` with depth; the
                wrist camera's frame parent is ``gripper`` (hand-eye
                calibration), the scene camera's is ``world`` (on a post at the table)
* ``ft``      - optional sensor returning force_x/y/z in N (UFACTORY 6-axis
                F/T sensor via a small module, or joint-current estimates)
* ``builtin`` motion service with the keep-out boxes passed as obstacles

Viam has no force-guarded linear move, so ``move_linear(max_force=...)``
steps in ``guard_step`` increments and reads the sensor between steps. That
is slow but bounded; replace it with the xArm SDK's force/impedance mode
via ``do_command`` on a custom module for anything faster.
"""

from __future__ import annotations

import asyncio
import os
import threading

import numpy as np

from ..geometry import Pose, interpolate_poses
from ..vision.types import CameraIntrinsics, CameraSpec, RGBDFrame, RGBFrame
from ..motion.safety import SafetyConfig
from .base import GripperState, MoveResult, RobotInterface, Wrench
from .viam_convert import pose_to_viam_dict, viam_dict_to_pose


class _Loop:
    """Run the async SDK from synchronous skills on a background event loop."""

    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self._thread.start()

    def run(self, coro, timeout=60):
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result(timeout)

    def close(self):
        self.loop.call_soon_threadsafe(self.loop.stop)
        self._thread.join()
        pending = asyncio.all_tasks(self.loop)
        for task in pending:
            task.cancel()
        if pending:
            self.loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        self.loop.close()


class ViamRobot(RobotInterface):
    def __init__(self, address: str | None = None, arm="arm", gripper="gripper",
                 cameras: dict[str, tuple[str, str]] | None = None, ft_sensor: str | None = None,
                 safety: SafetyConfig | None = None, guard_step: float = 0.002):
        from viam.components.arm import Arm
        from viam.components.camera import Camera
        from viam.components.gripper import Gripper
        from viam.robot.client import RobotClient
        from viam.services.motion import MotionClient

        self._io = _Loop()
        address = address or os.environ["VIAM_ADDRESS"]
        opts = RobotClient.Options.with_api_key(
            api_key=os.environ["VIAM_API_KEY"], api_key_id=os.environ["VIAM_API_KEY_ID"]
        )
        self.client = self._io.run(RobotClient.at_address(address, opts))
        self.arm = Arm.from_robot(self.client, arm)
        self.gripper_c = Gripper.from_robot(self.client, gripper)
        self.motion = MotionClient.from_robot(self.client, "builtin")
        # local name -> (viam component name, mount)
        cameras = cameras or {"wrist": ("wrist-cam", "wrist"), "scene": ("scene-cam", "base")}
        self._cams = {k: (Camera.from_robot(self.client, v[0]), v[1]) for k, v in cameras.items()}
        self._cam_names = {k: v[0] for k, v in cameras.items()}
        self.ft = None
        if ft_sensor:
            from viam.components.sensor import Sensor

            self.ft = Sensor.from_robot(self.client, ft_sensor)
        self.safety = safety or SafetyConfig()
        self.guard_step = guard_step
        self._tare = np.zeros(3)
        self._specs: dict[str, CameraSpec] = {}

    # helpers ---------------------------------------------------------------------
    def _viam_pose(self, p: Pose):
        from viam.proto.common import Pose as VPose

        return VPose(**pose_to_viam_dict(p))

    def update_world(self, zones) -> None:
        """Moving obstacles (the tracked head, the plate) added to the next motion plan."""
        self._dynamic_zones = tuple(zones)

    def _obstacles(self):
        """Keep-out zones as Viam obstacles so the motion planner avoids the user.

        TODO(feeding): also send the carried item/utensil as a geometry attached to
        the gripper frame (WorldState transforms), as the pick-and-place project did.
        """
        from viam.proto.common import (
            Geometry, GeometriesInFrame, Pose as VPose, RectangularPrism, Sphere, Vector3, WorldState,
        )

        geoms = []
        for z in tuple(self.safety.keep_out) + getattr(self, "_dynamic_zones", ()):
            if hasattr(z, "radius"):
                c = np.asarray(z.center) * 1000
                geoms.append(Geometry(center=VPose(x=c[0], y=c[1], z=c[2], o_z=1, theta=0),
                                      sphere=Sphere(radius_mm=z.radius * 1000), label=z.name))
            else:
                c, s = z.center * 1000, z.size * 1000
                geoms.append(Geometry(center=VPose(x=c[0], y=c[1], z=c[2], o_z=1, theta=0),
                                      box=RectangularPrism(dims_mm=Vector3(x=s[0], y=s[1], z=s[2])), label=z.name))
        return WorldState(obstacles=[GeometriesInFrame(reference_frame="world", geometries=geoms)])

    # RobotInterface ---------------------------------------------------------------
    def get_tcp_pose(self) -> Pose:
        p = self._io.run(self.arm.get_end_position())
        return viam_dict_to_pose({k: getattr(p, k) for k in ("x", "y", "z", "o_x", "o_y", "o_z", "theta")})

    def move_to_pose(self, pose: Pose, speed: float = 0.10) -> MoveResult:
        from viam.proto.common import PoseInFrame

        dest = PoseInFrame(reference_frame="world", pose=self._viam_pose(pose))
        try:
            ok = self._io.run(self.motion.move(
                component_name=self.gripper_c.get_resource_name(self.gripper_c.name),
                destination=dest, world_state=self._obstacles(),
            ), timeout=120)
        except Exception as e:  # planning failure, IK failure, timeout
            return MoveResult(False, f"motion: {e}", self.get_tcp_pose())
        return MoveResult(bool(ok), "", self.get_tcp_pose())

    def move_linear(self, pose: Pose, speed: float = 0.03, max_force: float | None = None) -> MoveResult:
        start = self.get_tcp_pose()
        dist = float(np.linalg.norm(pose.position - start.position))
        step = self.guard_step if max_force is not None else 0.02
        peak, travelled, prev = 0.0, 0.0, start
        for wp in interpolate_poses(start, pose, max(1, int(np.ceil(dist / step)))):
            self._io.run(self.arm.move_to_position(self._viam_pose(wp)))
            travelled += float(np.linalg.norm(wp.position - prev.position))
            prev = wp
            if max_force is not None:
                f = self.get_wrench().magnitude
                peak = max(peak, f)
                if f > max_force:
                    return MoveResult(False, "force_limit", self.get_tcp_pose(), peak, travelled)
        return MoveResult(True, "", self.get_tcp_pose(), peak, travelled)

    def open_gripper(self, width: float | None = None) -> GripperState:
        self._io.run(self.gripper_c.open())
        return GripperState(self.gripper_max_width, False)

    def close_gripper(self, force: float = 20.0) -> GripperState:
        holding = bool(self._io.run(self.gripper_c.grab()))  # grab() reports whether something was caught
        return GripperState(np.nan, holding, force)

    def _raw_force(self) -> np.ndarray:
        if self.ft is None:
            return np.zeros(3)  # no sensing: guarded moves degrade to travel limits only
        r = self._io.run(self.ft.get_readings())
        return np.array([r.get("force_x", 0.0), r.get("force_y", 0.0), r.get("force_z", 0.0)], dtype=float)

    def get_wrench(self) -> Wrench:
        return Wrench(self._raw_force() - self._tare)

    def tare_wrench(self) -> None:
        self._tare = self._raw_force()

    def cameras(self) -> dict[str, CameraSpec]:
        if not self._specs:
            for k, (cam, mount) in self._cams.items():
                props = self._io.run(cam.get_properties())
                ip = props.intrinsic_parameters
                self._specs[k] = CameraSpec(k, mount, CameraIntrinsics(
                    ip.focal_x_px, ip.focal_y_px, ip.center_x_px, ip.center_y_px, ip.width_px, ip.height_px))
        return self._specs

    def get_frames(self, names=None) -> dict[str, RGBDFrame]:
        out = {}
        for k in names or list(self._cams):
            cam = self._cams[k][0]
            images, _meta = self._io.run(cam.get_images())
            color = next(i for i in images if i.mime_type.startswith("image/jpeg") or i.mime_type.startswith("image/png"))
            depth = next(i for i in images if "dep" in i.mime_type)
            rgb = np.asarray(color.image.convert("RGB"))
            d = np.asarray(depth.image if hasattr(depth, "image") else depth.data, dtype=np.float32) / 1000.0
            out[k] = RGBDFrame(k, rgb, d, self.cameras()[k].intrinsics)
        return out

    def camera_pose(self, name: str) -> Pose:
        from viam.proto.common import Pose as VPose, PoseInFrame

        # The frame system knows both the static camera mount and the hand-eye offset.
        origin = PoseInFrame(reference_frame=self._cam_names[name], pose=VPose(o_z=1))
        p = self._io.run(self.client.transform_pose(origin, "world")).pose
        # Assumes the camera frame is configured with OpenCV optical axes (+z forward, +y down).
        return viam_dict_to_pose({k: getattr(p, k) for k in ("x", "y", "z", "o_x", "o_y", "o_z", "theta")})

    def stop(self) -> None:
        self._io.run(self.arm.stop())


class ViamCameraReader(RobotInterface):
    """Read only RGB images from existing Viam cameras, with sessions disabled.

    This adapter never binds arm, gripper, or motion clients. Unsupported robot
    operations fail locally, including ``stop``. ``close`` only disconnects the
    network client. The RGB-only stream cannot provide calibrated camera poses,
    intrinsics, or metric mouth/handle positions.
    """

    def __init__(self, address: str | None = None,
                 cameras: dict[str, str] | None = None, timeout_s: float = 10.0,
                 connect_timeout_s: float = 45.0):
        from viam.components.camera import Camera
        from viam.robot.client import RobotClient

        if not np.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("timeout_s must be positive and finite")
        if not np.isfinite(connect_timeout_s) or connect_timeout_s <= 0:
            raise ValueError("connect_timeout_s must be positive and finite")
        camera_names = {"wrist": "arm-camera"} if cameras is None else dict(cameras)
        if not camera_names or any(not alias or not name for alias, name in camera_names.items()):
            raise ValueError("cameras must map nonempty aliases to Viam camera names")
        address = address or os.environ["VIAM_ADDRESS"]
        opts = RobotClient.Options.with_api_key(
            api_key=os.environ["VIAM_API_KEY"], api_key_id=os.environ["VIAM_API_KEY_ID"],
            disable_sessions=True,
        )
        self.timeout_s = float(timeout_s)
        self.connect_timeout_s = float(connect_timeout_s)
        opts.dial_options.timeout = self.connect_timeout_s
        opts.dial_options.initial_connection_attempt_timeout = self.connect_timeout_s
        opts.dial_options.initial_connection_attempts = 1
        self._closed = True
        self._io = _Loop()
        self.client = None
        try:
            self.client = self._io.run(
                RobotClient.at_address(address, opts), timeout=self.connect_timeout_s + 1.0,
            )
            self._cams = {alias: Camera.from_robot(self.client, name) for alias, name in camera_names.items()}
        except BaseException:
            try:
                if self.client is not None:
                    self._io.run(self.client.close(), timeout=self.timeout_s)
            finally:
                self._io.close()
            raise
        self._closed = False

    def get_rgb_frames(self, names: list[str] | None = None) -> dict[str, RGBFrame]:
        if self._closed:
            raise RuntimeError("ViamCameraReader is closed")
        selected = list(self._cams) if names is None else list(names)
        unknown = set(selected).difference(self._cams)
        if unknown:
            raise KeyError(f"Unknown camera aliases: {sorted(unknown)}")
        out = {}
        for name in selected:
            images, metadata = self._io.run(
                self._cams[name].get_images(timeout=self.timeout_s), timeout=self.timeout_s + 1.0,
            )
            received_at = self.now()
            color = next((im for im in images if str(im.mime_type).split(";", 1)[0] in
                          {"image/jpeg", "image/png"}), None)
            if color is None:
                raise ValueError(f"Camera {name!r} returned no JPEG or PNG colour image")
            from io import BytesIO
            from PIL import Image

            with Image.open(BytesIO(color.data)) as image:
                rgb = np.array(image.convert("RGB"), dtype=np.uint8, copy=True)
            captured_at = None
            if metadata is not None and metadata.HasField("captured_at"):
                stamp = metadata.captured_at
                captured_at = float(stamp.seconds) + float(stamp.nanos) / 1_000_000_000
            out[name] = RGBFrame(name, rgb, received_at, captured_at,
                                 "unix" if captured_at is not None else None)
        return out

    def close(self) -> None:
        """Disconnect; never call StopAll, arm.stop, or a camera resource close."""
        if self._closed:
            return
        self._closed = True
        try:
            self._io.run(self.client.close(), timeout=self.timeout_s)
        finally:
            self._io.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    @staticmethod
    def _unsupported(*args, **kwargs):
        raise NotImplementedError("ViamCameraReader supports RGB camera reads only")

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
