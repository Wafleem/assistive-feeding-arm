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
from ..vision.types import CameraIntrinsics, CameraSpec, RGBDFrame
from ..motion.safety import SafetyConfig
from .base import GripperState, MoveResult, RobotInterface, Wrench
from .viam_convert import pose_to_viam_dict, viam_dict_to_pose


class _Loop:
    """Run the async SDK from synchronous skills on a background event loop."""

    def __init__(self):
        self.loop = asyncio.new_event_loop()
        threading.Thread(target=self.loop.run_forever, daemon=True).start()

    def run(self, coro, timeout=60):
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result(timeout)


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
            rgb, d = _rgb_and_depth(images)
            out[k] = RGBDFrame(k, rgb, d, self.cameras()[k].intrinsics, timestamp=self.now())
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


# --- Cameras only (no arm required): perception on the SO-101 bench rig -----------------


async def connect(address: str | None = None):
    """RobotClient from VIAM_ADDRESS / VIAM_API_KEY / VIAM_API_KEY_ID.

    ``VIAM_ADDRESS=localhost:<port>`` dials viam-server directly over TLS, without the cloud:
    on the Orin itself (port 8080), or from a laptop through an SSH tunnel over the Jetson's
    USB link (``ssh -L 18080:localhost:8080 ultraviolet@192.168.55.1``, then
    ``localhost:18080``). The venue Wi-Fi made cloud (WebRTC) dialling unreliable.
    The SDK skips certificate hostname checks for localhost only."""
    from viam.robot.client import RobotClient
    from viam.rpc.dial import DialOptions

    address = address or os.environ["VIAM_ADDRESS"]
    opts = RobotClient.Options.with_api_key(api_key=os.environ["VIAM_API_KEY"], api_key_id=os.environ["VIAM_API_KEY_ID"])
    if address.startswith(("localhost", "127.0.0.1")):
        opts.dial_options = DialOptions.with_api_key(os.environ["VIAM_API_KEY"], os.environ["VIAM_API_KEY_ID"])
        opts.dial_options.disable_webrtc = True
    return await RobotClient.at_address(address, opts)

# Logitech C270 at 640x480 (~60 deg diagonal FOV) until a checkerboard calibration exists.
C270_640x480 = CameraIntrinsics(780.0, 780.0, 320.0, 240.0, 640, 480)


def _rgb_and_depth(images):
    """NamedImages from ``Camera.get_images`` -> (RGB uint8 array, depth metres or zeros)."""
    from viam.media.utils.pil import viam_to_pil_image

    color = next((i for i in images if i.mime_type.startswith(("image/jpeg", "image/png"))), None)
    if color is None:
        raise RuntimeError(f"no colour image in {[i.mime_type for i in images]}")
    rgb = np.asarray(viam_to_pil_image(color).convert("RGB"))
    depth = next((i for i in images if "dep" in i.mime_type), None)
    if depth is None:  # plain webcam: no depth; detectors fall back to monocular priors
        return rgb, np.zeros(rgb.shape[:2], np.float32)
    return rgb, depth.bytes_to_depth_array().astype(np.float32) / 1000.0


class ViamCameraRig:
    """Read-only camera access for perception: the subset of ``RobotInterface`` that
    detectors use (``now``, ``get_frames``, ``camera_pose``, ``cameras``). Use it where
    there is no xArm, e.g. the SO-101 rig on ``orin-dev-kit``.

    Camera pose comes from the Viam frame system (the camera's ``frame`` in the
    machine config, e.g. ``webcam-2`` on ``follower-arm``), so it follows the arm.
    Credentials: ``VIAM_ADDRESS``, ``VIAM_API_KEY``, ``VIAM_API_KEY_ID`` (env or .env).
    """

    def __init__(self, cameras=("webcam-2",), address: str | None = None,
                 intrinsics: dict[str, CameraIntrinsics] | None = None, client=None, io=None):
        from viam.components.camera import Camera

        self._io = io or _Loop()
        self._address = address  # None + client given: reconnect via the env VIAM_ADDRESS is not attempted
        self._own_client = client is None or address is not None
        self.client = client or self._io.run(connect(address))
        self._names = tuple(cameras)
        self._cams = {c: self._find(Camera, c) for c in cameras}
        self._failures = 0
        self._k = dict(intrinsics or {})
        self.pose_error: str | None = None

    def _find(self, kind, name: str, wait: float = 15.0):
        """``kind.from_robot`` that rides out a machine reconfiguring (its resource list is
        briefly incomplete while a config change applies): refresh and retry for ``wait`` s."""
        import time

        from viam.errors import ResourceNotFoundError

        deadline = time.monotonic() + wait
        while True:
            try:
                return kind.from_robot(self.client, name)
            except ResourceNotFoundError:
                if time.monotonic() > deadline:
                    raise
                time.sleep(1.0)
                self._io.run(self.client.refresh(), timeout=10)

    def now(self) -> float:
        import time

        return time.monotonic()

    def intrinsics(self, name: str) -> CameraIntrinsics:
        if name not in self._k:
            ip = self._io.run(self._cams[name].get_properties()).intrinsic_parameters
            self._k[name] = (CameraIntrinsics(ip.focal_x_px, ip.focal_y_px, ip.center_x_px, ip.center_y_px,
                                              ip.width_px, ip.height_px) if ip.focal_x_px else C270_640x480)
        return self._k[name]

    def cameras(self) -> dict[str, CameraSpec]:
        return {c: CameraSpec(c, "wrist", self.intrinsics(c)) for c in self._cams}

    def reconnect(self) -> None:
        """Fresh connection (e.g. after viam-server restarted and the SDK's own reconnect stalled)."""
        from viam.components.camera import Camera

        old = self.client
        try:
            self._io.run(old.close(), timeout=3)
        except Exception:
            pass
        self.client = self._io.run(connect(self._address), timeout=30)
        self._cams = {c: self._find(Camera, c) for c in self._names}
        self._failures = 0

    def get_frames(self, names=None, timeout: float = 5.0) -> dict[str, RGBDFrame]:
        out = {}
        try:
            for name in names or list(self._cams):
                t = self.now()
                images, _ = self._io.run(self._cams[name].get_images(), timeout=timeout)
                rgb, depth = _rgb_and_depth(images)
                out[name] = RGBDFrame(name, rgb, depth, self.intrinsics(name), timestamp=t)
        except Exception:
            self._failures += 1
            if self._failures >= 2 and self._own_client:
                self.reconnect()
            raise
        self._failures = 0
        return out

    def camera_pose(self, name: str) -> Pose:
        """Camera pose in the base (world) frame; identity if the frame system can't say."""
        from viam.proto.common import Pose as VPose, PoseInFrame

        try:
            origin = PoseInFrame(reference_frame=name, pose=VPose(o_z=1))
            p = self._io.run(self.client.transform_pose(origin, "world"), timeout=5).pose
            self.pose_error = None
            return viam_dict_to_pose({k: getattr(p, k) for k in ("x", "y", "z", "o_x", "o_y", "o_z", "theta")})
        except Exception as e:  # no frame configured: positions stay in the camera frame
            self.pose_error = str(e)
            return Pose()

    def close(self) -> None:
        self._io.run(self.client.close())


class ViamVisionDetector:
    """A Viam vision service (e.g. YOLOv8 ``spoon-detector`` running on the Orin) behind the
    ``Detector`` protocol. Detections are fetched once per frame and filtered by label.
    The service runs on its own latest image of ``camera``, so boxes can lag the frame by
    one capture (~30 ms); fine for bowls, not for the face (MediaPipe runs locally)."""

    def __init__(self, rig: ViamCameraRig, service: str = "spoon-detector", camera: str = "webcam-2"):
        from viam.services.vision import VisionClient

        self.rig, self.camera = rig, camera
        self.vision = VisionClient.from_robot(rig.client, service)
        self._cache: tuple[float, list] | None = None

    def detect(self, frame: RGBDFrame, query: str):
        from ..vision.types import Detection

        if self._cache is None or self._cache[0] != frame.timestamp:
            dets = self.rig._io.run(self.vision.get_detections_from_camera(self.camera), timeout=10)
            self._cache = (frame.timestamp, dets)
        q = query.lower().strip()
        return [Detection(query, float(d.confidence), (float(d.x_min), float(d.y_min), float(d.x_max), float(d.y_max)))
                for d in self._cache[1] if d.class_name.lower() == q]
