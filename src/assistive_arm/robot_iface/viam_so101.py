"""SO101ViamRobot: the SO-101 follower (+ gripper, teleop, cameras) through Viam.

Joint-space first (degrees, SO-101 order: pan, lift, elbow, wrist flex, wrist roll):
small profiled moves (``move_joints`` waits until they settle). The module's ``Stop``
does NOT interrupt a joint move in progress, so callers near a person must keep every
move small. Speed is capped with the arm's ``set_speed`` DoCommand (floor ~12 deg/s).

Cameras come through ``ViamCameraRig`` on the same connection (``rig``), so perception
code (``MouthDetector.observe(robot)``) can take this object directly.

Not a full ``RobotInterface`` (no Cartesian guarded moves, no F/T sensor).
TODO(pick_place): gripper open/grab; Cartesian ``MoveToPosition`` with the orientation cone.
"""

from __future__ import annotations

import numpy as np

from .viam import ViamCameraRig, _Loop, connect


class SO101ViamRobot:
    def __init__(self, address: str | None = None, arm: str = "follower-arm", gripper: str = "follower-gripper",
                 teleop: str = "teleop", cameras=("webcam-2",), camera_address: str | None = None):
        """``camera_address``: a second, camera-only connection (e.g. ``localhost:18080``, an SSH
        tunnel over the Jetson's USB link) so video doesn't depend on the venue Wi-Fi; arm and
        teleop commands keep using ``address`` (the normal Viam cloud connection)."""
        from viam.components.arm import Arm
        from viam.services.generic import Generic

        self._io = _Loop()
        self.client = self._io.run(connect(address))
        self.arm = Arm.from_robot(self.client, arm)
        self.teleop_name = teleop
        self._teleop = Generic.from_robot(self.client, teleop) if teleop else None
        if camera_address:
            self.rig = ViamCameraRig(cameras, address=camera_address, io=self._io)
        else:
            self.rig = ViamCameraRig(cameras, client=self.client, io=self._io)
        self._arm_name = arm
        self._fast = None  # streamed setpoints go over the direct (camera) connection when there is one

    # perception passthrough (so detectors can use this object as their "robot")
    def now(self) -> float:
        return self.rig.now()

    def get_frames(self, names=None):
        return self.rig.get_frames(names)

    def camera_pose(self, name: str):
        return self.rig.camera_pose(name)

    # arm
    def joints(self, timeout: float = 10.0) -> np.ndarray:
        jp = self._io.run(self.arm.get_joint_positions(), timeout=timeout)
        return np.asarray(jp.values, float)

    def move_joints(self, q, timeout: float = 30.0) -> None:
        from viam.proto.component.arm import JointPositions

        self._io.run(self.arm.move_to_joint_positions(JointPositions(values=[float(v) for v in q])), timeout=timeout)

    def tool_pose(self, timeout: float = 10.0) -> tuple[np.ndarray, np.ndarray, float]:
        """(position m, tool +Z unit vector, roll theta deg) of the arm's end frame, base frame."""
        p = self._io.run(self.arm.get_end_position(), timeout=timeout)
        z = np.array([p.o_x, p.o_y, p.o_z], float)
        return np.array([p.x, p.y, p.z], float) / 1000.0, z / (np.linalg.norm(z) or 1.0), float(p.theta)

    def move_tool(self, position, direction, theta: float = 0.0, position_leeway_mm: float = 8.0,
                  orientation_leeway: float = 0.03, theta_leeway_deg: float = 180.0, timeout: float = 30.0) -> None:
        """Planned move (the module's own IK/planner, frame-system obstacles included) of the end
        frame to ``position`` with +Z along ``direction`` and roll ``theta`` (+- ``theta_leeway_deg``;
        180 = roll free). Keep the leeways tight near a person: the planner only promises the goal."""
        from viam.proto.common import Pose

        p, d = np.asarray(position, float) * 1000.0, np.asarray(direction, float)
        cloud = {"x": position_leeway_mm, "y": position_leeway_mm, "z": position_leeway_mm,
                 "ox": orientation_leeway, "oy": orientation_leeway, "oz": orientation_leeway,
                 "theta": float(theta_leeway_deg)}
        pose = Pose(x=p[0], y=p[1], z=p[2], o_x=d[0], o_y=d[1], o_z=d[2], theta=theta)
        self._io.run(self.arm.move_to_position(pose, extra={"pose_cloud": cloud}), timeout=timeout)

    def arm_command(self, cmd: dict, timeout: float = 10.0) -> dict:
        """The SO-101 module's DoCommand, e.g. {"command": "enter_manual_mode"} (hand-guiding:
        the arm holds against gravity and follows your hand) / {"command": "exit_manual_mode"}."""
        return dict(self._io.run(self.arm.do_command(cmd), timeout=timeout))

    def stream_joints(self, q) -> None:
        """Non-blocking streamed setpoint (what teleop uses): no servo profile, so the CALLER must
        keep successive setpoints close together (the face tracker limits speed and acceleration).
        Sent over the direct connection when available (lower latency than the cloud)."""
        from viam.proto.component.arm import JointPositions

        if self._fast is None:
            from viam.components.arm.client import ArmClient

            same = self.rig.client is self.client
            self._fast = self.arm if same else ArmClient(self._arm_name, self.rig.client._channel)
        self._io.run(self._fast.move_to_joint_positions(JointPositions(values=[float(v) for v in q]),
                                                        extra={"wait": False, "streamed": True}), timeout=2.0)

    def set_speed(self, degs_per_sec: float) -> float:
        out = self._io.run(self.arm.do_command({"set_speed": float(degs_per_sec)}), timeout=10)
        return float(out.get("speed_set", degs_per_sec))

    def stop(self) -> None:
        self._io.run(self.arm.stop(), timeout=10)

    # teleop (the follower must not be driven by the leader while servoing)
    def teleop_status(self) -> dict:
        return dict(self._io.run(self._teleop.do_command({"command": "status"}), timeout=10)) if self._teleop else {}

    def stop_teleop(self) -> None:
        if self._teleop:
            self._io.run(self._teleop.do_command({"command": "stop"}), timeout=10)

    def start_teleop(self) -> None:
        if self._teleop:
            self._io.run(self._teleop.do_command({"command": "start"}), timeout=10)

    def close(self) -> None:
        self._io.run(self.client.close())
