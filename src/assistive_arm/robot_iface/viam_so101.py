"""SO101ViamRobot: the pi-arm2 SO-101 follower + gripper through Viam. STUB (section 9).

Differences from ``ViamRobot`` (the xArm6 adapter):
* 5 joints + a separate gripper servo; joint-space moves are the primary API
  (see the contract proposal in ``pick_place/README.md``).
* No F/T sensor: ``move_linear(max_force=...)`` can't be force-guarded; use the
  servos' load/overload condition flags (``servo_position`` reports ``condition``) as a
  coarse contact signal, and keep speeds low.
* Speed/acceleration are set on the arm via DoCommand ``{"set_speed": ...}`` /
  ``{"set_acceleration": ...}``.
* Teleop state comes from the ``teleop`` generic service: ``{"command": "status"}``.

TODO(pick_place):
  1. Connect like ``ViamRobot`` (env VIAM_ADDRESS / VIAM_API_KEY / VIAM_API_KEY_ID).
  2. ``get_joint_positions`` / ``move_to_joint_positions`` via the Arm API (degrees).
  3. Gripper: ``open()`` / ``grab()`` (``grab`` returns whether something is held).
  4. ``teleop_running()`` and ``stop_teleop()`` via DoCommand on the teleop service.
  5. Cartesian methods: ``MoveToPosition`` with the module's orientation cone (later).
"""

from __future__ import annotations


class SO101ViamRobot:
    def __init__(self, address: str | None = None, arm: str = "follower-arm", gripper: str = "follower-gripper",
                 teleop: str = "teleop"):
        raise NotImplementedError("TODO(pick_place): SO101ViamRobot (see module docstring)")
