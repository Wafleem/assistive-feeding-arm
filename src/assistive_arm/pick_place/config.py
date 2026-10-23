"""Pick-and-place configuration (contract). Component names match the pi-arm2 machine."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PickPlaceConfig:
    arm: str = "follower-arm"
    gripper: str = "follower-gripper"
    teleop_service: str = "teleop"  # must be stopped before replay
    speed_deg_s: float = 30.0  # module default is 50; valid 3-180 (below ~12 is unreliable)
    first_run_speed_deg_s: float = 20.0
    acceleration_deg_s2: float = 200.0  # valid 50-500
    step_timeout_s: float = 10.0
    grip_settle_s: float = 0.5
    retry_on_empty_grip: int = 0  # hard-coded demo: don't retry blindly; stop and report
    pose_file_dir: str = "poses"
