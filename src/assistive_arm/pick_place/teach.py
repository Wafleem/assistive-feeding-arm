"""Teach poses with the leader arm (teleop running) and store them as JSON.

PLAN
* ``record_pose(robot, name)``: read the follower's joint angles (and gripper %) while
  teleop mirrors the leader; return a TaughtPose. Take the median of ~5 reads ~50 ms apart
  so a pose recorded mid-motion is rejected (spread > 1 deg => ask to hold still).
* ``save_library`` / ``load_library``: JSON in ``poses/<task>.json`` with the robot id;
  ``load_library`` refuses a file taught on a different robot.
* The interactive CLI lives in ``scripts/so101_teach.py`` (TODO).
"""

from __future__ import annotations

from pathlib import Path

from .types import PoseLibrary, TaughtPose


def record_pose(robot, name: str, samples: int = 5) -> TaughtPose:
    raise NotImplementedError("TODO(pick_place): record_pose")


def save_library(lib: PoseLibrary, path: Path) -> None:
    raise NotImplementedError("TODO(pick_place): save_library")


def load_library(path: Path, expect_robot: str | None = None) -> PoseLibrary:
    raise NotImplementedError("TODO(pick_place): load_library")
