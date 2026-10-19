"""One robot API, three backends. ``SimRobot`` needs ``mujoco``; ``ViamRobot`` needs ``viam-sdk``."""

from .base import GripperState, MoveResult, RobotInterface, Wrench
from .fake import FakeRobot, Graspable, SpringSurface


def make_robot(kind: str, **kwargs) -> RobotInterface:
    if kind == "fake":
        return FakeRobot(**kwargs)
    if kind == "sim":
        from .sim import SimRobot

        return SimRobot(**kwargs)
    if kind == "viam":
        from .viam import ViamRobot

        return ViamRobot(**kwargs)
    raise ValueError(f"unknown robot kind {kind!r} (fake | sim | viam)")


__all__ = ["FakeRobot", "Graspable", "GripperState", "MoveResult", "RobotInterface", "SpringSurface", "Wrench", "make_robot"]
