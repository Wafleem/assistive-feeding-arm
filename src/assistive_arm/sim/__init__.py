"""Section 7: simulation behaviour and metrics for the feeding scene. See README.md in this folder.

The robot itself is ``robot_iface.sim.SimRobot``; assets are in the top-level ``sim/`` folder.
"""

from .feeding_world import FeedingWorld, Scenario

__all__ = ["FeedingWorld", "Scenario"]
