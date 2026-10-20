"""Scene behaviour for sim/scenes/feeding.xml, plugged into SimRobot via ``hooks``.

PLAN
* HeadScript: drive the head_x/y/z/yaw actuators per scenario:
  "still", "drift" (slow +-1 cm sine), "lean_in" (moves 2 cm toward the bite),
  "turn_away" (at time T yaw 40 deg + back 5 cm in 0.5 s: MUST trigger a retreat),
  "never_opens" (jaw stays shut: MUST time out).
* Jaw: open after a delay; close when a bite is "taken" (food within 2 cm of the
  "mouth" site while open, held for 0.5 s) -> event "bite_taken", food removed.
* Metrics recorder: distance/lateral error at delivery, max TCP speed within 15 cm
  of the mouth, any arm contact with head geoms (SimRobot latches a stop),
  whether each scenario's required abort happened.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Scenario:
    name: str = "still"
    jaw_open_after: float = 2.0  # s
    turn_away_at: float | None = None  # s


@dataclass
class FeedingWorld:
    scenario: Scenario = field(default_factory=Scenario)
    events: list[str] = field(default_factory=list)

    def __call__(self, robot) -> None:
        """SimRobot hook, called every 20 ms tick."""
        raise NotImplementedError("TODO(sim): FeedingWorld hook")
