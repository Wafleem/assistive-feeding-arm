"""Scene behaviour for sim/scenes/feeding.xml, plugged into SimRobot via ``hooks``.

PLAN
* SCENE CHANGE FIRST (TODO(sim)): sim/scenes/feeding.xml still has a hinged-jaw head
  from the first scaffold. Replace it with the hackathon prop: a 3D-printed face plate
  (from the same CAD as the real print) with an always-open opening (~50 x 40 mm) and a
  removable catch cup behind it; keep the moving head base (x/y/z + yaw) so retreat
  logic can be tested. Remove the jaw joint/actuator and its test.
* Head scripts: "still", "drift" (slow +-1 cm), "moved_away" (slid 5 cm / turned 40 deg
  during the approach, like a teammate moving the prop by hand: MUST cause a retreat),
  "tracking_lost" (detector dropouts: MUST pause/retreat).
* Delivered = the bite's centre ends inside the catch-cup volume after release.
* Metrics: delivered per food, insertion point vs the opening centre (lateral error),
  max TCP speed within 15 cm of the face, arm contact with the face plate (must be 0),
  correct retreats per scenario. Expected perception pitfall to check: depth inside the
  opening is the catch cup, not the face plane.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Scenario:
    name: str = "still"
    move_away_at: float | None = None  # s into the approach, for "moved_away"


@dataclass
class FeedingWorld:
    scenario: Scenario = field(default_factory=Scenario)
    events: list[str] = field(default_factory=list)

    def __call__(self, robot) -> None:
        """SimRobot hook, called every 20 ms tick."""
        raise NotImplementedError("TODO(sim): FeedingWorld hook")
