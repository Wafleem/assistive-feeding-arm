"""The menu: canonical item names, what people call them, and how to acquire each.

The language layer maps "the strawberries" to "strawberry"; food perception
asks the detector for the canonical name; grasping picks the strategy.
Feasibility notes are for a PARALLEL-JAW gripper (85 mm stroke).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Strategy = Literal["pinch", "utensil_scoop", "utensil_skewer", "cup", "unsupported"]


@dataclass(frozen=True)
class MenuItem:
    name: str  # canonical, = the detector query
    synonyms: tuple[str, ...]
    strategy: Strategy
    feasibility: str  # one line, honest
    max_grip_force: float = 8.0  # N: soft foods bruise


MENU: tuple[MenuItem, ...] = (
    MenuItem("grape", ("grapes",), "pinch", "good: firm, 18-25 mm, pinch from the top"),
    MenuItem("strawberry", ("strawberries", "berry", "berries"), "pinch", "good if firm; grasp the body, not the leaves", 6.0),
    MenuItem("melon cube", ("melon", "melon cubes", "cantaloupe", "watermelon", "fruit cube", "cubes"), "pinch",
             "ok: 2 cm cubes; wet and soft, low force, may slip", 5.0),
    MenuItem("cracker", ("crackers", "biscuit", "cookie"), "pinch",
             "hard from flat on a plate (thin): stand it in a rack, or use a small cracker cube", 4.0),
    MenuItem("cheese cube", ("cheese",), "pinch", "good: cubes are the ideal finger food"),
    MenuItem("cup", ("water", "drink", "sip", "juice"), "cup",
             "plan: side grasp a 60-80 mm cup, straw toward the mouth; spill risk: lids"),
    MenuItem("yogurt", ("yoghurt", "pudding", "applesauce"), "utensil_scoop",
             "plan: grasp spoon from a fixed rest, scoop from a weighted bowl; stretch goal"),
    MenuItem("soup", (), "unsupported", "skip: spill risk near the face"),
)

_BY_WORD = {w: m for m in MENU for w in (m.name, *m.synonyms)}


def canonical(name: str) -> MenuItem | None:
    """'Strawberries' -> the strawberry entry; unknown words -> None (ask the user)."""
    key = " ".join(name.lower().replace("the ", "").split())
    return _BY_WORD.get(key) or _BY_WORD.get(key.rstrip("s"))


def menu_names() -> list[str]:
    return [m.name for m in MENU if m.strategy != "unsupported"]
