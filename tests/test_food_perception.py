"""Section 2 (food_perception). The vocabulary is real; finding food is a spec."""

import pytest

from assistive_arm.food_perception import canonical, menu_names
from conftest import todo


@pytest.mark.parametrize("said, name", [("strawberries", "strawberry"), ("Grapes", "grape"), ("melon", "melon cube"),
                                        ("water", "cup"), ("cheese", "cheese cube")])
def test_vocabulary_maps_what_people_say(said, name):
    assert canonical(said).name == name


def test_unknown_food_is_none_and_soup_is_not_offered():
    assert canonical("lasagna") is None
    assert "soup" not in menu_names()


@todo
def test_find_grapes_in_sim():
    mujoco = pytest.importorskip("mujoco")  # noqa: F841
    from assistive_arm.food_perception import FoodPerception
    from assistive_arm.robot_iface.sim import SimRobot
    from assistive_arm.vision import Locator

    r = SimRobot("feeding")
    items = FoodPerception(Locator(r.oracle_detector()), cameras=["scene"]).find(r, "grapes")
    assert len(items) >= 1 and all(it.name == "grape" for it in items)
    assert len(items[0].points) > 10
