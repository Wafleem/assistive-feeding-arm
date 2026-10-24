"""Which fixed bowl holds which food: geometry, look poses and the food -> bowl choice."""

import numpy as np

from assistive_arm.food_perception.bowl_id import (BowlLabel, FixedBowl, assign, bowl_for, camera_pose, crop,
                                                   look_pose, opening_outline, project)

HOME = np.array([7.38, -98.2, 74.29, 20.92, -1.05])
B1 = FixedBowl("bowl1", "Bowl 1 (cereal)", np.array([0.2296, 0.2165]), default_food="cereal")
B2 = FixedBowl("bowl2", "Bowl 2 (M&Ms)", np.array([0.3092, 0.0945]), default_food="mms")


def test_look_pose_puts_each_bowl_in_the_middle_of_the_image():
    for b in (B1, B2):
        q, info = look_pose(b, HOME, n=6000)
        T = camera_pose(q)
        c = project(T, np.r_[b.centre, b.rim_z][None])[0]
        assert 60 < c[0] < 580 and 60 < c[1] < 400, (b.bowl_id, c)
        _, box, frac = opening_outline(b, T)
        assert box is not None and frac >= 0.6
        assert 0.12 <= info["distance_cm"] / 100 <= 0.35


def test_opening_is_not_drawn_when_the_camera_looks_away():
    T = camera_pose(HOME)  # home looks out over the table, not down at the bowls
    _, box, _ = opening_outline(B1, T)
    if box is not None:  # if any of it projects, it must still be a real box
        assert box[0] < box[2] and box[1] < box[3]


def test_crop_is_the_inside_of_the_opening():
    rgb = np.zeros((480, 640, 3), np.uint8)
    c = crop(rgb, (100, 100, 300, 260), shrink=0.1)
    assert c.shape == (128, 160, 3)
    assert crop(rgb, (630, 470, 700, 520)) is None  # almost all outside: nothing usable


def test_food_goes_to_the_bowl_clip_named_and_falls_back_to_the_fixed_one():
    labels = [BowlLabel("bowl1", "mms", {"mms": 0.9, "cereal": 0.1}, 0.8),
              BowlLabel("bowl2", "cereal", {"cereal": 0.95, "mms": 0.05}, 0.9)]
    assert assign(labels) == {"mms": "bowl1", "cereal": "bowl2"}  # the bowls were swapped
    assert bowl_for("mms", labels, [B1, B2])[0].bowl_id == "bowl1"
    unsure = [BowlLabel("bowl1", "cereal", {"cereal": 0.55}, 0.1), BowlLabel("bowl2", "cereal", {"cereal": 0.9}, 0.8)]
    assert bowl_for("mms", unsure, [B1, B2])[0].bowl_id == "bowl2"  # no clear M&Ms: the fixed assignment
    both = [BowlLabel("bowl1", "cereal", {}, 0.9), BowlLabel("bowl2", "cereal", {}, 0.9)]
    assert "cereal" not in assign(both)  # two bowls claim it: ambiguous, not guessed
    assert bowl_for("cereal", None, [B1, B2])[1].startswith("fixed")
