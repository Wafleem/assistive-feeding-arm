"""Section 9 (pick_place). The sequence data is checked now; teaching and replay are specs."""

from pathlib import Path

import pytest

from assistive_arm.pick_place import (
    PICK_PLACE_STEPS, PickPlaceConfig, PoseLibrary, TaughtPose, load_library, required_poses, run_pick_and_place,
    save_library,
)
from conftest import todo


def test_sequence_grasps_between_pick_and_lift_and_ends_home():
    kinds = [(s.kind, s.pose) for s in PICK_PLACE_STEPS]
    assert kinds[0] == ("move", "home") and kinds[-1] == ("move", "home")
    i_pick, i_close, i_verify, i_lift = (kinds.index(k) for k in [("move", "pick"), ("gripper_close", None),
                                                                     ("verify_grip", None), ("move", "lift")])
    assert i_pick < i_close < i_verify < i_lift
    assert required_poses() == {"home", "pre_pick", "pick", "lift", "pre_place", "place", "retreat"}


def test_default_config_matches_pi_arm2_and_is_slow():
    c = PickPlaceConfig()
    assert (c.arm, c.gripper, c.teleop_service) == ("follower-arm", "follower-gripper", "teleop")
    assert 12 <= c.first_run_speed_deg_s <= c.speed_deg_s <= 50


def library():
    poses = {n: TaughtPose(n, [0.0, 10.0 * i, 20.0, 0.0, 0.0]) for i, n in enumerate(sorted(required_poses()))}
    return PoseLibrary("cube_to_bowl", "pi-arm2/follower-arm", poses)


@todo
def test_pose_library_round_trips_and_checks_the_robot(tmp_path: Path):
    p = tmp_path / "cube_to_bowl.json"
    save_library(library(), p)
    assert load_library(p, expect_robot="pi-arm2/follower-arm").poses["pick"].joints_deg == library().poses["pick"].joints_deg
    with pytest.raises(ValueError):
        load_library(p, expect_robot="pi-arm1/follower-arm")


@todo
def test_replay_refuses_a_library_missing_poses():
    lib = library()
    del lib.poses["place"]
    result = run_pick_and_place(robot=None, library=lib)
    assert not result.ok and "place" in result.reason
