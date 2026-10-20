"""The feeding scene and SimRobot (shared infra: these work now)."""

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from assistive_arm.geometry import Pose, look_rotation  # noqa: E402
from assistive_arm.robot_iface.sim import SimRobot  # noqa: E402
from assistive_arm.vision import Locator  # noqa: E402
from conftest import todo  # noqa: E402

PARK = Pose.from_matrix([0.15, -0.35, 0.25], look_rotation([0, 0, -1], [1, 0, 0]))


@pytest.fixture(scope="module")
def robot():
    r = SimRobot("feeding")
    try:
        r.get_frames(["scene"])
    except Exception as e:  # no OpenGL context: skip camera tests, physics tests still run
        pytest.skip(f"rendering unavailable: {e}")
    assert r.move_to_pose(PARK).ok
    yield r
    r.close()


def test_scene_loads_and_food_stays_on_the_plate():
    r = SimRobot("feeding")
    before = {b: r.body_pose(b).position.copy() for b in ("grape_1", "strawberry_1", "melon_cube_1", "cup")}
    r.wait(1.0)
    for b, p in before.items():
        assert np.linalg.norm(r.body_pose(b).position - p) < 0.005, b
    assert "user_contact" not in r.events


def test_head_and_jaw_are_actuated():
    r = SimRobot("feeding")
    m0 = r.site_pose("mouth").position.copy()
    r.data.ctrl[r.actuator("head_y")] = 0.05
    r.data.ctrl[r.actuator("jaw")] = 0.3
    r.wait(1.0)
    assert r.site_pose("mouth").position[1] - m0[1] > 0.04
    assert r.scene_state()["jaw"] > 0.2
    assert np.allclose(r.site_pose("mouth").z_axis, [0, -1, 0], atol=0.05)  # face normal toward the table


def test_fixed_camera_sees_every_item_and_the_mouth(robot):
    loc = Locator(robot.oracle_detector())
    for q, n in [("grape", 2), ("strawberry", 1), ("melon cube", 1), ("cup", 1), ("mouth", 1)]:
        found = loc.locate_all(robot, q, ["scene"])
        assert len(found) == n, q
        assert len(found[0].extras["points"]) > 10


def test_mouth_located_from_the_camera_matches_the_site(robot):
    t = Locator(robot.oracle_detector()).locate(robot, "mouth", ["scene"], support_query="face")
    assert np.linalg.norm(t.position - robot.site_pose("mouth").position) < 0.02
    assert np.dot(t.normal, [0, -1, 0]) > 0.8


@todo
def test_feeding_world_hook_runs():
    from assistive_arm.sim import FeedingWorld

    r = SimRobot("feeding", hooks=[FeedingWorld()])
    r.wait(0.1)
