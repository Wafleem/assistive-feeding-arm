"""MuJoCo end-to-end tests: the same executor, skills and perception code as on hardware,
with the oracle detector standing in for the vision model. ~10 s total."""

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from assistive_arm.executor import Executor  # noqa: E402
from assistive_arm.geometry import Pose, look_rotation  # noqa: E402
from assistive_arm.intent import parse  # noqa: E402
from assistive_arm.perception import Locator  # noqa: E402
from assistive_arm.robot_iface.sim import SimRobot  # noqa: E402
from assistive_arm.skills import DoorParams, SkillContext  # noqa: E402

YES = SkillContext(confirm=lambda m: True)


@pytest.fixture(scope="session")
def rendering():
    """Skip camera-based tests where there is no OpenGL context (e.g. a CI box
    without MUJOCO_GL=osmesa/egl set up). Physics-only tests still run."""
    r = SimRobot("elevator")
    try:
        r.get_frames(["wrist"])
    except Exception as e:
        pytest.skip(f"rendering unavailable: {e}")
    finally:
        r.close()


@pytest.fixture(scope="module")
def elevator(rendering):
    r = SimRobot("elevator")
    yield r
    r.close()


@pytest.mark.parametrize("scene", ["elevator", "bag", "door_lever", "door_knob"])
def test_scenes_load_and_hold_still(scene):
    r = SimRobot(scene)
    start = r.get_tcp_pose()
    r._step(0.5)
    assert r.get_tcp_pose().distance_to(start)[0] < 0.003
    assert "user_contact" not in r.events


def test_cameras_render_aligned_rgbd(elevator):
    frames = elevator.get_frames()
    assert set(frames) == {"wrist", "scene"}
    for f in frames.values():
        assert f.rgb.shape == (240, 320, 3) and f.depth.shape == (240, 320)
        assert 0.05 < np.median(f.depth) < 3.0
    assert {s.mount for s in elevator.cameras().values()} == {"wrist", "base"}


def test_oracle_locates_button_within_a_few_mm(elevator):
    elevator.reset()
    elevator.move_to_pose(Pose.from_matrix([0.20, -0.35, 0.20], look_rotation([1, 0, 0], [0, 0, 1])))
    loc = Locator(elevator.oracle_detector())
    gt = elevator.geom_position("button_7") + [-0.006, 0, 0]  # front face
    for cam in ("scene", "wrist"):
        t = loc.locate(elevator, "elevator button 7", [cam], support_query="elevator panel")
        if t is None:
            assert cam == "wrist"  # the wrist may not see it from the parked pose
            continue
        assert np.linalg.norm(t.position - gt) < 0.012, cam
        assert np.dot(t.normal, [-1, 0, 0]) > 0.98


@pytest.mark.parametrize("utterance, label", [("press button 7 on the elevator", "7"), ("press the lobby button", "L")])
def test_press_button_end_to_end(elevator, utterance, label):
    elevator.reset()
    log = Executor(elevator, Locator(elevator.oracle_detector()), YES).run(parse(utterance))
    assert log.result.status == "success", (log.result, log.notes)
    assert f"pressed:{label}" in elevator.events
    assert [e for e in elevator.events if e.startswith("pressed:")] == [f"pressed:{label}"]  # only that one
    assert "user_contact" not in elevator.events


def test_grab_bag_end_to_end(rendering):
    r = SimRobot("bag")
    log = Executor(r, Locator(r.oracle_detector()), YES).run(parse("grab the shopping bag"))
    assert log.result.status == "success", (log.result, log.notes)
    assert abs(log.result.evidence["mass_kg"] - 1.2) < 0.15
    assert r.body_pose("bag").position[2] > 0.0  # lifted off the counter (bag origin started at 0.02 m)
    assert r.gripper_state().holding


def test_open_lever_door_with_known_hinge(rendering):
    r = SimRobot("door_lever")
    hinge = r._base_world().inverse().transform_point([0.78, -1.10, 0.95])
    ex = Executor(r, Locator(r.oracle_detector()), YES, door_defaults=DoorParams(hinge_side="right", hinge_point=hinge))
    log = ex.run(parse("grab the door handle and open it"))
    assert log.result.status == "success", (log.result, log.notes)
    assert r.scene_state()["door"] > np.deg2rad(15)


def test_door_never_reports_success_it_did_not_achieve(rendering):
    """With an estimated hinge the arc is a few cm off. Whatever happens, a
    'success' must mean the door really is open."""
    r = SimRobot("door_lever")
    log = Executor(r, Locator(r.oracle_detector()), YES,
                   door_defaults=DoorParams(hinge_side="right", door_width=0.71)).run(parse("open the door"))
    if log.result.status == "success":
        assert r.scene_state()["door"] > np.deg2rad(15)
    assert "user_contact" not in r.events


def test_sim_stops_on_contact_with_the_user():
    r = SimRobot("elevator")
    # Bypass the safety envelope on purpose: drive the TCP straight at the user's chest.
    res = r.move_linear(Pose.from_matrix([-0.15, 0.25, 0.30], look_rotation([0, 1, 0], [0, 0, 1])), 0.1)
    assert not res.ok
    assert res.reason in ("unreachable", "collision")  # refused at planning, or stopped on contact
    if "user_contact" in r.events:  # a contact latches a stop until reset()
        assert r.move_to_pose(r.get_tcp_pose()).reason == "stopped"


def test_wheelchair_sway_is_tolerated_by_guarded_press(rendering):
    r = SimRobot("elevator", sway=0.004)
    log = Executor(r, Locator(r.oracle_detector()), YES).run(parse("press button 5 on the elevator"))
    assert log.result.status == "success", (log.result, log.notes)
    assert "pressed:5" in r.events
