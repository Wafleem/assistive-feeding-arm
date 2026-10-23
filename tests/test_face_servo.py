"""Face visual servoing on the SO-101 rig: decisions only (no arm, no camera)."""

import numpy as np
import pytest

from assistive_arm.feeding.face_servo import (
    FaceServoConfig, FaceServoController, FaceView, ImageJacobian, joint_target, target_pixel,
)
from assistive_arm.mouth_perception import MouthState

J = ImageJacobian.from_probes([-40.0, 2.0], [3.0, 35.0], probe_deg=3.0)  # px per probe of 3 deg


def view(px=(320, 240), dist=0.6, front=0.95):
    return FaceView(px, dist, front, (640, 480), 780.0, 780.0)


def mstate(speed=0.0, age=0.0):
    return MouthState(0.0, np.zeros(3), np.array([speed, 0, 0]), np.array([0, 0, -1.0]), 0.5, True, 0.9, age)


def on_target(dist=0.6, cfg=None):
    t = target_pixel(view(dist=dist), cfg or FaceServoConfig())
    return (float(t[0]), float(t[1]))


def test_centring_moves_the_face_toward_the_spoon_line_and_converges():
    c = FaceServoController(J, approach=False)
    px = np.array([500.0, 300.0])
    for _ in range(30):
        cmd = c.step(view(tuple(px)), mstate())
        if cmd.kind != "center":
            break
        px = px + J.J @ cmd.d_offset  # the image responds as calibrated
        assert np.all(np.abs(cmd.d_offset) <= c.cfg.max_step_deg + 1e-9)
    assert cmd.kind == "hold" and "centred" in cmd.reason
    assert np.linalg.norm(px - on_target()) < c.cfg.center_tol_px


def test_approach_only_when_centred_and_still_and_stops_at_the_stop_distance():
    cfg = FaceServoConfig(stop_tip_distance=0.20)
    c = FaceServoController(J, cfg)
    assert c.step(view(on_target(0.6), dist=0.6), mstate(speed=0.08)).kind == "hold"  # head not still
    cmd = c.step(view(on_target(0.6), dist=0.6), mstate())
    assert cmd.kind == "approach" and cmd.ds == cfg.approach_ds
    assert c.step(view(on_target(0.31), dist=0.31), mstate()).kind == "done"  # tip 0.19 m


def test_retreat_when_the_face_is_lost_or_moves_during_the_approach():
    c = FaceServoController(J)
    assert c.step(view(on_target()), mstate()).kind == "approach"
    cmd = c.step(None, None)
    assert cmd.kind == "retreat" and c.s == 0.0
    assert c.step(view(on_target()), mstate()).kind == "approach"
    assert c.step(view(on_target()), mstate(speed=0.3)).kind == "retreat"
    assert c.step(view(on_target()), mstate()).kind == "approach"
    assert c.step(view(on_target(), front=0.3), mstate()).kind == "retreat"


def test_too_close_retreats_and_lost_before_approaching_only_holds():
    c = FaceServoController(J)
    assert c.step(None, None).kind == "hold"
    assert c.step(view(on_target()), mstate()).kind == "approach"
    assert c.step(view(on_target(0.2), dist=0.2), mstate()).kind == "retreat"  # tip 8 cm < 20 - 2


def test_offsets_are_limited_in_total():
    cfg = FaceServoConfig(max_offset_deg=(5.0, 5.0))
    c = FaceServoController(J, cfg, approach=False)
    for _ in range(20):
        c.step(view((630, 470)), mstate())  # the face never moves: offsets saturate
    assert np.all(np.abs(c.offset) <= 5.0 + 1e-9)


def test_joint_target_interpolates_and_adds_offsets():
    look, near = [30, -90, 80, 20, 0], [30, -60, 50, 10, 0]
    q = joint_target(look, near, 0.5, [2.0, -1.0])
    assert np.allclose(q, [32, -75, 65, 14, 0])


def test_stop_distance_floor_cannot_be_configured_away():
    with pytest.raises(ValueError):
        FaceServoConfig(stop_tip_distance=0.01)


def test_degenerate_probes_are_rejected():
    with pytest.raises(ValueError):
        ImageJacobian.from_probes([0.0, 0.0], [3.0, 35.0], 3.0)
