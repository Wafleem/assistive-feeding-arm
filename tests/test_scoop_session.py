"""Teach the bowl and Scoop from the dashboard session, on the fake arm with the real SO-101
kinematics (tests/test_face_track_session.py): planned before anything moves, streamed, back
where it started, and stopped when the real arm can't follow."""

import json
from dataclasses import replace

import numpy as np

from assistive_arm.grasping.scoop import BowlTarget, ScoopConfig, _Spoon, _solve_seeded, pan_axis_xy
from assistive_arm.orchestration.face_servo_session import FaceServoSession
from assistive_arm.robot_iface.so101_kinematics import fk, joint_limits
from test_face_track_session import LOOK, KinArm, SeenFace, level_first, wait

A = np.radians(-25.0)
BOTTOM = np.r_[pan_axis_xy() + 0.31 * np.array([np.cos(A), np.sin(A)]), 0.008]
# 10x the real speeds so a whole scoop streams in a few seconds
FAST = replace(ScoopConfig(dip_deg=50.0), max_joint_speed_deg=250.0, air_speed=0.6, dip_speed=0.2,
               scoop_speed=0.2, max_rotate_deg_s=300.0)


def tip_on_the_bottom() -> np.ndarray:
    """Joints with the spoon tip on BOTTOM (as if hand-guided there)."""
    c = ScoopConfig()
    sp = _Spoon(c, np.r_[np.cos(A), np.sin(A), 0.0])
    lim = joint_limits()
    q = _solve_seeded(sp, BOTTOM, np.radians(-40.0), LOOK, lim[:, 0] + 3, lim[:, 1] - 3, c)
    assert q is not None
    return q


def tip(q):
    T = fk(q)
    return T[:3, 3] + ScoopConfig().spoon_tip_from_tool * T[:3, 2]


def session(arm, tmp_path):
    s = FaceServoSession(arm, SeenFace(arm), poses_path=tmp_path / "p.json", settle=0.0)
    s.scoop_cfg = replace(FAST, spoon_tip_from_tool=s.ext_cfg.spoon_tip_from_tool)
    return s


def test_scoop_without_a_taught_bowl_explains_itself_and_does_not_move(tmp_path):
    arm = KinArm(LOOK)
    s = session(arm, tmp_path)
    s.start()
    try:
        level_first(s)
        s.command("scoop")
        assert wait(lambda: s.mode == "idle" and "teach the bowl first" in s.reason), s.reason
        assert arm.streamed == []
    finally:
        s.close()


def test_teach_bowl_from_the_spoon_tip_then_scoop_and_come_back(tmp_path):
    arm = KinArm(tip_on_the_bottom())
    s = session(arm, tmp_path)
    s.start()
    try:
        s.command("teach_bowl", rim_radius_cm=6, depth_cm=4)
        saved = json.loads((tmp_path / "p.json").read_text())["bowl"]
        assert np.allclose(np.asarray(saved["bottom_mm"]) / 1000, BOTTOM, atol=0.001)
        assert saved["rim_radius_mm"] == 60 and saved["depth_mm"] == 40
        arm.move_joints(LOOK)
        level_first(s)
        start = arm.q.copy()
        s.command("plan_scoop")  # dry run
        assert arm.streamed == [] and s.last_scoop["ok"], s.last_scoop
        s.command("scoop")
        assert wait(lambda: s.mode == "idle" and "scooped and back" in s.reason, timeout=60), s.reason
        assert np.allclose(arm.q, start)
        tips = np.array([tip(q) for q in arm.streamed])
        assert tips[:, 2].min() - BOTTOM[2] < 0.01  # down to the bottom
        assert np.linalg.norm(tips[np.argmin(tips[:, 2])][:2] - BOTTOM[:2]) < 0.06  # inside the bowl
        assert s.state()["bowl"]["depth_mm"] == 40
    finally:
        s.close()


def test_scoop_stops_when_the_arm_falls_behind(tmp_path):
    arm = KinArm(LOOK)
    s = session(arm, tmp_path)
    s.taught.bowl = {"bottom_mm": [round(v * 1000, 1) for v in BOTTOM], "rim_radius_mm": 60.0, "depth_mm": 40.0}
    s.start()
    try:
        level_first(s)
        arm.sag = 15.0  # measured joints 15 deg off the commands (blocked, or stalled servos)
        s.command("scoop")
        assert wait(lambda: s.mode == "idle" and "behind the scoop" in s.reason, timeout=30), s.reason
        assert 0 < len(arm.streamed) < 60  # stopped within the first second or so
    finally:
        s.close()


def test_unreachable_taught_bowl_is_refused_before_anything_moves(tmp_path):
    arm = KinArm(LOOK)
    s = session(arm, tmp_path)
    s.taught.bowl = {"bottom_mm": [150.0, -60.0, 8.0], "rim_radius_mm": 60.0, "depth_mm": 40.0}  # 12 cm out: too close
    s.start()
    try:
        level_first(s)
        s.command("scoop")
        assert wait(lambda: s.mode == "idle" and "scoop refused" in s.reason, timeout=30), s.reason
        assert arm.streamed == []
    finally:
        s.close()
