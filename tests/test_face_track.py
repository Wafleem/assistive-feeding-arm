"""Continuous level tracker on the real SO-101 kinematics (simulated, perfect joint tracking)."""

import numpy as np
import pytest

from assistive_arm.feeding.face_track import LevelTracker, TrackConfig
from assistive_arm.robot_iface.so101_kinematics import fk

Q0 = np.array([22.0, -89.1, 89.3, -12.7, -7.6])  # a real pose from the rig (spoon tilted ~12 deg up)


def run(face_fn, seconds=6.0, q0=Q0, **cfg):
    tr = LevelTracker(TrackConfig(**cfg))
    tr.reset(q0)
    qs, vs, errs = [tr.q.copy()], [tr.v.copy()], []
    n = int(seconds * tr.cfg.rate_hz)
    for k in range(n):
        f = face_fn(k / tr.cfg.rate_hz)
        qs.append(tr.step(f))
        vs.append(tr.v.copy())
        errs.append(dict(tr.last))
    return tr, np.array(qs), np.array(vs), errs


def face_ahead(q, dist=0.6, dx=0.0, dz=0.0):
    T = fk(q)
    d = T[:3, 2].copy(); d[2] = 0; d /= np.linalg.norm(d)
    side = np.cross([0, 0, 1.0], d)
    return T[:3, 3] + dist * d + dx * side + np.array([0, 0, dz])


def test_converges_onto_an_offset_face_level_and_smooth():
    face = face_ahead(Q0, dist=0.6, dx=-0.12, dz=0.08)  # 12 cm to the right, 8 cm up
    tr, qs, vs, errs = run(lambda t: face)
    e = errs[-1]
    assert e["miss_cm"] <= 1.2 and abs(e["pitch_deg"]) <= 1.0 and abs(e["reach_cm"]) <= 1.0, e
    acc = np.abs(np.diff(vs, axis=0)) * tr.cfg.rate_hz
    assert acc.max() <= tr.cfg.max_acc_deg + 1e-6  # no jerks
    assert np.abs(vs[:, :4]).max(axis=0).round(6).tolist() <= list(tr.cfg.max_vel_deg)
    assert np.allclose(qs[:, 4], Q0[4])  # wrist roll held: bowl stays up


def test_settles_without_hunting():
    face = face_ahead(Q0, dist=0.6, dx=0.05, dz=-0.03)
    tr, qs, vs, errs = run(lambda t: face, seconds=8.0)
    tail = vs[-40:]  # last 2 s
    assert np.abs(tail).max() < 0.5  # at rest, deg/s
    assert errs[-1]["miss_cm"] <= 1.2


def test_follows_a_slowly_moving_face():
    base = face_ahead(Q0, dist=0.6)
    side = np.cross([0, 0, 1.0], base / np.linalg.norm(base))
    tr, qs, vs, errs = run(lambda t: base + 0.06 * np.sin(0.6 * t) * side, seconds=12.0)  # 6 cm sway
    late = [e["miss_cm"] for e in errs[-120:]]
    assert max(late) <= 3.0, max(late)


def test_a_lost_face_decelerates_smoothly_to_a_stop():
    face = face_ahead(Q0, dist=0.6, dx=-0.15)
    tr = LevelTracker(); tr.reset(Q0)
    for _ in range(10):
        tr.step(face)
    moving = np.abs(tr.v).max()
    assert moving > 1.0
    speeds = []
    for _ in range(20):
        tr.step(None)
        speeds.append(np.abs(tr.v).max())
    assert speeds[-1] == pytest.approx(0.0, abs=1e-9)
    assert all(b <= a + 1e-9 for a, b in zip(speeds, speeds[1:]))  # monotonic slow-down
    assert np.abs(np.diff(speeds)).max() <= tr.cfg.max_acc_deg / tr.cfg.rate_hz + 1e-6


def test_lips_far_above_reach_tip_up_within_the_limit_without_drifting():
    face = face_ahead(Q0, dist=0.9, dz=0.55)  # lips ~0.7 m up: beyond even a tipped-up spoon
    tr, qs, vs, errs = run(lambda t: face, seconds=15.0)
    e = errs[-1]
    assert -1.0 <= e["tilt_deg"] <= tr.cfg.max_tip_up_deg + 1.0 and e["tip_up_allowed"], e
    assert e["lips_above_reach_cm"] > 5, e  # still out of reach: reported, not chased
    tail = np.abs(vs[-40:, :4]).max(axis=1)
    assert tail.max() < 2.0 and tail[-1] <= tail[0]  # slowing to a stop, not drifting into a folded pose
    assert (qs[:, 1:4] >= np.minimum(tr.lo[1:4], Q0[1:4]) - 1e-9).all() and (qs[:, 1:4] <= tr.hi[1:4] + 1e-9).all()


# --- extend on the tracker -----------------------------------------------------------------
from assistive_arm.feeding.face_track import ExtendConfig, extend_decision  # noqa: E402


def test_extend_decision_rules():
    c = ExtendConfig(stop_tip_distance=0.20)
    ok = dict(face_seen=True, age=0.05, head_speed=0.0, miss_cm=0.8, tilt_deg=0.5, tip=0.40, travelled=0.0, held_s=0.0)
    assert extend_decision(c, **ok)[0] == 1
    assert extend_decision(c, **{**ok, "miss_cm": 3.0})[0] == 0  # centre first
    assert extend_decision(c, **{**ok, "tilt_deg": 4.0})[0] == 0  # level first
    assert extend_decision(c, **{**ok, "head_speed": 0.08})[0] == 0  # wait for stillness
    assert extend_decision(c, **{**ok, "head_speed": 0.2, "travelled": 0.05})[0] == -1
    assert extend_decision(c, **{**ok, "face_seen": False, "travelled": 0.05})[0] == -1
    assert extend_decision(c, **{**ok, "face_seen": False})[0] == 0  # nothing to back out of
    assert extend_decision(c, **{**ok, "tip": 0.20, "travelled": 0.1, "held_s": 1.0})[0] == 0  # holding
    assert extend_decision(c, **{**ok, "tip": 0.20, "travelled": 0.1, "held_s": 9.0})[0] == -1  # hold over
    assert extend_decision(c, **{**ok, "tip": 0.15, "travelled": 0.1})[0] == -1  # too close
    with pytest.raises(ValueError):
        ExtendConfig(stop_tip_distance=0.01)


def test_tracker_extends_level_to_the_stop_distance_then_backs_out():
    c = ExtendConfig(stop_tip_distance=0.20, hold_s=1.0)
    face = face_ahead(Q0, dist=0.55, dz=0.0)
    tr = LevelTracker(); tr.reset(Q0); tr.set_extending(True)
    dt = 1 / tr.cfg.rate_hz
    for _ in range(80):  # centre and level first
        tr.step(face, dt)
    reach, tips, pitches, held, phase = [], [], [], 0.0, "in"
    for k in range(1200):
        e = tr.last
        tip = tr.tip_distance(face, c.spoon_tip_from_tool)
        d, why = extend_decision(c, True, 0.05, 0.0, e["miss_cm"], e["pitch_deg"], tip, tr.travelled, held)
        if phase == "out" or d < 0:
            phase, d = "out", -1
        held = held + dt if why.startswith("at the stop") else 0.0
        before = tr.reach0
        tr.advance(d, dt, c)
        assert tr.reach0 - before <= c.approach_speed * dt + 1e-12
        tr.step(face, dt)
        reach.append(tr.travelled); tips.append(tip); pitches.append(tr.last["pitch_deg"])
        if phase == "out" and tr.travelled <= 1e-4 and abs(tr.actual_travelled()) < 0.01:
            break
    assert min(tips) >= c.stop_tip_distance - 0.02 and min(tips) <= c.stop_tip_distance + 0.02  # got there, not past
    assert max(abs(p) for p in pitches) <= 2.5  # spoon stayed level all the way
    assert tr.travelled <= 1e-4 and abs(tr.actual_travelled()) < 0.01 and max(reach) > 0.05  # in, then back out



def test_high_lips_tip_the_spoon_up_never_down_and_keep_the_face_in_view():
    # The real case: the spoon starts tipped up ~24 deg at lips ~12 cm above the cap.
    q0 = np.array([61.0, -96.4, 71.8, 6.5, -3.3])
    T0 = fk(q0)
    d = T0[:3, 2].copy(); d[2] = 0; d /= np.linalg.norm(d)
    face = T0[:3, 3] + 0.45 * d + np.array([0, 0, 0.0])
    face[2] = 0.44  # 10 cm above the 34 cm cap
    tr, qs, vs, errs = run(lambda t: face, seconds=10.0, q0=q0)
    tilts = np.array([np.degrees(np.arcsin(fk(q)[2, 2])) for q in qs])
    assert tilts.min() >= -1.0  # never tipped down below level
    assert tilts[-1] <= tr.cfg.max_tip_up_deg + 1.0  # within the allowed tip-up
    offaxis = [np.degrees(np.arccos(np.clip(np.dot(fk(q)[:3, 2], (face - fk(q)[:3, 3]) / np.linalg.norm(face - fk(q)[:3, 3])), -1, 1))) for q in qs]
    assert max(offaxis) < 17.0  # the lips never leave the C270's vertical field of view (~+-17 deg)
    assert errs[-1]["miss_cm"] <= 3.0, errs[-1]
