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
