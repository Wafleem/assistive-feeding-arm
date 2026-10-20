"""Spec for section 1 (mouth_perception). Remove @todo as you implement."""

import numpy as np

from conftest import todo

from assistive_arm.mouth_perception import (
    ConstantVelocityKF, HeadMotionMonitor, MouthObservation, MouthState, MouthTracker, ReadyGate, TrackerParams,
)

N = np.array([0.0, -1.0, 0.0])


def obs(t, p, open_ratio=1.0, conf=1.0, n=N):
    return MouthObservation(t, np.asarray(p, float), np.asarray(n, float), open_ratio, conf)


@todo
def test_kf_tracks_constant_velocity():
    rng = np.random.default_rng(0)
    kf = ConstantVelocityKF()
    v = np.array([0.02, -0.01, 0.0])
    for k in range(60):  # 2 s at 30 Hz, 4 mm noise
        t = k / 30
        kf.update(np.array([0.3, 0.3, 0.3]) + v * t + rng.normal(0, 0.004, 3), t)
    assert np.linalg.norm(kf.x[3:] - v) < 0.035  # velocity noise budget at sigma_accel=0.3, see kalman.py
    pred, vel = kf.predict_ahead(0.2)
    assert np.allclose(pred, kf.x[:3] + 0.2 * kf.x[3:]) and np.allclose(vel, kf.x[3:])


@todo
def test_kf_gates_outliers():
    kf = ConstantVelocityKF()
    for k in range(30):
        kf.update([0.3, 0.3, 0.3], k / 30)
    assert not kf.update([0.3, 0.3, 0.6], 1.0)  # 30 cm jump in one frame: a misdetection
    assert kf.rejected == 1 and abs(kf.x[2] - 0.3) < 0.01


@todo
def test_latency_compensation_reduces_lag_error():
    v = np.array([0.0, 0.03, 0.0])  # slow lean toward the spoon
    tr = MouthTracker(TrackerParams(latency=0.15))
    for k in range(45):
        tr.update(obs(k / 30, [0.3, 0.3 + v[1] * k / 30, 0.3]))
    now = 44 / 30
    truth_at_actuation = np.array([0.3, 0.3 + v[1] * (now + 0.15), 0.3])
    comp = tr.state(now).position
    raw = tr.state(now, compensate_latency=False).position
    assert np.linalg.norm(comp - truth_at_actuation) < 0.3 * np.linalg.norm(raw - truth_at_actuation)


@todo
def test_open_state_has_hysteresis():
    tr = MouthTracker()
    tr.update(obs(0.0, [0.3, 0.3, 0.3], open_ratio=0.5))
    assert tr.is_open
    tr.update(obs(0.1, [0.3, 0.3, 0.3], open_ratio=0.30))  # between closed and open thresholds
    assert tr.is_open
    tr.update(obs(0.2, [0.3, 0.3, 0.3], open_ratio=0.1))
    assert not tr.is_open


def state(t=0.0, p=(0.3, 0.3, 0.3), v=(0, 0, 0), n=N, open_=True, conf=1.0, age=0.0):
    return MouthState(t, np.asarray(p, float), np.asarray(v, float), np.asarray(n, float), 1.0 if open_ else 0.0,
                      open_, conf, age)


@todo
def test_ready_gate_needs_open_and_still_for_the_hold_time():
    g = ReadyGate()
    assert not g.update(state(open_=False), 0.0)
    assert not g.update(state(), 0.1)  # open, but the hold timer just started
    assert g.update(state(), 0.7)
    assert not g.update(state(v=(0.06, 0, 0)), 0.8) and "moving" in g.reason
    assert not g.update(state(), 0.9)  # timer restarted
    assert not g.update(state(age=1.0), 2.0) and g.reason == "tracking stale"


@todo
def test_head_motion_monitor_triggers_retreat():
    m = HeadMotionMonitor()
    m.start(state())
    assert m.check(state()).ok
    assert "fast" in m.check(state(v=(0.3, 0, 0))).reason
    assert "moved" in m.check(state(p=(0.3, 0.36, 0.3))).reason
    turned = np.array([np.sin(np.radians(40)), -np.cos(np.radians(40)), 0])
    assert "turned" in m.check(state(n=turned)).reason
    assert "closed" in m.check(state(open_=False)).reason
    assert not m.check(None).ok
