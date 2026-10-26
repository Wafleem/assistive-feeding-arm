import pytest
"""Smooth streamed motion (motion/smoothing.py) and how the session uses it: the recorded
scoop is resampled, smoothed, capped and eased in and out; streams follow the wall clock
(no catch-up burst after a slow call); a glide starts from the last setpoint."""

import json
import time
from pathlib import Path

import numpy as np

from assistive_arm.motion.smoothing import (MIN_JERK_PEAK, SmoothingConfig, ease_in_out, gaussian_smooth, limit_speed,
                                            min_jerk, peaks, point_to_point, resample)
from assistive_arm.orchestration.face_servo_session import FaceServoSession
from test_face_track_session import LOOK, KinArm, SeenFace

DEMO = Path(__file__).resolve().parents[1] / "poses" / "so101_scoop_demo.json"


def speeds(t, q):
    return np.abs(np.diff(q, axis=0) / np.diff(t)[:, None]).max(axis=1)


def test_min_jerk_is_rest_to_rest_with_the_documented_peak():
    s = np.linspace(0, 1, 2001)
    p = min_jerk(s)
    v = np.gradient(p, s)
    a = np.gradient(v, s)
    assert p[0] == 0 and p[-1] == 1
    assert abs(v[0]) < 1e-3 and abs(v[-1]) < 1e-3
    assert abs(a[1]) < 0.1 and abs(a[-2]) < 0.1  # a cosine profile starts at pi^2/2 ~ 4.9
    assert abs(v.max() - MIN_JERK_PEAK) < 1e-3


def test_resample_makes_uneven_samples_even_and_keeps_the_ends():
    t = np.array([0.0, 0.005, 0.048, 0.05, 0.05, 0.09, 0.13])
    q = np.c_[t * 100, -t * 50]
    tu, qu = resample(t, q, 30.0)
    assert np.allclose(np.diff(tu), np.diff(tu)[0])
    assert tu[-1] == t[-1] - t[0] and np.allclose(qu[-1], q[-1]) and np.allclose(qu[0], q[0])
    assert np.allclose(qu[:, 0], tu * 100)  # linear stays linear


def test_gaussian_smoothing_removes_jitter_without_lag_and_holds_the_ends():
    rate = 30.0
    t = np.arange(0, 3, 1 / rate)
    clean = np.c_[40 * np.sin(2 * np.pi * 0.5 * t)]
    noisy = clean + np.random.default_rng(0).normal(0, 1.5, clean.shape)
    out = gaussian_smooth(noisy, rate, 0.05)
    rms = lambda e: float(np.sqrt(np.mean(e[5:-5] ** 2)))
    assert rms(out - clean) < 0.7 * rms(noisy - clean)  # light on purpose: 1.5 samples, the scoop keeps its depth
    assert np.allclose(out[[0, -1]], noisy[[0, -1]])
    step = gaussian_smooth(np.c_[(t > 1.5).astype(float)], rate, 0.05)[:, 0]
    assert abs(t[np.argmin(np.abs(step - 0.5))] - 1.5) < 1 / rate  # zero phase: the edge doesn't move


def test_limit_speed_stretches_time_until_under_both_caps():
    t = np.linspace(0, 1, 31)
    q = np.c_[300 * min_jerk(t)]  # peak 562 deg/s
    t2, k = limit_speed(t, q, 200.0, 2000.0)
    v, a = peaks(t2, q)
    assert k > 1 and v <= 200.0 + 1e-6 and a <= 2000.0 + 1e-6
    assert limit_speed(t, q, 1e4, 1e6)[1] == 1.0


def test_ease_in_out_starts_and_ends_at_rest_on_the_same_path():
    t = np.linspace(0, 2, 61)
    q = np.c_[60 * t, 10 * np.ones_like(t)]  # constant 60 deg/s from the first sample
    te, qe = ease_in_out(t, q, 0.4, 30.0)
    v = speeds(te, qe)
    assert abs(te[-1] - 2.4) < 1e-9
    assert v[0] < 1.0 and v[-1] < 1.0 and v.max() <= 60.0 + 1e-6
    assert np.allclose(qe[0], q[0]) and np.allclose(qe[-1], q[-1])


def test_point_to_point_peaks_at_the_requested_speed():
    t, q = point_to_point([0, 0, 0, 0, 0], [90, -30, 0, 10, 0], 60.0, 30.0)
    assert np.allclose(q[0], 0) and np.allclose(q[-1], [90, -30, 0, 10, 0])
    assert abs(speeds(t, q).max() - 60.0) < 1.0


def session(arm, tmp_path):
    return FaceServoSession(arm, SeenFace(arm), poses_path=tmp_path / "p.json", settle=0.0)


@pytest.mark.skip(reason="session streaming from the smoothing branch is not adopted on main (motion tuned live on the robot)")
def test_recorded_scoop_replays_smoothly(tmp_path):
    s = session(KinArm(LOOK), tmp_path)
    try:
        demo = json.loads(DEMO.read_text())
        t, q, rep = s._replay_path(demo)
        c = SmoothingConfig()
        v = speeds(t, q)
        assert np.allclose(np.diff(t), np.diff(t)[0])  # even: equal steps at equal times
        assert v[0] < 2.0 and v[-1] < 2.0  # eased: no kick after the glide that ends at rest
        assert rep["peak_speed_degs"] <= c.max_speed + 1 and rep["peak_accel_degs2"] <= c.max_accel + 1
        # the raw recording, as streamed before: 500 deg/s and 30000 deg/s^2 spikes from its uneven timestamps
        raw_t, raw_q = np.asarray(demo["t"], float), np.asarray(demo["q"], float)
        keep = np.r_[True, np.diff(raw_t) > 1e-6]
        assert peaks(raw_t[keep], raw_q[keep])[1] > 5 * rep["peak_accel_degs2"]
    finally:
        s.close()


class StallingArm(KinArm):
    """Streaming calls that sometimes take 0.3 s (the cloud link on venue Wi-Fi)."""

    def __init__(self, q):
        super().__init__(q)
        self.sent_at = []

    def stream_joints(self, q):
        super().stream_joints(q)
        self.sent_at.append(time.monotonic())
        if len(self.streamed) == 5:
            time.sleep(0.3)


@pytest.mark.skip(reason="session streaming from the smoothing branch is not adopted on main (motion tuned live on the robot)")
def test_a_slow_call_does_not_cause_a_burst_of_catch_up_setpoints(tmp_path):
    arm = StallingArm(LOOK)
    s = session(arm, tmp_path)
    try:
        t, q = point_to_point(LOOK, np.asarray(LOOK) + [30, 0, 0, 0, 0], 60.0, 30.0)
        s._stream_path(t, q)
        gaps = np.diff(arm.sent_at)
        assert np.sum(gaps < 0.01) <= 1  # the old loop sent ~9 setpoints back to back here
        assert len(arm.streamed) < len(t) - 5  # the stalled stretch was skipped, not replayed late
        assert np.allclose(arm.streamed[-1], q[-1])
    finally:
        s.close()


class SaggingArm(KinArm):
    """Measured joints 3 deg under the setpoints (gravity sag, servo lag)."""

    def joints(self, timeout=None):
        return self.q.copy() - 3.0


@pytest.mark.skip(reason="session streaming from the smoothing branch is not adopted on main (motion tuned live on the robot)")
def test_a_glide_after_a_stream_starts_from_the_last_setpoint_not_the_measured_joints(tmp_path):
    arm = SaggingArm(LOOK)
    s = session(arm, tmp_path)
    try:
        start = np.asarray(LOOK, float) + [10, 0, 0, 0, 0]
        s._send(start)
        n = len(arm.streamed)
        s._glide(start + [20, 0, 0, 0, 0])
        first = arm.streamed[n]
        assert np.max(np.abs(first - start)) < 0.5  # from the measured joints it would step back 3 deg
    finally:
        s.close()
