"""Smooth joint trajectories for streamed servo arms (the SO-101's setpoint stream).

A streamed setpoint has no servo profile behind it: whatever the stream does, the servos do.
So a trajectory is made smooth before it is streamed:

* ``resample``: a recording's uneven timestamps (teleop at "30 Hz" arrives 5-48 ms apart)
  onto an even grid, so equal steps go out at equal times;
* ``gaussian_smooth``: zero-phase smoothing (no lag), ends held, so jitter goes but the
  motion's timing and extremes stay;
* ``limit_speed``: stretch time (never clip a position) until peak speed and acceleration
  are under the caps;
* ``ease_in_out``: playback starts and ends at rest with zero acceleration (a minimum-jerk
  ramp of the playback rate), so joining a move that ended at rest is not a kick;
* ``min_jerk``: the rest-to-rest profile for point-to-point moves (zero speed AND zero
  acceleration at both ends; a cosine profile jumps in acceleration at both ends).

Pure numpy, no robot: ``t`` in s, ``q`` an (n, joints) array in whatever unit the caller
streams (degrees for the SO-101), and the caps in that unit per s and per s^2.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SmoothingConfig:
    rate_hz: float = 30.0  # setpoints per second (the streaming rate)
    sigma_s: float = 0.05  # Gaussian smoothing; the recorded scoop loses < 0.3 deg of depth at 0.05
    ease_s: float = 0.4  # playback ramps from rest to full rate (and back) over this long
    max_speed: float = 200.0  # deg/s, any joint (STS3215 servos manage ~270 unloaded)
    max_accel: float = 2000.0  # deg/s^2, any joint


def min_jerk(s):
    """Rest-to-rest progress 0 -> 1 for s in [0, 1]: 10s^3 - 15s^4 + 6s^5. Peak rate 1.875x the
    mean (a cosine profile peaks at 1.571x), zero speed and acceleration at both ends."""
    s = np.clip(np.asarray(s, float), 0.0, 1.0)
    return s ** 3 * (10.0 - 15.0 * s + 6.0 * s ** 2)


MIN_JERK_PEAK = 1.875  # peak speed / mean speed of ``min_jerk``


def sample(t, q, at):
    """q at time(s) ``at`` (linear between samples, held outside [t0, tn])."""
    t, q = np.asarray(t, float), np.asarray(q, float)
    at = np.atleast_1d(np.asarray(at, float))
    out = np.stack([np.interp(at, t, q[:, j]) for j in range(q.shape[1])], axis=1)
    return out if out.shape[0] > 1 else out[0]


def resample(t, q, rate_hz: float):
    """Even time grid from t[0] at ``rate_hz`` (repeated timestamps dropped), last sample kept."""
    t, q = np.asarray(t, float), np.asarray(q, float)
    keep = np.r_[True, np.diff(t) > 1e-6]
    t, q = t[keep] - t[0], q[keep]
    n = max(2, int(round(t[-1] * rate_hz)) + 1)
    tu = np.linspace(0.0, t[-1], n)
    return tu, sample(t, q, tu)


def gaussian_smooth(q, rate_hz: float, sigma_s: float):
    """Zero-phase Gaussian smoothing of evenly sampled ``q``; the end samples are held (padded),
    so the trajectory still starts and ends where it did."""
    q = np.asarray(q, float)
    s = sigma_s * rate_hz
    if s <= 0.0:
        return q.copy()
    r = int(np.ceil(3.0 * s))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / s) ** 2)
    k /= k.sum()
    pad = np.vstack([np.repeat(q[:1], r, 0), q, np.repeat(q[-1:], r, 0)])
    out = np.stack([np.convolve(pad[:, j], k, "valid") for j in range(q.shape[1])], axis=1)
    out[0], out[-1] = q[0], q[-1]
    return out


def peaks(t, q) -> tuple[float, float]:
    """(peak speed, peak acceleration) over all joints, by finite differences."""
    t, q = np.asarray(t, float), np.asarray(q, float)
    if len(t) < 3:
        return 0.0, 0.0
    v = np.diff(q, axis=0) / np.diff(t)[:, None]
    tm = 0.5 * (t[1:] + t[:-1])
    a = np.diff(v, axis=0) / np.diff(tm)[:, None]
    return float(np.abs(v).max()), float(np.abs(a).max())


def limit_speed(t, q, max_speed: float, max_accel: float):
    """Times stretched uniformly (positions untouched) so peak speed <= ``max_speed`` and peak
    acceleration <= ``max_accel``. Returns (t, factor); factor 1 when already under the caps."""
    v, a = peaks(t, q)
    factor = max(1.0, v / max_speed if max_speed > 0 else 1.0, np.sqrt(a / max_accel) if max_accel > 0 else 1.0)
    t = np.asarray(t, float)
    return t[0] + (t - t[0]) * factor, float(factor)


def ease_in_out(t, q, ease_s: float, rate_hz: float):
    """Play ``(t, q)`` with a playback rate that rises 0 -> 1 over ``ease_s`` (minimum-jerk) and
    falls back at the end: same path, starts and ends at rest. The result lasts ``ease_s``
    longer and is sampled evenly at ``rate_hz``. A trajectory shorter than ``ease_s`` gets
    a shorter ramp."""
    t, q = np.asarray(t, float), np.asarray(q, float)
    dur = float(t[-1] - t[0])
    ramp = min(ease_s, dur)
    if ramp <= 0.0:
        return t.copy(), q.copy()
    total = dur + ramp
    n = max(2, int(np.ceil(total * rate_hz)) + 1)
    T = np.linspace(0.0, total, n)
    rate = np.minimum(min_jerk(T / ramp), min_jerk((total - T) / ramp))
    u = np.r_[0.0, np.cumsum(0.5 * (rate[1:] + rate[:-1]) * np.diff(T))]
    u *= dur / u[-1]  # numerical: land exactly on the end
    return T, sample(t - t[0], q, u)


def point_to_point(q0, q1, speed: float, rate_hz: float, min_duration_s: float = 0.6):
    """Synchronised rest-to-rest move (every joint starts and arrives together) with a
    minimum-jerk profile whose peak joint speed is ``speed``. Returns (t, q), q[0] = q0."""
    q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
    span = float(np.max(np.abs(q1 - q0)))
    dur = max(min_duration_s, span / speed * MIN_JERK_PEAK if speed > 0 else min_duration_s)
    n = max(2, int(np.ceil(dur * rate_hz)) + 1)
    t = np.linspace(0.0, dur, n)
    return t, q0 + min_jerk(t / dur)[:, None] * (q1 - q0)
