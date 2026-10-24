#!/usr/bin/env python3
"""EXPERIMENTAL (not wired into anything): bowl centre / rim height from ONE gripper-camera frame.

Idea (docs/plans/BOWL_VISION_PLAN.md): a bowl's rim is a horizontal circle of known radius.
Back-project rim edge pixels through the camera pose (SO-101 FK + the webcam-2 frame from the
Viam config) onto a horizontal plane z = h. Only at the TRUE rim height do the points form a
circle of the right radius; at a wrong h they form an ellipse stretched along the view. So we
scan h and fit a circle at each, and keep the h whose circle is roundest with a radius near
the prior. Output: rim centre (x, y), rim height h, radius r, all in the arm base frame.

    # 1) synthetic error budget at the proposed survey pose (no camera, no arm, numpy only)
    .venv/bin/python scripts/bowl_rim_fit_experimental.py --selftest

    # 2) a captured frame + the joints it was taken at (+ optional bowl box from YOLO/OWL)
    .venv/bin/python scripts/bowl_rim_fit_experimental.py --image frame.jpg \
        --joints -51 -92 25 93 0 [--box x0 y0 x1 y1] [--radius-mm 75] [--debug out.jpg]

Mode 2 needs opencv (already in the venv) and has NOT been validated on a real survey frame
yet (none exists: the arm was never sent to a look-at-bowls pose). On a synthetic render with
line clutter it recovered the centre to < 1 mm WITH a bowl box, and locked onto clutter
WITHOUT one: always pass --box (YOLO "bowl" / OWLv2 / a SAM mask's box). Read-only: never
talks to the robot.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from assistive_arm.robot_iface.so101_kinematics import fk  # noqa: E402

# webcam-2 in the Viam config: parent follower-arm, translation (-21, -8, 90) mm, -90 deg about z.
CAM_T_MM = (-21.0, -8.0, 90.0)
CAM_RZ_DEG = -90.0
K = np.array([[772.5, 0, 320.0], [0, 772.5, 240.0], [0, 0, 1]])  # C270_640x480 (uncalibrated)
W, H = 640, 480
OVERLAY_ROWS = 455  # the dashboard stream draws a status bar below this row: ignore it

SURVEY_Q = np.array([-51.0, -92.0, 25.0, 93.0, 0.0])  # a reachable look-down pose for Bowl 1 (FK search)
BOWL1 = np.array([0.2296, 0.2165])  # scene centre of the calibrated Bowl 1 (m)
RIM_Z = 0.094  # m, Bowl 1 rim height from the spoon-tip calibration (base frame)
RIM_R = 0.075  # m, outer rim radius of a ~15 cm paper bowl


def _rz(deg: float) -> np.ndarray:
    t = np.radians(deg)
    M = np.eye(4)
    M[:2, :2] = [[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]]
    return M


def camera_pose(q_deg, d_rot_deg=(0.0, 0.0, 0.0), d_t_m=(0.0, 0.0, 0.0)) -> np.ndarray:
    """Base <- camera (OpenCV optical axes), optionally perturbed (for the error budget)."""
    T = np.eye(4)
    T[:3, 3] = np.asarray(CAM_T_MM) / 1000.0 + np.asarray(d_t_m)
    T = fk(np.asarray(q_deg, float)) @ T @ _rz(CAM_RZ_DEG)
    if any(d_rot_deg):
        from math import cos, radians, sin
        rx, ry, rz = (radians(a) for a in d_rot_deg)
        Rx = np.array([[1, 0, 0], [0, cos(rx), -sin(rx)], [0, sin(rx), cos(rx)]])
        Ry = np.array([[cos(ry), 0, sin(ry)], [0, 1, 0], [-sin(ry), 0, cos(ry)]])
        Rz = np.array([[cos(rz), -sin(rz), 0], [sin(rz), cos(rz), 0], [0, 0, 1]])
        T[:3, :3] = T[:3, :3] @ Rz @ Ry @ Rx
    return T


def project(T: np.ndarray, P: np.ndarray) -> np.ndarray:
    Pc = (np.linalg.inv(T) @ np.c_[P, np.ones(len(P))].T)[:3].T
    uv = (K @ Pc.T).T
    out = uv[:, :2] / uv[:, 2:3]
    out[Pc[:, 2] <= 0] = np.nan
    return out


def backproject(T: np.ndarray, uv: np.ndarray, z: float) -> np.ndarray:
    """Pixels -> points on the horizontal plane z (base frame); NaN where the ray misses it."""
    rays = np.c_[(uv[:, 0] - K[0, 2]) / K[0, 0], (uv[:, 1] - K[1, 2]) / K[1, 1], np.ones(len(uv))]
    d = rays @ T[:3, :3].T
    o = T[:3, 3]
    with np.errstate(divide="ignore", invalid="ignore"):
        s = (z - o[2]) / d[:, 2]
    P = o + s[:, None] * d
    P[s <= 0] = np.nan
    return P[:, :2]


def circle_fit(xy: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Kasa algebraic fit: (centre, radius, rms radial residual)."""
    A = np.c_[2 * xy, np.ones(len(xy))]
    b = (xy ** 2).sum(1)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    c = sol[:2]
    r = float(np.sqrt(max(sol[2] + c @ c, 1e-12)))
    res = np.linalg.norm(xy - c, axis=1) - r
    return c, r, float(np.sqrt(np.mean(res ** 2)))


def fit_rim(T: np.ndarray, uv: np.ndarray, r_prior: float = RIM_R, z_range=(0.03, 0.16),
            z_fixed: float | None = None, r_weight: float = 0.3) -> dict:
    """Rim centre/height from rim pixels. ``z_fixed``: trust a known rim height instead of scanning."""
    def score(z):
        xy = backproject(T, uv, z)
        xy = xy[np.all(np.isfinite(xy), axis=1)]
        if len(xy) < 8:
            return np.inf, None
        c, r, rms = circle_fit(xy)
        return rms + r_weight * abs(r - r_prior), (c, r, rms)

    if z_fixed is not None:
        _, (c, r, rms) = score(z_fixed)
        return {"centre": c, "z": z_fixed, "r": r, "rms": rms}
    zs = np.arange(z_range[0], z_range[1], 0.002)
    costs = [score(z)[0] for z in zs]
    z0 = zs[int(np.argmin(costs))]
    zs = np.arange(z0 - 0.002, z0 + 0.002, 0.0002)  # refine
    best = min(zs, key=lambda z: score(z)[0])
    _, (c, r, rms) = score(best)
    return {"centre": c, "z": float(best), "r": r, "rms": rms}


def rim_pixels(T, centre_xy, z, r, n=180) -> np.ndarray:
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    P = np.c_[centre_xy[0] + r * np.cos(a), centre_xy[1] + r * np.sin(a), np.full(n, z)]
    uv = project(T, P)
    ok = np.all(np.isfinite(uv), 1) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < OVERLAY_ROWS)
    return uv[ok]


def selftest(trials: int = 300, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    T0 = camera_pose(SURVEY_Q)
    uv = rim_pixels(T0, BOWL1, RIM_Z, RIM_R)
    rng_m = np.linalg.norm(T0[:3, 3] - np.r_[BOWL1, RIM_Z])
    print(f"survey pose {SURVEY_Q.tolist()}: camera {np.round(T0[:3, 3] * 1000)} mm, looking "
          f"{np.degrees(np.arcsin(-T0[2, 2])):.0f} deg below horizontal, {rng_m * 100:.0f} cm from the rim centre; "
          f"{len(uv)}/180 rim samples in the image")
    print("centre error (mm), median / 95th percentile over", trials, "trials")
    cases = {
        "2 px edge noise only": dict(px=2.0),
        "+ camera mount 1 deg / 3 mm": dict(px=2.0, rot=1.0, t=0.003),
        "+ mount 2 deg / 5 mm": dict(px=2.0, rot=2.0, t=0.005),
        "+ mount 1 deg, then reference-bowl bias removed": dict(px=2.0, rot=1.0, t=0.003, debias=True),
    }
    for name, c in cases.items():
        errs = {"known rim z": [], "rim z 5 mm off": [], "rim z fitted": []}
        for _ in range(trials):
            dr = rng.normal(0, c.get("rot", 0) / np.sqrt(3), 3)
            dt = rng.normal(0, c.get("t", 0) / np.sqrt(3), 3)
            T_true = camera_pose(SURVEY_Q, dr, dt)  # where the camera really is
            shift = rng.uniform(-0.03, 0.03, 2)  # the bowl was moved up to 3 cm
            truth = BOWL1 + shift
            meas = rim_pixels(T_true, truth, RIM_Z, RIM_R)
            meas = meas + rng.normal(0, c["px"], meas.shape)
            ref = None
            if c.get("debias"):  # one reference view of the bowl at its spoon-calibrated position
                ref = rim_pixels(T_true, BOWL1, RIM_Z, RIM_R)
                ref = ref + rng.normal(0, c["px"], ref.shape)
            for k, kw in (("known rim z", {"z_fixed": RIM_Z}), ("rim z 5 mm off", {"z_fixed": RIM_Z + 0.005}),
                          ("rim z fitted", {})):
                bias = fit_rim(T0, ref, **kw)["centre"] - BOWL1 if ref is not None else 0.0
                est = fit_rim(T0, meas, **kw)["centre"] - bias
                errs[k].append(np.linalg.norm(est - truth) * 1000)
        print(f"  {name:48s} " + "  ".join(f"{k}: {np.median(v):4.1f}/{np.percentile(v, 95):4.1f}" for k, v in errs.items()))


def from_image(args) -> None:
    import cv2

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"can't read {args.image}")
    T = camera_pose(args.joints)
    if T[2, 3] - args.rim_z_mm / 1000 < 0.05 or -T[2, 2] < 0.2:
        raise SystemExit(f"not a survey view: camera {T[2, 3] * 1000:.0f} mm high (rim {args.rim_z_mm:.0f}), "
                         f"looking {np.degrees(np.arcsin(-T[2, 2])):.0f} deg down; need >= 50 mm above the rim, >= 12 deg down")
    gray = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    edges = cv2.Canny(gray, 40, 120)
    edges[OVERLAY_ROWS:] = 0
    if args.box:
        x0, y0, x1, y1 = (int(v) for v in args.box)
        m = np.zeros_like(edges)
        pad = 20
        m[max(0, y0 - pad):y1 + pad, max(0, x0 - pad):x1 + pad] = 1
        edges *= m
    v, u = np.nonzero(edges)
    uv = np.c_[u, v].astype(float)
    r0 = args.radius_mm / 1000
    # RANSAC on the plane z = prior: 3 edge points -> circle; inliers within 6 mm of it, radius near the prior
    rng = np.random.default_rng(0)
    z0 = args.rim_z_mm / 1000
    xy = backproject(T, uv, z0)
    # inlier band = 2 px on the image, whatever metres that is at each point (far points are coarser)
    px = np.maximum(np.linalg.norm(backproject(T, uv + [1, 0], z0) - xy, axis=1),
                    np.linalg.norm(backproject(T, uv + [0, 1], z0) - xy, axis=1))
    ok = np.all(np.isfinite(xy), 1) & np.isfinite(px)
    uv, xy, tol = uv[ok], xy[ok], np.clip(2.0 * px[ok], 0.0005, 0.008)
    best = (0, None)
    for _ in range(4000):
        s = xy[rng.choice(len(xy), 3, replace=False)]
        c, r, _ = circle_fit(s)
        if abs(r - r0) > 0.015:
            continue
        inl = np.abs(np.linalg.norm(xy - c, axis=1) - r) < tol
        if inl.sum() > best[0]:
            best = (int(inl.sum()), inl)
    if best[1] is None:
        raise SystemExit("no rim-like circle found")
    rim_uv = uv[best[1]]
    fz = fit_rim(T, rim_uv, r0)
    fk_ = fit_rim(T, rim_uv, r0, z_fixed=args.rim_z_mm / 1000)
    for name, f in (("rim z fitted", fz), ("rim z prior", fk_)):
        bad = abs(f["r"] - r0) > 0.015 or f["rms"] > 0.004
        print(f"{'REJECTED ' if bad else ''}{name}: centre ({f['centre'][0] * 1000:.1f}, {f['centre'][1] * 1000:.1f}) mm, "
              f"rim z {f['z'] * 1000:.1f} mm, r {f['r'] * 1000:.1f} mm, rms {f['rms'] * 1000:.1f} mm, "
              f"{len(rim_uv)} rim pixels")
    if args.debug:
        out = img.copy()
        for p in rim_uv.astype(int):
            cv2.circle(out, tuple(p), 1, (0, 255, 0), -1)
        a = np.linspace(0, 2 * np.pi, 120)
        P = np.c_[fz["centre"][0] + fz["r"] * np.cos(a), fz["centre"][1] + fz["r"] * np.sin(a), np.full(120, fz["z"])]
        pts = project(T, P)
        pts = pts[np.all(np.isfinite(pts), 1)].astype(np.int32)
        cv2.polylines(out, [pts], True, (0, 0, 255), 1)
        cv2.imwrite(args.debug, out)
        print("wrote", args.debug)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--trials", type=int, default=300)
    ap.add_argument("--image")
    ap.add_argument("--joints", type=float, nargs=5, default=SURVEY_Q.tolist())
    ap.add_argument("--box", type=float, nargs=4)
    ap.add_argument("--radius-mm", type=float, default=RIM_R * 1000)
    ap.add_argument("--rim-z-mm", type=float, default=RIM_Z * 1000)
    ap.add_argument("--debug")
    a = ap.parse_args()
    if a.image:
        from_image(a)
    else:
        selftest(a.trials)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
