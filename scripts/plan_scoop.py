#!/usr/bin/env python3
"""Offline scoop planner for the SO-101 feeding arm (no robot, no network).

    .venv/bin/python scripts/plan_scoop.py                 # keyframes, checks, stream, timing
    .venv/bin/python scripts/plan_scoop.py --food-mm 25    # assumed food depth above the inner floor
    .venv/bin/python scripts/plan_scoop.py --json          # hand-off poses only (pre_scoop, carry)
    .venv/bin/python scripts/plan_scoop.py --radius-sweep  # which bowl distances the scoop fits

Plan: docs/plans/SCOOP_PLAN.md. Companion: scripts/plan_home_transit.py (HOME, OVER_*, PRE_SCOOP_*).

Geometry (mm, base frame = world, z = 0 at the table top):
  bowls    PLY /home/ultraviolet/bowl-150x60.ply (read 2026-10-23): an open frustum, origin at the
           bottom centre, r 50 mm at z 0, r 75 mm at z 60 (wall leans out 22.6 deg). Modelled as
           a 4 mm shell (inner floor z 4, inner wall 4.3 mm inside the mesh): ASSUMED thickness.
  spoon    tip 133 mm along tool +z (the Viam "spoon" box: 143 mm centred at 61.5 mm)
  bodies   (tool frame; +z along the spoon, +x down at the spoon-level roll)
             tip      the point at z 133: >= 8 mm off the inner floor and walls
             blade    z 100..128, 25 mm wide (tool y), 12 mm thick: >= 3 mm off walls/rim/floor
             jaws     z 60..100, radius 15 (jaw tips at 99.9 mm, docs/SO101_SETUP.md): >= 5 mm, above food
             gripper  z -10..60, radius 25 (the 50 x 45 mm Viam box): >= 5 mm, above food
             camera   C270 corners x -36..-6, y -45..29, z 62..92: >= 5 mm, above food
             links    upper arm, forearm, wrist as capsules r 30: >= 5 mm off both bowls, >= 10 mm off the table
  food     a flat surface FOOD_MM above the inner floor; only the tip and blade may go below it
Pan: + turns the arm toward -y (the person); the bowls (+y) need NEGATIVE pan. The spoon's
vertical plane contains the pan axis (x = 38.84, y = 0) to within 0.2 mm, so the tip can only
move radially (lift/elbow/flex) or on an arc about that axis (pan): no sideways scoop.

PLANNING TOOL ONLY: it never imports viam and never talks to the arm.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from assistive_arm.robot_iface.so101_kinematics import fk, joint_limits, load  # noqa: E402

# ---------------------------------------------------------------- scene and calibration
LEVEL_PITCH = -4.83  # tool pitch (deg) with the spoon level (poses/so101_face.json spoon_level)
ROLL = -2.81  # wrist roll of that calibration: held through the whole scoop
PAN_AXIS = np.array([38.8353, 0.0])
TIP = 133.0
BOWLS = {"bowl-right": np.array([316.2, 173.8]), "bowl-left": np.array([316.2, 323.8])}
# Same distance from the pan axis as bowl-right, 31.5 deg further round (no overlap, 30 mm gap).
SUGGESTED_LEFT = np.array([184.2, 292.9])
R_BOT, R_RIM, H = 50.0, 75.0, 60.0
WALL, FLOOR = 4.0, 4.0
W_IN = WALL / np.cos(np.arctan((R_RIM - R_BOT) / H))  # horizontal inset of the inner wall
SHELL = np.array([(0.0, 0.0), (R_BOT, 0.0), (R_RIM, H), (R_RIM - W_IN, H),
                  (R_BOT - W_IN + (R_RIM - R_BOT) * FLOOR / H, FLOOR), (0.0, FLOOR)])
LIMIT_MARGIN = 5.0
TABLE_MIN = 10.0
TIP_MIN = 8.0
CARRY = (-72.5, 32.5)  # lift, elbow of the transit plan's carry/OVER posture (spoon level)
PRE_TIP_Z = 140.0  # transit plan's PRE_SCOOP: spoon tip over the bowl centre at this height, level
RATE_HZ = 30.0

LIM = joint_limits()

# Planar geometry of the lift/elbow/flex chain (from the SVA file; checked against fk below).
S = np.array([30.3992, 116.6])  # shoulder-lift axis: (radius from the pan axis, z)
L1, L2 = np.hypot(28.0, 112.57), np.hypot(134.9, 5.2)
PSI0 = np.degrees(np.arctan2(112.57, 28.0))
DELTA0 = np.degrees(np.arctan2(5.2, 134.9)) - PSI0
WRIST = 61.1  # flex axis -> tool origin, along the tool axis


# ---------------------------------------------------------------- kinematics
def _rot(axis, deg):
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    t = np.radians(deg)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * K @ K


def joint_points(q):
    """Positions (mm) of the shoulder-lift, elbow, flex, roll axes and the tool frame (4x4)."""
    T, pts, i = np.eye(4), [], 0
    for c in load():
        if c[0] == "fixed":
            T = T @ c[1]
        else:
            pts.append(T[:3, 3].copy())
            M = np.eye(4)
            M[:3, :3] = _rot(c[1], float(q[i]))
            T = T @ M
            i += 1
    return pts[1:], T  # drop the pan axis point


def spoon_pitch(q) -> float:
    """Spoon pitch from level (deg, + = tip up)."""
    return float(np.degrees(np.arcsin(np.clip(fk(q)[2, 2], -1, 1)))) - LEVEL_PITCH


def tip_of(q):
    T = fk(q)
    return T[:3, 3] * 1000 + TIP * T[:3, 2]


def heading(xy) -> float:
    d = np.asarray(xy, float) - PAN_AXIS
    return float(-np.degrees(np.arctan2(d[1], d[0])))


def ik(pan, rho, z, pitch, elbow_up=True):
    """Joints with the spoon tip at radius ``rho`` from the pan axis, height ``z`` (mm), spoon
    ``pitch`` deg from level, in the plane of ``pan``. Closed form (tool pitch = -(lift+elbow+flex))."""
    phi = np.radians(pitch + LEVEL_PITCH)
    d = np.array([np.cos(phi), np.sin(phi)])
    w = np.array([rho, z]) - (TIP + WRIST) * d - S
    c = (w @ w - L1 * L1 - L2 * L2) / (2 * L1 * L2)
    if abs(c) > 1:
        return None
    b = np.arccos(c) * (-1 if elbow_up else 1)
    t1 = np.arctan2(w[1], w[0]) - np.arctan2(L2 * np.sin(b), L1 + L2 * np.cos(b))
    q1 = PSI0 - np.degrees(t1)
    q2 = DELTA0 - np.degrees(b)
    return np.array([pan, q1, q2, -(pitch + LEVEL_PITCH) - q1 - q2, ROLL])


def limit_margin(q) -> float:
    q = np.asarray(q, float)
    return float(np.min(np.minimum(q - LIM[:, 0], LIM[:, 1] - q)))


# ---------------------------------------------------------------- bowl geometry
def _seg_dist(p, a, b):
    ab = b - a
    t = np.clip(np.dot(p - a, ab) / np.dot(ab, ab), 0, 1)
    return float(np.linalg.norm(p - (a + t * ab)))


def _inside_poly(p, poly):
    x, y, inside = p[0], p[1], False
    for i in range(len(poly)):
        (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % len(poly)]
        if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
            inside = not inside
    return inside


def shell_dist(p, c) -> float:
    """Signed distance (mm) from 3D point p to the bowl shell at centre c (< 0 inside the wall).
    Exact for a surface of revolution: the distance in the (radius, z) half-plane."""
    m = np.array([np.hypot(p[0] - c[0], p[1] - c[1]), p[2]])
    d = min(_seg_dist(m, SHELL[i], SHELL[i + 1]) for i in range(len(SHELL) - 1))
    return -d if _inside_poly(m, SHELL) else d


def in_cavity(p, c) -> bool:
    r = np.hypot(p[0] - c[0], p[1] - c[1])
    return FLOOR <= p[2] <= H and r < R_BOT - W_IN + (R_RIM - R_BOT) * p[2] / H


# ---------------------------------------------------------------- bodies and the per-pose check
CAM = np.array([[x, y, z] for x in (-36.0, -6.0) for y in (-45.0, 29.0) for z in (62.0, 92.0)] + [[-21.0, -8.0, 90.0]])


def bodies(q):
    """(name, points (n, 3) mm, radius mm, required surface clearance mm, may enter food)."""
    (s, e, w, p), T = joint_points(q)
    R, o = T[:3, :3], T[:3, 3]

    def axis(z0, z1, n):
        return np.array([o + R[:, 2] * z for z in np.linspace(z0, z1, n)])

    def seg(a, b, n=6):
        return np.array([a + (b - a) * t for t in np.linspace(0, 1, n)])

    blade = np.vstack([axis(100, 128, 5) + R[:, 1] * y for y in (-12.5, 0.0, 12.5)])  # 25 mm wide, 12 mm thick
    return [("blade", blade, 6.0, 3.0, True),
            ("jaws", axis(60, 100, 5), 15.0, 5.0, False),
            ("gripper", axis(-10, 60, 6), 25.0, 5.0, False),
            ("camera", (R @ CAM.T).T + o, 0.0, 5.0, False),
            ("links", np.vstack([seg(s, e), seg(e, w), seg(w, p)]), 30.0, 5.0, False)]


def check_pose(q, bowl, bowls, food_mm):
    """Clearances (mm) of one pose; ``bowl`` = the bowl being scooped (food lives there)."""
    q = np.asarray(q, float)
    z_food = FLOOR + food_mm
    tip = tip_of(q)
    out = {"margin": limit_margin(q), "pitch": spoon_pitch(q), "tip": tip}
    inside = in_cavity(tip, bowl)
    out["tip_floor"] = tip[2] - FLOOR if inside else np.inf
    out["tip_shell"] = min(shell_dist(tip, c) for c in bowls)
    out["table"] = tip[2] - TABLE_MIN if all(np.hypot(*(tip[:2] - c)) > R_RIM for c in bowls) else np.inf
    out["food"] = np.inf  # lowest non-spoon body surface above the food (mm), inside the cavity
    out["dry_depth"] = np.inf  # deepest food (mm above the inner floor) this pose keeps off the jaws/camera
    worst, who = np.inf, ""
    for name, pts, rad, need, wet in bodies(q):
        for pt in pts:
            for c in bowls:
                d = shell_dist(pt, c) - rad - need
                if d < worst:
                    worst, who = d, name
            if all(np.hypot(*(pt[:2] - c)) > R_RIM for c in bowls):  # over the bare table
                out["table"] = min(out["table"], pt[2] - rad - TABLE_MIN)
            if not wet and np.hypot(*(pt[:2] - bowl)) < R_RIM and pt[2] - rad < H:
                low = pt[2] - rad
                out["food"] = min(out["food"], low - z_food - 5.0)
                out["dry_depth"] = min(out["dry_depth"], low - 5.0 - FLOOR)
    out["shell"], out["shell_body"] = worst, who
    out["ok"] = bool(out["margin"] >= LIMIT_MARGIN and out["tip_floor"] >= TIP_MIN and out["tip_shell"] >= TIP_MIN
                     and out["table"] >= 0 and out["shell"] >= 0 and out["food"] >= 0)
    return out


# ---------------------------------------------------------------- the scoop template
# (name, u = tip offset from the bowl centre along the pan direction (+ = away from the base),
#  tip z (mm), spoon pitch from level (deg, - = tip down), phase). The tip runs on the radial line
# through the bowl centre. Far-wall push scoop: enter tip-down just inside the near half, push
# out along the floor while flattening, ride up the far wall curling, then lift (level from the
# rim plane up) and retreat. The jaws and gripper must clear the NEAR rim, which is what limits
# how flat the spoon can be inside the bowl (see the plan).
TEMPLATE = [
    ("pre_scoop", 0, PRE_TIP_Z, 0, "approach"),  # = PRE_SCOOP_* of the transit plan (tip over the centre)
    ("hover", -25, 80, -58, "approach"),  # tilted tip-down, tip 20 mm above the rim plane
    ("plunge", -15, 16, -62, "plunge"),  # in along (roughly) the spoon axis, tip 12 mm off the floor
    ("drag1", 5, 14, -62, "scoop"),  # push out along the floor, still steep (jaws stay out of the food)
    ("drag2", 25, 15, -57, "scoop"),
    ("wall", 38, 24, -45, "scoop"),  # tip reaches the far wall and starts up it
    ("curl", 43, 40, -28, "scoop"),  # curl up the far wall: food trapped against the wall
    ("rise", 44, 64, -12, "lift"),  # leaving the food; flattening as the near rim allows
    ("rim", 36, 92, 0, "lift"),  # level from here (tip 32 mm above the rim plane)
    ("lift_clear", 0, PRE_TIP_Z, 0, "lift"),  # = pre_scoop again (transit plan's scoop exit), level
    ("carry", None, None, None, "retreat"),  # = OVER_k = READY_k of the transit plan (carry posture)
]


def build(bowl_name, bowl, food_mm):
    pan = heading(bowl)
    rb = float(np.linalg.norm(bowl - PAN_AXIS))
    rows = []
    for name, u, z, p, phase in TEMPLATE:
        if name == "carry":
            q = np.array([pan, CARRY[0], CARRY[1], -LEVEL_PITCH - CARRY[0] - CARRY[1], ROLL])
        else:
            q = ik(pan, rb + u, z, p)
            if q is None or limit_margin(q) < 0:
                q2 = ik(pan, rb + u, z, p, elbow_up=False)
                q = q2 if q2 is not None else q
        rows.append({"name": name, "phase": phase, "u": u, "z": z, "pitch": p, "q": q})
    return pan, rb, rows


def joint_line(qa, qb, n=25):
    return [qa + (qb - qa) * t for t in np.linspace(0, 1, n)]


# ---------------------------------------------------------------- timing (streamed setpoints)
SPEED = {"approach": 40.0, "plunge": 20.0, "scoop": 15.0, "lift": 25.0, "retreat": 40.0}  # mm/s spoon tip
PITCH_RATE = 20.0  # deg/s spoon pitch
JOINT_VEL = np.array([20.0, 20.0, 20.0, 25.0, 10.0])  # deg/s caps (pan, lift, elbow, flex, roll)
JOINT_ACC = 80.0  # deg/s^2 (TrackConfig.max_acc_deg)
SPACE = {"approach": "joint", "plunge": "task", "scoop": "task", "lift": "task", "retreat": "joint"}


def seg_time(a, b):
    ta, tb = tip_of(a["q"]), tip_of(b["q"])
    return max(np.linalg.norm(tb - ta) / SPEED[b["phase"]], abs(spoon_pitch(b["q"]) - spoon_pitch(a["q"])) / PITCH_RATE,
               float(np.max(np.abs(b["q"] - a["q"]) / JOINT_VEL)))


def _spline(P, s, knots):
    """Catmull-Rom through the rows of P at parameter values ``knots`` (0..1), evaluated at s."""
    j = min(int(np.searchsorted(knots, s, side="right")) - 1, len(P) - 2)
    h = knots[j + 1] - knots[j]
    x = (s - knots[j]) / h
    m = []
    for k in (j, j + 1):  # tangents (zero at the phase ends: the phase starts and ends at rest anyway)
        if k == 0 or k == len(P) - 1:
            m.append(np.zeros(P.shape[1]))
        else:
            m.append((P[k + 1] - P[k - 1]) / (knots[k + 1] - knots[k - 1]) * h)
    h00, h10, h01, h11 = 2 * x**3 - 3 * x**2 + 1, x**3 - 2 * x**2 + x, -2 * x**3 + 3 * x**2, x**3 - x**2
    return h00 * P[j] + h10 * m[0] + h01 * P[j + 1] + h11 * m[1]


def stream(rows, bowl, bowls, food_mm, pan, rb):
    """Each phase = one move that starts and ends at rest: a Catmull-Rom spline through its
    keyframes (in (u, z, pitch) with IK per sample for the in-bowl phases; in joints for the
    approach and the retreat), timed with a min-jerk profile, then slowed until every joint is
    under JOINT_VEL and JOINT_ACC. Sampled at RATE_HZ; every sample is checked."""
    phases, out = [], []
    for i in range(1, len(rows)):
        if phases and phases[-1][0] == rows[i]["phase"]:
            phases[-1][1].append(i)
        else:
            phases.append((rows[i]["phase"], [i]))
    t0 = 0.0
    for phase, idx in phases:
        ks = [idx[0] - 1] + idx
        times = np.array([seg_time(rows[i - 1], rows[i]) for i in idx])
        knots = np.r_[0, np.cumsum(times)] / times.sum()
        if SPACE[phase] == "joint":
            P = np.array([rows[k]["q"] for k in ks])
            to_q = lambda x: x  # noqa: E731
        else:
            P = np.array([[rows[k]["u"], rows[k]["z"], rows[k]["pitch"]] for k in ks], float)
            to_q = lambda x: ik(pan, rb + x[0], x[1], x[2])  # noqa: E731
        T = 1.875 * times.sum()
        for _ in range(4):
            n = max(3, int(np.ceil(T * RATE_HZ)))
            qs = []
            for k in range(n + 1):
                x = k / n
                qs.append(to_q(_spline(P, 10 * x**3 - 15 * x**4 + 6 * x**5, knots)))
            qs = np.array(qs)
            dt = T / n
            v = np.diff(qs, axis=0) / dt
            acc = np.diff(v, axis=0) / dt
            k = max(1.0, float(np.max(np.abs(v) / JOINT_VEL)), float(np.sqrt(np.max(np.abs(acc)) / JOINT_ACC)))
            if k <= 1.001:
                break
            T *= k * 1.02
        checks = [check_pose(q, bowl, bowls, food_mm) for q in qs]
        out.append({"phase": phase, "t0": t0, "T": T, "n": len(qs), "vmax": np.max(np.abs(v), axis=0),
                    "amax": float(np.max(np.abs(acc))), "checks": checks, "q": qs,
                    "tip_v": float(np.max(np.linalg.norm(np.diff([tip_of(q) for q in qs], axis=0), axis=1)) / dt)})
        t0 += T
    return out


# ---------------------------------------------------------------- report
def fmt_q(q):
    return "[" + ", ".join(f"{v:7.2f}" for v in q) + "]"


def worst(checks):
    keys = ("margin", "tip_floor", "tip_shell", "table", "shell", "food", "dry_depth")
    w = {k: min(c[k] for c in checks) for k in keys}
    w["ok"] = all(c["ok"] for c in checks)
    w["shell_body"] = min(checks, key=lambda c: c["shell"])["shell_body"]
    return w


def f(v):
    return "  inf" if not np.isfinite(v) else f"{v:5.1f}"


def report(bowl_name, bowl, bowls, food_mm):
    pan, rb, rows = build(bowl_name, bowl, food_mm)
    print(f"\n=== {bowl_name} at ({bowl[0]:.1f}, {bowl[1]:.1f}) mm: pan {pan:+.2f} deg, {rb:.1f} mm from the pan axis, "
          f"food {food_mm:.0f} mm deep ===")
    if any(r["q"] is None for r in rows):
        print("  UNREACHABLE keyframes: " + ", ".join(r["name"] for r in rows if r["q"] is None))
        return {"bowl": bowl_name, "ok": False, "rows": rows, "pan": pan}
    print(f"{'keyframe':10} {'joints (pan, lift, elbow, flex, roll) deg':46} {'tip xyz mm':20} {'u':>4} {'pitch':>6} "
          f"{'marg':>5} {'tipFl':>5} {'tipWl':>5} {'shell':>5} {'(body)':8} {'food':>5} {'table':>5} ok")
    for r in rows:
        c = check_pose(r["q"], bowl, bowls, food_mm)
        r["check"] = c
        t = c["tip"]
        u = "" if r["u"] is None else f"{r['u']:+4.0f}"
        print(f"{r['name']:10} {fmt_q(r['q']):46} ({t[0]:5.0f},{t[1]:5.0f},{t[2]:5.0f}) {u:>4} {c['pitch']:+6.1f} "
              f"{c['margin']:5.1f} {f(c['tip_floor'])} {f(c['tip_shell'])} {f(c['shell'])} {c['shell_body']:8} "
              f"{f(c['food'])} {f(c['table'])} {'OK' if c['ok'] else 'FAIL'}")
    print("  (mm unless noted; marg = deg inside the joint limits, need >= 5; tipFl = tip above the inner floor, need >= 8;"
          " tipWl = tip to the nearest shell; shell = worst body surface clearance beyond its required margin, need >= 0;"
          " food = jaws/gripper/camera above the food surface + 5 mm, need >= 0; table: need >= 0)")
    print("\n  joint-space straight lines between keyframes (25 samples each):")
    allok = True
    for a, b in zip(rows, rows[1:]):
        w = worst([check_pose(q, bowl, bowls, food_mm) for q in joint_line(a["q"], b["q"])])
        dev = max(np.linalg.norm(tip_of(q) - (tip_of(a["q"]) * (1 - t) + tip_of(b["q"]) * t))
                  for q, t in zip(joint_line(a["q"], b["q"]), np.linspace(0, 1, 25)))
        allok &= w["ok"]
        print(f"  {a['name']:>10} -> {b['name']:10} marg {w['margin']:5.1f} tipFl {f(w['tip_floor'])} "
              f"shell {f(w['shell'])} ({w['shell_body']:7}) food {f(w['food'])} table {f(w['table'])} "
              f"tip off the straight line {dev:4.1f} mm  {'OK' if w['ok'] else 'FAIL'}")
    print(f"\n  streamed plan ({RATE_HZ:.0f} Hz, min-jerk per phase, IK per sample):")
    st = stream(rows, bowl, bowls, food_mm, pan, rb)
    for p in st:
        w = worst(p["checks"])
        allok &= w["ok"]
        print(f"  {p['phase']:9} t {p['t0']:5.1f}-{p['t0'] + p['T']:5.1f} s ({p['n']:3d} setpoints)  max deg/s "
              f"{np.round(p['vmax'], 1).tolist()} acc {p['amax']:3.0f} tip {p['tip_v']:3.0f} mm/s | marg {w['margin']:4.1f} "
              f"tipFl {f(w['tip_floor'])} shell {f(w['shell'])} food {f(w['food'])} table {f(w['table'])} "
              f"{'OK' if w['ok'] else 'FAIL'}")
    wet = [c for p in st if p["phase"] in ("plunge", "scoop") for c in p["checks"]]
    dry = min(c["dry_depth"] for p in st for c in p["checks"])
    names = [r["name"] for r in rows]
    loaded = [spoon_pitch(q) for p in st for q in p["q"]][sum(p["n"] for p in st if p["phase"] in ("approach", "plunge")):]
    print(f"  total {st[-1]['t0'] + st[-1]['T']:.1f} s; deepest food that stays off the jaws/gripper/camera: {dry:.0f} mm; "
          f"spoon pitch from 'curl' on: {spoon_pitch(rows[names.index('curl')]['q']):+.0f} deg, level (+-1) from "
          f"'rim' (tip z {rows[names.index('rim')]['z']:.0f}) on")
    # sag sensitivity at the bottom of the scoop
    qb = rows[[r["name"] for r in rows].index("drag1")]["q"]
    sens = []
    for j in (1, 2, 3):
        d = np.zeros(5)
        d[j] = 1.0
        sens.append(tip_of(qb + d) - tip_of(qb))
    print("  sag sensitivity at drag1 (tip mm per +1 deg): " + ", ".join(
        f"{n} dz {s[2]:+.1f} / radial {np.dot(s[:2], (bowl - PAN_AXIS) / rb):+.1f}" for n, s in zip(("lift", "elbow", "flex"), sens)))
    use = []
    for r in rows:
        if r["u"] is not None:
            phi = np.radians(r["pitch"] + LEVEL_PITCH)
            w = np.array([rb + r["u"], r["z"]]) - (TIP + WRIST) * np.array([np.cos(phi), np.sin(phi)]) - S
            use.append((float(np.linalg.norm(w)), r["name"]))
    m = max(use)
    print(f"  reach: worst keyframe '{m[1]}' needs shoulder->flex {m[0]:.0f} of {L1 + L2:.0f} mm "
          f"({L1 + L2 - m[0]:.0f} mm spare); radial band where the template passes: --radius-sweep")
    return {"bowl": bowl_name, "ok": allok and all(r["check"]["ok"] for r in rows), "rows": rows, "pan": pan,
            "time": st[-1]["t0"] + st[-1]["T"], "dry": dry}


def radius_sweep(food_mm):
    print("\nTemplate fit vs bowl distance from the pan axis (pan of bowl-right, other bowl ignored):")
    for r in range(240, 441, 5):
        b = PAN_AXIS + r * np.array([np.cos(np.radians(32.07)), np.sin(np.radians(32.07))])
        pan, rb, rows = build("x", b, food_mm)
        if any(x["q"] is None for x in rows):
            print(f"  r {r}: unreachable")
            continue
        qs = []
        for a, c in zip(rows, rows[1:]):
            qs += joint_line(a["q"], c["q"], 12)
        w = worst([check_pose(q, b, [b], food_mm) for q in qs])
        print(f"  r {r}: marg {w['margin']:5.1f} tipFl {f(w['tip_floor'])} shell {f(w['shell'])} ({w['shell_body']:7}) "
              f"food {f(w['food'])} {'OK' if w['ok'] else 'FAIL'}")


def tolerance(bowl_name, food_mm):
    """Plan for the CONFIGURED bowl pose, check against the bowl displaced (radial/tangential)."""
    b0 = BOWLS[bowl_name]
    pan, rb, rows = build(bowl_name, b0, food_mm)
    rad = (b0 - PAN_AXIS) / rb
    tan = np.array([-rad[1], rad[0]])
    qs = []
    for a, c in zip(rows, rows[1:]):
        qs += joint_line(a["q"], c["q"], 12)
    print(f"\nBowl pose error tolerance for {bowl_name} (plan for the configured pose, bowl actually shifted):")
    for name, d in (("radial", rad), ("tangential", tan)):
        row = []
        for e in (-20, -10, -5, 5, 10, 20):
            b = b0 + e * d
            w = worst([check_pose(q, b, [b], food_mm) for q in qs])
            row.append(f"{e:+d} mm {'OK' if w['ok'] else 'FAIL'} (tipFl {f(w['tip_floor']).strip()}, shell {w['shell']:.1f} {w['shell_body']})")
        print(f"  {name:10}: " + "; ".join(row))


def handoff(results):
    out = {}
    for r in results:
        rows = {x["name"]: x for x in r["rows"]}
        if rows["pre_scoop"]["q"] is None:
            continue
        out[r["bowl"]] = {"pan_deg": round(r["pan"], 2),
                          "pre_scoop_tip_mm": [round(float(v), 1) for v in tip_of(rows["pre_scoop"]["q"])],
                          "carry_tip_mm": [round(float(v), 1) for v in tip_of(rows["carry"]["q"])],
                          "pre_scoop": [round(float(v), 2) for v in rows["pre_scoop"]["q"]],
                          "carry": [round(float(v), 2) for v in rows["carry"]["q"]],
                          "scoop_feasible": bool(r["ok"])}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--food-mm", type=float, default=15.0, help="food depth above the inner floor (mm)")
    ap.add_argument("--json", action="store_true", help="print only the hand-off poses")
    ap.add_argument("--radius-sweep", action="store_true")
    a = ap.parse_args()
    # self-check: closed-form IK against the module's FK
    q = ik(-30.0, 300.0, 50.0, -40.0)
    tip = tip_of(q)
    want = np.r_[PAN_AXIS + 300.0 * np.array([np.cos(np.radians(30)), np.sin(np.radians(30))]), 50.0]
    assert np.linalg.norm(tip - want) < 0.5 and abs(spoon_pitch(q) + 40) < 0.01, (tip, want)
    bowls = list(BOWLS.values())
    res = []
    out = sys.stdout
    if a.json:
        sys.stdout = open("/dev/null", "w")
    for name, b in BOWLS.items():
        res.append(report(name, b, bowls, a.food_mm))
    alt = [BOWLS["bowl-right"], SUGGESTED_LEFT]
    res_alt = report("bowl-left (suggested spot)", SUGGESTED_LEFT, alt, a.food_mm)
    if a.radius_sweep:
        radius_sweep(a.food_mm)
        tolerance("bowl-right", a.food_mm)
    sys.stdout = out
    h = handoff(res + [res_alt])
    print("\nHAND-OFF POSES (joints deg: pan, lift, elbow, flex, roll)")
    print("{\n" + ",\n".join(f"  {json.dumps(k)}: {json.dumps(v)}" for k, v in h.items()) + "\n}")


if __name__ == "__main__":
    main()
