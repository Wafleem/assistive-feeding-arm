#!/usr/bin/env python3
"""Offline check of the SO-101 feeding HOME pose and transit waypoints (no robot, no network).

    .venv/bin/python scripts/plan_home_transit.py            # waypoint + segment tables
    .venv/bin/python scripts/plan_home_transit.py --search   # also re-run the carry-posture search

Plan: docs/plans/HOME_AND_TRANSIT_PLAN.md. Everything here is forward kinematics of
``robot_iface/so101_kinematics.py`` plus simple geometry from the orin-dev-kit Viam config:
  table   z = 0 plane (conservative: treated as infinite, the real top ends at y = -76 mm)
  bowls   solid cylinders r 75 mm, h 60 mm (the PLY is a frustum: r 50 mm bottom, 75 mm rim)
  spoon   capsule from the tool origin to the tip (133 mm along tool +z), radius 25 mm
  camera  sphere r 20 mm at (-21, -8, 90) mm in the tool frame, looking along tool +z
  arm     the upper_arm / lower_arm / wrist link boxes from the kinematics file
Segments are checked under two interpolation models: SYNC (straight joint-space line, all
joints arrive together) and UNSYNC (every joint at the same deg/s, each stops when it arrives).

PLANNING TOOL ONLY: it never talks to the arm. Promote the pose table into a config dataclass
(feeding/config.py or orchestration) once the team has reviewed the plan.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from assistive_arm.feeding.face_track import TrackConfig, max_level_reach, solve_level_pose  # noqa: E402
from assistive_arm.robot_iface.so101_kinematics import CHAIN, DATA, fk, joint_limits, load  # noqa: E402

# ---------------------------------------------------------------- scene (mm, base frame = world)
LEVEL_PITCH = -4.83  # tool pitch (deg) at which the spoon is level (poses/so101_face.json spoon_level)
ROLL = -2.81  # wrist roll during the spoon-level calibration: held fixed in every transit
PAN_AXIS = np.array([38.8353, 0.0])  # base link translation: the pan joint's vertical axis
SPOON_TIP = 133.0  # tool origin -> spoon tip along tool +z
SPOON_R = 25.0
CAM_OFF = np.array([-21.0, -8.0, 90.0])
CAM_R = 20.0
CAM_HALF_FOV = 28.0  # deg, assumed cone (webcam ~ 60-70 deg HFOV, ~45 deg VFOV): verify
BOWLS = {"bowl-right": np.array([316.2, 173.8]), "bowl-left": np.array([316.2, 323.8])}
BOWL_R, BOWL_H = 75.0, 60.0
LIPS = np.array([20.0, -490.0, 270.0])  # middle of the measured (0..40, -490, 270)
CLEAR_MM = 50.0  # required clearance: table and bowls (spoon capsule, camera, links)
SELF_MM = 10.0  # spoon / camera vs the arm's own base and shoulder boxes
LIMIT_MARGIN = 5.0  # deg inside the joint limits
TILT_TOL = 5.0  # deg spoon tilt allowed while loaded
LIPS_MIN_MM = 200.0  # spoon tip to lips outside the tracker (ExtendConfig.stop_tip_distance, first runs)

LIM = joint_limits()
G = 9.81


# ---------------------------------------------------------------- kinematics helpers
def _rot(axis, deg):
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    t = np.radians(deg)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * K @ K


def frames(q) -> dict:
    """Every link frame (4x4, mm) along the chain."""
    T, out, i = np.eye(4), {}, 0
    for name, c in zip(CHAIN, load()):
        if c[0] == "fixed":
            T = T @ c[1]
        else:
            M = np.eye(4)
            M[:3, :3] = _rot(c[1], float(q[i]))
            T = T @ M
            i += 1
        out[name] = T.copy()
    return out


def _link_boxes() -> dict:
    k = json.loads(Path(DATA).read_text())
    out = {}
    for l in k["links"]:
        g = l.get("geometry")
        if g and g.get("type") == "box":
            t = g.get("translation", {})
            out[l["id"]] = (np.array([t.get("x", 0), t.get("y", 0), t.get("z", 0)], float),
                            np.array([g["x"], g["y"], g["z"]], float) / 2)
    return out


BOXES = _link_boxes()
_G3 = np.array(np.meshgrid(*[[-1, 0, 1]] * 3)).reshape(3, -1).T  # 27 sample points per box


def box_points(F, name):
    c, h = BOXES[name]
    return (F[:3, :3] @ (c + _G3 * h).T).T + F[:3, 3]


def in_box_dist(p, F, name):
    """Distance (mm) from point p to the box of link ``name`` (0 inside)."""
    c, h = BOXES[name]
    local = F[:3, :3].T @ (p - F[:3, 3]) - c
    return float(np.linalg.norm(np.maximum(np.abs(local) - h, 0.0)))


def pitch(q) -> float:
    return float(np.degrees(np.arcsin(np.clip(fk(q)[2, 2], -1, 1))))


def tilt(q) -> float:
    """Spoon tilt from level (deg, + = tip up)."""
    return pitch(q) - LEVEL_PITCH


def heading(xy) -> float:
    """Pan angle (deg) that points the arm at a world xy (pan + turns toward -y)."""
    d = np.asarray(xy, float) - PAN_AXIS
    return float(-np.degrees(np.arctan2(d[1], d[0])))


def spoon_pts(q, n=12):
    T = fk(q)
    p, z = T[:3, 3] * 1000, T[:3, 2]
    return np.array([p + z * s for s in np.linspace(0, SPOON_TIP, n)]), T


def camera_pt(T):
    return T[:3, 3] * 1000 + T[:3, :3] @ CAM_OFF


def bowl_dist(p, c):
    rho = np.hypot(p[0] - c[0], p[1] - c[1])
    dr, dz = max(0.0, rho - BOWL_R), max(0.0, p[2] - BOWL_H, -p[2])
    return float(np.hypot(dr, dz))


def shoulder_torque(q) -> float:
    """Gravity torque proxy on the shoulder lift (N m, + = pulls the arm out/down). Point masses
    (kg, guesses): elbow servo 0.07, wrist-flex servo 0.07, wrist roll + gripper 0.10 at the tool
    origin, jaws + spoon + camera + food 0.08 at 50 mm along tool z, links 0.015 each."""
    f = frames(q)
    s, e, w, t = f["shoulder"][:3, 3], f["upper_arm"][:3, 3], f["lower_arm"][:3, 3], f["tool"]
    tip = t[:3, 3] + t[:3, :3] @ [0, 0, 50.0]
    h = np.array([np.cos(np.radians(-q[0])), np.sin(np.radians(-q[0])), 0.0])
    pts = [(0.07, e), (0.015, (s + e) / 2), (0.07, w), (0.015, (e + w) / 2), (0.10, t[:3, 3]), (0.08, tip)]
    return G * sum(m * float(np.dot(p - s, h)) for m, p in pts) / 1000


def view_angle(T, target) -> float:
    """Angle (deg) between the camera boresight (tool +z) and the camera->target ray."""
    c = camera_pt(T)
    v = np.asarray(target, float) - c
    return float(np.degrees(np.arccos(np.clip(np.dot(v / np.linalg.norm(v), T[:3, 2]), -1, 1))))


def check(q) -> dict:
    """Every per-pose quantity used by the tables."""
    q = np.asarray(q, float)
    sp, T = spoon_pts(q)
    cam = camera_pt(T)
    f = frames(q)
    links = np.vstack([box_points(f[n], n) for n in ("upper_arm", "lower_arm", "wrist")])
    table = min(sp[:, 2].min() - SPOON_R, cam[2] - CAM_R, links[:, 2].min())
    bowls = min(min(min(bowl_dist(p, c) for p in sp) - SPOON_R, bowl_dist(cam, c) - CAM_R,
                    min(bowl_dist(p, c) for p in links)) for c in BOWLS.values())
    selfc = min(in_box_dist(p, f[n], n) for p in list(sp) + [cam] for n in ("base", "shoulder"))
    margin = float(np.min(np.minimum(q - LIM[:, 0], LIM[:, 1] - q)))
    tip = sp[-1]
    return {"tool": T[:3, 3] * 1000, "tip": tip, "tilt": tilt(q), "table": table, "bowls": bowls, "self": selfc,
            "margin": margin, "torque": shoulder_torque(q), "lips": float(np.linalg.norm(tip - LIPS)),
            "see_lips": view_angle(T, LIPS),
            "see_bowl": min(view_angle(T, [*c, BOWL_H / 2]) for c in BOWLS.values())}


# ---------------------------------------------------------------- poses
def level_posture(lift, elbow, pan=0.0):
    """(pan, lift, elbow, flex, roll) with flex chosen so the spoon is level
    (tool pitch = -(lift + elbow + flex) exactly on the SO-101)."""
    return np.array([pan, lift, elbow, -LEVEL_PITCH - lift - elbow, ROLL])


def tip_over(xy, tip_z, seed, tilt_deg=0.0):
    """Joints with the spoon tip over world ``xy`` at height ``tip_z`` (mm), spoon at
    ``tilt_deg`` from level, by base pan + ``solve_level_pose`` (shoulder, elbow, flex)."""
    pan = heading(xy)
    p = LEVEL_PITCH + tilt_deg
    d = np.asarray(xy, float) - PAN_AXIS
    r_tip = float(np.linalg.norm(d))
    r_tool = r_tip - SPOON_TIP * np.cos(np.radians(p))
    z_tool = tip_z - SPOON_TIP * np.sin(np.radians(p))
    u = d / r_tip
    xy_tool = PAN_AXIS + u * r_tool
    q0 = np.asarray(seed, float).copy()
    q0[0] = pan
    q, ok = solve_level_pose(q0, float(np.linalg.norm(xy_tool)) / 1000, z_tool / 1000, p)
    return np.round(q, 2), ok


def carry_search(n=12):
    """Rank level postures for the carry/home posture: low shoulder torque, compact,
    tool 240-285 mm high (lips ~270), every clearance and margin met."""
    rows = []
    for lift in np.arange(-90, -39, 2.5):
        for elbow in np.arange(0, 91, 2.5):
            q = level_posture(lift, elbow)
            if abs(q[3]) > 90:
                continue
            c = check(q)
            r = float(np.hypot(*(c["tool"][:2] - PAN_AXIS)))
            if not (240 <= c["tool"][2] <= 285 and 90 <= r <= 150):
                continue
            if c["margin"] < 10 or c["table"] < CLEAR_MM or c["self"] < SELF_MM:
                continue
            rows.append((abs(c["torque"]), q, r, c))
    rows.sort(key=lambda x: x[0])
    print(f"\nCarry/home posture search (pan 0, level): best {n} by |shoulder torque|")
    print(f"{'lift':>6} {'elbow':>6} {'flex':>6} | {'r mm':>5} {'z mm':>5} | {'|tau| Nm':>8} {'margin':>6} {'self':>5}")
    for t, q, r, c in rows[:n]:
        print(f"{q[1]:6.1f} {q[2]:6.1f} {q[3]:6.2f} | {r:5.0f} {c['tool'][2]:5.0f} | {t:8.3f} {c['margin']:6.1f} {c['self']:5.0f}")


# The chosen carry posture (lift, elbow): HOME, READY and the sweep all use it (see the plan).
CARRY = (-72.5, 32.5)
HOME_PAN = 0.0
FACE_PAN = 93.0  # lips heading from the pan axis is ~90-95 deg; the arm "looks" from ~94
PEEK_DOWN = 30.0  # deg of extra wrist flex (spoon tip down) to look into a bowl: empty spoon only
PRE_TIP_Z = 140.0  # spoon tip above the table over a bowl: capsule bottom 115 mm, rim 60 mm


def poses() -> dict:
    P = {"HOME": level_posture(*CARRY, HOME_PAN)}
    for name, c in BOWLS.items():
        k = name.split("-")[1].upper()
        P[f"OVER_{k}"] = level_posture(*CARRY, round(heading(c), 2))
        P[f"PEEK_{k}"] = P[f"OVER_{k}"] + np.array([0, 0, 0, PEEK_DOWN, 0])  # EMPTY spoon only
        q, ok = tip_over(c, PRE_TIP_Z, P["HOME"])
        if not ok:
            print(f"WARNING: PRE_SCOOP_{k} not reached exactly")
        P[f"PRE_SCOOP_{k}"] = q
    P["FACE_START"] = level_posture(*CARRY, FACE_PAN)
    P["LOOK_SAVED"] = np.array([12.04, -98.2, 70.07, 16.88, -7.3])
    P["CLOSE_TAUGHT"] = np.array([94.0, 9.1, -29.9, 16.2, -2.7])  # taught close pose (spoon tipped up)
    P["FACE_NEAR"] = level_posture(9.0, -30.0, 94.0)  # the same posture, spoon level (tracker output)
    return P


# Each segment: (name, from, to, loaded, how). "sync" = streamed straight joint line.
SEGMENTS = [
    ("home -> over bowl-right", "HOME", "OVER_RIGHT", False, "pan only"),
    ("over -> peek right", "OVER_RIGHT", "PEEK_RIGHT", False, "flex only"),
    ("over -> pre-scoop right", "OVER_RIGHT", "PRE_SCOOP_RIGHT", False, "level line"),
    ("carry right -> over", "PRE_SCOOP_RIGHT", "OVER_RIGHT", True, "level line"),
    ("sweep right -> face", "OVER_RIGHT", "FACE_START", True, "pan only"),
    ("home -> over bowl-left", "HOME", "OVER_LEFT", False, "pan only"),
    ("over -> pre-scoop left", "OVER_LEFT", "PRE_SCOOP_LEFT", False, "level line"),
    ("carry left -> over", "PRE_SCOOP_LEFT", "OVER_LEFT", True, "level line"),
    ("sweep left -> face", "OVER_LEFT", "FACE_START", True, "pan only"),
    ("face near -> face start", "FACE_NEAR", "FACE_START", True, "level line"),
    ("face start -> home", "FACE_START", "HOME", False, "pan only"),
    ("face start -> over right", "FACE_START", "OVER_RIGHT", False, "pan only"),
    ("face start -> over left", "FACE_START", "OVER_LEFT", False, "pan only"),
    ("saved look -> home (today)", "LOOK_SAVED", "HOME", False, "joint line"),
]


def path(qa, qb, model, n=80):
    qa, qb = np.asarray(qa, float), np.asarray(qb, float)
    if model == "sync":
        return [qa + (qb - qa) * s for s in np.linspace(0, 1, n)]
    d = qb - qa  # unsync: each joint at the same speed, stops on arrival
    tmax = float(np.max(np.abs(d)))
    return [qa + np.sign(d) * np.minimum(np.abs(d), t) for t in np.linspace(0, tmax, n)]


def segment_report(qa, qb, loaded, model):
    cs = [check(q) for q in path(qa, qb, model)]
    worst = {
        "tilt": max(cs, key=lambda c: abs(c["tilt"]))["tilt"],
        "table": min(c["table"] for c in cs), "bowls": min(c["bowls"] for c in cs),
        "self": min(c["self"] for c in cs), "margin": min(c["margin"] for c in cs),
        "lips": min(c["lips"] for c in cs), "torque": max(abs(c["torque"]) for c in cs),
    }
    ok = (worst["table"] >= CLEAR_MM and worst["bowls"] >= CLEAR_MM and worst["self"] >= SELF_MM
          and worst["margin"] >= LIMIT_MARGIN and (not loaded or abs(worst["tilt"]) <= TILT_TOL))
    return worst, ok


def xyz(v):
    return "(" + ", ".join(f"{x:.0f}" for x in v) + ")"


def fmt(v):
    return "[" + ", ".join(f"{x:7.2f}" for x in v) + "]"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--search", action="store_true", help="re-run the carry-posture search")
    a = ap.parse_args()
    if a.search:
        carry_search()

    P = poses()
    print("\nWAYPOINTS (joints deg: pan, lift, elbow, flex, roll; xyz mm, base frame; tilt = spoon from level, + = tip up)")
    print(f"{'pose':16} {'joints':45} {'tool xyz':22} {'tip xyz':22} {'tilt':>5} {'table':>5} {'bowls':>5} "
          f"{'self':>4} {'marg':>4} {'tau':>5} {'lips':>4} {'cam>lips':>8} {'cam>bowl':>8}")
    for k, q in P.items():
        c = check(q)
        print(f"{k:16} {fmt(q):45} {xyz(c['tool']):22} {xyz(c['tip']):22} {c['tilt']:+5.1f} {c['table']:5.0f} "
              f"{c['bowls']:5.0f} {c['self']:4.0f} {c['margin']:4.1f} {c['torque']:+5.2f} {c['lips']:4.0f} "
              f"{c['see_lips']:8.0f} {c['see_bowl']:8.0f}")
    print(f"  (clearances mm, need >= {CLEAR_MM:.0f} table/bowls, >= {SELF_MM:.0f} self; marg = deg to the nearest "
          f"limit, need >= {LIMIT_MARGIN:.0f}; tau = shoulder gravity proxy N m; lips = tip to lips mm; "
          f"cam>X = deg off the camera axis, in view if <= {CAM_HALF_FOV:.0f})")

    print("\nSEGMENTS (worst case over 80 samples; SYNC = streamed straight joint line, UNSYNC = same deg/s per joint)")
    print(f"{'segment':28} {'loaded':6} {'how':11} | {'model':6} {'tilt':>6} {'table':>5} {'bowls':>5} {'self':>4} "
          f"{'marg':>5} {'lips':>5} {'|tau|':>5} | ok")
    for name, a_, b_, loaded, how in SEGMENTS:
        for model in ("sync", "unsync"):
            w, ok = segment_report(P[a_], P[b_], loaded, model)
            print(f"{name if model == 'sync' else '':28} {('yes' if loaded else 'no') if model == 'sync' else '':6} "
                  f"{how if model == 'sync' else '':11} | {model:6} {w['tilt']:+6.1f} {w['table']:5.0f} {w['bowls']:5.0f} "
                  f"{w['self']:4.0f} {w['margin']:5.1f} {w['lips']:5.0f} {w['torque']:5.2f} | {'PASS' if ok else 'FAIL'}")

    # Camera: from which pan does the sweep see the lips?
    for k in ("HOME", "FACE_START"):
        el = frames(P[k])["upper_arm"][:3, 3]
        print(f"elbow joint at {k}: ({el[0]:.0f}, {el[1]:.0f}, {el[2]:.0f}) mm (behind the pan axis if x < 39)")
    q = P["FACE_START"].copy()
    seen = [p for p in np.arange(P["OVER_LEFT"][0], FACE_PAN + 0.1, 1.0)
            if view_angle(fk(np.r_[p, q[1:]]), LIPS) <= CAM_HALF_FOV]
    print(f"\nSweep at the carry posture: lips inside the {CAM_HALF_FOV:.0f} deg camera cone from pan "
          f"{min(seen):.0f} deg on" if seen else "\nSweep: lips never inside the camera cone")
    # Extend from FACE_START: how far can the tracker move in with a level spoon?
    r_now = float(np.linalg.norm(fk(q)[:3, 3][:2]))
    r_max = max_level_reach(q, LEVEL_PITCH, TrackConfig())
    tip_gap = check(q)["lips"]
    print(f"Extend from FACE_START: reach {r_now * 1000:.0f} -> {r_max * 1000:.0f} mm available "
          f"(+{(r_max - r_now) * 1000:.0f} mm); spoon tip to lips at the start {tip_gap:.0f} mm "
          f"(needs ~{tip_gap - 30:.0f} mm to reach the 3 cm stop)")


if __name__ == "__main__":
    main()
