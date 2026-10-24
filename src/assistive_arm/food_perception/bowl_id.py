"""Which fixed bowl holds which food: look at each bowl, crop its opening, ask CLIP on the Orin.

The bowls do not move (their positions come from the Viam 3D scene and the spoon-tip
calibration, ``poses/so101_bowl*.json``), so nothing here localises bowls. For each bowl:

* ``opening_outline``: the bowl's rim circle (scene centre, rim height, rim radius) projected
  through the live camera pose into the image: the "opening", drawn on the video feed and
  used to crop.
* ``look_pose``: joints that look down into the bowl (most of the opening in view, 12-35 cm), by a seeded
  search over SO-101 forward kinematics (cached per bowl).
* ``crop``: the opening's bounding box (slightly shrunk to the inside) as an image crop; the
  crops go to the ``bowl-contents`` vision service on the Orin (CLIP on the GPU).
* ``assign``: labels -> {food: bowl_id}, refusing ambiguous answers (the caller then falls back
  to the fixed assignment or asks).

Pure numpy (+ OpenCV only for JPEG encoding), no robot calls.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field

import numpy as np

from ..robot_iface.so101_kinematics import fk, joint_limits

# webcam-2 in the Viam config: parent follower-arm, (-21, -8, 90) mm, -90 deg about z (OpenCV optical axes)
CAM_T_M = np.array([-0.021, -0.008, 0.090])
CAM_RZ_DEG = -90.0
K_C270 = np.array([[772.5, 0.0, 320.0], [0.0, 772.5, 240.0], [0.0, 0.0, 1.0]])
IMG_W, IMG_H = 640, 480


def camera_offset() -> np.ndarray:
    """Tool <- camera transform from the Viam config (used to plan look poses offline)."""
    t = np.radians(CAM_RZ_DEG)
    T = np.eye(4)
    T[:2, :2] = [[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]]
    T[:3, 3] = CAM_T_M
    return T


def camera_pose(q_deg, offset: np.ndarray | None = None) -> np.ndarray:
    """Base <- camera (4x4, metres) at joints ``q_deg``."""
    return fk(np.asarray(q_deg, float)) @ (camera_offset() if offset is None else offset)


@dataclass(frozen=True)
class FixedBowl:
    bowl_id: str
    name: str
    centre: np.ndarray  # (2,) m, rim centre in the arm base frame (Viam scene geometry)
    rim_z: float = 0.060  # m (scene: 150 x 60 mm bowl on the table at z = 0)
    rim_radius: float = 0.075  # m
    default_food: str | None = None  # the fixed assignment when vision is unavailable


@dataclass
class BowlLabel:
    bowl_id: str
    label: str | None
    scores: dict = field(default_factory=dict)
    margin: float = 0.0
    box: tuple | None = None  # (x0, y0, x1, y1) of the opening in the look view


def rim_points(b: FixedBowl, n: int = 72, scale: float = 1.0) -> np.ndarray:
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    r = b.rim_radius * scale
    return np.c_[b.centre[0] + r * np.cos(a), b.centre[1] + r * np.sin(a), np.full(n, b.rim_z)]


def project(T_cam: np.ndarray, P: np.ndarray, K: np.ndarray = K_C270) -> np.ndarray:
    """Base-frame points -> pixels (NaN behind the camera)."""
    Pc = (np.linalg.inv(T_cam) @ np.c_[P, np.ones(len(P))].T)[:3].T
    uv = (K @ Pc.T).T
    with np.errstate(divide="ignore", invalid="ignore"):
        out = uv[:, :2] / uv[:, 2:3]
    out[Pc[:, 2] <= 1e-3] = np.nan
    return out


def opening_outline(b: FixedBowl, T_cam: np.ndarray, K: np.ndarray = K_C270, scale: float = 1.0):
    """(outline pixels (N, 2), bounding box or None if not in view, fraction of the rim in view)."""
    uv = project(T_cam, rim_points(b, scale=scale), K)
    ok = np.all(np.isfinite(uv), axis=1)
    if not np.any(ok):
        return uv, None, 0.0
    inside = ok & (uv[:, 0] >= 0) & (uv[:, 0] < IMG_W) & (uv[:, 1] >= 0) & (uv[:, 1] < IMG_H)
    p = uv[ok]
    box = (float(p[:, 0].min()), float(p[:, 1].min()), float(p[:, 0].max()), float(p[:, 1].max()))
    vis = (max(0, box[0]) < min(IMG_W, box[2])) and (max(0, box[1]) < min(IMG_H, box[3]))
    return uv, (box if vis else None), float(inside.mean())


def look_pose(b: FixedBowl, q_from, n: int = 20000, seed: int = 0, dist=(0.12, 0.35),
              min_rim_in_view: float = 0.6) -> tuple[np.ndarray, dict]:
    """Joints that look down into the bowl: its centre near the image centre, most of the rim in
    view (the C270 can't get far enough back to see all of it), 12-35 cm away, close to
    ``q_from``. Wrist roll is kept from ``q_from``."""
    rng = np.random.default_rng(seed)
    q0 = np.asarray(q_from, float)
    lim = joint_limits()
    target = np.r_[b.centre, b.rim_z]
    pan0 = -np.degrees(np.arctan2(target[1], target[0] - 0.0388))  # + pan turns toward -y
    rim = rim_points(b, n=36)
    best, best_cost, info = None, np.inf, {}
    for _ in range(n):
        q = q0.copy()
        q[0] = pan0 + rng.uniform(-30, 30)
        q[1] = rng.uniform(-100, 40)
        q[2] = rng.uniform(-60, 96)
        q[3] = rng.uniform(-94, 94)
        if np.any(q[:4] < lim[:4, 0] + 2) or np.any(q[:4] > lim[:4, 1] - 2):
            continue
        T = camera_pose(q)
        if T[2, 3] < b.rim_z + 0.05:  # camera above the rim
            continue
        c = project(T, target[None])[0]
        if not np.all(np.isfinite(c)) or not (60 < c[0] < IMG_W - 60 and 60 < c[1] < IMG_H - 80):
            continue
        uv = project(T, rim)
        ok = np.all(np.isfinite(uv), axis=1)
        frac = float((ok & (uv[:, 0] >= 0) & (uv[:, 0] < IMG_W) & (uv[:, 1] >= 0) & (uv[:, 1] < IMG_H - 24)).mean())
        if frac < min_rim_in_view:
            continue
        d = float(np.linalg.norm(T[:3, 3] - target))
        if not dist[0] <= d <= dist[1]:
            continue
        cost = (np.hypot(c[0] - IMG_W / 2, c[1] - IMG_H / 2) / 100 + 3.0 * (1 - frac)
                + 0.01 * float(np.abs(q[:4] - q0[:4]).sum()))
        if cost < best_cost:
            best, best_cost = q, cost
            info = {"distance_cm": round(d * 100, 1), "centre_px": [round(float(c[0])), round(float(c[1]))],
                    "rim_in_view": round(frac, 2), "camera_height_cm": round(float(T[2, 3]) * 100, 1)}
    if best is None:
        raise RuntimeError(f"no pose found that looks into {b.name}")
    return np.round(best, 2), info


def crop(rgb: np.ndarray, box, shrink: float = 0.12) -> np.ndarray | None:
    """The opening's box, shrunk toward its centre (the inside of the bowl, less rim and table)."""
    x0, y0, x1, y1 = box
    dx, dy = (x1 - x0) * shrink, (y1 - y0) * shrink
    h, w = rgb.shape[:2]
    a, b_ = int(max(0, x0 + dx)), int(max(0, y0 + dy))
    c, d = int(min(w, x1 - dx)), int(min(h, y1 - dy))
    if c - a < 16 or d - b_ < 16:
        return None
    return rgb[b_:d, a:c].copy()


def jpeg_b64(rgb: np.ndarray, quality: int = 90) -> str:
    import cv2

    ok, buf = cv2.imencode(".jpg", rgb[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return base64.b64encode(buf.tobytes()).decode()


def assign(labels: list[BowlLabel], foods=("cereal", "mms"), min_margin: float = 0.2) -> dict[str, str]:
    """{food: bowl_id} for every food that exactly one bowl was confidently labelled with."""
    out = {}
    for f in foods:
        hits = [b for b in labels if b.label == f and b.margin >= min_margin]
        if len(hits) == 1:
            out[f] = hits[0].bowl_id
    return out


def bowl_for(food: str, labels: list[BowlLabel] | None, bowls: list[FixedBowl]) -> tuple[FixedBowl, str]:
    """The bowl to scoop ``food`` from, and why: the vision label if it is clear, else the fixed one."""
    if labels:
        a = assign(labels)
        if food in a:
            return next(b for b in bowls if b.bowl_id == a[food]), "seen (CLIP on the Orin)"
    for b in bowls:
        if b.default_food == food:
            return b, "fixed assignment (no clear vision label)"
    raise RuntimeError(f"no bowl for {food!r}")


# ---------------------------------------------------------------- sweeping mode
def pan_to_centre(b: FixedBowl, q_posture, iters: int = 30) -> tuple[np.ndarray, float]:
    """Same posture (lift/elbow/flex/roll), base pan turned so the bowl centre sits in the middle
    of the image horizontally. Returns (joints, fraction of the rim in view)."""
    q = np.asarray(q_posture, float).copy()
    target = np.r_[b.centre, b.rim_z][None]
    for _ in range(iters):
        u = project(camera_pose(q), target)[0, 0]
        if not np.isfinite(u):
            break
        err = u - IMG_W / 2
        if abs(err) < 2:
            break
        q1 = q.copy(); q1[0] += 0.5
        du = (project(camera_pose(q1), target)[0, 0] - u) / 0.5
        if not np.isfinite(du) or abs(du) < 1e-3:
            break
        q[0] -= float(np.clip(err / du, -10, 10))
    _, _, frac = opening_outline(b, camera_pose(q))
    return np.round(q, 2), frac


def sweep_poses(bowls: list[FixedBowl], q_from, look=look_pose) -> list[tuple[FixedBowl, np.ndarray]]:
    """One look-down posture for all bowls (found for the first bowl), each bowl reached by turning
    the base only: a single smooth sweep, nearest bowl first from ``q_from``'s heading."""
    if not bowls:
        return []
    q0 = np.asarray(q_from, float)
    first = min(bowls, key=lambda b: abs(-np.degrees(np.arctan2(b.centre[1], b.centre[0] - 0.0388)) - q0[0]))
    posture, _ = look(first, q0)
    out = [(b, pan_to_centre(b, posture)[0]) for b in bowls]
    return sorted(out, key=lambda bq: abs(bq[1][0] - q0[0]))


def vote_crops(rgb: np.ndarray, bottom_skip: int = 70) -> list[np.ndarray]:
    """Crops of the inside of the bowl at a look-down view (the bowl fills the middle of the image):
    centre, left, right and upper, avoiding the bottom band where the gripper/spoon may show."""
    h, w = rgb.shape[:2]
    H = h - bottom_skip
    s = int(min(w, H) * 0.55)
    cx, cy = w // 2, H // 2
    boxes = [(cx - s // 2, cy - s // 2), (cx - s // 2 - s // 3, cy - s // 2), (cx - s // 2 + s // 3, cy - s // 2),
             (cx - s // 2, max(0, cy - s // 2 - s // 4))]
    out = []
    for x, y in boxes:
        x, y = int(np.clip(x, 0, w - s)), int(np.clip(y, 0, H - s))
        out.append(rgb[y:y + s, x:x + s].copy())
    return out


def combine(results: list[dict]) -> dict:
    """Average CLIP's per-crop scores into one label for the bowl."""
    names = sorted({k for r in results for k in r["scores"]})
    mean = {n: float(np.mean([r["scores"].get(n, 0.0) for r in results])) for n in names}
    order = sorted(mean, key=lambda n: -mean[n])
    margin = mean[order[0]] - (mean[order[1]] if len(order) > 1 else 0.0)
    return {"label": order[0], "scores": {n: round(v, 4) for n, v in mean.items()}, "margin": round(margin, 4),
            "votes": [r["label"] for r in results]}
