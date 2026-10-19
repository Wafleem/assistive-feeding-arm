"""Pure conversions between this package's poses and Viam's (no viam import).

Viam poses are millimetres plus an *orientation vector* (o_x, o_y, o_z, theta
in degrees): (o_x, o_y, o_z) is where the frame's +z axis points and theta is
the spin about it. RDK builds the rotation as ZYZ Euler angles
(lon, lat, theta) with lat = acos(o_z) and lon = atan2(o_y, o_x)
(see rdk ``spatialmath/orientationVector.go``). Check a few poses against the
live arm (``arm.get_end_position``) before relying on this for contact moves.
"""

from __future__ import annotations

import numpy as np

from ..geometry import Pose

_EPS = 1e-4


def _rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def _ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1.0, 0], [-s, 0, c]])


def ov_to_mat(ox: float, oy: float, oz: float, theta_deg: float) -> np.ndarray:
    v = np.array([ox, oy, oz], dtype=float)
    v = v / np.linalg.norm(v)
    lat = np.arccos(np.clip(v[2], -1.0, 1.0))
    lon = np.arctan2(v[1], v[0]) if 1 - abs(v[2]) > _EPS else 0.0
    return _rz(lon) @ _ry(lat) @ _rz(np.deg2rad(theta_deg))


def mat_to_ov(r) -> tuple[float, float, float, float]:
    r = np.asarray(r, dtype=float)
    oz = float(np.clip(r[2, 2], -1.0, 1.0))
    lat = np.arccos(oz)
    if np.sin(lat) > 1e-6 and 1 - abs(oz) > _EPS:
        lon = np.arctan2(r[1, 2], r[0, 2])
        th = np.arctan2(r[2, 1], -r[2, 0])
    elif oz > 0:  # +z up: R = Rz(theta)
        lon, th = 0.0, np.arctan2(r[1, 0], r[0, 0])
    else:  # +z down: R = Ry(pi) Rz(theta)
        lon, th = 0.0, np.arctan2(r[1, 0], r[1, 1])
    v = r[:, 2]
    return float(v[0]), float(v[1]), float(v[2]), float(np.rad2deg(th))


def pose_to_viam_dict(p: Pose) -> dict:
    ox, oy, oz, th = mat_to_ov(p.rot)
    x, y, z = (p.position * 1000.0).tolist()
    return {"x": x, "y": y, "z": z, "o_x": ox, "o_y": oy, "o_z": oz, "theta": th}


def viam_dict_to_pose(d: dict) -> Pose:
    return Pose.from_matrix(
        np.array([d["x"], d["y"], d["z"]]) / 1000.0,
        ov_to_mat(d["o_x"], d["o_y"], d["o_z"], d["theta"]),
    )
