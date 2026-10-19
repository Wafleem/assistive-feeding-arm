"""Small, dependency-free (numpy only) rigid-body helpers.

Conventions used everywhere in this package:

* Units are SI: metres, radians, newtons. Only ``robot_iface.viam_convert``
  converts to Viam's millimetres / degrees.
* Quaternions are ``(w, x, y, z)`` (MuJoCo order).
* Every pose handed to a skill is in the **arm base frame** (``base``): origin
  at the xArm6 base flange, +x forward of the wheelchair, +y toward the seated
  user (arm mounted on the right armrest), +z up.
* The tool frame's +z axis points out of the gripper (the approach direction),
  the fingers close along tool +/-y.
* Camera poses use the OpenCV optical convention (+z forward, +y down).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _vec(v) -> np.ndarray:
    return np.asarray(v, dtype=float).reshape(3)


def normalize(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < 1e-12:
        raise ValueError("cannot normalise a zero vector")
    return v / n


def quat_to_mat(q) -> np.ndarray:
    w, x, y, z = normalize(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def mat_to_quat(m) -> np.ndarray:
    m = np.asarray(m, dtype=float)
    tr = np.trace(m)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        q = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    q = normalize(q)
    return q if q[0] >= 0 else -q


def axis_angle_to_mat(axis, angle: float) -> np.ndarray:
    k = normalize(axis)
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(angle) * kx + (1 - np.cos(angle)) * kx @ kx


def rotation_angle_between(r_a, r_b) -> float:
    """Geodesic angle (rad) between two rotation matrices."""
    c = (np.trace(np.asarray(r_a).T @ np.asarray(r_b)) - 1) / 2
    return float(np.arccos(np.clip(c, -1.0, 1.0)))


def look_rotation(z_axis, x_hint=(1.0, 0.0, 0.0)) -> np.ndarray:
    """Rotation whose +z is ``z_axis`` and whose +x is as close as possible to ``x_hint``."""
    z = normalize(z_axis)
    x = _vec(x_hint) - np.dot(x_hint, z) * z
    if np.linalg.norm(x) < 1e-6:  # hint parallel to z: pick any perpendicular
        x = np.cross(z, [0.0, 1.0, 0.0] if abs(z[1]) < 0.9 else [1.0, 0.0, 0.0])
    x = normalize(x)
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


@dataclass
class Pose:
    """Rigid transform: position (m) + rotation, stored as a quaternion (w, x, y, z)."""

    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    quat: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))

    def __post_init__(self):
        self.position = _vec(self.position)
        self.quat = normalize(np.asarray(self.quat, dtype=float).reshape(4))

    @classmethod
    def from_matrix(cls, position, rot) -> "Pose":
        return cls(position, mat_to_quat(rot))

    @property
    def rot(self) -> np.ndarray:
        return quat_to_mat(self.quat)

    @property
    def z_axis(self) -> np.ndarray:
        return self.rot[:, 2]

    def as_matrix(self) -> np.ndarray:
        t = np.eye(4)
        t[:3, :3] = self.rot
        t[:3, 3] = self.position
        return t

    def __matmul__(self, other: "Pose") -> "Pose":
        return Pose.from_matrix(self.position + self.rot @ other.position, self.rot @ other.rot)

    def inverse(self) -> "Pose":
        r_t = self.rot.T
        return Pose.from_matrix(-r_t @ self.position, r_t)

    def transform_point(self, p) -> np.ndarray:
        return self.rot @ _vec(p) + self.position

    def transform_vector(self, v) -> np.ndarray:
        return self.rot @ _vec(v)

    def translated(self, delta) -> "Pose":
        return Pose(self.position + _vec(delta), self.quat)

    def distance_to(self, other: "Pose") -> tuple[float, float]:
        """(translation error m, rotation error rad)."""
        return (
            float(np.linalg.norm(self.position - other.position)),
            rotation_angle_between(self.rot, other.rot),
        )

    def __repr__(self) -> str:
        p = np.round(self.position, 4).tolist()
        q = np.round(self.quat, 4).tolist()
        return f"Pose(pos={p}, quat={q})"


def tool_pose_facing(point, approach_dir, x_hint=(0.0, 0.0, 1.0), standoff: float = 0.0) -> Pose:
    """Tool pose at ``point - standoff * approach_dir`` with tool +z along ``approach_dir``.

    ``approach_dir`` is the direction the gripper travels to make contact, i.e.
    the negated surface normal for a button.
    """
    d = normalize(approach_dir)
    return Pose.from_matrix(_vec(point) - standoff * d, look_rotation(d, x_hint))


def interpolate_poses(a: Pose, b: Pose, n: int) -> list[Pose]:
    """``n`` poses from a (exclusive) to b (inclusive): linear position, slerp rotation."""
    qa, qb = a.quat, b.quat
    if np.dot(qa, qb) < 0:
        qb = -qb
    out = []
    for i in range(1, n + 1):
        t = i / n
        dot = np.clip(np.dot(qa, qb), -1.0, 1.0)
        if dot > 0.9995:
            q = qa + t * (qb - qa)
        else:
            th = np.arccos(dot)
            q = (np.sin((1 - t) * th) * qa + np.sin(t * th) * qb) / np.sin(th)
        out.append(Pose(a.position + t * (b.position - a.position), q))
    return out


def arc_waypoints(start: Pose, center, axis, angle: float, n: int) -> list[Pose]:
    """Rotate ``start`` about the line (center, axis) in ``n`` steps up to ``angle``.

    Used for both door swings (axis = hinge) and lever/knob turns (axis = the
    handle's spindle). Orientation rotates with the handle so the grasp is kept.
    """
    c = _vec(center)
    out = []
    for i in range(1, n + 1):
        r = axis_angle_to_mat(axis, angle * i / n)
        out.append(Pose.from_matrix(c + r @ (start.position - c), r @ start.rot))
    return out
