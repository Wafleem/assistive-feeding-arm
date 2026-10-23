"""SO-101 forward kinematics from the module's own SVA kinematics file (pure numpy).

``data/so101_kinematics.json`` is what ``Arm.GetKinematics`` returns for ``devrel:so101:arm``
(module 0.10.0). Viam's SVA convention: each link is a fixed translation (mm) + rotation
(euler roll/pitch/yaw, applied as Rz(yaw) Ry(pitch) Rx(roll)) from its parent; each joint
rotates about its axis by the joint angle (degrees). Checked against the arm's reported
end poses on 2026-10-23: 0.0-0.2 mm, 0.0-0.3 deg.

Used by the continuous face tracker: joint-space servoing at 20 Hz needs the tool pose and
its Jacobian locally, without a network round trip per evaluation.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

CHAIN = ("base", "1", "shoulder", "2", "upper_arm", "3", "lower_arm", "4", "wrist", "5", "tool")
DATA = Path(__file__).parent / "data" / "so101_kinematics.json"


def _axis_angle(axis, deg: float) -> np.ndarray:
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    t = np.radians(deg)
    x, y, z = a
    c, s, C = np.cos(t), np.sin(t), 1 - np.cos(t)
    return np.array([[c + x * x * C, x * y * C - z * s, x * z * C + y * s],
                     [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
                     [z * x * C - y * s, z * y * C + x * s, c + z * z * C]])


def _euler_zyx(roll: float, pitch: float, yaw: float) -> np.ndarray:
    return (_axis_angle([0, 0, 1], np.degrees(yaw)) @ _axis_angle([0, 1, 0], np.degrees(pitch))
            @ _axis_angle([1, 0, 0], np.degrees(roll)))


@lru_cache(maxsize=4)
def load(path: str = str(DATA)) -> tuple:
    """Pre-computed chain: tuple of ("fixed", 4x4) / ("joint", axis, min, max)."""
    k = json.loads(Path(path).read_text())
    links = {l["id"]: l for l in k["links"]}
    joints = {j["id"]: j for j in k["joints"]}
    chain = []
    for name in CHAIN:
        if name in links:
            l = links[name]
            t = l.get("translation", {})
            o = l.get("orientation", {}).get("value", {})
            M = np.eye(4)
            M[:3, :3] = _euler_zyx(o.get("roll", 0.0), o.get("pitch", 0.0), o.get("yaw", 0.0))
            M[:3, 3] = [t.get("x", 0.0), t.get("y", 0.0), t.get("z", 0.0)]
            chain.append(("fixed", M))
        else:
            j = joints[name]
            chain.append(("joint", (j["axis"]["x"], j["axis"]["y"], j["axis"]["z"]), j["min"], j["max"]))
    return tuple(chain)


def joint_limits(path: str = str(DATA)) -> np.ndarray:
    """(5, 2) min/max degrees."""
    return np.array([[c[2], c[3]] for c in load(path) if c[0] == "joint"])


def fk(q_deg, path: str = str(DATA)) -> np.ndarray:
    """4x4 pose of the tool frame in the arm base frame (metres)."""
    T = np.eye(4)
    i = 0
    for c in load(path):
        if c[0] == "fixed":
            T = T @ c[1]
        else:
            M = np.eye(4)
            M[:3, :3] = _axis_angle(c[1], float(q_deg[i]))
            T = T @ M
            i += 1
    T = T.copy()
    T[:3, 3] /= 1000.0
    return T
