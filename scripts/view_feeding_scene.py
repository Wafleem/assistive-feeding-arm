#!/usr/bin/env python3
"""Look at the feeding scene: save both camera views, or open the interactive viewer.

    python scripts/view_feeding_scene.py --render out/        # PNGs from the wrist and fixed cameras
    python scripts/view_feeding_scene.py --viewer              # macOS: run with .venv/bin/mjpython
    python scripts/view_feeding_scene.py --render out/ --park  # arm parked out of the camera's view

This is the starting point for sections 1, 2 and 7: everything the perception
code will see in sim comes from these cameras.
"""

from __future__ import annotations

import argparse
import struct
import sys
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from assistive_arm.geometry import Pose, look_rotation  # noqa: E402
from assistive_arm.robot_iface.sim import SimRobot  # noqa: E402

PARK = Pose.from_matrix([0.15, -0.35, 0.25], look_rotation([0, 0, -1], [1, 0, 0]))


def write_png(path: Path, img: np.ndarray) -> None:
    if img.dtype != np.uint8:  # depth -> grey
        d = np.nan_to_num(img, posinf=0)
        d = np.clip(d / max(1e-6, float(np.percentile(d[d > 0], 99)) if (d > 0).any() else 1.0), 0, 1)
        img = np.repeat((255 * (1 - d))[..., None], 3, axis=2).astype(np.uint8)
    h, w, _ = img.shape
    raw = b"".join(b"\x00" + img[y].tobytes() for y in range(h))

    def chunk(t, data):
        return struct.pack(">I", len(data)) + t + data + struct.pack(">I", zlib.crc32(t + data) & 0xFFFFFFFF)

    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--render", type=Path)
    ap.add_argument("--viewer", action="store_true")
    ap.add_argument("--park", action="store_true", help="move the arm out of the fixed camera's view first")
    a = ap.parse_args()
    robot = SimRobot("feeding", image_size=(640, 480), viewer=a.viewer)
    if a.park:
        print("park:", robot.move_to_pose(PARK).ok)
    if a.render:
        a.render.mkdir(parents=True, exist_ok=True)
        for name, f in robot.get_frames().items():
            write_png(a.render / f"feeding_{name}_rgb.png", f.rgb)
            write_png(a.render / f"feeding_{name}_depth.png", f.depth)
        print(f"wrote {a.render}/feeding_*_{{rgb,depth}}.png")
    print("oracle vocabulary:", ", ".join(robot.oracle_detector().known_queries()))
    print("mouth site:", np.round(robot.site_pose("mouth").position, 3), "scene joints:",
          {k: round(v, 3) for k, v in robot.scene_state().items()})
    if a.viewer:
        input("Enter to close the viewer")
    robot.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
