#!/usr/bin/env python3
"""Run one spoken-style request in MuJoCo, end to end.

    python scripts/sim_demo.py --scene elevator --say "press button 7 on the elevator"
    python scripts/sim_demo.py --scene bag --say "grab the shopping bag" --viewer
    python scripts/sim_demo.py --scene door_lever --say "open the door" --known-hinge
    python scripts/sim_demo.py --scene elevator --render out/   # save both camera views

Confirmation prompts are answered "yes" automatically unless --ask is given.
"""

from __future__ import annotations

import argparse
import struct
import sys
import time
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from assistive_arm.executor import Executor  # noqa: E402
from assistive_arm.intent import parse  # noqa: E402
from assistive_arm.perception import Locator, NoisyDetector  # noqa: E402
from assistive_arm.robot_iface.sim import SimRobot  # noqa: E402
from assistive_arm.skills import DoorParams, SkillContext  # noqa: E402

DEFAULT_SAY = {
    "elevator": "press button 7 on the elevator",
    "bag": "grab the shopping bag",
    "door_lever": "grab the door handle and open it",
    "door_knob": "turn the knob and pull the door open",
}
SIM_HINGE_WORLD = [0.78, -1.10, 0.95]  # door scenes: where the hinge really is


def write_png(path: Path, img: np.ndarray) -> None:
    """Minimal PNG writer (RGB uint8 or float depth), no Pillow needed."""
    if img.dtype != np.uint8:
        d = np.nan_to_num(img, posinf=0)
        d = np.clip(d / max(1e-6, np.percentile(d[d > 0], 99) if (d > 0).any() else 1), 0, 1)
        img = np.repeat((255 * (1 - d))[..., None], 3, axis=2).astype(np.uint8)
    h, w, _ = img.shape
    raw = b"".join(b"\x00" + img[y].tobytes() for y in range(h))

    def chunk(t, data):
        return struct.pack(">I", len(data)) + t + data + struct.pack(">I", zlib.crc32(t + data) & 0xFFFFFFFF)

    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default="elevator", choices=sorted(DEFAULT_SAY))
    ap.add_argument("--say", help="the user's request (default: a typical one for the scene)")
    ap.add_argument("--viewer", action="store_true", help="open the MuJoCo viewer (macOS: run with mjpython)")
    ap.add_argument("--render", type=Path, help="save RGB + depth from both cameras to this folder and exit")
    ap.add_argument("--sway", type=float, default=0.0, help="wheelchair sway amplitude in metres, e.g. 0.004")
    ap.add_argument("--noisy", action="store_true", help="degrade the oracle detector (jitter, no masks, misses)")
    ap.add_argument("--known-hinge", action="store_true", help="door scenes: give the skill the true hinge")
    ap.add_argument("--ask", action="store_true", help="ask y/n on the terminal instead of auto-confirming")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    robot = SimRobot(a.scene, sway=a.sway, seed=a.seed, viewer=a.viewer)
    if a.render:
        a.render.mkdir(parents=True, exist_ok=True)
        for name, f in robot.get_frames().items():
            write_png(a.render / f"{a.scene}_{name}_rgb.png", f.rgb)
            write_png(a.render / f"{a.scene}_{name}_depth.png", f.depth)
        print(f"wrote camera images to {a.render}/")
        return 0

    detector = robot.oracle_detector()
    if a.noisy:
        detector = NoisyDetector(detector, seed=a.seed)
    door = DoorParams(hinge_side="right", door_width=0.71)
    if a.known_hinge:
        door.hinge_point = robot._base_world().inverse().transform_point(SIM_HINGE_WORLD)
    confirm = (lambda m: input(f"{m} [y/N] ").strip().lower().startswith("y")) if a.ask else (lambda m: print(f"? {m} -> yes") or True)
    ctx = SkillContext(confirm=confirm, log=lambda m: print(f"  {m}"))

    text = a.say or DEFAULT_SAY[a.scene]
    intent = parse(text)
    print(f'heard: "{text}"\nintent: {intent.to_dict()}')
    t0 = time.time()
    log = Executor(robot, Locator(detector), ctx, door_defaults=door).run(intent)
    print(f"plan: {log.plan.describe()}")
    for t in log.targets:
        print(f"  located '{t.label}' with {t.camera} camera at {np.round(t.position, 3)} normal {np.round(t.normal, 2)}")
    for n in log.notes:
        print(f"  note: {n}")
    r = log.result
    print(f"result: {r.status} - {r.message}\nevidence: {r.evidence}")
    print(f"sim events: {sorted(robot.events)}  scene: { {k: round(v, 3) for k, v in robot.scene_state().items()} }")
    print(f"wall time {time.time() - t0:.1f}s, sim time {robot.data.time:.1f}s")
    if a.viewer:
        input("press Enter to close the viewer")
    robot.close()
    return 0 if r.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
