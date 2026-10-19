#!/usr/bin/env python3
"""Benchmark the whole pipeline in MuJoCo: success rate, contact force, user contact.

    python scripts/eval_sim.py                 # 3 seeds, every task
    python scripts/eval_sim.py --seeds 10 --sway 0.004 --noisy

Every row is a fresh scene. A run counts as success only if the skill says so
AND the world agrees (button latched, bag held off the counter, door open).
This is the number to track per commit, and the harness a VLA policy is
scored with (swap the executor for ``run_vla``).
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from assistive_arm.executor import Executor  # noqa: E402
from assistive_arm.intent import parse  # noqa: E402
from assistive_arm.perception import Locator, NoisyDetector  # noqa: E402
from assistive_arm.robot_iface.sim import SimRobot  # noqa: E402
from assistive_arm.skills import DoorParams, SkillContext  # noqa: E402

TASKS = (
    [("elevator", f"press button {b} on the elevator", f"pressed:{b}") for b in "123456789"]
    + [("elevator", "press the lobby button", "pressed:L"), ("bag", "grab the shopping bag", "bag_held"),
       ("door_lever", "grab the door handle and open it", "door_open"), ("door_knob", "turn the knob and open the door", "door_open")]
)


def world_says_done(robot: SimRobot, check: str) -> bool:
    if check.startswith("pressed:"):
        return [e for e in robot.events if e.startswith("pressed:")] == [check]  # that button and no other
    if check == "bag_held":
        return robot.gripper_state().holding and robot.body_pose("bag").position[2] > 0.0
    if check == "door_open":
        return robot.scene_state()["door"] > np.deg2rad(15)
    raise ValueError(check)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--sway", type=float, default=0.0)
    ap.add_argument("--noisy", action="store_true")
    ap.add_argument("--known-hinge", action="store_true")
    a = ap.parse_args()

    stats = defaultdict(lambda: {"n": 0, "ok": 0, "claimed": 0, "false_claims": 0, "user_contact": 0, "peak": []})
    t0 = time.time()
    for seed in range(a.seeds):
        for scene, text, check in TASKS:
            r = SimRobot(scene, sway=a.sway, seed=seed)
            det = NoisyDetector(r.oracle_detector(), seed=seed) if a.noisy else r.oracle_detector()
            door = DoorParams(hinge_side="right", door_width=0.71)
            if a.known_hinge:
                door.hinge_point = r._base_world().inverse().transform_point([0.78, -1.10, 0.95])
            log = Executor(r, Locator(det), SkillContext(confirm=lambda m: True), door_defaults=door).run(parse(text))
            s = stats[f"{scene}: {text}"]
            done = world_says_done(r, check)
            claimed = log.result.status == "success"
            s["n"] += 1
            s["ok"] += done
            s["claimed"] += claimed
            s["false_claims"] += claimed and not done
            s["user_contact"] += "user_contact" in r.events
            s["peak"].append(log.result.evidence.get("peak_force", 0.0))
            r.close()
    print(f"{'task':58s} {'world ok':>9s} {'claimed':>8s} {'FALSE':>6s} {'user hit':>8s} {'peak N':>7s}")
    tot = defaultdict(int)
    for k, s in stats.items():
        print(f"{k:58s} {s['ok']:>4d}/{s['n']:<4d} {s['claimed']:>8d} {s['false_claims']:>6d} {s['user_contact']:>8d} {max(s['peak']):>7.1f}")
        for f in ("n", "ok", "false_claims", "user_contact"):
            tot[f] += s[f]
    print(f"\ntotal {tot['ok']}/{tot['n']} world-verified, {tot['false_claims']} false success claims, "
          f"{tot['user_contact']} user contacts, {time.time() - t0:.0f}s")
    return 0 if tot["false_claims"] == 0 and tot["user_contact"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
