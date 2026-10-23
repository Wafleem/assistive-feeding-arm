"""FaceServoSession: the SO-101 "sweep to the person, find the face, centre, extend" sequence,
as debuggable steps for the dashboard (dashboard/server.py).

Two threads:
* perception (always on): grab a frame, run the mouth detector, update the tracker,
  render the overlay JPEG, publish a snapshot (``snapshot()``);
* motion (one action at a time): ``sweep``, ``find_face``, ``calibrate``, ``center``,
  ``extend``, ``retreat``. ``stop`` always wins: it ends the action and stops the arm.

Every decision near the face is made by ``feeding.face_servo.FaceServoController`` (pure,
tested); this module only sequences, waits for FRESH frames after each move (never acts
on an image taken before the arm settled), and logs.

Taught poses and the image Jacobian persist in ``poses/so101_face.json``.
"""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from ..feeding.face_servo import (
    FLEX, PAN, FaceServoConfig, FaceServoController, FaceView, ImageJacobian, joint_target, target_pixel,
    tip_distance,
)
from ..mouth_perception.tracker import MouthTracker
from ..mouth_perception.types import MouthState


@dataclass
class Taught:
    look: list[float] | None = None
    near: list[float] | None = None
    jacobian: dict | None = None
    notes: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> "Taught":
        return cls(**json.loads(path.read_text())) if path.exists() else cls()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2) + "\n")


class _OneFrame:
    """Lets the detector see exactly the frame the loop grabbed (one grab, one pose query)."""

    def __init__(self, robot, frames):
        self.robot, self.frames, self._pose = robot, frames, {}

    def now(self):
        return self.robot.now()

    def get_frames(self, names=None):
        return {k: v for k, v in self.frames.items() if names is None or k in names}

    def camera_pose(self, name):
        if name not in self._pose:
            self._pose[name] = self.robot.camera_pose(name)
        return self._pose[name]


class Stopped(Exception):
    pass


class FaceServoSession:
    ACTIONS = ("sweep", "find_face", "calibrate", "center", "extend", "retreat")

    def __init__(self, robot, detector, camera: str = "webcam-2", poses_path="poses/so101_face.json",
                 cfg: FaceServoConfig | None = None, render=None, probe_deg: float = 4.0, settle: float = 0.15):
        self.robot, self.det, self.camera = robot, detector, camera
        self.path = Path(poses_path)
        self.taught = Taught.load(self.path)
        self.cfg = cfg or FaceServoConfig()
        self.render = render  # (frame, session_snapshot) -> jpeg bytes; None = no preview
        self.probe_deg, self.settle = probe_deg, settle
        self.tracker = MouthTracker()
        self.ctrl: FaceServoController | None = None
        self.mode, self.reason = "idle", "ready"
        self.log: deque = deque(maxlen=300)
        self._snap: dict = {}
        self._jpeg: bytes | None = None
        self._lock = threading.Lock()
        self._new_frame = threading.Condition()
        self._stop = threading.Event()
        self._running = True
        self._worker: threading.Thread | None = None
        self.last_move_end = 0.0
        self.fps = 0.0

    # ---------------------------------------------------------------- perception
    def start(self) -> None:
        threading.Thread(target=self._perception_loop, daemon=True).start()

    def _perception_loop(self) -> None:
        last = time.monotonic()
        while self._running:
            try:
                t0 = self.robot.now()
                frames = self.robot.get_frames([self.camera])
                frame = frames.get(self.camera)
                if frame is None:
                    time.sleep(0.1)
                    continue
                obs = self.det.observe(_OneFrame(self.robot, frames))
                self.tracker.update(obs)
                now = self.robot.now()
                state = self.tracker.state(now)
                view = None
                m = getattr(self.det, "last", None)
                if obs is not None and m is not None and self.det.last_landmarks is not None:
                    lm = self.det.last_landmarks
                    k = frame.intrinsics
                    lips = 0.5 * (lm[13, :2] + lm[14, :2]) * [k.width, k.height]
                    view = FaceView((float(lips[0]), float(lips[1])), m.distance, m.frontalness,
                                    (k.width, k.height), k.fx, k.fy)
                t = time.monotonic()
                self.fps = 0.8 * self.fps + 0.2 / max(t - last, 1e-3)
                last = t
                snap = {"frame_t": t0, "view": view, "state": state, "obs": obs, "frame": frame}
                with self._lock:
                    self._snap = snap
                if self.render is not None:
                    jpeg = self.render(frame, self)
                    with self._lock:
                        self._jpeg = jpeg
                with self._new_frame:
                    self._new_frame.notify_all()
            except Exception as e:  # keep the preview alive through network hiccups
                self._event("perception", f"error: {e}")
                time.sleep(0.5)

    def jpeg(self) -> bytes | None:
        with self._lock:
            return self._jpeg

    def latest(self) -> dict:
        with self._lock:
            return dict(self._snap)

    def fresh(self, after: float, timeout: float = 3.0) -> dict:
        """The first snapshot whose frame was grabbed after ``after`` (robot clock)."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self._check_stop()
            s = self.latest()
            if s and s["frame_t"] > after:
                return s
            with self._new_frame:
                self._new_frame.wait(0.2)
        return {}

    # ---------------------------------------------------------------- commands
    def command(self, action: str, **kw) -> dict:
        if action == "stop":
            return self.stop("stop pressed")
        if action == "teleop_start":
            self._require_idle()
            self.robot.start_teleop()
            return self._event("teleop", "started")
        if action == "teleop_stop":
            self.robot.stop_teleop()
            return self._event("teleop", "stopped")
        if action in ("teach_look", "teach_near"):
            q = [round(float(v), 2) for v in self.robot.joints()]
            setattr(self.taught, action[6:], q)
            self.taught.save(self.path)
            return self._event(action, f"saved {q}")
        if action == "set_params":
            self._require_idle()
            fields = {k: float(v) for k, v in kw.items() if k in ("stop_tip_distance", "joint_speed_degs")}
            self.cfg = FaceServoConfig(**{**asdict(self.cfg), **fields})  # validates the floor
            return self._event("params", str(fields))
        if action not in self.ACTIONS:
            raise ValueError(f"unknown action {action!r}")
        self._require_idle()
        self._stop.clear()
        self.mode, self.reason = action, "starting"
        self._worker = threading.Thread(target=self._run, args=(action,), daemon=True)
        self._worker.start()
        return {"ok": True, "mode": action}

    def stop(self, why: str) -> dict:
        self._stop.set()
        try:
            self.robot.stop()
        except Exception as e:
            self._event("stop", f"arm stop error: {e}")
        w = self._worker
        if w is not None and w is not threading.current_thread():
            w.join(timeout=5.0)  # a joint move in flight can't be interrupted: let it settle
        self.mode, self.reason = "idle", why
        return self._event("stop", why)

    def close(self) -> None:
        self._running = False
        self.stop("shutdown")

    def _require_idle(self) -> None:
        if self._worker is not None and self._worker.is_alive() and self._stop.is_set():
            self._worker.join(timeout=5.0)
        if self._worker is not None and self._worker.is_alive():
            raise RuntimeError(f"busy: {self.mode} (press STOP first)")

    def _check_stop(self) -> None:
        if self._stop.is_set():
            raise Stopped()

    def _event(self, kind: str, msg: str) -> dict:
        e = {"t": time.strftime("%H:%M:%S"), "kind": kind, "msg": msg}
        self.log.appendleft(e)
        return {"ok": True, **e}

    def _run(self, action: str) -> None:
        try:
            self.robot.stop_teleop()  # the leader must not drive the follower during any action
            self.robot.set_speed(self.cfg.joint_speed_degs)
            getattr(self, f"_do_{action}")()
            if self.mode == action:
                self.mode = "idle"
        except Stopped:
            self.mode = "idle"
        except Exception as e:
            self.mode, self.reason = "idle", f"{action} failed: {e}"
            self._event(action, f"FAILED: {e}")

    def _move(self, q) -> None:
        self._check_stop()
        self.robot.move_joints(q)
        self.last_move_end = self.robot.now()
        self._check_stop()

    def _need(self, *names) -> None:
        missing = [n for n in names if getattr(self.taught, n) is None]
        if missing:
            raise RuntimeError(f"teach {', '.join(missing)} first")

    def _face_px(self, n: int = 4) -> np.ndarray | None:
        """Median lips pixel over n fresh frames after the last move (None if the face is lost)."""
        pts, after = [], self.last_move_end + self.settle
        for _ in range(n):
            s = self.fresh(after)
            if not s or s["view"] is None:
                return None
            pts.append(s["view"].lips_px)
            after = s["frame_t"]
        return np.median(np.asarray(pts), axis=0)

    # ---------------------------------------------------------------- actions
    def _do_sweep(self) -> None:
        self._need("look")
        self.reason = "moving to the look pose"
        self._move(self.taught.look)
        self.reason = "at the look pose"
        self._event("sweep", "at the look pose")

    def _do_find_face(self) -> None:
        self._need("look")
        for d_pan in (0, -8, 8, -16, 16, -24, 24):
            q = np.array(self.taught.look, float)
            q[PAN] += d_pan
            self.reason = f"looking at pan {d_pan:+d} deg"
            self._move(q)
            if self._face_px(2) is not None:
                self.reason = f"face found at pan {d_pan:+d} deg"
                self._event("find_face", self.reason)
                return
        self.reason = "no face in the sweep"
        self._event("find_face", self.reason)

    def _do_calibrate(self) -> None:
        q0 = self.robot.joints()
        self.last_move_end = self.robot.now()
        p0 = self._face_px()
        if p0 is None:
            raise RuntimeError("no face visible: find the face first, and hold still")
        shifts = []
        for j in (PAN, FLEX):
            q = q0.copy()
            q[j] += self.probe_deg
            self.reason = f"probing joint {j + 1} by {self.probe_deg:+.0f} deg"
            self._move(q)
            p = self._face_px()
            self._move(q0)
            if p is None:
                raise RuntimeError("face lost during the probe: hold still and retry")
            shifts.append(p - p0)
        jac = ImageJacobian.from_probes(shifts[0], shifts[1], self.probe_deg)
        self.taught.jacobian = jac.to_dict()
        self.taught.save(self.path)
        self.reason = f"calibrated: J = {np.round(jac.J, 1).tolist()} px/deg"
        self._event("calibrate", self.reason)

    def _controller(self, approach: bool) -> FaceServoController:
        self._need("look", "near", "jacobian")
        ctrl = FaceServoController(ImageJacobian.from_dict(self.taught.jacobian), self.cfg, approach)
        q, look = self.robot.joints(), np.array(self.taught.look, float)
        others = [i for i in range(5) if i not in (PAN, FLEX)]
        if np.max(np.abs(q[others] - look[others])) > 4.0:
            raise RuntimeError("the arm is not at the look pose: press Sweep first")
        lim = np.asarray(self.cfg.max_offset_deg)
        ctrl.offset = np.clip([q[PAN] - look[PAN], q[FLEX] - look[FLEX]], -lim, lim)
        return ctrl

    def _servo(self, approach: bool) -> None:
        self.ctrl = ctrl = self._controller(approach)
        self.last_move_end = self.robot.now()
        while True:
            s = self.fresh(self.last_move_end + self.settle, timeout=1.0)
            view, state = (s.get("view"), s.get("state")) if s else (None, None)
            cmd = ctrl.step(view, state)
            self.reason = f"{cmd.kind}: {cmd.reason}"
            if cmd.kind in ("center", "approach"):
                self._event(cmd.kind, cmd.reason + (f" (s={ctrl.s:.2f})" if cmd.kind == "approach" else ""))
                self._move(joint_target(self.taught.look, self.taught.near, ctrl.s, ctrl.offset))
            elif cmd.kind == "retreat":
                self._event("retreat", cmd.reason)
                self._move(joint_target(self.taught.look, self.taught.near, 0.0, ctrl.offset))
            elif cmd.kind == "done":
                self._event("done", cmd.reason)
                self.mode = "done"
                return
            else:
                self._check_stop()
                time.sleep(0.05)
                self.last_move_end = min(self.last_move_end, self.robot.now())

    def _do_center(self) -> None:
        self._servo(approach=False)

    def _do_extend(self) -> None:
        self._servo(approach=True)

    def _do_retreat(self) -> None:
        self._need("look")
        off = self.ctrl.offset if self.ctrl is not None else np.zeros(2)
        near = self.taught.near or self.taught.look
        self._move(joint_target(self.taught.look, near, 0.0, off))
        if self.ctrl is not None:
            self.ctrl.s = 0.0
        self.reason = "retreated to the look pose"
        self._event("retreat", self.reason)

    # ---------------------------------------------------------------- dashboard state
    def state(self) -> dict:
        s = self.latest()
        view, st = s.get("view"), s.get("state")
        face = None
        if view is not None:
            err = np.asarray(view.lips_px) - target_pixel(view, self.cfg)
            face = {"lips_px": [round(v, 1) for v in view.lips_px], "error_px": [round(float(e), 1) for e in err],
                    "camera_distance_cm": round(view.camera_distance * 100, 1),
                    "tip_distance_cm": round(tip_distance(view, self.cfg) * 100, 1),
                    "frontalness": round(view.frontalness, 2)}
        if isinstance(st, MouthState):
            face = face or {}
            face.update({"open": st.is_open, "open_ratio": round(st.open_ratio, 2), "speed_cm_s": round(st.speed * 100, 1),
                         "age_s": round(st.age, 2), "position_mm": [round(float(v) * 1000) for v in st.position]})
        try:
            joints = [round(float(v), 1) for v in self.robot.joints()]
        except Exception as e:
            joints = f"error: {e}"
        return {
            "mode": self.mode, "reason": self.reason, "fps": round(self.fps, 1), "joints": joints,
            "taught": {"look": self.taught.look, "near": self.taught.near, "calibrated": self.taught.jacobian is not None},
            "servo": None if self.ctrl is None else {"s": round(self.ctrl.s, 2),
                                                     "offset_deg": [round(float(v), 1) for v in self.ctrl.offset]},
            "params": {"stop_tip_distance_cm": round(self.cfg.stop_tip_distance * 100, 1),
                       "floor_cm": round(self.cfg.min_tip_distance_floor * 100, 1),
                       "joint_speed_degs": self.cfg.joint_speed_degs},
            "face": face, "log": list(self.log)[:60],
        }
