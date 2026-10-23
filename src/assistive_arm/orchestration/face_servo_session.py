"""FaceServoSession: the SO-101 "sweep to the person, find the face, centre, extend" sequence,
as debuggable steps for the dashboard (dashboard/server.py).

Two threads:
* perception (always on): grab a frame, run the mouth detector, update the tracker,
  render the overlay JPEG, publish a snapshot (``snapshot()``);
* motion (one action at a time): ``home``, ``search``, ``sweep``, ``find_face``, ``center``,
  ``extend``, ``retreat``, ``auto_center``. ``stop`` always wins: it ends the action and stops the arm.

Every decision near the face is made by ``feeding.face_servo.PointingServo`` (pure, tested):
no calibration; the face's 3D position (camera + frame system) and the tool pose give the
direction to the lips, and the arm's own planner (``MoveToPosition``) re-aims the spoon in
small steps. This module only sequences, uses face fixes taken after the arm settled
(median of a few, so dropped frames don't matter), and logs.

Taught poses persist in ``poses/so101_face.json``.
"""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from ..feeding.face_servo import PAN, FaceServoConfig, FaceView, PointingConfig, PointingServo, target_pixel, tip_distance
from ..mouth_perception.tracker import MouthTracker
from ..mouth_perception.types import MouthState


@dataclass
class Taught:
    home: list[float] | None = None  # "start centered"; default: the look posture with the base pan at 0
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
    ACTIONS = ("home", "search", "sweep", "find_face", "center", "auto_center", "extend", "retreat")

    def __init__(self, robot, detector, camera: str = "webcam-2", poses_path="poses/so101_face.json",
                 cfg: FaceServoConfig | None = None, render=None, probe_deg: float = 4.0, settle: float = 0.15):
        self.robot, self.det, self.camera = robot, detector, camera
        self.path = Path(poses_path)
        self.taught = Taught.load(self.path)
        self.cfg = cfg or FaceServoConfig()
        self.pcfg = PointingConfig(stop_tip_distance=self.cfg.stop_tip_distance,
                                   joint_speed_degs=self.cfg.joint_speed_degs)
        self.pointer: PointingServo | None = None
        self.last_cmd = None
        self.render = render  # (frame, session_snapshot) -> jpeg bytes; None = no preview
        self.probe_deg, self.settle = probe_deg, settle
        self.tracker = MouthTracker()
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
            except Exception as e:  # keep the preview alive through network hiccups (the rig reconnects)
                self._event("perception", f"error: {type(e).__name__} {e}".strip())
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
        if action in ("teach_home", "teach_look", "teach_near"):
            q = [round(float(v), 2) for v in self.robot.joints()]
            setattr(self.taught, action[6:], q)
            self.taught.save(self.path)
            return self._event(action, f"saved {q}")
        if action == "set_params":
            self._require_idle()
            fields = {k: float(v) for k, v in kw.items() if k in ("stop_tip_distance", "joint_speed_degs")}
            self.cfg = FaceServoConfig(**{**asdict(self.cfg), **fields})  # validates the floor
            self.pcfg = PointingConfig(**{**asdict(self.pcfg), **fields})
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
    def home_pose(self) -> np.ndarray:
        if self.taught.home is not None:
            return np.array(self.taught.home, float)
        self._need("look")
        q = np.array(self.taught.look, float)
        q[PAN] = 0.0
        return q

    def _do_home(self) -> None:
        self.reason = "moving to start (centered)"
        self._move(self.home_pose())
        self.reason = "at start (centered)"
        self._event("home", f"at start {np.round(self.home_pose(), 1).tolist()}")

    def _do_search(self, step_deg: float = 6.0) -> None:
        """Pan from start toward the look side in small steps, looking for a face after each;
        then search around the look pose. Stops at the first face."""
        self._need("look")
        start, look = self.home_pose(), np.array(self.taught.look, float)
        n = max(1, int(np.ceil(abs(look[PAN] - start[PAN]) / step_deg)))
        path = [start + (look - start) * k / n for k in range(n + 1)]
        for d in (8, -8, 16, -16, 24, -24):
            q = look.copy()
            q[PAN] += d
            path.append(q)
        for q in path:
            self.reason = f"searching at pan {q[PAN]:.0f} deg"
            self._move(q)
            if self._face_px(2) is not None:
                self.reason = f"face found at pan {q[PAN]:.0f} deg: Calibrate or Center next"
                self._event("search", self.reason)
                return
        self.reason = "no face found: check the person is in view of the look pose"
        self._event("search", self.reason)

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

    def _face_target(self, need: int = 2, timeout: float = 1.5):
        """Median base-frame mouth position from fixes taken after the arm settled.
        Dropped frames are skipped: ``need`` good fixes within ``timeout`` (one is enough if
        the tracker is fresh). Returns (position or None, latest MouthState or None)."""
        after, pts, st = self.last_move_end + self.settle, [], None
        deadline = time.monotonic() + timeout
        while len(pts) < need and time.monotonic() < deadline:
            s = self.fresh(after, timeout=max(0.05, deadline - time.monotonic()))
            if not s:
                break
            after = s["frame_t"]
            st = s.get("state") or st
            if s.get("obs") is not None:
                pts.append(np.asarray(s["obs"].position, float))
        if not pts:
            return None, st
        return np.median(np.asarray(pts), axis=0), st

    def _point_loop(self, approach: bool) -> None:
        self.pointer = ctrl = PointingServo(self.pcfg, approach)
        self.last_move_end = self.robot.now()
        while True:
            face, st = self._face_target()
            pos, z, theta = self.robot.tool_pose()
            cmd = self.last_cmd = ctrl.step(pos, z, face, st)
            self.reason = f"{cmd.kind}: {cmd.reason}"
            if cmd.kind in ("point", "approach"):
                self._event(cmd.kind, cmd.reason)
                q0 = self.robot.joints()
                self._check_stop()
                self.robot.move_tool(cmd.position, cmd.direction, theta, self.pcfg.position_leeway_mm,
                                     self.pcfg.orientation_leeway)
                self.last_move_end = self.robot.now()
                jump = float(np.max(np.abs(self.robot.joints() - q0)))
                if jump > self.pcfg.max_joint_jump_deg:
                    self.stop(f"planner moved a joint {jump:.0f} deg in one step: stopped (check the pose)")
                    raise Stopped()
                self._check_stop()
            elif cmd.kind == "retreat":
                self._event("retreat", cmd.reason)
                self._do_retreat()
                return
            elif cmd.kind == "done":
                self._event("done", cmd.reason)
                self.mode = "done"
                return
            else:
                self._check_stop()
                time.sleep(0.1)

    def _do_center(self) -> None:
        self._point_loop(approach=False)

    def _do_extend(self) -> None:
        self._point_loop(approach=True)

    def _do_auto_center(self) -> None:
        """One button, no calibration: find the face (sweep from start toward the look side if
        it isn't in view), then keep the spoon pointed at the lips until STOP."""
        face, _ = self._face_target(need=1, timeout=1.0)
        if face is None:
            if self.taught.look is None:
                raise RuntimeError("no face in view: point the arm at the person, or teach the look pose")
            self._do_search()
            if "face found" not in self.reason:
                return
        self.mode = "center"
        self._point_loop(approach=False)

    def _do_retreat(self) -> None:
        if self.taught.look is not None:
            self._move(self.taught.look)
            self.reason = "retreated to the look pose"
        else:  # no taught pose: back straight out along the spoon by what extend travelled
            travelled = self.pointer.travelled if self.pointer else 0.0
            if travelled > 0:
                pos, z, theta = self.robot.tool_pose()
                self.robot.move_tool(pos - travelled * z, z, theta)
                self.last_move_end = self.robot.now()
            self.reason = "retreated along the spoon line"
        if self.pointer is not None:
            self.pointer.travelled = 0.0
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
        try:  # short timeout: a slow cloud link must not freeze the page
            self._last_joints = [round(float(v), 1) for v in self.robot.joints(timeout=1.5)]
            joints = self._last_joints
        except Exception as e:
            last = getattr(self, "_last_joints", None)
            joints = f"{last} (stale: {type(e).__name__})" if last else f"unavailable ({type(e).__name__})"
        return {
            "mode": self.mode, "reason": self.reason, "fps": round(self.fps, 1), "joints": joints,
            "taught": {"home": self.taught.home, "look": self.taught.look, "near": self.taught.near, "calibrated": self.taught.jacobian is not None},
            "servo": None if self.last_cmd is None else {
                "aim_error_deg": round(self.last_cmd.angle_deg, 1),
                "tip_distance_cm": round(self.last_cmd.tip_distance * 100, 1),
                "travelled_cm": round((self.pointer.travelled if self.pointer else 0.0) * 100, 1)},
            "params": {"stop_tip_distance_cm": round(self.pcfg.stop_tip_distance * 100, 1),
                       "floor_cm": round(self.cfg.min_tip_distance_floor * 100, 1),
                       "joint_speed_degs": self.cfg.joint_speed_degs},
            "face": face, "log": list(self.log)[:60],
        }
