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

from ..feeding.face_track import LevelTracker, TrackConfig
from ..geometry import Pose
from ..robot_iface.so101_kinematics import fk as so101_fk
from ..feeding.face_servo import (
    FLEX, PAN, FaceServoConfig, FaceView, PointingConfig, PointingServo, line_miss, spoon_tilt_deg, target_pixel, tip_distance,
)

ROLL = 4  # SO-101 wrist roll joint index
FLEX_J = 3
from ..mouth_perception.tracker import MouthTracker
from ..mouth_perception.types import MouthState


@dataclass
class Taught:
    home: list[float] | None = None  # "start centered"; default: the look posture with the base pan at 0
    look: list[float] | None = None
    near: list[float] | None = None
    jacobian: dict | None = None
    spoon_level: dict | None = None  # {"tool_pitch_deg": ..., "wrist_roll_deg": ...}: tool pose when the SPOON is level
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
            override = getattr(self, "pose_fn", None)
            self._pose[name] = override(self.frames[name].timestamp) if override else self.robot.camera_pose(name)
        return self._pose[name]


def _tool_pitch(q) -> float:
    """Tool +Z pitch above horizontal (deg), from local SO-101 kinematics."""
    return float(np.degrees(np.arcsin(np.clip(so101_fk(q)[2, 2], -1, 1))))


def _r(v, n: int = 4):
    return None if v is None else [round(float(x), n) for x in np.asarray(v, float).ravel()]


def _mm(v):
    return None if v is None else [round(float(x) * 1000, 1) for x in np.asarray(v, float).ravel()]


class Stopped(Exception):
    pass


class FaceServoSession:
    ACTIONS = ("home", "search", "sweep", "find_face", "center", "auto_center", "extend", "retreat")

    def __init__(self, robot, detector, camera: str = "webcam-2", poses_path="poses/so101_face.json",
                 cfg: FaceServoConfig | None = None, render=None, probe_deg: float = 4.0, settle: float = 0.15,
                 allow_extend: bool = True, log_dir=None):
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
        self.allow_extend = allow_extend
        self.track_cfg = TrackConfig()
        self.track: LevelTracker | None = None
        self.servo_lag = 0.12  # s: streamed setpoint -> arm actually there (module docs: 40-160 ms)
        self._pose_fn = None  # camera pose from commanded joints while tracking
        self._q_hist: deque = deque(maxlen=400)
        self.hand_guide = False
        self._apply_level()
        self.plan_log = Path(log_dir) / f"face_servo-{time.strftime('%Y%m%d')}.jsonl" if log_dir else None
        self.plan_failures = 0
        self.last_plan_error = ""

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
                one = _OneFrame(self.robot, frames)
                one.pose_fn = self._pose_fn
                obs = self.det.observe(one)
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
        if action in ("hand_guide_on", "hand_guide_off"):
            self._require_idle()
            if action == "hand_guide_on":
                self.robot.stop_teleop()  # the leader must not fight your hand
                self.robot.arm_command({"command": "enter_manual_mode"})
                self.hand_guide = True
                return self._event("level", "hand-guide ON: tilt the wrist by hand until the spoon is level")
            self.robot.arm_command({"command": "exit_manual_mode"})
            self.hand_guide = False
            return self._event("level", "hand-guide off: arm holds its pose")
        if action == "nudge_spoon":
            self._require_idle()
            deg = float(np.clip(float(kw.get("deg", 0.0)), -10.0, 10.0))  # + = spoon tip up
            self.robot.stop_teleop()
            q = self.robot.joints()
            p0 = _tool_pitch(q)
            q1 = q.copy(); q1[FLEX] += 1.0
            sign = 1.0 if _tool_pitch(q1) > p0 else -1.0  # which way wrist flex tips the spoon up
            q1 = q.copy(); q1[FLEX] += sign * deg
            self.robot.move_joints(q1)
            self.hand_guide = False  # any motion command ends manual mode
            return self._event("level", f"spoon tip {'up' if deg > 0 else 'down'} {abs(deg):g} deg (tool pitch {_tool_pitch(q1):+.1f})")
        if action == "save_level":
            self._require_idle()
            q = self.robot.joints()
            self.taught.spoon_level = {"tool_pitch_deg": round(_tool_pitch(q), 2), "wrist_roll_deg": round(float(q[ROLL]), 2),
                                       "joints": [round(float(v), 2) for v in q], "t": time.strftime("%Y-%m-%dT%H:%M:%S")}
            self.taught.save(self.path)
            self._apply_level()
            return self._event("level", f"saved: spoon is level when the tool pitch is {self.taught.spoon_level['tool_pitch_deg']:+.2f} deg")
        if action == "set_params":
            self._require_idle()
            fields = {k: float(v) for k, v in kw.items() if k in ("stop_tip_distance", "joint_speed_degs")}
            self.cfg = FaceServoConfig(**{**asdict(self.cfg), **fields})  # validates the floor
            self.pcfg = PointingConfig(**{**asdict(self.pcfg), **fields})
            return self._event("params", str(fields))
        if action not in self.ACTIONS:
            raise ValueError(f"unknown action {action!r}")
        if action == "extend" and not self.allow_extend:
            raise RuntimeError("Extend is disabled (start the dashboard with --allow-extend)")
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
        self.plan_failures = 0
        self.last_move_end = self.robot.now()
        _, z0, self._theta0 = self.robot.tool_pose()  # hold this wrist roll (bowl up) throughout
        self._roll0 = float(self.robot.joints()[ROLL])
        while True:
            face, st = self._face_target()
            pos, z, theta = self.robot.tool_pose()
            theta = self._theta0
            self._aim_info = {"spoon_tilt_deg": round(spoon_tilt_deg(z), 1),
                              "miss_cm": None if face is None else round(line_miss(pos, z, face) * 100, 1)}
            cmd = self.last_cmd = ctrl.step(pos, z, face, st)
            self.reason = f"{cmd.kind}: {cmd.reason}"
            if cmd.kind in ("point", "approach"):
                self._event(cmd.kind, cmd.reason)
                q0 = self.robot.joints()
                self._check_stop()
                if not self._plan_move(cmd, pos, z, theta, face, q0):
                    self.plan_failures += 1
                    self._event("planner", f"no solution ({self.plan_failures}/3): {self.last_plan_error[:160]}")
                    if self.plan_failures >= 3:
                        raise RuntimeError("the arm's planner found no solution 3 times in a row: "
                                           f"{self.last_plan_error[:200]} (log: {self.plan_log})")
                    self.last_move_end = self.robot.now()
                    continue
                self.plan_failures = 0
                self.last_move_end = self.robot.now()
                q1 = self.robot.joints()
                jump = float(np.max(np.abs(q1 - q0)))
                if self.pcfg.level_spoon and abs(q1[ROLL] - self._roll0) > self.pcfg.roll_leeway_deg + 6.0:
                    self.stop(f"wrist roll drifted {q1[ROLL] - self._roll0:+.0f} deg (spill risk): stopped")
                    raise Stopped()
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

    def _plan_move(self, cmd, pos, z, theta, face, q0) -> bool:
        """One planned MoveToPosition; on failure retry once with a 3x wider pose cloud.
        Every attempt is appended to the JSONL plan log (inputs, result, error, timing)."""
        for wide in (False, True):
            lw = self.pcfg.position_leeway_mm * (3.0 if wide else 1.0)
            ol = self.pcfg.orientation_leeway * (3.0 if wide else 1.0)
            t0, err = time.monotonic(), ""
            roll_lw = (self.pcfg.roll_leeway_deg * (2.0 if wide else 1.0)) if self.pcfg.level_spoon else 180.0
            try:
                self.robot.move_tool(cmd.position, cmd.direction, theta, lw, ol, roll_lw)
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
            self._log_plan({
                "kind": cmd.kind, "ok": not err, "error": err, "wide": wide, "duration_s": round(time.monotonic() - t0, 3),
                "aim_error_deg": round(cmd.angle_deg, 2), "tip_distance_m": round(cmd.tip_distance, 3),
                "tool_pos_mm": _mm(pos), "tool_z": _r(z), "theta_deg": round(theta, 1),
                "target_pos_mm": _mm(cmd.position), "target_z": _r(cmd.direction),
                "leeway_mm": lw, "leeway_orient": ol, "leeway_roll_deg": roll_lw,
                "spoon_tilt_target_deg": round(spoon_tilt_deg(cmd.direction), 2), "face_mm": _mm(face) if face is not None else None,
                "joints_before": _r(q0, 1), "joints_after": _r(self._joints_or_none(), 1),
            })
            if not err:
                return True
            self.last_plan_error = err
        return False

    def _joints_or_none(self):
        try:
            return self.robot.joints(timeout=2.0)
        except Exception:
            return None

    def _log_plan(self, rec: dict) -> None:
        if self.plan_log is None:
            return
        try:
            self.plan_log.parent.mkdir(parents=True, exist_ok=True)
            with self.plan_log.open("a") as f:
                f.write(json.dumps({"t": time.strftime("%Y-%m-%dT%H:%M:%S"), **rec}) + "\n")
        except OSError:
            pass

    def _apply_level(self) -> None:
        """Use the saved spoon-level calibration as "level" for tracking and level aiming."""
        lvl = (self.taught.spoon_level or {}).get("tool_pitch_deg", 0.0)
        self.track_cfg = TrackConfig(**{**asdict(self.track_cfg), "pitch_target_deg": float(lvl)})
        self.pcfg = PointingConfig(**{**asdict(self.pcfg), "level_pitch_deg": float(lvl)})

    # ---------------------------------------------------------------- continuous tracking
    def _q_at(self, t: float) -> np.ndarray:
        """Commanded joints at time t (robot clock), from the setpoint history."""
        best = self._q_hist[0][1]
        for ti, qi in self._q_hist:
            if ti > t:
                break
            best = qi
        return best

    def _camera_offset(self, q) -> np.ndarray:
        """Camera pose relative to the tool frame (constant: the camera is on the gripper)."""
        cam = self.robot.camera_pose(self.camera).as_matrix()
        return np.linalg.inv(so101_fk(q)) @ cam

    def _track_loop(self) -> None:
        """Smooth continuous centring: stream joint setpoints at ``track_cfg.rate_hz``."""
        c = self.track_cfg
        dt = 1.0 / c.rate_hz
        q_meas = self.robot.joints()
        self.track = tr = LevelTracker(c)
        tr.reset(q_meas)
        offset = self._camera_offset(q_meas)
        self._q_hist.clear()
        self._q_hist.append((self.robot.now(), tr.q.copy()))

        def pose_at(t):
            T = so101_fk(self._q_at(t - self.servo_lag)) @ offset
            return Pose.from_matrix(T[:3, 3], T[:3, :3])

        self._pose_fn = pose_at
        next_t = last_check = time.monotonic()
        tick = 0
        try:
            while True:
                self._check_stop()
                now = self.robot.now()
                st = self.tracker.state(now)
                face = st.position if (st is not None and st.age <= 0.5) else None
                q = tr.step(face, dt)
                self.robot.stream_joints(q)
                self._q_hist.append((self.robot.now(), q.copy()))
                e = tr.last
                if face is not None and e:
                    self.reason = "tracking: miss {miss_cm} cm, spoon tilt {pitch_deg} deg".format(**e)
                    if e.get("lips_above_reach_cm"):
                        self.reason += f" | lips {e['lips_above_reach_cm']} cm above a level spoon's reach (spoon held level at the cap)"
                else:
                    self.reason = "tracking: waiting for the face (holding still)"
                if time.monotonic() - last_check > 0.5:  # is the real arm following?
                    last_check = time.monotonic()
                    lag = float(np.max(np.abs(self.robot.joints(timeout=1.0) - self._q_at(self.robot.now() - 0.3))))
                    if lag > 10.0:
                        self.stop(f"arm is {lag:.0f} deg behind its commands (blocked?): stopped")
                        raise Stopped()
                tick += 1
                if tick % 4 == 0:
                    self._log_plan({"kind": "track", "q_cmd": _r(q, 2), "v": _r(tr.v, 2), **e,
                                    "face_mm": _mm(face) if face is not None else None,
                                    "face_age_s": None if st is None else round(st.age, 3)})
                next_t += dt
                pause = next_t - time.monotonic()
                if pause > 0:
                    time.sleep(pause)
                else:
                    next_t = time.monotonic()  # fell behind: don't burst to catch up
        finally:
            self._pose_fn = None

    def _do_center(self) -> None:
        self._track_loop()

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
        self._track_loop()

    def _do_retreat(self) -> None:
        travelled = self.pointer.travelled if self.pointer else 0.0
        if travelled > 0 and self.pcfg.level_spoon:  # back straight out, level, in small planned steps
            pos, z, _ = self.robot.tool_pose()
            theta = getattr(self, "_theta0", 0.0)
            az = np.arctan2(pos[1], pos[0])
            d = np.array([np.cos(az), np.sin(az), 0.0])
            while travelled > 1e-3:
                step = min(self.pcfg.approach_step, travelled)
                pos = pos - step * d
                self.robot.move_tool(pos, d, theta, self.pcfg.position_leeway_mm * 3, self.pcfg.orientation_leeway,
                                     self.pcfg.roll_leeway_deg * 2)
                travelled -= step
            self.last_move_end = self.robot.now()
            self.pointer.travelled = 0.0
            self.reason = "retreated straight out, spoon level"
            self._event("retreat", self.reason)
            return
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
            "track": dict(self.track.last) if (self.track is not None and self.mode == "center") else None,
            "servo": None if self.last_cmd is None else {
                **getattr(self, "_aim_info", {}),
                "aim_error_deg": round(self.last_cmd.angle_deg, 1),
                "tip_distance_cm": round(self.last_cmd.tip_distance * 100, 1),
                "travelled_cm": round((self.pointer.travelled if self.pointer else 0.0) * 100, 1)},
            "params": {"stop_tip_distance_cm": round(self.pcfg.stop_tip_distance * 100, 1),
                       "floor_cm": round(self.cfg.min_tip_distance_floor * 100, 1),
                       "joint_speed_degs": self.cfg.joint_speed_degs},
            "face": face, "log": list(self.log)[:60], "allow_extend": self.allow_extend,
            "level": {"saved_tool_pitch_deg": (self.taught.spoon_level or {}).get("tool_pitch_deg"),
                      "spoon_tilt_now_deg": None if isinstance(joints, str) else round(
                          _tool_pitch(np.array(self._last_joints)) - float((self.taught.spoon_level or {}).get("tool_pitch_deg", 0.0)), 1),
                      "hand_guide": self.hand_guide},
            "planner": {"failures_in_a_row": self.plan_failures, "last_error": self.last_plan_error[:300],
                        "log_file": str(self.plan_log) if self.plan_log else None},
        }
