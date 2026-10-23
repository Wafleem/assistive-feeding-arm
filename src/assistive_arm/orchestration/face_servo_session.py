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

from ..feeding.face_track import (ExtendConfig, LevelTracker, TrackConfig, approach_scale, extend_decision, jog_level,
                                  max_level_reach)
from ..geometry import Pose
from ..robot_iface.so101_kinematics import fk as so101_fk
from ..feeding.face_servo import (
    FLEX, PAN, FaceServoConfig, FaceView, PointingConfig, PointingServo, line_miss, spoon_tilt_deg, target_pixel, tip_distance,
)

ROLL = 4  # SO-101 wrist roll joint index
FLEX_J = 3
from ..mouth_perception.depth_range import DepthRangeCorrector
from ..mouth_perception.kalman import KalmanParams
from ..mouth_perception.tracker import MouthTracker, TrackerParams
from ..mouth_perception.types import MouthState


@dataclass
class Taught:
    home: list[float] | None = None  # "start centered"; default: the look posture with the base pan at 0
    look: list[float] | None = None
    near: list[float] | None = None
    jacobian: dict | None = None
    spoon_level: dict | None = None  # {"tool_pitch_deg": ..., "wrist_roll_deg": ...}: tool pose when the SPOON is level
    # Taught close pose: {"joints", "lips_in_tool_mm": [x, y, z], "t"}: where the lips were, in the
    # tool frame, with the spoon held where Extend should stop (z = along the spoon).
    close: dict | None = None
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
    ACTIONS = ("home", "search", "sweep", "find_face", "level_spoon", "jog", "center", "auto_center", "extend", "retreat",
               "feed")
    LEVEL_TOL_DEG = 5.0  # centring/extend only start with the spoon at least this level

    def __init__(self, robot, detector, camera: str = "webcam-2", poses_path="poses/so101_face.json",
                 cfg: FaceServoConfig | None = None, render=None, probe_deg: float = 4.0, settle: float = 0.15,
                 allow_extend: bool = True, log_dir=None, depth_camera: str | None = None, depth_detector=None,
                 depth_render=None):
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
        # Face-position smoothing: monocular landmarks at ~1 m jitter by ~1-2 cm, so the filter must
        # not trust a single frame to 4 mm (the old default): the arm would chase the jitter.
        self.tracker = MouthTracker(TrackerParams(kalman=KalmanParams(sigma_meas=0.012, sigma_accel=0.3)))
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
        self.depth_camera, self.depth_det = depth_camera, depth_detector
        self.depth = DepthRangeCorrector()
        self.depth_status = "off" if not (depth_camera and depth_detector) else "starting"
        self.depth_render = depth_render  # (frame, landmarks or None, session) -> jpeg bytes
        self._depth_jpeg: bytes | None = None
        self.track_cfg = TrackConfig()
        self.ext_cfg = ExtendConfig(stop_tip_distance=self.cfg.stop_tip_distance)
        self.track: LevelTracker | None = None
        self.servo_lag = 0.12  # s: streamed setpoint -> arm actually there (module docs: 40-160 ms)
        self.sag_gain = 1.5  # 1/s: servo-sag integrator (0 = off)
        self.sag_max = 6.0  # deg: most it will add to any joint
        self._pose_fn = None  # camera pose from commanded joints while tracking
        self._q_hist: deque = deque(maxlen=400)
        self._qm_hist: deque = deque(maxlen=400)  # MEASURED joints while tracking (servo sag included)
        self._ext_reach_start: float | None = None  # reach where the last extend began (for Retreat)
        self.hand_guide = False
        self._apply_level()
        self._apply_close()
        self.plan_log = Path(log_dir) / f"face_servo-{time.strftime('%Y%m%d')}.jsonl" if log_dir else None
        self.plan_failures = 0
        self.last_plan_error = ""

    # ---------------------------------------------------------------- perception
    def start(self) -> None:
        threading.Thread(target=self._perception_loop, daemon=True).start()
        if self.depth_camera and self.depth_det is not None:
            threading.Thread(target=self._depth_loop, daemon=True).start()

    def _depth_loop(self, period: float = 0.25) -> None:
        """Metric depth (a few Hz) teaches the fast face pipeline this person's eye span."""
        while self._running:
            t0 = time.monotonic()
            try:
                frame = self.robot.get_frames([self.depth_camera]).get(self.depth_camera)
                if frame is None or frame.depth is None or not np.any(frame.depth > 0):
                    self.depth_status = "no depth frame"
                else:
                    found = self.depth_det.landmarks(frame.rgb, self.robot.now())
                    if self.depth_render is not None:
                        jpg = self.depth_render(frame, None if found is None else found[0], self)
                        with self._lock:
                            self._depth_jpeg = jpg
                    if found is None:
                        self.depth_status = "depth ok, no face in it"
                    else:
                        params = getattr(self.det, "params", None)
                        if params is not None and self.depth.update(found[0], frame.intrinsics, frame.depth, params):
                            if self.depth.ready:
                                params.eye_span_m = self.depth.eye_span_m  # the 20 Hz pipeline now ranges metrically
                        self.depth_status = "ok" if self.depth.ready else f"learning ({len(self.depth.samples)} samples)"
            except Exception as e:
                self.depth_status = f"error: {type(e).__name__} {e}"[:160]
            time.sleep(max(0.0, period - (time.monotonic() - t0)))

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

    def depth_jpeg(self) -> bytes | None:
        with self._lock:
            return self._depth_jpeg

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
        if action == "teach_close":
            return self._event("close", self._teach_close())
        if action == "forget_close":
            self._require_idle()
            self.taught.close = None
            self.taught.save(self.path)
            self._apply_close()
            return self._event("close", "close pose forgotten: Extend uses the stop distance again")
        if action == "save_level":
            self._require_idle()
            q = self.robot.joints()
            self.taught.spoon_level = {"tool_pitch_deg": round(_tool_pitch(q), 2), "wrist_roll_deg": round(float(q[ROLL]), 2),
                                       "joints": [round(float(v), 2) for v in q], "t": time.strftime("%Y-%m-%dT%H:%M:%S")}
            self.taught.save(self.path)
            self._apply_level()
            return self._event("level", f"saved: spoon is level when the tool pitch is {self.taught.spoon_level['tool_pitch_deg']:+.2f} deg")
        if action == "set_tuning":
            return self._event("tuning", self._set_tuning(**kw))
        if action == "set_params":
            if "aim_raise_cm" in kw and self.taught.close:
                self.taught.close["raised_cm"] = float(np.clip(float(kw.pop("aim_raise_cm")), -5.0, 8.0))
                self.taught.save(self.path)
                self._apply_close()
                if self.track is not None:
                    self.track.cfg = self.track_cfg  # live, also while tracking
                return self._event("params", f"spoon aim raised {self.taught.close['raised_cm']:g} cm")
            self._require_idle()
            if "close_margin" in kw:
                m = float(np.clip(float(kw.pop("close_margin")), self.ext_cfg.min_tip_distance_floor, 0.20))
                self.ext_cfg = ExtendConfig(**{**asdict(self.ext_cfg), "close_margin": m})
            fields = {k: float(v) for k, v in kw.items() if k in ("stop_tip_distance", "joint_speed_degs")}
            self.cfg = FaceServoConfig(**{**asdict(self.cfg), **fields})  # validates the floor
            self.pcfg = PointingConfig(**{**asdict(self.pcfg), **fields})
            if "stop_tip_distance" in fields:
                self.ext_cfg = ExtendConfig(**{**asdict(self.ext_cfg), "stop_tip_distance": fields["stop_tip_distance"]})
            return self._event("params", str(fields))
        if action not in self.ACTIONS:
            raise ValueError(f"unknown action {action!r}")
        if action in ("extend", "feed") and not self.allow_extend:
            raise RuntimeError("Extend is disabled (start the dashboard with --allow-extend)")
        if action in ("extend", "retreat", "center", "auto_center") and self._tracking_live():
            if action in ("center", "auto_center"):
                return {"ok": True, "mode": self.mode, "note": "already tracking"}
            self._ext_request = action  # the running loop switches over: no STOP, no re-level
            return self._event(action, "requested while tracking")
        self._stop.clear()
        self._kw = kw
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

    def _tracking_live(self) -> bool:
        return (self._worker is not None and self._worker.is_alive() and not self._stop.is_set()
                and getattr(self, "_in_track_loop", False))

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
        """Turn the base from its current heading toward the look pose's, in small steps,
        looking for a face after each; then search around that heading. Stops at the first face."""
        self._need("look")
        # Turn ONLY the base, from where the arm is now to the look pose's heading: the rest of
        # the posture (and a just-levelled spoon) stays as it is.
        start = self.robot.joints()
        look_pan = float(self.taught.look[PAN])
        n = max(1, int(np.ceil(abs(look_pan - start[PAN]) / step_deg)))
        path = []
        for k in range(n + 1):
            q = start.copy()
            q[PAN] = start[PAN] + (look_pan - start[PAN]) * k / n
            path.append(q)
        for d in (8, -8, 16, -16, 24, -24):
            q = start.copy()
            q[PAN] = look_pan + d
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

    BASE_VEL = (120.0, 80.0, 80.0, 60.0)

    def _set_tuning(self, gain=None, speed_pct=None, accel=None, deadband_cm=None, smoothing_cm=None,
                    rate_hz=None) -> str:
        """Live smoothness tuning. Applies to the running tracker too."""
        c = asdict(self.track_cfg)
        if gain is not None:
            c["gain"] = float(np.clip(float(gain), 0.3, 4.0))
        if speed_pct is not None:
            k = float(np.clip(float(speed_pct), 5.0, 150.0)) / 100.0
            c["max_vel_deg"] = tuple(round(v * k, 2) for v in self.BASE_VEL)
        if accel is not None:
            c["max_acc_deg"] = float(np.clip(float(accel), 10.0, 800.0))
        if deadband_cm is not None:
            c["deadband_m"] = float(np.clip(float(deadband_cm), 0.2, 4.0)) / 100.0
        if rate_hz is not None:
            c["rate_hz"] = float(np.clip(float(rate_hz), 10.0, 60.0))
        self.track_cfg = TrackConfig(**c)
        if self.track is not None:
            self.track.cfg = self.track_cfg
        if smoothing_cm is not None:
            kp = KalmanParams(sigma_meas=float(np.clip(float(smoothing_cm), 0.2, 5.0)) / 100.0, sigma_accel=0.3)
            self.tracker.p = TrackerParams(**{**asdict(self.tracker.p), "kalman": kp})
            self.tracker.kf.p = kp
        t = self.tuning()
        return ", ".join(f"{k} {v}" for k, v in t.items())

    def tuning(self) -> dict:
        c, kp = self.track_cfg, self.tracker.kf.p
        return {"gain": c.gain, "speed_pct": round(100 * c.max_vel_deg[0] / self.BASE_VEL[0]), "accel": c.max_acc_deg,
                "deadband_cm": round(c.deadband_m * 100, 2), "smoothing_cm": round(kp.sigma_meas * 100, 2),
                "rate_hz": c.rate_hz}

    def _apply_level(self) -> None:
        """Use the saved spoon-level calibration as "level" for tracking and level aiming."""
        lvl = (self.taught.spoon_level or {}).get("tool_pitch_deg", 0.0)
        self.track_cfg = TrackConfig(**{**asdict(self.track_cfg), "pitch_target_deg": float(lvl)})
        self.pcfg = PointingConfig(**{**asdict(self.pcfg), "level_pitch_deg": float(lvl)})

    def _apply_close(self) -> None:
        """Use the taught close pose: aim the measured spoon line, stop where it was taught."""
        c = self.taught.close
        if c and c.get("lips_in_tool_measured_mm") is not None:
            # aim = where the lips were (measured) with the spoon raised by ``raised_cm`` (straight up
            # at the taught pose, expressed in the tool frame): the spoon sits lower than the camera thinks
            up = so101_fk(np.asarray(c["joints"], float))[:3, :3].T @ np.array([0.0, 0.0, 1.0])
            lips = np.asarray(c["lips_in_tool_measured_mm"], float) / 1000 - float(c.get("raised_cm", 0.0)) / 100 * up
            c["lips_in_tool_mm"] = [round(float(v) * 1000, 1) for v in lips]
        off = (0.0, 0.0) if not c else (c["lips_in_tool_mm"][0] / 1000, c["lips_in_tool_mm"][1] / 1000)
        self.track_cfg = TrackConfig(**{**asdict(self.track_cfg), "aim_offset_m": off})
        # Taught: aim the measured spoon line; stop by the spoon tip's distance to the lips.
        fwd = None if not c else self.ext_cfg.spoon_tip_from_tool
        self.ext_cfg = ExtendConfig(**{**asdict(self.ext_cfg), "close_fwd": fwd})
        # The taught pose is reachable by definition: let the comfort posture include it (3 deg spare).
        lo, hi = np.asarray(TrackConfig.posture_lo, float), np.asarray(TrackConfig.posture_hi, float)
        if c:
            qc = np.asarray(c["joints"][:4], float)
            lo, hi = np.minimum(lo, qc - 3.0), np.maximum(hi, qc + 3.0)
        self.track_cfg = TrackConfig(**{**asdict(self.track_cfg), "posture_lo": tuple(lo), "posture_hi": tuple(hi)})

    def _teach_close(self) -> str:
        """Record the close pose: the arm where the spoon should stop (hand-guided), the person
        still. The lips come from the latest face fix (up to 5 s old: very close, the camera may
        no longer see the whole face), in the base frame, so the arm's pose doesn't matter."""
        self._require_idle()
        st = self.tracker.state(self.robot.now())
        if st is None or st.age > 5.0:
            raise RuntimeError("no recent face fix: let the camera see the person, then move in and teach again")
        q = self.robot.joints()
        T = so101_fk(q)
        lips = np.linalg.inv(T) @ np.r_[np.asarray(st.position, float), 1.0]
        x, y, z = (float(v) for v in lips[:3])
        if not (0.03 < z < 0.6 and abs(x) < 0.15 and abs(y) < 0.15):
            raise RuntimeError(f"lips at {x * 100:+.1f}/{y * 100:+.1f}/{z * 100:.1f} cm in the tool frame: "
                               "not in front of the spoon: check the face fix and teach again")
        self.taught.close = {"joints": [round(float(v), 2) for v in q],
                             "lips_in_tool_mm": [round(x * 1000, 1), round(y * 1000, 1), round(z * 1000, 1)],
                             "lips_base_mm": _mm(st.position), "face_age_s": round(float(st.age), 2),
                             "spoon_tilt_deg": round(self._spoon_tilt(q), 1),
                             "t": time.strftime("%Y-%m-%dT%H:%M:%S")}
        self.taught.save(self.path)
        self._apply_close()
        Tc = so101_fk(q)
        line = self.track_cfg.aim_offset_m
        return (f"close pose saved: lips {z * 100:.1f} cm ahead of the tool along the spoon, spoon line "
                f"{line[0] * 100:+.1f}/{line[1] * 100:+.1f} cm off the tool axis (tool x/y); Extend stops here. "
                f"Tool at {Tc[2, 3] * 100:.1f} cm height, face fix {st.age:.1f} s old")

    # ---------------------------------------------------------------- continuous tracking
    def _q_at(self, t: float) -> np.ndarray:
        """Commanded joints at time t (robot clock), from the setpoint history."""
        best = self._q_hist[0][1]
        for ti, qi in self._q_hist:
            if ti > t:
                break
            best = qi
        return best

    def _qm_at(self, t: float) -> np.ndarray:
        best = self._qm_hist[0][1]
        for ti, qi in self._qm_hist:
            if ti > t:
                break
            best = qi
        return best

    def _camera_offset(self, q) -> np.ndarray:
        """Camera pose relative to the tool frame (constant: the camera is on the gripper)."""
        cam = self.robot.camera_pose(self.camera).as_matrix()
        return np.linalg.inv(so101_fk(q)) @ cam

    def _track_loop(self, extend: bool = False, retreat_only: bool = False, once: bool = False) -> None:
        """Smooth continuous centring (and, with ``extend``, moving in to the stop distance and
        back out): stream joint setpoints at ``track_cfg.rate_hz``."""
        c = self.track_cfg
        dt = 1.0 / c.rate_hz
        q_meas = self.robot.joints()
        self.track = tr = LevelTracker(c)
        tr.reset(q_meas)
        if retreat_only and self._ext_reach_start is not None:
            tr.reach_start, extend = self._ext_reach_start, True  # back out to where extend began
        elif extend:
            self._ext_reach_start = tr.reach_start
        tr.set_extending(extend)
        offset = self._camera_offset(q_meas)
        self._q_hist.clear()
        self._q_hist.append((self.robot.now(), tr.q.copy()))

        self._qm_hist.clear()
        fast = getattr(self.robot, "joints_fast", None)

        def pose_at(t):
            # The camera is where the arm REALLY is: measured joints (gravity sag included) at the
            # frame's time; commanded joints minus the servo lag only if measurements are missing.
            q_at = self._qm_at(t) if self._qm_hist else self._q_at(t - self.servo_lag)
            T = so101_fk(q_at) @ offset
            return Pose.from_matrix(T[:3, 3], T[:3, :3])

        self._pose_fn = pose_at
        next_t = last_check = time.monotonic()
        tick = 0
        held_since, backing_out = None, retreat_only
        prog = [0.0, time.monotonic()]  # (best actual travel, when): stall detection while extending
        # Servo-sag feedback: the servos settle short of their setpoints under gravity. An
        # integrator per joint adds the measured shortfall back onto the setpoint, so the arm
        # really is where the tracker thinks (spoon level, aim and reach all from the same q).
        sag = np.zeros(5)
        self._sag = sag
        self._ext_request = None
        self._in_track_loop = True
        try:
            while True:
                self._check_stop()
                now = self.robot.now()
                st = self.tracker.state(now)
                face = st.position if (st is not None and st.age <= 0.8) else None  # coasts through short dropouts
                req, self._ext_request = self._ext_request, None
                if req == "extend" and not extend:
                    extend, backing_out, held_since = True, False, None
                    prog = [0.0, time.monotonic()]
                    tr.reach0 = tr.reach_start = float(np.linalg.norm(so101_fk(tr.q)[:3, 3][:2]))
                    self._ext_reach_start = tr.reach_start
                    tr.set_extending(True)
                    self._check_extend_reach(face, tr.q)
                    self.mode = "extend"
                elif req == "retreat" and extend:
                    backing_out = True
                ext_reason, ext_log = "", {}
                if extend:
                    e0 = tr.last or {}
                    if face is None:
                        tip = 9.9
                    elif self.ext_cfg.close_fwd is not None:
                        tip = tr.remaining(face, self.ext_cfg)
                    else:
                        tip = tr.tip_distance(face, self.ext_cfg.spoon_tip_from_tool)
                    held = (time.monotonic() - held_since) if held_since else 0.0
                    direction, ext_reason = extend_decision(
                        self.ext_cfg, face is not None, st.age if st is not None else 9.9,
                        st.speed if st is not None else 0.0, e0.get("miss_cm", 99.0), e0.get("pitch_deg", 99.0),
                        tip, tr.travelled, held)
                    if direction < 0:
                        backing_out = True
                    if backing_out:  # once backing out, finish backing out (then stop)
                        direction = -1
                        if tr.travelled <= 1e-4 and abs(tr.actual_travelled()) < 0.01:  # the ARM is back
                            self._ext_reach_start = None
                            self._event("extend", f"backed out to the start ({ext_reason or 'retreat'})")
                            if retreat_only or once:  # once: one bite (Feed), then the caller goes home
                                self.reason = f"backed out to the start ({ext_reason or 'retreat'})"
                                self.mode = "done"
                                return
                            # keep tracking (spoon level, on the lips): press Extend again any time
                            extend, backing_out, held_since = False, False, None
                            tr.set_extending(False)
                            self.mode = "center"
                            continue
                    lead = tr.travelled - tr.actual_travelled()  # target ahead of the arm: at its limit
                    if direction > 0 and lead > 0.015:
                        direction, ext_reason = 0, "arm at its reach limit here: not moving in further"
                    if direction > 0 and tr.travelled >= getattr(self, "_ext_max_travel", 9.9):
                        direction, ext_reason = 0, "at the arm's reach for a level spoon: not moving in further"
                    # The eating hold starts at the stop OR as far in as the arm can go; re-aiming
                    # there doesn't restart it, only moving in again does.
                    # Stalled (no real progress for a while, e.g. re-aiming at a limit): this is as
                    # close as it gets here, so the eating hold starts anyway.
                    act = tr.actual_travelled()
                    if direction < 0 or act > prog[0] + 0.005:
                        prog = [act, time.monotonic()]
                    elif not held_since and not backing_out and time.monotonic() - prog[1] > 2.5:
                        ext_reason = "not getting closer: holding here"
                    if not backing_out:
                        if ext_reason.startswith(("at the stop", "arm at its reach", "at the arm's reach", "not getting closer")):
                            held_since = held_since or time.monotonic()
                        if held_since and direction > 0:
                            direction = 0  # eating: the spoon stays put (aim still tracks the lips)
                        if held_since and time.monotonic() - held_since >= self.ext_cfg.hold_s:
                            direction, ext_reason, backing_out = -1, "hold time over: backing out", True
                        elif held_since and not ext_reason.startswith("at the stop"):
                            ext_reason = "holding here" if ext_reason.startswith(("moving in", "re-centring")) else ext_reason
                            ext_reason += f" (holding {time.monotonic() - held_since:.0f}/{self.ext_cfg.hold_s:.0f} s)"
                    if ext_reason != getattr(self, "_last_ext_reason", None) and ext_reason.split(":")[0] != \
                            (getattr(self, "_last_ext_reason", "") or "").split(":")[0]:
                        self._event("extend", ext_reason)
                    self._last_ext_reason = ext_reason
                    ext_log = {"ext_dir": direction, "ext_reason": ext_reason, "tip_cm": round(tip * 100, 1),
                               "travel_target_cm": round(tr.travelled * 100, 1),
                               "travel_actual_cm": round(tr.actual_travelled() * 100, 1),
                               "head_speed_cm_s": None if st is None else round(st.speed * 100, 1)}
                    scale = 1.0
                    if direction > 0:  # slow down smoothly as the aim drifts or the arm nears its limit
                        scale = approach_scale(self.ext_cfg, e0.get("miss_cm", 99.0)) * \
                            float(np.clip((0.015 - lead) / 0.01, 0.0, 1.0))
                    ext_log["approach_scale"] = round(scale, 2)
                    stop_at = (max(self.ext_cfg.close_margin, self.ext_cfg.min_tip_distance_floor)
                               if self.ext_cfg.close_fwd is not None else self.ext_cfg.stop_tip_distance)
                    tr.advance(direction, dt, self.ext_cfg, scale, room=(tip - stop_at) if face is not None else 0.0)
                # Eating hold: the person moves onto the spoon, so the arm keeps still (no
                # re-centring). The face is still watched: a fast head move still backs out.
                frozen = extend and held_since is not None and not backing_out
                q = tr.step(None if frozen else face, dt, blind=extend and backing_out)
                if self._qm_hist and self.sag_gain > 0:
                    t_m, q_m = self._qm_hist[-1]
                    err = self._q_at(t_m - self.servo_lag) - q_m  # setpoint (desired) vs where it is
                    err[ROLL] = 0.0
                    ok = np.abs(err) < 4.0  # bigger = moving/blocked, not sag: don't wind up
                    sag += np.where(ok, self.sag_gain * err * dt, 0.0)
                    np.clip(sag, -self.sag_max, self.sag_max, out=sag)
                self.robot.stream_joints(q + sag)
                self._q_hist.append((self.robot.now(), q.copy()))  # the DESIRED joints
                if fast is not None:
                    try:
                        self._qm_hist.append((self.robot.now(), np.asarray(fast(timeout=0.3), float)))
                    except Exception:
                        pass
                c, dt = tr.cfg, 1.0 / tr.cfg.rate_hz  # tuning can change live
                e = tr.last
                if extend and ext_reason:
                    self.reason = f"extend: {ext_reason} | travelled {tr.travelled * 100:.1f} cm"
                elif face is not None and e:
                    self.reason = "tracking: miss {miss_cm} cm, spoon tilt {tilt_deg} deg".format(**e)
                    if e.get("lips_above_reach_cm"):
                        self.reason += f" | lips {e['lips_above_reach_cm']} cm above a level spoon's reach (spoon held level at the cap)"
                else:
                    self.reason = "tracking: waiting for the face (holding still)"
                if time.monotonic() - last_check > 0.5:  # is the real arm following?
                    last_check = time.monotonic()
                    q_real = self._qm_hist[-1][1] if self._qm_hist else self.robot.joints(timeout=1.0)
                    lag = float(np.max(np.abs(q_real - self._q_at(self.robot.now() - 0.3))))
                    if lag > 20.0:  # faster motion lags more; still catches a blocked arm
                        self.stop(f"arm is {lag:.0f} deg behind its commands (blocked?): stopped")
                        raise Stopped()
                tick += 1
                if tick % 4 == 0:
                    Tq = so101_fk(q)
                    self._log_plan({"kind": "extend" if extend else "track", "q_cmd": _r(q, 2), "v": _r(tr.v, 2), **e,
                                    **ext_log, "tool_mm": _mm(Tq[:3, 3]),
                                    "cam_mm": _mm(pose_at(self.robot.now()).position),
                                    "face_mm": _mm(face) if face is not None else None,
                                    "face_age_s": None if st is None else round(st.age, 3),
                                    "sag_deg": _r(sag, 2),
                                    "eye_span_mm": round(getattr(self.det, "params").eye_span_m * 1000, 1)
                                    if hasattr(self.det, "params") else None})
                next_t += dt
                pause = next_t - time.monotonic()
                if pause > 0:
                    time.sleep(pause)
                else:
                    next_t = time.monotonic()  # fell behind: don't burst to catch up
        finally:
            self._pose_fn = None
            self._in_track_loop = False

    def _spoon_tilt(self, q=None) -> float:
        """Spoon tilt from the saved level (deg, + = tip up)."""
        q = self.robot.joints() if q is None else q
        return _tool_pitch(q) - float((self.taught.spoon_level or {}).get("tool_pitch_deg", 0.0))

    def _require_level(self) -> None:
        """Level the spoon first if it isn't (slow wrist-only move), instead of refusing."""
        tilt = self._spoon_tilt()
        if abs(tilt) > self.LEVEL_TOL_DEG:
            self._event("level", f"spoon {tilt:+.0f} deg from level: levelling first")
            self._do_level_spoon()

    def _do_level_spoon(self, speed_degs: float = 30.0) -> None:
        """Slowly turn only the wrist flex until the spoon is level (two passes for accuracy)."""
        try:
            self.robot.set_speed(speed_degs)
            for _ in range(2):
                q = self.robot.joints()
                tilt = self._spoon_tilt(q)
                if abs(tilt) < 0.5:
                    break
                q1 = q.copy(); q1[FLEX] += 1.0
                slope = _tool_pitch(q1) - _tool_pitch(q)  # deg of spoon tilt per deg of wrist flex
                if abs(slope) < 0.2:
                    raise RuntimeError("the wrist can't level the spoon from this pose")
                target = q.copy()
                target[FLEX] = float(np.clip(q[FLEX] - tilt / slope, -94.0, 94.0))
                self.reason = f"levelling the spoon ({tilt:+.1f} deg)"
                self._move(target)
        finally:
            self.robot.set_speed(self.cfg.joint_speed_degs)
        self.reason = f"spoon level ({self._spoon_tilt():+.1f} deg)"
        self._event("level", self.reason)

    def _do_jog(self, speed_degs: float = 40.0) -> None:
        """Debug jog with the spoon kept level: yaw (+ = to the right, seen from behind the arm
        looking out along the spoon) by base pan only; up/down by shoulder+elbow+wrist together."""
        kw = getattr(self, "_kw", {}) or {}
        yaw = float(np.clip(float(kw.get("yaw_deg", 0.0)), -10.0, 10.0))
        dz = float(np.clip(float(kw.get("dz_cm", 0.0)), -5.0, 5.0)) / 100.0
        self._require_level()
        q = self.robot.joints()
        T0, T1 = so101_fk(q), so101_fk(q + np.array([1.0, 0, 0, 0, 0]))
        az0, az1 = np.arctan2(T0[1, 2], T0[0, 2]), np.arctan2(T1[1, 2], T1[0, 2])  # spoon heading
        pan_per_right = -1.0 if az1 > az0 else 1.0  # turning right = heading decreasing
        level = float((self.taught.spoon_level or {}).get("tool_pitch_deg", 0.0))
        target, rep = jog_level(q, pan_per_right * yaw, dz, level)
        if not rep["ok"]:
            raise RuntimeError(f"out of the arm's reach with a level spoon here (gets within {rep['miss_cm']} cm): try a smaller step")
        try:
            self.robot.set_speed(speed_degs)
            what = (f"yaw {'right' if yaw > 0 else 'left'} {abs(yaw):g} deg" if yaw else "") + \
                   (f"{' + ' if yaw else ''}{'up' if dz > 0 else 'down'} {abs(dz) * 100:g} cm" if dz else "")
            self.reason = f"jog: {what}"
            self._move(target)
        finally:
            self.robot.set_speed(self.cfg.joint_speed_degs)
        self.reason = (f"jogged {what}: spoon height {rep['height_cm']} cm, reach {rep['reach_cm']} cm, "
                       f"tilt {self._spoon_tilt():+.1f} deg")
        self._event("jog", self.reason)

    def _do_center(self) -> None:
        self._require_level()
        self._track_loop()

    def _do_extend(self) -> None:
        self._require_level()
        self._check_extend_reach()
        self.mode = "extend"
        self._track_loop(extend=True)

    def _check_extend_reach(self, face=None, q=None) -> None:
        """How far the arm can move in from here (caps the approach) and how far it needs to.
        Reports only: Extend then goes as far as it can and holds there."""
        if face is None:
            st = self.tracker.state(self.robot.now())
            face = None if (st is None or st.age > 2.0) else np.asarray(st.position, float)
        q = self.robot.joints() if q is None else np.asarray(q, float)
        if face is None:
            self._ext_max_travel = 9.9  # no fix right now: the loop waits for the face anyway
            self._event("extend", "check skipped: no face fix yet (the approach waits for one)")
            return
        T = so101_fk(q)
        level = float((self.taught.spoon_level or {}).get("tool_pitch_deg", 0.0))
        if self.ext_cfg.close_fwd is not None:  # taught close pose: distance to go along the spoon
            probe = LevelTracker(self.track_cfg)
            probe.reset(q)
            tip = probe.remaining(face, self.ext_cfg)
            need = tip - self.ext_cfg.close_margin
        else:
            tip = float(np.linalg.norm(face - T[:3, 3])) - self.ext_cfg.spoon_tip_from_tool
            need = tip - self.ext_cfg.stop_tip_distance
        avail = max_level_reach(q, level, self.track_cfg) - float(np.linalg.norm(T[:3, 3][:2]))
        what = "spoon tip to lips"
        self._event("extend", f"check: {tip * 100:.0f} cm {what}, needs {need * 100:.0f} cm "
                              f"more reach, the arm has {avail * 100:.0f} cm at this height")
        if need > avail + 0.01:
            self._event("extend", f"note: the person is {(need - avail) * 100:.0f} cm beyond the arm's reach here: "
                                  f"the spoon will stop {(need - avail) * 100:.0f} cm short")
        self._ext_max_travel = max(avail, 0.0) + 0.005

    # ---------------------------------------------------------------- one-button feed
    FEED_GLIDE_DEGS = 60.0  # deg/s for the big joint moves (home, sweep), streamed and synchronised

    def _glide(self, target, speed_degs: float | None = None, what: str = "") -> None:
        """Smooth synchronised joint move: every joint starts and arrives together (cosine
        profile), streamed at the tracker rate. Keeps the spoon level between two level poses
        (and, for a pan-only sweep, exactly level the whole way)."""
        q0 = np.asarray(self.robot.joints(), float)
        q1 = np.asarray(target, float)
        span = float(np.max(np.abs(q1 - q0)))
        if span < 0.3:
            return
        rate = self.track_cfg.rate_hz
        dur = max(0.6, span / (speed_degs or self.FEED_GLIDE_DEGS) * 1.6)  # cosine peak = 1.57 x mean speed
        n = int(dur * rate)
        t0 = time.monotonic()
        for i in range(1, n + 1):
            self._check_stop()
            s = 0.5 - 0.5 * np.cos(np.pi * i / n)
            self.robot.stream_joints(q0 + s * (q1 - q0))
            if what:
                self.reason = f"{what} ({100 * i // n}%)"
            pause = t0 + i / rate - time.monotonic()
            if pause > 0:
                time.sleep(pause)
        deadline = time.monotonic() + 2.0  # let the servos arrive
        while time.monotonic() < deadline:
            self._check_stop()
            if float(np.max(np.abs(np.asarray(self.robot.joints(timeout=1.0), float) - q1))) < 3.0:
                break
            time.sleep(0.05)
        self.last_move_end = self.robot.now()

    def _person_pan(self) -> float:
        """Where the person usually is: the taught close pose's heading, else the look pose's."""
        if self.taught.close:
            return float(self.taught.close["joints"][PAN])
        self._need("look")
        return float(self.taught.look[PAN])

    def _do_feed(self) -> None:
        """One bite: home -> sweep to where the person usually sits -> find the face -> centre
        and extend (stop 3 cm from the lips, hold 10 s) -> back out -> home. STOP at any time
        leaves the arm where it is."""
        home = self.home_pose()
        self._event("feed", "going home")
        self._glide(home, what="feed: going home")
        sweep = home.copy()
        sweep[PAN] = self._person_pan()  # pan only: the spoon's tilt can't change on the way
        self._event("feed", f"sweeping to the person (pan {sweep[PAN]:.0f} deg)")
        self._glide(sweep, what="feed: sweeping to the person")
        face, _ = self._face_target(need=1, timeout=1.5)
        for d in (12.0, -12.0, 24.0, -24.0):  # not in view: look around the usual spot (pan only)
            if face is not None:
                break
            q = sweep.copy()
            q[PAN] = float(np.clip(sweep[PAN] + d, -108.0, 108.0))
            self._glide(q, what=f"feed: looking for the face (pan {q[PAN]:.0f} deg)")
            face, _ = self._face_target(need=1, timeout=1.0)
        if face is None:
            self._event("feed", "no face found: going home")
            self._glide(home, what="feed: no face, going home")
            self.reason = "feed: no face found near the person's usual spot (back home)"
            return
        self._require_level()
        self._check_extend_reach()
        self.mode = "extend"
        self._track_loop(extend=True, once=True)
        self.mode = "feed"
        self._event("feed", "bite done: going home")
        self._glide(home, what="feed: going home")
        self.reason = "feed done: back home"
        self._event("feed", self.reason)

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
        self._require_level()
        self.mode = "center"
        self._track_loop()

    def _do_retreat(self) -> None:
        if self._ext_reach_start is not None:  # after a (stopped) smooth extend: glide back out, level
            self.mode = "extend"
            self._track_loop(retreat_only=True)
            return
        self._do_retreat_planned()

    def _do_retreat_planned(self) -> None:
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
            "taught": {"home": self.taught.home, "look": self.taught.look, "near": self.taught.near, "calibrated": self.taught.jacobian is not None,
                       "close": self.taught.close, "close_margin_cm": round(self.ext_cfg.close_margin * 100, 1)},
            "track": ({**self.track.last, "travelled_cm": round(self.track.travelled * 100, 1),
                       "sag_deg": _r(getattr(self, "_sag", np.zeros(5)), 1)}
                      if (self.track is not None and self.mode in ("center", "extend")) else None),
            "servo": None if self.last_cmd is None else {
                **getattr(self, "_aim_info", {}),
                "aim_error_deg": round(self.last_cmd.angle_deg, 1),
                "tip_distance_cm": round(self.last_cmd.tip_distance * 100, 1),
                "travelled_cm": round((self.pointer.travelled if self.pointer else 0.0) * 100, 1)},
            "params": {"stop_tip_distance_cm": round(self.ext_cfg.stop_tip_distance * 100, 1),
                       "floor_cm": round(self.cfg.min_tip_distance_floor * 100, 1),
                       "joint_speed_degs": self.cfg.joint_speed_degs},
            "face": face, "log": list(self.log)[:60], "allow_extend": self.allow_extend, "tuning": self.tuning(),
            "depth": {"status": self.depth_status, **self.depth.last,
                      "eye_span_mm": round(self.depth.eye_span_m * 1000, 1), "samples": len(self.depth.samples)},
            "level": {"saved_tool_pitch_deg": (self.taught.spoon_level or {}).get("tool_pitch_deg"),
                      "spoon_tilt_now_deg": None if isinstance(joints, str) else round(
                          _tool_pitch(np.array(self._last_joints)) - float((self.taught.spoon_level or {}).get("tool_pitch_deg", 0.0)), 1),
                      "hand_guide": self.hand_guide},
            "planner": {"failures_in_a_row": self.plan_failures, "last_error": self.last_plan_error[:300],
                        "log_file": str(self.plan_log) if self.plan_log else None},
        }
