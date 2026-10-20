"""MuJoCo implementation of RobotInterface.

* Motion: Cartesian interpolation at 50 Hz with damped-least-squares IK,
  planned as a dry run before anything moves, executed on joint position
  actuators (the real xArm is also position-controlled). ``move_linear`` is
  always a straight line; ``move_to_pose`` falls back to a joint-space move
  (IK with restarts) when the straight line hits a joint limit, standing in
  for Viam's motion planner.
* Force: a 3-axis force sensor between the flange and the gripper, standing
  in for the UFACTORY F/T sensor. Guarded moves stop on it.
* Cameras: "wrist" (on the gripper) and "scene" (fixed on a post at the table),
  rendered as RGB + metric depth + geom segmentation (the segmentation feeds
  the oracle detector only).
* World: scene-specific behaviour (scripted head motion, the jaw, bite
  detection) plugs in through ``hooks``: callables run every 20 ms tick.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable

import numpy as np

from ..geometry import Pose, interpolate_poses
from ..vision.detectors import SegmentationOracleDetector
from ..vision.types import CameraIntrinsics, CameraSpec, RGBDFrame
from .base import GripperState, MoveResult, RobotInterface, Wrench

REPO_ROOT = Path(__file__).resolve().parents[3]
SCENES = REPO_ROOT / "sim" / "scenes"
ARM_JOINTS = [f"joint{i}" for i in range(1, 7)]
# Ready pose: tool pointing forward, TCP ~(0.38, 0, 0.30) m in the base frame,
# elbow up and tucked so the arm stays out of the user's lap and line of sight.
HOME_Q = np.array([0.0, -0.186, 0.081, 0.0, -1.466, 0.0])
CONTROL_DT = 0.02

# Geoms whose names start with these belong to the person; any arm contact with
# them latches a stop (the sim's stand-in for the arm's collision detection).
USER_PREFIXES = ("user_", "head_", "face", "lip_", "jaw", "chin")


def oracle_label(geom_name: str) -> str:
    """'grape_2' -> 'grape', 'melon_cube_1' -> 'melon cube', 'mouth' -> 'mouth'."""
    return re.sub(r"_\d+$", "", geom_name).replace("_", " ")


def scene_path(name: str) -> Path:
    p = Path(name)
    return p if p.suffix == ".xml" else SCENES / f"{name}.xml"


class SimRobot(RobotInterface):
    def __init__(self, scene: str = "feeding", image_size=(320, 240), depth_noise: float = 0.0,
                 seed: int = 0, viewer: bool = False, hooks: list[Callable[["SimRobot"], None]] | None = None):
        import mujoco

        self.mj = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(scene_path(scene)))
        self.data = mujoco.MjData(self.model)
        self._ik = mujoco.MjData(self.model)
        m = self.model
        jid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in ARM_JOINTS]
        self.qadr = np.array([m.jnt_qposadr[j] for j in jid])
        self.dadr = np.array([m.jnt_dofadr[j] for j in jid])
        self.jrange = m.jnt_range[jid].copy()
        self.act = np.array([mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, j) for j in ARM_JOINTS])
        self.act_grip = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper")
        self.q_finger = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "finger_l")]
        self.tcp = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "tcp")
        self.ft_site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "ft_site")
        self.base_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "arm_base")
        s = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "ft_force")
        self.ft_adr = m.sensor_adr[s]
        self.n_substeps = max(1, int(round(CONTROL_DT / m.opt.timestep)))
        self.hooks = list(hooks or [])
        self.depth_noise = depth_noise
        self.rng = np.random.default_rng(seed)
        self.w, self.h = image_size
        self._renderer = None
        self._tare = np.zeros(3)
        self._stopped = False
        self.events: set[str] = set()
        self.user_geoms = {i for i in range(m.ngeom) if (self._geom_name(i) or "").startswith(USER_PREFIXES)}
        body_ids = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b)
                    for b in ("link1", "link2", "link3", "link4", "link5", "link6", "gripper", "finger_l", "finger_r")}
        self.arm_geoms = {i for i in range(m.ngeom) if m.geom_bodyid[i] in body_ids}
        self.viewer = None
        self.reset()
        if viewer:
            import mujoco.viewer

            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

    # setup ---------------------------------------------------------------------------
    def _geom_name(self, i):
        return self.mj.mj_id2name(self.model, self.mj.mjtObj.mjOBJ_GEOM, i)

    def _joint_q(self, name):
        j = self.mj.mj_name2id(self.model, self.mj.mjtObj.mjOBJ_JOINT, name)
        return None if j < 0 else float(self.data.qpos[self.model.jnt_qposadr[j]])

    def reset(self, q=HOME_Q):
        self.mj.mj_resetData(self.model, self.data)
        self.data.qpos[self.qadr] = q
        self.data.ctrl[self.act] = q
        self.mj.mj_forward(self.model, self.data)
        self._stopped = False
        self.events.clear()
        self._step(0.3)

    # simulation loop -------------------------------------------------------------------
    def _scene_hooks(self):
        for hook in self.hooks:
            hook(self)

    def _user_contact(self) -> bool:
        d = self.data
        for i in range(d.ncon):
            g1, g2 = d.contact[i].geom1, d.contact[i].geom2
            if (g1 in self.user_geoms and g2 in self.arm_geoms) or (g2 in self.user_geoms and g1 in self.arm_geoms):
                return True
        return False

    def _tick(self):
        self._scene_hooks()
        self.mj.mj_step(self.model, self.data, nstep=self.n_substeps)
        if self._user_contact():
            # Like the real arm's collision detection: stop, and stay stopped until reset().
            self.events.add("user_contact")
            self._stopped = True
        if self.viewer is not None:
            self.viewer.sync()

    def _step(self, seconds: float):
        for _ in range(max(1, int(seconds / CONTROL_DT))):
            self._tick()

    # kinematics ------------------------------------------------------------------------
    def _base_world(self) -> Pose:
        return Pose.from_matrix(self.data.xpos[self.base_body], self.data.xmat[self.base_body].reshape(3, 3))

    def get_tcp_pose(self) -> Pose:
        w = Pose.from_matrix(self.data.site_xpos[self.tcp], self.data.site_xmat[self.tcp].reshape(3, 3))
        return self._base_world().inverse() @ w

    def joint_positions(self) -> np.ndarray:
        return self.data.qpos[self.qadr].copy()

    def ik(self, target: Pose, q_seed=None, iters: int = 100, tol=(5e-4, 5e-3)) -> tuple[np.ndarray, bool]:
        """Damped least squares on the TCP site; target in the base frame."""
        mj, m, d = self.mj, self.model, self._ik
        d.qpos[:] = self.data.qpos
        d.qpos[self.qadr] = self.data.qpos[self.qadr] if q_seed is None else q_seed
        tw = self._base_world() @ target
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        for _ in range(iters):
            mj.mj_kinematics(m, d)
            mj.mj_comPos(m, d)
            pos = d.site_xpos[self.tcp]
            rot = d.site_xmat[self.tcp].reshape(3, 3)
            ep = tw.position - pos
            r_err = tw.rot @ rot.T
            ang = np.arccos(np.clip((np.trace(r_err) - 1) / 2, -1, 1))
            if ang < 1e-9:
                er = np.zeros(3)
            else:
                er = ang / (2 * np.sin(ang)) * np.array([r_err[2, 1] - r_err[1, 2], r_err[0, 2] - r_err[2, 0], r_err[1, 0] - r_err[0, 1]])
            if np.linalg.norm(ep) < tol[0] and np.linalg.norm(er) < tol[1]:
                return d.qpos[self.qadr].copy(), True
            mj.mj_jacSite(m, d, jp, jr, self.tcp)
            j = np.vstack([jp[:, self.dadr], jr[:, self.dadr]])
            e = np.concatenate([ep, 0.5 * er])
            j[3:] *= 0.5
            lam = 0.02  # damped least squares
            dq = j.T @ np.linalg.solve(j @ j.T + lam**2 * np.eye(6), e)
            dq = np.clip(dq, -0.3, 0.3)
            d.qpos[self.qadr] = np.clip(d.qpos[self.qadr] + dq, self.jrange[:, 0], self.jrange[:, 1])
        return d.qpos[self.qadr].copy(), False

    # RobotInterface: motion -----------------------------------------------------------------
    def _raw_force(self) -> np.ndarray:
        r_site = self.data.site_xmat[self.ft_site].reshape(3, 3)
        f_world = r_site @ self.data.sensordata[self.ft_adr:self.ft_adr + 3]
        # MuJoCo reports the force the child (gripper) receives from the parent; the
        # external load on the tool is the opposite of that.
        return -(self._base_world().rot.T @ f_world)

    def _plan_cartesian(self, start: Pose, target: Pose, speed: float, q0) -> list[np.ndarray] | None:
        """Dry-run IK along the straight line; None if any waypoint is unreachable."""
        dist = float(np.linalg.norm(target.position - start.position))
        ang = start.distance_to(target)[1]
        n = max(1, int(np.ceil(max(dist / (speed * CONTROL_DT), ang / (0.8 * CONTROL_DT)))))
        qs, q = [], q0
        for wp in interpolate_poses(start, target, n):
            q, ok = self.ik(wp, q)
            if not ok:
                q, ok = self.ik(wp, q, iters=300)
            if not ok or np.max(np.abs(q - (qs[-1] if qs else q0))) > 0.3:  # no flips mid-line
                return None
            qs.append(q)
        return qs

    def _path_hits_user(self, qs, clearance: float = 0.04, every: int = 2) -> bool:
        """Dry-run a joint path: does any arm geom come within ``clearance`` of the
        seated user? The margin covers tracking lag between plan and execution."""
        mj, m, d = self.mj, self.model, self._ik
        fromto = np.zeros(6)
        for q in list(qs[::every]) + [qs[-1]]:
            d.qpos[:] = self.data.qpos
            d.qpos[self.qadr] = q
            mj.mj_kinematics(m, d)
            for g_arm in self.arm_geoms:
                for g_user in self.user_geoms:
                    if mj.mj_geomDistance(m, d, g_arm, g_user, clearance, fromto) < clearance:
                        return True
        return False

    def _plan_joint(self, target: Pose, speed: float, q0) -> list[np.ndarray] | None:
        """Free-space fallback (what a motion planner would give): IK with restarts,
        then a straight joint-space path, rejected if any of it touches the user.
        Prefers solutions near the current one and away from joint limits.
        The TCP path is NOT a straight line."""
        seeds = [q0, HOME_Q] + [self.rng.uniform(self.jrange[:, 0], self.jrange[:, 1]) for _ in range(40)]
        candidates = []
        for sd in seeds:
            q, ok = self.ik(target, sd, iters=300)
            if not ok:
                continue
            margin = np.minimum(q - self.jrange[:, 0], self.jrange[:, 1] - q)
            candidates.append((np.linalg.norm(q - q0) + 2.0 * np.sum(np.clip(0.4 - margin, 0, None)), q))
        dist = float(np.linalg.norm(target.position - self.get_tcp_pose().position))

        def segment(a, b):
            n = max(1, int(np.ceil(np.max(np.abs(b - a)) / (0.6 * CONTROL_DT))), int(np.ceil(dist / (speed * CONTROL_DT))))
            return [a + (b - a) * (i / n) for i in range(1, n + 1)]

        ranked = [q for _, q in sorted(candidates, key=lambda c: c[0])]
        for best in ranked:  # straight joint-space line
            qs = segment(q0, best)
            if not self._path_hits_user(qs):
                return qs
        for best in ranked:  # via the tucked ready pose, a known-safe transit configuration
            qs = segment(q0, HOME_Q) + segment(HOME_Q, best)
            if not self._path_hits_user(qs):
                return qs
        return None

    def _execute(self, qs, target: Pose, max_force: float | None) -> MoveResult:
        start = self.get_tcp_pose()
        peak = 0.0
        for q in qs:
            self.data.ctrl[self.act] = q
            self._tick()
            f = float(np.linalg.norm(self._raw_force() - self._tare))
            peak = max(peak, f)
            if self._stopped:
                self._hold()
                return MoveResult(False, "collision" if "user_contact" in self.events else "stopped",
                                  self.get_tcp_pose(), peak, self._travel(start))
            if max_force is not None and f > max_force:
                self._hold()
                return MoveResult(False, "force_limit", self.get_tcp_pose(), peak, self._travel(start))
        for _ in range(25):  # settle on the final target
            if self.get_tcp_pose().distance_to(target)[0] < 0.002:
                break
            self._tick()
            if max_force is not None:
                f = float(np.linalg.norm(self._raw_force() - self._tare))
                peak = max(peak, f)
                if f > max_force:
                    self._hold()
                    return MoveResult(False, "force_limit", self.get_tcp_pose(), peak, self._travel(start))
        err = self.get_tcp_pose().distance_to(target)[0]
        return MoveResult(err < 0.01, "" if err < 0.01 else f"tracking error {err:.3f} m",
                          self.get_tcp_pose(), peak, self._travel(start))

    def _cartesian(self, target: Pose, speed: float, max_force: float | None, allow_joint=False) -> MoveResult:
        if self._stopped:
            return MoveResult(False, "stopped", self.get_tcp_pose())
        q0 = self.data.ctrl[self.act].copy()
        qs = self._plan_cartesian(self.get_tcp_pose(), target, speed, q0)
        if qs is not None and self._path_hits_user(qs):
            qs = None
        if qs is None and allow_joint:
            qs = self._plan_joint(target, speed, q0)
        if qs is None:
            return MoveResult(False, "unreachable", self.get_tcp_pose())
        return self._execute(qs, target, max_force)

    def _travel(self, start: Pose) -> float:
        return float(np.linalg.norm(self.get_tcp_pose().position - start.position))

    def _hold(self):
        """Freeze where we are (back off the commanded penetration)."""
        self.data.ctrl[self.act] = self.data.qpos[self.qadr]
        self._step(0.06)

    def move_to_pose(self, pose: Pose, speed: float = 0.10) -> MoveResult:
        return self._cartesian(pose, speed, None, allow_joint=True)

    def move_linear(self, pose: Pose, speed: float = 0.03, max_force: float | None = None) -> MoveResult:
        return self._cartesian(pose, speed, max_force)

    # gripper ---------------------------------------------------------------------------
    def _gripper_state(self, force=0.0) -> GripperState:
        width = self.gripper_max_width - 2 * float(self.data.qpos[self.q_finger])
        return GripperState(width, 0.004 < width < self.gripper_max_width - 0.004, force)

    def gripper_state(self) -> GripperState:
        return self._gripper_state()

    def open_gripper(self, width: float | None = None) -> GripperState:
        w = self.gripper_max_width if width is None else width
        self.data.ctrl[self.act_grip] = (self.gripper_max_width - w) / 2
        self._step(0.5)
        return self._gripper_state()

    def close_gripper(self, force: float = 20.0, speed: float = 0.05) -> GripperState:
        """Ramp the jaws shut at ``speed`` m/s per finger (the real gripper has a speed
        setting too); slamming them knocks free-standing objects away."""
        target = self.gripper_max_width / 2
        c = float(self.data.ctrl[self.act_grip])
        while c < target:
            c = min(target, c + speed * CONTROL_DT)
            self.data.ctrl[self.act_grip] = c
            self._tick()
        self._step(0.3)
        return self._gripper_state(force)

    # sensing ---------------------------------------------------------------------------
    def get_wrench(self) -> Wrench:
        return Wrench(self._raw_force() - self._tare)

    def tare_wrench(self) -> None:
        self._step(0.1)
        self._tare = self._raw_force()

    def cameras(self) -> dict[str, CameraSpec]:
        out = {}
        for name, mount in (("wrist", "wrist"), ("scene", "base")):
            cid = self.mj.mj_name2id(self.model, self.mj.mjtObj.mjOBJ_CAMERA, name)
            if cid >= 0:
                out[name] = CameraSpec(name, mount, CameraIntrinsics.from_fovy(self.model.cam_fovy[cid], self.w, self.h))
        return out

    def _render(self):
        if self._renderer is None:
            self._renderer = self.mj.Renderer(self.model, self.h, self.w)
        return self._renderer

    def get_frames(self, names=None) -> dict[str, RGBDFrame]:
        r = self._render()
        specs = self.cameras()
        out = {}
        for name in names or list(specs):
            r.update_scene(self.data, camera=name)
            rgb = r.render().copy()
            r.enable_depth_rendering()
            r.update_scene(self.data, camera=name)
            depth = r.render().astype(np.float32).copy()
            r.disable_depth_rendering()
            r.enable_segmentation_rendering()
            r.update_scene(self.data, camera=name)
            seg_raw = r.render().copy()
            r.disable_segmentation_rendering()
            seg = np.where(seg_raw[..., 1] == int(self.mj.mjtObj.mjOBJ_GEOM), seg_raw[..., 0], -1)
            if self.depth_noise > 0:  # D435-like: noise grows with z^2
                depth = depth + self.rng.normal(0, 1, depth.shape).astype(np.float32) * self.depth_noise * depth**2
            out[name] = RGBDFrame(name, rgb, depth, specs[name].intrinsics, seg, float(self.data.time))
        return out

    def camera_pose(self, name: str) -> Pose:
        cid = self.mj.mj_name2id(self.model, self.mj.mjtObj.mjOBJ_CAMERA, name)
        r_mj = self.data.cam_xmat[cid].reshape(3, 3)
        cam_world = Pose.from_matrix(self.data.cam_xpos[cid], r_mj @ np.diag([1.0, -1.0, -1.0]))  # -> OpenCV axes
        return self._base_world().inverse() @ cam_world

    def stop(self) -> None:
        self._stopped = True
        self._hold()

    def now(self) -> float:
        return float(self.data.time)

    def wait(self, seconds: float) -> None:
        self._step(seconds)

    # sim-only helpers ------------------------------------------------------------------
    def oracle_detector(self) -> SegmentationOracleDetector:
        """Ground-truth instance masks: one instance per body, labelled by ``oracle_label``."""
        instances: dict[tuple[str, int], list[int]] = {}
        for i in range(self.model.ngeom):
            n = self._geom_name(i)
            if n:
                instances.setdefault((oracle_label(n), int(self.model.geom_bodyid[i])), []).append(i)
        out: dict[str, list[list[int]]] = {}
        for (label, _), geoms in sorted(instances.items()):
            out.setdefault(label, []).append(geoms)
        return SegmentationOracleDetector(out)

    def body_pose(self, name: str) -> Pose:
        b = self.mj.mj_name2id(self.model, self.mj.mjtObj.mjOBJ_BODY, name)
        w = Pose.from_matrix(self.data.xpos[b], self.data.xmat[b].reshape(3, 3))
        return self._base_world().inverse() @ w

    def geom_position(self, name: str) -> np.ndarray:
        g = self.mj.mj_name2id(self.model, self.mj.mjtObj.mjOBJ_GEOM, name)
        return self._base_world().inverse().transform_point(self.data.geom_xpos[g])

    def scene_state(self) -> dict:
        """Every scalar joint that isn't the arm's (head slides, jaw, ...)."""
        out = {}
        for j in range(self.model.njnt):
            if int(self.model.jnt_type[j]) in (int(self.mj.mjtJoint.mjJNT_SLIDE), int(self.mj.mjtJoint.mjJNT_HINGE)):
                name = self.mj.mj_id2name(self.model, self.mj.mjtObj.mjOBJ_JOINT, j)
                if name and name not in ARM_JOINTS and not name.startswith("finger"):
                    out[name] = float(self.data.qpos[self.model.jnt_qposadr[j]])
        return out

    def site_pose(self, name: str) -> Pose:
        sid = self.mj.mj_name2id(self.model, self.mj.mjtObj.mjOBJ_SITE, name)
        w = Pose.from_matrix(self.data.site_xpos[sid], self.data.site_xmat[sid].reshape(3, 3))
        return self._base_world().inverse() @ w

    def actuator(self, name: str) -> int:
        return self.mj.mj_name2id(self.model, self.mj.mjtObj.mjOBJ_ACTUATOR, name)

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        if self.viewer is not None:
            self.viewer.close()
