"""Scripted spoon scoop for the SO-101 bench rig (spoon taped in the fixed jaw), pure numpy.

The motion, in the vertical plane through the base axis and the bowl (the SO-101 has no wrist
yaw, so the spoon always points along that plane, away from the base):

  1. over   tip above the bowl's near side, ``hover`` above the rim, spoon level (or already
            partly tipped where a level spoon can't get there)
  2. dip    down to ``bottom_clearance`` above the bottom, turning tip-down by ``dip_deg``: first
            the turn then down, or down to the rim then the turn, or both at once, whichever
            the arm can follow (it tips a long spoon steeply only low down, and holds it level
            only higher up). The dip point is as far to the near side as the handle allows.
  3. scoop  the spoon turns back up to level (+ ``end_tip_up_deg``) while the tip slides along
            the bottom toward the far wall. Tip-down, the spoon's hollow faces the far wall, so
            turning back up drives the hollow through the food. The tip rises only as much as
            the handle needs to clear the near rim.
  4. lift   straight up to ``hover`` above the rim, spoon (nearly) level
  5. back   to the joints the scoop started from

Every sample is solved with local IK (pan, lift, elbow, flex; wrist roll held, so the hollow
stays the side that was levelled) and checked BEFORE anything moves: IK within ``ik_tol_m`` /
``ik_tol_deg``, inside the joint limits, no joint jumps, and the spoon (tip to tool, plus the
jaw under the tool) inside the bowl wherever it is below the rim and clear of the table
elsewhere. A plan that fails any check is refused with a reason; nothing is ever clipped.

Then the plan is timed: each phase starts and ends at rest (cosine profile), with tip speed,
spoon rotation rate and joint speed capped. The result is setpoints at ``rate_hz`` for a
streamed joint executor (``SO101ViamRobot.stream_joints``, wired in orchestration).

The bowl is plain geometry (``BowlTarget``): a taught spoon-tip position (hand-guide the tip
to the inside bottom centre) or a ``food_perception`` bowl (rim centre minus depth), converted
by orchestration, so this module depends on neither.

TODO(grasping): "spoon loaded?" check after the lift (VLM on the gripper camera, or the food
level re-detected); bowl depth and bottom radius per prop (docs/PROPS.md); bottom contact from
the servos' load reading (the SO-101 has no F/T sensor, so the plan never touches the bottom).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from ..robot_iface.so101_kinematics import fk as so101_fk
from ..robot_iface.so101_kinematics import joint_limits, load

ROLL = 4
ARM_JOINTS = (0, 1, 2, 3)  # pan, lift, elbow, flex: the IK's unknowns (roll held)


@dataclass(frozen=True)
class BowlTarget:
    """A bowl, base frame (m). ``bottom`` is the INSIDE bottom centre (where a hand-guided spoon
    tip touches); the rim is ``depth`` above it."""

    bottom: np.ndarray
    rim_radius: float  # inside radius at the rim
    depth: float  # rim height above the inside bottom
    bottom_radius: float | None = None  # inside radius at the bottom (None: 0.6 x rim, a typical bowl)

    @property
    def r_bottom(self) -> float:
        return 0.6 * self.rim_radius if self.bottom_radius is None else self.bottom_radius

    def radius_at(self, h: float) -> float:
        """Inside radius at height h above the bottom (straight-sided cone)."""
        return self.r_bottom + (self.rim_radius - self.r_bottom) * float(np.clip(h / self.depth, 0.0, 1.0))


@dataclass(frozen=True)
class ScoopConfig:
    # spoon (tool frame: +Z out along the spoon, +X down when level)
    spoon_tip_from_tool: float = 0.133  # m, as feeding.ExtendConfig (from the Viam scene)
    spoon_half_width: float = 0.018  # m, half the spoon bowl's width
    spoon_bowl_length: float = 0.05  # m: the spoon bowl's length back from the tip (wider than the handle)
    handle_half_width: float = 0.006  # m
    jaw_below_tool: float = 0.02  # m: the gripper's lowest point under the tool frame (checked too)
    # the motion
    dip_deg: float = 90.0  # spoon tips down this far from level before the dip ("rotate ~90 deg")
    min_dip_deg: float = 30.0  # out of reach at dip_deg: try less, in steps of dip_step_deg, down to this
    dip_step_deg: float = 5.0
    end_tip_up_deg: float = 3.0  # spoon ends this much tip-up (food slides back, not off); keep < 5
    dip_routes: tuple = ("turn_then_down", "down_then_turn", "turn_while_down")  # tried in order
    dip_route: str = "turn_then_down"  # the one being planned (set by plan_scoop)
    entry_frac: float = 0.5  # tip enters at most this fraction of the bottom radius toward the near side
    entry_slack: float = 0.005  # m further in than the handle's rim clearance strictly needs
    min_scoop_length: float = 0.03  # m of bottom the tip must sweep
    exit_frac: float = 0.75  # scoop ends with the tip at this fraction of the rim radius, far side
    bottom_clearance: float = 0.006  # m above the bottom: no force sensing, so never press on it
    rim_clearance: float = 0.015  # m: anything over the wall stays this far above the rim
    sweep_height_margin: float = 0.003  # m added to the lowest clear tip height while scooping
    wall_margin: float = 0.006  # m kept from the inside wall (on top of the spoon's half width)
    wall_thickness: float = 0.006  # m
    table_z: float = 0.0  # m, table top (base frame)
    table_clearance: float = 0.015  # m
    hover: float = 0.05  # m above the rim for the over / lift poses
    # timing (each phase starts and ends at rest)
    rate_hz: float = 30.0  # streamed setpoints per second
    max_joint_speed_deg: float = 25.0  # deg/s, any joint (peak)
    air_speed: float = 0.06  # m/s tip speed (peak) moving over and back
    dip_speed: float = 0.02  # m/s tip speed (peak) going down into the bowl and up out of it
    scoop_speed: float = 0.02  # m/s tip speed (peak) along the bottom
    max_rotate_deg_s: float = 30.0  # deg/s spoon rotation (peak)
    # planning
    step_m: float = 0.003  # path sample spacing (the long lever: 1 deg of spoon tilt ~ 4 deg of wrist)
    step_deg: float = 0.5
    sweep_samples: int = 120
    ik_tol_m: float = 0.002
    ik_tol_deg: float = 1.0
    max_jump_deg: float = 6.0  # between neighbouring samples
    max_split: int = 3  # halve a step up to this many times before calling a jump a branch flip
    limit_margin_deg: float = 3.0
    # Elbow kept bent the way it starts (deg, SO-101 elbow_flex): crossing straight (0) mid-scoop
    # flips the whole arm through a large swing while the spoon is in the food.
    elbow_min_deg: float = 8.0


@dataclass
class ScoopPlan:
    ok: bool
    reason: str
    t: np.ndarray = field(default_factory=lambda: np.zeros(0))  # (N,) s
    q: np.ndarray = field(default_factory=lambda: np.zeros((0, 5)))  # (N, 5) deg, setpoints at rate_hz
    phases: list = field(default_factory=list)  # [(name, t_start, t_end)]
    tip: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))  # (N, 3) spoon tip, m
    tilt_deg: np.ndarray = field(default_factory=lambda: np.zeros(0))  # (N,) spoon pitch from level, + = tip up
    report: dict = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return float(self.t[-1]) if len(self.t) else 0.0


class _FastFK:
    """``so101_kinematics.fk`` with the joint rotations precomputed (Rodrigues), ~4x faster:
    the planner calls FK some 10^5 times."""

    def __init__(self):
        self.ops = []
        for op in load():
            if op[0] == "fixed":
                self.ops.append((op[1][:3, :3].copy(), op[1][:3, 3] / 1000.0))
            else:
                a = np.asarray(op[1], float) / np.linalg.norm(op[1])
                K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
                self.ops.append((K, np.outer(a, a)))
        self.fixed = [len(o[1].shape) == 1 for o in self.ops]

    def __call__(self, q) -> np.ndarray:
        R, p, i = np.eye(3), np.zeros(3), 0
        I = np.eye(3)
        for (A, B), fixed in zip(self.ops, self.fixed):
            if fixed:
                p = p + R @ B
                R = R @ A
            else:
                t = np.radians(q[i]); i += 1
                c, s = np.cos(t), np.sin(t)
                R = R @ (c * I + s * A + (1 - c) * B)
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R, p
        return T


_fast_fk = None


def _fk_for(fk):
    global _fast_fk
    if fk is not so101_fk:
        return fk
    if _fast_fk is None:
        _fast_fk = _FastFK()
    return _fast_fk


def _unit(v) -> np.ndarray:
    v = np.asarray(v, float)
    return v / (np.linalg.norm(v) or 1.0)


class _Spoon:
    """Spoon tip and in-plane pitch from joints. Pitch is measured in the vertical plane along
    ``u`` (horizontal, base axis -> bowl), so it runs smoothly past straight down."""

    def __init__(self, c: ScoopConfig, u, fk=so101_fk):
        self.c, self.u, self.fk = c, np.asarray(u, float), fk

    def task(self, q) -> np.ndarray:
        T = self.fk(q)
        z = T[:3, 2]
        tip = T[:3, 3] + self.c.spoon_tip_from_tool * z
        return np.r_[tip, np.arctan2(z[2], float(np.dot(z[:2], self.u[:2])))]

    def points(self, q) -> tuple[np.ndarray, np.ndarray]:
        """Points from the tip along the handle to the tool frame, plus the jaw under the tool
        (base frame), and each one's half width across the spoon plane."""
        T = self.fk(q)
        z, x, o = T[:3, 2], T[:3, 0], T[:3, 3]
        back, hw = spoon_profile(self.c)
        return np.vstack([o + (self.c.spoon_tip_from_tool - back)[:, None] * z, o + self.c.jaw_below_tool * x]), \
            np.r_[hw, self.c.handle_half_width]


def spoon_profile(c: ScoopConfig, n: int = 16) -> tuple[np.ndarray, np.ndarray]:
    """Distances back from the tip (m) of the spoon's check points, and their half widths."""
    back = np.linspace(0.0, c.spoon_tip_from_tool, n)
    return back, np.where(back <= c.spoon_bowl_length, c.spoon_half_width, c.handle_half_width)


def _ik(sp: _Spoon, q0, tip, pitch, lo, hi, iters: int = 60) -> tuple[np.ndarray, float, float]:
    """Joints for (tip position, pitch rad), roll held. Damped least squares on a numeric
    Jacobian over pan, lift, elbow, flex. Returns (joints, position error m, pitch error deg)."""
    q = np.asarray(q0, float).copy()
    target = np.r_[tip, pitch]
    w = np.array([1.0, 1.0, 1.0, 0.2])  # 0.2 m per rad: 1 deg ~ 3.5 mm
    for _ in range(iters):
        f = sp.task(q)
        e = (target - f) * w
        e[3] = (np.arctan2(np.sin(target[3] - f[3]), np.cos(target[3] - f[3]))) * w[3]
        if np.max(np.abs(e)) < 2e-5:
            break
        J = np.zeros((4, 4))
        for i, j in enumerate(ARM_JOINTS):
            dq = np.zeros(5); dq[j] = 0.05
            d = sp.task(q + dq) - f
            d[3] = np.arctan2(np.sin(d[3]), np.cos(d[3]))
            J[:, i] = d * w / 0.05
        step = J.T @ np.linalg.solve(J @ J.T + 1e-7 * np.eye(4), e)
        q_new = np.clip(q[:4] + np.clip(step, -4.0, 4.0), lo[:4], hi[:4])
        if np.max(np.abs(q_new - q[:4])) < 1e-4:
            break  # stuck (a joint limit): more iterations won't help
        q[:4] = q_new
    f = sp.task(q)
    dp = float(np.degrees(abs(np.arctan2(np.sin(target[3] - f[3]), np.cos(target[3] - f[3])))))
    return q, float(np.linalg.norm(target[:3] - f[:3])), dp


def spoon_collision(points: np.ndarray, half_widths, bowl: BowlTarget, c: ScoopConfig) -> str | None:
    """Why these spoon points collide with the bowl or the table, or None if they're clear.
    The spoon lies in the vertical plane through the base axis, so a point's widest reach from
    the bowl's axis is hypot(its distance, its half width) (the bowl is round)."""
    for p, hw in zip(points, half_widths):
        d = float(np.hypot(np.linalg.norm(p[:2] - bowl.bottom[:2]), hw))
        h = float(p[2] - bowl.bottom[2])
        if d > bowl.rim_radius + c.wall_thickness + c.wall_margin:  # outside the bowl: only the table
            if p[2] < c.table_z + c.table_clearance:
                return f"table ({(p[2] - c.table_z) * 100:.1f} cm above it)"
            continue
        if h >= bowl.depth + c.rim_clearance:
            continue  # above the rim with room to spare
        if h < 0:
            return f"bowl bottom ({h * 100:.1f} cm)"
        if d > bowl.radius_at(min(h, bowl.depth)) - c.wall_margin:
            return f"bowl wall ({d * 100:.1f} cm from the axis, {h * 100:.1f} cm up)"
    return None


def _sweep_heights(bowl: BowlTarget, c: ScoopConfig, u, xs, pitches) -> np.ndarray:
    """Lowest tip height (above the bottom) at each sweep sample at which the spoon line
    clears the bowl, never going back down once it rises."""
    back, hw = spoon_profile(c)
    hs = []
    grid = np.arange(c.bottom_clearance, bowl.depth + c.rim_clearance + c.hover, 0.002)
    for x, p in zip(xs, pitches):
        d = np.r_[np.cos(p) * u[:2], np.sin(p)]
        h_ok = grid[-1]
        for h in grid:
            tip = bowl.bottom + np.r_[x * u[:2], h]
            if spoon_collision(tip - back[:, None] * d, hw, bowl, c) is None:
                h_ok = h
                break
        hs.append(h_ok)
    hs = np.asarray(hs) + c.sweep_height_margin
    k = int(np.argmin(hs))
    hs[k:] = np.maximum.accumulate(hs[k:])
    return hs


def _segment(a_tip, b_tip, a_pitch, b_pitch, c: ScoopConfig) -> tuple[np.ndarray, np.ndarray]:
    """Tip samples a -> b on a straight line, with the pitch interpolated."""
    dist = float(np.linalg.norm(b_tip - a_tip))
    n = max(2, int(np.ceil(max(dist / c.step_m, np.degrees(abs(b_pitch - a_pitch)) / c.step_deg))) + 1)
    s = np.linspace(0.0, 1.0, n)
    return a_tip + s[:, None] * (b_tip - a_tip), a_pitch + s * (b_pitch - a_pitch)


def pan_axis_xy(fk=so101_fk) -> np.ndarray:
    """Where the base pan axis meets the table plane (x, y), m: it is NOT the base frame's origin
    (about 39 mm in front of it on the SO-101), and the spoon always points away from it."""
    pts = np.array([fk(np.r_[a, -40.0, 60.0, 20.0, 0.0])[:2, 3] for a in (-30.0, 0.0, 30.0)])
    (x1, y1), (x2, y2), (x3, y3) = pts
    d = 2 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    ux = ((x1**2 + y1**2) * (y2 - y3) + (x2**2 + y2**2) * (y3 - y1) + (x3**2 + y3**2) * (y1 - y2)) / d
    uy = ((x1**2 + y1**2) * (x3 - x2) + (x2**2 + y2**2) * (x1 - x3) + (x3**2 + y3**2) * (x2 - x1)) / d
    return np.array([ux, uy])


def _joint_line(qa, qb, c: ScoopConfig) -> np.ndarray:
    n = max(2, int(np.ceil(np.max(np.abs(np.asarray(qb) - qa)) / c.step_deg)) + 1)
    return np.asarray(qa, float) + np.linspace(0.0, 1.0, n)[:, None] * (np.asarray(qb, float) - qa)


def _solve_seeded(sp: _Spoon, tip, pitch, q_near, lo, hi, c: ScoopConfig) -> np.ndarray | None:
    """IK from a few seeds (the given joints, then elbow-up postures facing the target): the
    converged solution closest to ``q_near``, or None."""
    v = np.asarray(tip[:2]) - pan_axis_xy(sp.fk)
    az = float(np.degrees(np.arctan2(v[1], v[0])))
    q_near = np.asarray(q_near, float)
    pan_sign = 1.0 if sp.task(q_near + np.r_[1.0, 0, 0, 0, 0])[1] > sp.task(q_near)[1] else -1.0
    pan = pan_sign * az  # pan that faces the target (the sign depends on the module's convention)
    seeds = [q_near] + [np.r_[pan, l, e, f, q_near[ROLL]] for l, e, f in ((-40, 40, 60), (-20, 10, 80), (-60, 70, 40))]
    best = None
    for s0 in seeds:
        q, e_pos, e_deg = _ik(sp, s0, tip, pitch, lo, hi, iters=150)
        if e_pos <= c.ik_tol_m and e_deg <= c.ik_tol_deg:
            d = float(np.max(np.abs(q - q_near)))
            if best is None or d < best[0]:
                best = (d, q)
    return None if best is None else best[1]


def plan_scoop(q_start, bowl: BowlTarget, cfg: ScoopConfig | None = None, level_pitch_deg: float = 0.0,
               fk=so101_fk, limits=None) -> ScoopPlan:
    """The whole scoop from ``q_start`` (deg) and back, checked and timed. ``level_pitch_deg``
    is the tool pitch at which the spoon is level (the dashboard's saved spoon level).

    With a long spoon the SO-101 can't tip it straight down near the table (FK: about 60-70 deg
    at most, and level only with the tip 35+ cm from the base axis). If ``dip_deg`` is out of
    reach, smaller dips are tried down to ``min_dip_deg``; ``report["dip_deg"]`` says which ran."""
    c = cfg or ScoopConfig()
    fk = _fk_for(fk)

    def attempt(dip):
        first = None
        for route in c.dip_routes:
            plan = _plan_once(q_start, bowl, replace(c, dip_deg=dip, dip_route=route), level_pitch_deg, fk, limits)
            if plan.ok:
                plan.report["dip_route"] = route
                return plan
            first = first or plan
            if "at the bottom" in plan.reason:
                break  # no route can help
        return first

    first = attempt(c.dip_deg)
    if first.ok or first.reason.startswith(("over:", "bowl too small")):
        return first
    # the deepest dip that works (not monotone: a shallow dip can foul the near rim), so walk down
    for dip in np.arange(c.dip_deg - c.dip_step_deg, c.min_dip_deg - 1e-9, -c.dip_step_deg):
        plan = attempt(float(dip))
        if plan.ok:
            plan.reason = f"ok with the spoon tipped {dip:g} deg (not {c.dip_deg:g}: out of the arm's reach)"
            return plan
    first.reason += f" (and with any dip down to {c.min_dip_deg:g} deg)"
    return first


def _plan_once(q_start, bowl: BowlTarget, c: ScoopConfig, level_pitch_deg: float, fk, limits) -> ScoopPlan:
    q_start = np.asarray(q_start, float)
    lim = joint_limits() if limits is None else np.asarray(limits, float)
    lo, hi = lim[:, 0] + c.limit_margin_deg, lim[:, 1] - c.limit_margin_deg
    lo[2] = max(lo[2], c.elbow_min_deg)
    if q_start[2] < c.elbow_min_deg:
        return ScoopPlan(False, f"start with the elbow bent (elbow flex {q_start[2]:.0f} deg < {c.elbow_min_deg:g})")
    lo, hi = np.minimum(lo, q_start), np.maximum(hi, q_start)  # a start inside the margin is not a jump
    bowl = BowlTarget(np.asarray(bowl.bottom, float), bowl.rim_radius, bowl.depth, bowl.bottom_radius)
    u = np.r_[_unit(bowl.bottom[:2] - pan_axis_xy(fk)), 0.0]  # pan axis -> bowl: the spoon points this way
    sp = _Spoon(c, u, fk)
    level = np.radians(level_pitch_deg)
    x_exit = c.exit_frac * bowl.rim_radius
    if (c.spoon_half_width > bowl.r_bottom - c.wall_margin
            or np.hypot(x_exit, c.spoon_half_width) > bowl.rim_radius - c.wall_margin):
        return ScoopPlan(False, f"bowl too small for the spoon (rim radius {bowl.rim_radius * 100:.1f} cm)")
    # dip as far to the near side as the handle allows: at this tilt it must clear the near rim
    rise = bowl.depth + c.rim_clearance - c.bottom_clearance
    x_entry = max(-c.entry_frac * bowl.r_bottom,
                  -(bowl.rim_radius - c.wall_margin) + rise / np.tan(np.radians(c.dip_deg)) + c.entry_slack)
    if x_entry > x_exit - c.min_scoop_length:
        return ScoopPlan(False, f"dip: a {c.dip_deg:g} deg spoon can't get under the rim of a "
                                f"{bowl.depth * 100:.1f} cm deep bowl and still scoop {c.min_scoop_length * 100:.0f} cm")
    top = bowl.depth + c.rim_clearance + c.hover

    def at(x, h):
        return bowl.bottom + np.r_[x * u[:2], h]

    # fail fast: the tipped spoon at the bottom is the hardest pose to reach (same for every route)
    if c.dip_route == c.dip_routes[0] and _solve_seeded(sp, at(x_entry, c.bottom_clearance),
                                                       level - np.radians(c.dip_deg), q_start, lo, hi, c) is None:
        return ScoopPlan(False, f"dip: out of the arm's reach with the spoon tipped {c.dip_deg:g} deg at the bottom")
    # 1: over: a joint-space move (in the air; checked below) to the spoon level above the entry
    # the empty spoon may already be tipped over the bowl if a level one can't get there
    for over_pitch in (level, level - np.radians(c.dip_deg) / 2, level - np.radians(c.dip_deg)):
        q_over = _solve_seeded(sp, at(x_entry, top), over_pitch, q_start, lo, hi, c)
        if q_over is not None:
            break
    if q_over is None:
        return ScoopPlan(False, f"over: the bowl at {np.round(bowl.bottom * 100, 1).tolist()} cm is out of the "
                                f"arm's reach {top * 100:.0f} cm above its bottom, level or tipped up to {c.dip_deg:g} deg")
    pts, pitches, names = [], [], []

    def add(name, tips, pp):
        pts.append(tips); pitches.append(pp); names.append(name)

    # 2-3: tip down, dip
    # 2: dip, along whichever (height, tilt) route the arm can follow at this bowl (the SO-101 can
    # tip a long spoon steeply only low down, and hold it level only higher up)
    rim = bowl.depth + c.rim_clearance
    dipped = level - np.radians(c.dip_deg)
    route = {"turn_then_down": [(top, over_pitch), (rim, dipped), (c.bottom_clearance, dipped)],
             "down_then_turn": [(top, over_pitch), (rim, over_pitch), (c.bottom_clearance, dipped)],
             "turn_while_down": [(top, over_pitch), (c.bottom_clearance, dipped)]}[c.dip_route]
    for h, p in route[1:-1]:  # fail fast on the route's corners too
        if _solve_seeded(sp, at(x_entry, h), p, q_over, lo, hi, c) is None:
            return ScoopPlan(False, f"dip: out of the arm's reach ({c.dip_route}, spoon tilt "
                                    f"{np.degrees(p - level):+.0f} deg {h * 100:.1f} cm above the bottom)")
    tips, pp = [at(x_entry, route[0][0])], [route[0][1]]
    for (ha, pa), (hb, pb) in zip(route, route[1:]):
        t_, p_ = _segment(at(x_entry, ha), at(x_entry, hb), pa, pb, c)
        tips += list(t_[1:]); pp += list(p_[1:])
    add("dip", np.array(tips), np.array(pp))
    # 4: scoop: turn back up while the tip slides along the bottom, rising only where it must
    s = np.linspace(0.0, 1.0, c.sweep_samples)
    sw_pitch = (level - np.radians(c.dip_deg)) + s * np.radians(c.dip_deg + c.end_tip_up_deg)
    sw_x = x_entry + s * (x_exit - x_entry)
    sw_h = _sweep_heights(bowl, c, u, sw_x, sw_pitch)
    sw_h[0] = c.bottom_clearance
    add("scoop", np.array([at(x, h) for x, h in zip(sw_x, sw_h)]), sw_pitch)
    # 5: lift
    end_pitch = level + np.radians(c.end_tip_up_deg)
    add("lift", *_segment(at(x_exit, sw_h[-1]), at(x_exit, top), end_pitch, end_pitch, c))

    # solve every sample in the bowl, warm-started along the path; joint-space moves over and back
    worst = {"pos_mm": 0.0, "pitch_deg": 0.0}
    over = _joint_line(q_start, q_over, c)
    qs, phase_of, q = list(over), ["over"] * len(over), q_over.copy()
    prev = sp.task(q)
    for name, tips, pp in zip(names, pts, pitches):
        for tip, p in zip(tips, pp):
            # where a joint moves fast (the long lever near a stretched arm), solve in between too;
            # a jump that survives halving the step is a branch flip
            for k in range(1, c.max_split + 2):
                n = 2 ** (k - 1)
                trial, q_try, bad = [], q.copy(), None
                for f in np.arange(1, n + 1) / n:
                    tgt_tip, tgt_p = prev[:3] + f * (tip - prev[:3]), prev[3] + f * (p - prev[3])
                    q_new, e_pos, e_deg = _ik(sp, q_try, tgt_tip, tgt_p, lo, hi)
                    worst["pos_mm"] = max(worst["pos_mm"], e_pos * 1000)
                    worst["pitch_deg"] = max(worst["pitch_deg"], e_deg)
                    if e_pos > c.ik_tol_m or e_deg > c.ik_tol_deg:
                        return ScoopPlan(False, f"{name}: out of the arm's reach (tip off by {e_pos * 1000:.0f} mm, "
                                                f"spoon off by {e_deg:.0f} deg at {np.round(tgt_tip * 100, 1).tolist()} cm)",
                                         report=worst)
                    jump = float(np.max(np.abs(q_new - q_try)))
                    if jump > c.max_jump_deg:
                        bad = jump
                        break
                    trial.append(q_new); q_try = q_new
                if bad is None:
                    break
            else:
                return ScoopPlan(False, f"{name}: joint jump of {bad:.0f} deg (IK branch flip)", report=worst)
            for q_new in trial:
                hit = spoon_collision(*sp.points(q_new), bowl, c)
                if hit is not None:
                    return ScoopPlan(False, f"{name}: the spoon would hit the {hit}", report=worst)
                qs.append(q_new); phase_of.append(name)
            q, prev = trial[-1], np.r_[tip, p]
    back = _joint_line(q, q_start, c)[1:]
    qs += list(back); phase_of += ["back"] * len(back)
    names = ["over"] + names + ["back"]
    for name, qq in zip(phase_of, qs):
        if name in ("over", "back") and (hit := spoon_collision(*sp.points(qq), bowl, c)) is not None:
            return ScoopPlan(False, f"{name}: the spoon would hit the {hit} on the way (start from a higher pose)",
                             report=worst)
    qs = np.asarray(qs)

    # time it: per-sample cost at the caps, then a cosine (rest-to-rest) profile per phase
    t_out, q_out, phases, t0 = [np.zeros(1)], [qs[:1]], [], 0.0
    names_arr = np.asarray(phase_of)
    tasks = np.array([sp.task(qq) for qq in qs])
    speed = {"over": c.air_speed, "back": c.air_speed, "dip": c.dip_speed,
             "lift": c.dip_speed, "scoop": c.scoop_speed}
    i0 = 0
    for name in names:
        idx = np.where(names_arr == name)[0]
        idx = np.r_[i0, idx] if idx[0] != i0 else idx
        seg_q, seg_task = qs[idx], tasks[idx]
        dq = np.max(np.abs(np.diff(seg_q, axis=0)), axis=1) / c.max_joint_speed_deg
        dtip = np.linalg.norm(np.diff(seg_task[:, :3], axis=0), axis=1) / speed[name]
        drot = np.degrees(np.abs(np.diff(seg_task[:, 3]))) / c.max_rotate_deg_s
        cost = np.r_[0.0, np.cumsum(np.maximum.reduce([dq, dtip, drot]))]
        T = 0.5 * np.pi * cost[-1]  # cosine profile: peak = pi/2 x mean, so peaks sit at the caps
        if T > 0:
            n = max(2, int(np.ceil(T * c.rate_hz)) + 1)
            tt = np.linspace(0.0, T, n)
            prog = 0.5 * (1 - np.cos(np.pi * tt / T)) * cost[-1]
            seg = np.array([np.interp(prog, cost, seg_q[:, j]) for j in range(5)]).T
            t_out.append(t0 + tt[1:]); q_out.append(seg[1:])
            phases.append((name, round(t0, 3), round(t0 + T, 3)))
            t0 += T
        i0 = idx[-1]
    t, qq = np.concatenate(t_out), np.vstack(q_out)
    for x in qq:  # what is streamed, not just the samples it was timed from
        if (hit := spoon_collision(*sp.points(x), bowl, c)) is not None:
            return ScoopPlan(False, f"timed path: the spoon would hit the {hit}", report=worst)
    task = np.array([sp.task(x) for x in qq])
    tilt = np.degrees(task[:, 3]) - level_pitch_deg
    dip = [p for p in phases if p[0] == "dip"][0]
    i_bottom = int(np.searchsorted(t, dip[2]))
    report = {**{k: round(v, 2) for k, v in worst.items()},
              "duration_s": round(float(t[-1]), 1),
              "lowest_tip_above_bottom_mm": round(float(np.min(task[:, 2]) - bowl.bottom[2]) * 1000, 1),
              "deepest_tilt_deg": round(float(np.min(tilt)), 1),
              "tilt_at_bottom_deg": round(float(tilt[min(i_bottom, len(tilt) - 1)]), 1),
              "end_tilt_deg": round(float(tilt[-1]), 1),
              "scoop_length_cm": round(float(x_exit - x_entry) * 100, 1)}
    report["dip_deg"] = c.dip_deg
    return ScoopPlan(True, "ok", t, qq, phases, task[:, :3], tilt, report)
