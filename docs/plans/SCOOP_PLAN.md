# SO-101 scoop motion for bowl-right and bowl-left (plan)

Status: **plan only**. Nothing here has run on the arm. Every number comes from offline forward
kinematics (`robot_iface/so101_kinematics.py`) and the bowl mesh. Reproduce it with:

```
.venv/bin/python scripts/plan_scoop.py                 # keyframes, checks, stream, timing
.venv/bin/python scripts/plan_scoop.py --radius-sweep  # + which bowl distances fit, bowl-pose tolerance
.venv/bin/python scripts/plan_scoop.py --json          # hand-off poses only
.venv/bin/python scripts/plan_scoop.py --food-mm 20    # another food depth (default 15)
```

Units: deg and mm in the arm base frame (= Viam world; table top z = 0). Joints are
`[pan, lift, elbow, flex, roll]`. "Pitch" is the spoon's angle from level (level = tool pitch
-4.83 deg; + = tip up). Roll is held at the spoon-level calibration value, -2.81, throughout.

## 0. Hand-off poses (agree with `HOME_AND_TRANSIT_PLAN.md`)

```json
{
  "bowl-right": {"pan_deg": -32.07, "scoop": "FEASIBLE",
                 "pre_scoop": [-32.07, -17.96, 54.42, -31.63, -2.81],
                 "carry":     [-32.07, -72.50, 32.50,  44.83, -2.81]},
  "bowl-left":  {"pan_deg": -49.42, "scoop": "NOT FEASIBLE at its configured pose (426 mm from the pan axis)",
                 "pre_scoop": [-49.42,  27.12, -4.23, -18.06, -2.81],
                 "carry":     [-49.42, -72.50, 32.50,  44.83, -2.81]},
  "bowl-left @ suggested (184.2, 292.9)": {"pan_deg": -63.61, "scoop": "FEASIBLE",
                 "pre_scoop": [-63.61, -18.14, 54.58, -31.61, -2.81],
                 "carry":     [-63.61, -72.50, 32.50,  44.83, -2.81]}
}
```

How these names map to the transit plan:
- **`pre_scoop`** = its `PRE_SCOOP_k`: tip over the bowl centre at z 140 (80 mm above the rim),
  spoon level, empty. It matches that table to within 0.06 deg (solver tolerance).
- **`carry`** = its `OVER_k` = `READY_k`: the carry posture (lift -72.5, elbow 32.5), level, at
  the bowl's pan. The loaded sweep to the person starts here.
- **The scoop exits through `pre_scoop`** (its "CARRY_k" exit, "the simplest exit"). So the
  exit line `PRE_SCOOP_k -> OVER_k` is that plan's already-checked segment ("carry k -> over",
  SYNC PASS).
- **Level while loaded.** From the end of the lift (tip z 92, 32 mm above the rim) through
  `pre_scoop`, `carry` and the pan-only sweep, the spoon is level to +-1.1 deg in FK. That is
  inside the required +-5 deg, provided gravity sag is corrected (section 5). Pitch =
  -(lift + elbow + flex) exactly, so a synchronised joint line between level poses stays level
  and the pan never tilts the spoon.

The other three hand-off checks also pass: the joint margins (pre_scoop 42 deg, carry
27.5 deg), the bowl clearance at pre_scoop (55 mm in that plan's model, 56 mm beyond margin in
mine) and the loaded-segment model (streamed SYNC lines).

Two things differ from the transit plan:
1. It says bowl-left could be scooped "on its near half" or after moving "about 6 cm closer".
   This plan finds that the near half doesn't work: the gripper hits the near rim (section 3).
   The bowl must sit at **<= 335 mm** from the pan axis, about 9 cm closer.
2. Its solid-cylinder bowl check (>= 50 mm) must not be applied inside the scoop. The scoop is
   joint-streamed and checked against the real frustum shell here.

## 1. Geometry that drives the design

- **Pan sign.** + pan turns toward -y (the person, taught close pose at pan +94). The bowls (+y)
  need **negative** pan:
  - bowl-right: **-32.07 deg**, 327 mm from the pan axis;
  - bowl-left: **-49.42 deg**, 426 mm.
  
  The pan axis is at (38.8, 0), not at the origin. That is why the pans are not the
  atan2(y, x) values of 29 and 46 deg. The sweep to the person (about +93) is therefore 125 deg
  from bowl-right and 142 deg from bowl-left.
- **No wrist yaw.** The spoon's vertical plane contains the pan axis (to 0.2 mm). The tip can
  only move radially (lift, elbow, flex), or on an arc about the pan axis (pan), which drags the
  spoon sideways. Roll only spins the spoon about its own axis. **Every scoop stroke is
  therefore radial**, along the line from the pan axis through the bowl centre.
- **The spoon always points away from the base.** Pointing it back toward the base would need
  tool pitch > 90 deg plus roll 180, and is not practical. A spoon pushed tip-first with its
  concave face leading collects food. So the stroke **pushes outward, toward the far wall**:
  it enters tip-down in the near half, drags out along the floor, and curls up the far wall,
  which traps the food.
- **Bowl mesh** (`bowl-150x60.ply`, read from the Orin; units are **metres** in the file): an
  open frustum, **r 50 mm at the bottom (z 0), r 75 at the rim (z 60)**, origin at the bottom
  centre, walls leaning out 22.6 deg. The model adds an ASSUMED 4 mm wall and 4 mm floor.
  A `bowl-150x70.ply` also exists: if the real bowl is 70 mm tall, rerun with `H = 70`.
- **The spoon sticks out only ~33 mm past the jaws.** Tip at 133 mm, jaw tips at 99.9 mm, and
  the C270 camera at 62-92 mm, 6-36 mm above the axis. Only the spoon bowl may enter the food;
  the jaws, gripper and camera must clear the food and the **near rim**, which they pass over
  while the tip is in the bowl. This one constraint sets everything:
  - **Steep in the food** (-57 to -62 deg): it keeps the jaws above about 20 mm of food.
  - **Can't be level inside the bowl.** With the tip below the rim and the spoon flat, the
    gripper's 50 mm body crosses the near rim. The spoon can only be level from tip z ~90 up.
  - **Reach.** A steep spoon needs the wrist far out. At the floor the steepest reachable
    pitch drops about 0.3 deg per mm of bowl distance (-69 deg at 320 mm, -58 at 360, -38 at
    420).

## 2. Scoop strategy (one template, both bowls)

Keyframes are given as (u, z, pitch):
- **u** = tip offset from the bowl centre along the pan direction (+ = away from the base), mm;
- **z** = tip height, mm;
- **pitch** = spoon pitch, deg.

The same template fits any bowl 245-335 mm from the pan axis; only the pan changes.

| # | keyframe | u | tip z | pitch | what happens |
|---|---|---|---|---|---|
| 0 | pre_scoop | 0 | 140 | 0 | hand-off: level over the centre, 80 mm above the rim |
| 1 | hover | -25 | 80 | -58 | tilt tip-down while descending (empty); tip 20 mm above the rim plane |
| 2 | plunge | -15 | 16 | -62 | in along roughly the spoon axis; tip 12 mm off the inner floor |
| 3 | drag1 | +5 | 14 | -62 | push out along the floor, concave face leading |
| 4 | drag2 | +25 | 15 | -57 | ... |
| 5 | wall | +38 | 24 | -45 | tip reaches the far wall and starts up it |
| 6 | curl | +43 | 40 | -28 | curl up the far wall: food trapped between the spoon and the wall |
| 7 | rise | +44 | 64 | -12 | out of the food; flattens as fast as the near rim allows |
| 8 | rim | +36 | 92 | 0 | **level from here on** (tip 32 mm above the rim) |
| 9 | lift_clear | 0 | 140 | 0 | = pre_scoop: back over the centre, level |
| 10 | carry | | | 0 | = OVER_k (joint line, level throughout) |

**Unavoidable deviation from the brief:** "lift out vertically with the spoon level or <= 10 deg
tip-up" is not possible from inside this bowl with this gripper. Between `curl` and `rim` the
loaded spoon goes from -28 deg to level while rising; it is within -5 deg from t = 33.8 s (tip
z 78). That is about **3.7 s tip-down while loaded**. Cohesive food (yogurt, oatmeal, mashed
food, rice) stays put; loose food (peas, cereal) may lose some. Mitigations:
- curl harder at the wall: limited by the near rim;
- a deeper spoon bowl;
- a spoon bent ~20 deg at the neck so it sits tip-up relative to the handle (best fix; then
  re-teach `spoon_level`).

## 3. Waypoint tables (script output, food depth 15 mm)

Clearances are in mm **beyond** each body's required margin (section 4); "food" means the
jaws/gripper/camera above the food surface + 5 mm.

### bowl-right (316.2, 173.8): pan -32.07, 327 mm from the pan axis: FEASIBLE

| keyframe | joints (deg) | tip xyz (mm) | pitch | limit margin | tip above floor | worst shell clearance (body) | food |
|---|---|---|---|---|---|---|---|
| pre_scoop | [-32.07, -17.96, 54.42, -31.63, -2.81] | (316, 174, 140) | 0.0 | 42.4 | - | 56.1 (gripper) | - |
| hover | [-32.07, 12.81, -24.54, 74.56, -2.81] | (295, 160, 80) | -58.0 | 20.4 | - | 37.8 (blade) | - |
| plunge | [-32.07, 23.92, -15.89, 58.80, -2.81] | (304, 166, 16) | -62.0 | 36.2 | 12.0 | 5.5 (gripper) | 7.3 |
| drag1 | [-32.07, 37.38, -36.61, 66.06, -2.81] | (321, 176, 14) | -62.0 | 28.9 | 10.0 | 5.4 (blade) | 5.3 |
| drag2 | [-32.07, 41.30, -40.54, 61.07, -2.81] | (337, 187, 15) | -57.0 | 33.9 | 11.0 | 6.1 (blade) | 5.1 |
| wall | [-32.07, 28.35, -13.28, 34.76, -2.81] | (348, 194, 24) | -45.0 | 60.2 | 20.0 | 9.5 (blade) | 10.2 |
| curl | [-32.07, 17.93, 15.44, -0.54, -2.81] | (353, 196, 40) | -28.0 | 77.9 | 36.0 | 11.9 (blade) | 18.9 |
| rise | [-32.07, 17.14, 30.22, -30.53, -2.81] | (354, 197, 64) | -12.0 | 64.5 | - | 7.6 (gripper) | 34.6 |
| rim | [-32.07, 16.50, 39.00, -50.67, -2.81] | (347, 193, 92) | 0.0 | 44.3 | - | 11.0 (gripper) | - |
| lift_clear | = pre_scoop | (316, 174, 140) | 0.0 | 42.4 | - | 56.1 | - |
| carry | [-32.07, -72.50, 32.50, 44.83, -2.81] | (229, 119, 252) | 0.0 | 27.5 | - | 183 | - |

Reach and margins:
- **Radial reach margin is small: +10 mm.** The worst keyframe, `drag2`, needs a shoulder-to-flex
  distance of 241 of the 251 mm available. The template passes for bowl centres at
  **245-335 mm** from the pan axis, and bowl-right is at 327.
- **Joint margins** are >= 20 deg at every keyframe and stream sample.
- **Tolerance to bowl-pose error.** The plan is made for the configured pose, then checked
  against a shifted bowl:

  | direction | tolerance | first failure |
  |---|---|---|
  | radial | **+-5 mm** | +-10 mm: the margin is used up (blade or gripper); real contact at about +-15-20 mm |
  | tangential | **+-10 mm** | +-20 mm: the blade or the camera |

### bowl-left (316.2, 323.8): pan -49.42, 426 mm: NOT FEASIBLE as placed

- **Out of reach.** `hover` through `rise` can't be reached at 426 mm.
- **Best reachable tip-down poses hit the near rim.** At the floor the steepest reachable
  pitch is about -36 deg (u 0) or -42 deg (u -20). At those angles the gripper and jaws pass
  **18-23 mm into the near rim**, and the jaws go below a 15 mm food layer (food clearance
  -3.4 mm). A "skim" with the tip at z 40 and -30 deg still puts the gripper 6 mm into the rim
  and only touches food deeper than 35 mm.
- **Only the level poses are reachable.** A level tip reaches 471-474 mm, so `pre_scoop`
  (above) and `carry` work, and the transit plan's poses for this bowl are still valid.

**Fix: move bowl-left to <= 335 mm from the pan axis.** The suggested spot keeps bowl-right's
distance and goes 31.5 deg further round, leaving a 30 mm gap between the rims: **(184.2, 292.9)**,
pan -63.61. Every check passes there, with both bowls as obstacles:

| keyframe | joints (deg) | tip xyz (mm) | pitch | limit margin | tip above floor | worst shell clearance | food |
|---|---|---|---|---|---|---|---|
| pre_scoop | [-63.61, -18.14, 54.58, -31.61, -2.81] | (184, 293, 140) | 0.0 | 42.3 | - | 56.1 | - |
| hover | [-63.61, 12.60, -24.25, 74.48, -2.81] | (173, 270, 80) | -58.0 | 20.5 | - | 37.8 | - |
| plunge | [-63.61, 23.73, -15.60, 58.70, -2.81] | (178, 279, 16) | -62.0 | 36.3 | 12.0 | 5.5 | 7.3 |
| drag1 | [-63.61, 37.11, -36.16, 65.88, -2.81] | (187, 297, 14) | -62.0 | 29.1 | 10.0 | 5.4 | 5.3 |
| drag2 | [-63.61, 41.00, -40.04, 60.86, -2.81] | (195, 315, 15) | -57.0 | 34.1 | 11.0 | 6.1 | 5.1 |
| wall | [-63.61, 28.17, -12.99, 34.65, -2.81] | (201, 327, 24) | -45.0 | 46.4 | 20.0 | 9.5 | 10.2 |
| curl | [-63.61, 17.79, 15.65, -0.61, -2.81] | (203, 331, 40) | -28.0 | 46.4 | 36.0 | 11.9 | 18.9 |
| rise | [-63.61, 17.02, 30.42, -30.61, -2.81] | (204, 332, 64) | -12.0 | 46.4 | - | 7.6 | 34.6 |
| rim | [-63.61, 16.40, 39.18, -50.75, -2.81] | (200, 325, 92) | 0.0 | 44.2 | - | 11.0 | - |
| lift_clear | = pre_scoop | (184, 293, 140) | 0.0 | 42.3 | - | 56.1 | - |
| carry | [-63.61, -72.50, 32.50, 44.83, -2.81] | (139, 201, 252) | 0.0 | 27.5 | - | 183 | - |

At this spot the sweep to the person grows to about 157 deg. Alternatively, move both bowls to
about 290 mm from the pan axis, the middle of the 245-335 band, for maximum tolerance.

## 4. What is checked (every keyframe, every joint-space line, every streamed setpoint)

| body (tool frame, z along the spoon) | model | requirement |
|---|---|---|
| tip | point at z 133 | >= 8 mm above the inner floor and off the walls (plan: 10 mm) |
| spoon blade | z 100-128, 25 mm wide, 12 mm thick | >= 3 mm off walls, rim and floor; may be in food |
| jaws | z 60-100, r 15 | >= 5 mm off the bowl, and >= 5 mm above the food |
| gripper | z -10-60, r 25 (the Viam 50 x 45 box) | >= 5 mm off the bowl, and >= 5 mm above the food |
| camera | C270 box corners (x -36..-6, y -45..29, z 62..92) | >= 5 mm off the bowl, and >= 5 mm above the food |
| upper arm, forearm, wrist | capsules r 30 | >= 5 mm off both bowls; >= 10 mm above the table |
| joints | | >= 5 deg inside the limits |

The bowl is a frustum shell, and the distance is exact in the (radius, z) half-plane. Food is a
flat layer: **15 mm design depth; the plan keeps the jaws and camera dry up to 20 mm.**

Results for bowl-right and the suggested bowl-left:
- **All 11 keyframes, all 10 joint-space lines (25 samples each) and all ~1,440 streamed setpoints
  pass.**
- Worst values: margin 20 deg, tip 9.8 mm above the floor, body clearance +5.1 mm, food +4.7 mm,
  table +60 mm.
- **Straight joint lines are fine** in the bowl: the tip leaves the straight task-space line by
  <= 4 mm. In `pre_scoop -> hover` (35 mm off) and `lift_clear -> carry` (26 mm) the tip is
  clear of everything.

## 5. Timing, speeds and sag

Streamed at 30 Hz, five phases. Each phase starts and ends at rest: a Catmull-Rom spline through
its keyframes, with min-jerk timing, slowed until every joint is <= 20 deg/s (flex 25) and
<= 80 deg/s^2 (`TrackConfig.max_acc_deg`). The in-bowl phases use IK per setpoint.

| phase | keyframes | time (s) | tip speed (peak) | peak joint speed (lift, elbow, flex) |
|---|---|---|---|---|
| approach (empty) | pre_scoop -> hover | 0.0-12.2 | 19 mm/s | 7, 18, 25 deg/s |
| plunge | hover -> plunge | 12.2-18.3 | 30 mm/s (mean 15) | 6, 5, 7 |
| scoop | -> drag1 -> drag2 -> wall -> curl | 18.3-30.1 | 13 mm/s | 9, 20, 21 |
| lift (loaded) | -> rise -> rim -> lift_clear | 30.1-39.0 | 32 mm/s | 18, 11, 24 |
| retreat (loaded, level) | -> carry | 39.0-47.8 | 52 mm/s | 18, 7, 25 |

- **About 48 s for the first runs.** After about 5 clean dry runs, approach and retreat can go
  2x (they are in free air), giving about 35 s. Keep plunge and scoop slow: food drag adds load
  to servos that are already sagging.
- **Sag sensitivity at `drag1`** (tip mm per +1 deg of error): lift 5.3 down, elbow 3.7 down,
  flex 1.3 down.
  - 1 deg of uncorrected shoulder sag uses half of the 10 mm floor clearance. 2 deg puts the tip
    on the floor.
  - The drag poses are the most stretched of the whole flow (the shoulder-to-flex distance is
    241 of 251 mm), so expect the largest sag here.
- **Margins to use:**
  - **Floor.** Run the first scoops with the whole template raised 10 mm (tip 20-25 mm off the
    floor) and an empty or water-free bowl. Lower it only after measuring the tip height at
    `drag1` against FK.
  - **Sag feed-forward.** Hold 1 s at `hover`, record measured minus commanded on lift, elbow
    and flex, and add it as a constant offset for the plunge and scoop phases.
  - **Freeze the session's sag integrator in the food.** Food resistance looks exactly like sag,
    so the integrator would push the spoon harder into the bowl floor. Re-enable it for lift and
    retreat, where the load is real and level matters: three joints sagging 1.5-2 deg each would
    otherwise tip the loaded spoon about -5 deg.
  - **Optional tip-up bias.** A +2 deg bias on `carry` (still "level or a little tip-up") gives
    extra margin against that sag.
  - **Lag stop.** In the bowl, stop at a lag of > 4 deg (the tracker uses 10). On a stop in the
    food, **don't** improvise. Stream the executed setpoints back in reverse, slowly, to `hover`,
    then to `pre_scoop`. That path is known to be clear.

## 6. Risks: what must be measured on the real arm (empty spoon first)

1. **Spoon geometry.**
   - **Tip position.** The tip at 133 mm on the tool axis comes from the Viam box, not a
     measurement. The taught close pose shows the spoon line about 15 mm off the tool axis
     (`lips_in_tool_mm` x 7-32, y 15). Level is tool pitch -4.83, so the spoon is not parallel
     to the tool z. Measure the tip in the tool frame: touch a marked point from two
     orientations. A 10 mm error in the tip's z offset is the same as a 10 mm radial bowl error,
     which is beyond the +-5 mm tolerance.
   - **Other dimensions.** Measure the spoon-bowl length and depth, and the jaw-tip and camera
     positions relative to the tip.
2. **Bowl pose.** The configured centres must be right to **+-5 mm radially and +-10 mm
   tangentially**, or the plan must be regenerated from a measured pose. Planning plus checks
   takes about 4 s in Python, so it can run per scoop. Also check:
   - the mesh origin (the bottom centre is assumed);
   - the units: the PLY is in metres; if Viam reads it as mm, the scene obstacle is 0.15 mm wide;
   - the real wall and floor thickness (4 mm assumed);
   - whether the bowl is weighted or fixed. The drag pushes outward on the far wall, so a light
     bowl will slide.
3. **Food level.** The plan is designed for 15 mm of food and is safe to 20 mm. Deeper food
   wets the jaws and camera; shallower food gives a smaller bite. Check the level with `PEEK_k`
   (transit plan) before each scoop, and refuse above 20 mm.
4. **Table height and base mounting.** z = 0 must be the table top at the bowls. Measure the
   tip z at `pre_scoop` and at `drag1` (with the bowl removed) with a ruler.
5. **Sag and reach.**
   - Measure the joint error at `hover`, `drag2` and `curl`.
   - **Reach margin.** `drag2` is 10 mm from full extension. If sag or a bowl error eats that,
     the IK fails, and the plan must say so and refuse; never clip.
6. **The loaded tip-down window** (3.7 s, -28 deg to -5 deg): test with the target food. If it
   spills, consider a bent spoon (section 2) or a shallower template plus more food depth.
7. **"Loaded" is not verified.** Don't report a successful scoop without evidence (AGENTS.md
   section 4): PEEK at the spoon or a VLM check, as in the transit plan's Q8.

## 7. How it plugs into the streamed-setpoint session (proposal; no edits made)

1. **A pure planner (after review)**, e.g. `src/assistive_arm/grasping/scoop.py`. Section 3,
   strategy `utensil_scoop`, behind `acquire_with_utensil` in `grasping/acquisition.py`. It
   contains:
   - a frozen `ScoopConfig` holding every number above (template, speeds, clearances, food
     depth, spoon and gripper geometry), in SI units;
   - `plan_scoop(bowl_xy, cfg) -> ScoopPlan`, which returns the setpoints (deg, at `rate_hz`),
     a phase label per setpoint, and the check report. It raises or returns `ok=False` with a
     reason when any check fails or the bowl is outside 245-335 mm.

   It uses only `so101_kinematics` (closed-form IK: pitch = -(lift + elbow + flex)). The script
   becomes a thin CLI over it.
2. **`FaceServoSession._do_scoop`** (a new `"scoop"` in `ACTIONS`, bowl id via `self._kw`, run
   by the existing `_run` worker so STOP and error handling come for free):
   1. **Plan and refuse.** Plan from the bowl pose: the config now, later
      `food_perception/bowls.py`. Show the failing check on the dashboard and refuse to move.
   2. **Get to `pre_scoop`.** Require it, reached by the transit plan's
      `_stream_line(OVER_k -> PRE_SCOOP_k)`, with `|tilt| <= 2` deg.
   3. **Stream the plan.** Use a loop like the tail of `_track_loop`. Per tick:
      - `_check_stop()`;
      - `self.robot.stream_joints(q + sag)`;
      - append to `_q_hist`, and read `joints_fast` into `_qm_hist`;
      - the lag check: 4 deg in the plunge/scoop phases, 10 deg elsewhere;
      - sleep to the next tick at `track_cfg.rate_hz` (30 Hz; the largest step per tick is
        0.83 deg, small enough for `stream_joints`, which has no servo profile).
   4. **Handle sag per phase.** Freeze the sag integrator in plunge and scoop, after the hover
      feed-forward. Run it in lift and retreat.
   5. **Finish.** End at `carry` (= READY_k). Set the mode to "loaded"; the transit plan's
      `_do_sweep_to_person` then takes over, then `auto_center` and `extend` (unchanged).
3. **STOP.** `stop()` already stops streaming and the arm. A new `scoop_backout` action replays
   the executed setpoints in reverse to `hover`, then `pre_scoop`. It is the only motion offered
   while the tip is below rim + 30 mm. The transit plan's Home will "lift the tip straight up";
   in the bowl that would drag the gripper over the near rim at a steep spoon, so it must use
   this backout instead.
4. **Viam obstacles.** The scoop never calls the motion planner, so the bowl obstacles stay in
   the config for every planned move. `move_tool` must not be used for any scoop segment.
5. **Tests** (FakeRobot, offline):
   - every setpoint of `plan_scoop` passes the checks;
   - pitch is +-1 deg from `rim` on;
   - setpoint steps are <= 1 deg per tick;
   - STOP mid-scoop stops streaming;
   - the backout reverses the executed setpoints exactly;
   - a bowl at 426 mm is refused with a reach reason.

## Files

- `scripts/plan_scoop.py`: the offline planner and checker (new; no robot or network calls).
- `docs/plans/SCOOP_PLAN.md`: this plan.
- Companion: `scripts/plan_home_transit.py`, `docs/plans/HOME_AND_TRANSIT_PLAN.md`.
