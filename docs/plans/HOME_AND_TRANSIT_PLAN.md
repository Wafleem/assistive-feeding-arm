# SO-101 feeding: HOME pose, transits and the full feeding flow (plan)

Status: **plan only**. Nothing here has been run on the arm. Every number comes from offline
forward kinematics (`robot_iface/so101_kinematics.py`, checked against the arm to 0.2 mm) and
simple geometry. Reproduce it with:

```
.venv/bin/python scripts/plan_home_transit.py            # waypoint + segment tables
.venv/bin/python scripts/plan_home_transit.py --search   # also the carry-posture search
```

Units in this doc are deg and mm in the arm base frame (= Viam world; table top z = 0).
Joints are `[pan, shoulder lift, elbow, wrist flex, wrist roll]`. "Tilt" is the spoon's angle from
level (+ = tip up), with level = tool pitch -4.83 deg (`poses/so101_face.json` `spoon_level`).

## 1. Three facts that drive the design

1. **Spoon pitch is linear in the joints.** On the SO-101, tool pitch = -(lift + elbow + flex),
   exactly. Pan and roll don't change it. So:
   - level means `flex = 4.83 - lift - elbow`;
   - a pan-only move never tilts the spoon;
   - a straight, *synchronised* joint-space line between two level poses stays level the whole way.
2. **Synchronisation decides spill and collision.** If every joint instead runs at the same
   deg/s and stops when it arrives (UNSYNC in the tables), the same level-to-level lines tip the
   spoon by 19-37 deg, and the spoon hits a bowl (-8 to -15 mm clearance). We don't know which one
   the SO-101 module's `MoveToJointPositions` does (open question Q1). Until we know, **every move
   that changes more than one joint is streamed** as a synchronised line through
   `stream_joints`, the way the tracker already drives the arm. Pan-only moves can use
   `move_joints`.
3. **A level camera can't see the bowls.** The camera looks along the spoon (tool +z). With the
   spoon level, the bowls are 41-69 deg off its axis. The person comes into view on the way over
   (section 5). Looking into a bowl needs an empty spoon tipped down (`PEEK_*`, section 3).

## 2. HOME and the carry posture

One **carry posture** is used for HOME, above each bowl, READY and FACE_START. Only the pan
changes between them, so the long moves never tilt the spoon:

| joint | lift | elbow | flex | roll |
|---|---|---|---|---|
| carry | **-72.5** | **32.5** | **44.83** | **-2.81** |

**HOME = `[0.0, -72.5, 32.5, 44.83, -2.81]`**

| property | value | requirement |
|---|---|---|
| tool / spoon tip | (131, 0, 263) / (264, 0, 252) mm | |
| spoon tilt | 0.0 deg (level) | level or a little tip-up |
| clearance: table / bowls | 154 / 194 mm (spoon capsule r 25, camera, link boxes) | >= 50 |
| clearance: own base and shoulder | 102 mm | >= 10 |
| joint-limit margin | 27.5 deg (lift) | >= 5 |
| shoulder gravity proxy | 0.07 N m | face pose 0.61, pre-scoop left 0.72 |
| radius from the pan axis: tool / tip | 92 / 225 mm | compact |

Why this pose:
- **Low shoulder load.** The upper arm leans back and the forearm folds forward, so the wrist
  and spoon sit almost over the shoulder axis. That gives about 1/9 of the gravity torque of the
  face poses, so there is little sag. The `--search` ranking puts it 4th of all level postures;
  the top three are within 0.007 N m.
- **Compact.** The spoon tip is only 225 mm from the pan axis.
- **At lips height.** The tool is at z 263 mm (lips about 270), so the sweep needs no height change.
- **Inside the tracker's comfort box.** `TrackConfig.posture_lo/hi` allow lift >= -97,
  elbow >= -60 and flex <= 80, so `LevelTracker.reset()` starts from it without clamping.
- **HOME pan 0** points the spoon over the table, away from the person. It is 32 and 49 deg
  from the bowls and 93 deg from the person.
- **Roll is held** at the spoon-level calibration's -2.81 deg in every transit. If `save_level`
  is re-taught, recompute flex as `-pitch_level - lift - elbow`.

What HOME does not give:
- **No useful camera view.** The camera points at the far table and room. The person is
  113 deg off its axis and the bowls are 69 deg off (fact 3). Pointing the camera at the person
  would put HOME in the person's space, so it doesn't.
- **The elbow sits behind the base**, at (-30, -18, 177) mm, past the table's back edge
  (x = -26). See Q6.

The current default home is the look pose with pan 0 (`[0, -98.2, 70.07, 16.88, -7.3]`). It is
1.8 deg from the lift limit and has the spoon 16 deg tip-up. Replace it.

## 3. Waypoints (script output)

| pose | joints (deg) | tool xyz (mm) | tip xyz (mm) | tilt | table / bowls clear (mm) | margin | tau (N m) | tip-lips (mm) | camera to lips / bowl (deg) |
|---|---|---|---|---|---|---|---|---|---|
| HOME | [0, -72.5, 32.5, 44.83, -2.81] | (131, 0, 263) | (264, 0, 252) | 0.0 | 154 / 194 | 27.5 | 0.07 | 547 | 113 / 69 |
| OVER_RIGHT (= READY_1) | [-32.07, -72.5, 32.5, 44.83, -2.81] | (117, 49, 263) | (229, 119, 252) | 0.0 | 154 / 168 | 27.5 | 0.07 | 645 | 139 / 47 |
| PEEK_RIGHT (empty only) | [-32.07, -72.5, 32.5, 74.83, -2.81] | (108, 43, 233) | (201, 101, 157) | -30 | 132 / 90 | 20.2 | 0.04 | 628 | - / **12 (in view)** |
| PRE_SCOOP_RIGHT | [-32.07, -18.01, 54.44, -31.60, -2.81] | (204, 103, 151) | (316, 174, 140) | 0.0 | 115 / 55 | 42.4 | 0.40 | 738 | 145 / 54 |
| OVER_LEFT (= READY_2) | [-49.42, -72.5, 32.5, 44.83, -2.81] | (99, 70, 263) | (185, 171, 252) | 0.0 | 154 / 175 | 27.5 | 0.07 | 681 | 152 / 41 |
| PEEK_LEFT (empty only) | [-49.42, -72.5, 32.5, 74.83, -2.81] | (92, 62, 233) | (163, 145, 157) | -30 | 132 / 101 | 20.2 | 0.04 | 660 | - / **2 (in view)** |
| PRE_SCOOP_LEFT | [-49.42, 27.09, -4.18, -18.08, -2.81] | (230, 223, 151) | (316, 324, 140) | 0.0 | 111 / 55 | 60.6 | 0.72 | 876 | 159 / 68 |
| FACE_START | [93, -72.5, 32.5, 44.83, -2.81] | (34, -92, 263) | (27, -225, 252) | 0.0 | 154 / 210 | 17.0 | 0.07 | **266** | **5 (in view)** / 132 |
| FACE_NEAR (tracker, level) | [94, 9, -30, 25.83, -2.81] | (20, -260, 271) | (11, -392, 260) | 0.0 | 165 / 294 | 16.0 | 0.61 | 99 | 10 / 142 |
| LOOK_SAVED (today) | [12.04, -98.2, 70.07, 16.88, -7.3] | (128, -19, 208) | (256, -46, 234) | +16.1 | 115 / 205 | **1.8** | 0.05 | 504 | 100 / 87 |

Notes on the table:
- **Bowl headings from the pan axis** at (38.8, 0): bowl-right is pan -32.07 at 327 mm,
  bowl-left is pan -49.42 at 426 mm. **Both bowls are on the arm's left (+y)**; the person is on
  its right (-y).
- **PRE_SCOOP** holds the tip over the bowl centre at z 140, level. The spoon capsule's bottom
  is then 55 mm above the rim. This is the proposed hand-off to the scoop motion (section 6).
- **PEEK** is OVER plus 30 deg of wrist flex, to point the camera into a bowl for food
  perception. Use it only with an empty spoon.
- **FACE_NEAR** is the taught close pose `[94, 9.1, -29.9, 16.2]` re-levelled. It stands for
  where tracking and Extend leave the arm.

## 4. Segment checks (80 samples each, both interpolation models)

| segment | loaded | how | SYNC: tilt, table / bowl clear (mm) | UNSYNC: tilt, table / bowl clear (mm) |
|---|---|---|---|---|
| HOME -> OVER_k | no | pan only (`move_joints`) | 0.0, 154 / 168: PASS | same: PASS |
| OVER_k -> PEEK_k | no | flex only | -30 (empty), 132 / 90: PASS | same: PASS |
| OVER_RIGHT -> PRE_SCOOP_RIGHT | no | streamed level line | 0.0, 115 / 55: PASS | -21.9, 45 / **-15**: FAIL (hits the bowl) |
| OVER_LEFT -> PRE_SCOOP_LEFT | no | streamed level line | 0.0, 111 / 55: PASS | +36.7, 111 / 55: PASS (empty) |
| PRE_SCOOP_RIGHT -> OVER_RIGHT | **yes** | streamed level line | 0.0: PASS | **+21.9**: FAIL |
| PRE_SCOOP_LEFT -> OVER_LEFT | **yes** | streamed level line | 0.0: PASS | **-36.7**, bowl **-8**: FAIL |
| OVER_k -> FACE_START (the sweep) | **yes** | pan only | 0.0, tip-lips >= 266: PASS | same: PASS |
| FACE_NEAR -> FACE_START (retract) | **yes** | streamed level line | 0.0: PASS | **-19.0**: FAIL |
| FACE_START -> HOME / OVER_k | no | pan only | PASS | PASS |

What the checks show:
- **Every segment passes when synchronised.** All joint margins stay >= 16 deg on the planned
  segments, and self-clearance stays >= 93 mm.
- **Unsynchronised multi-joint moves fail**, which is why they are streamed.
- **Moving from today's saved look pose fails** (margin 1.8 deg at its start).
- **Camera on the way over:** with the cone assumed at +-28 deg, the lips come into view from
  pan **74 deg** onward.
- **Reach for Extend from FACE_START:** `max_level_reach` gives +202 mm, but reaching the 3 cm
  stop needs about 236 mm. At lips z 270 the spoon stops about 6 cm short. This is an existing
  reach limit, and `_check_extend_reach` already reports it. The person or prop should sit
  about 4 cm closer, or slightly lower.
- **Scoop reach (for the scoop plan):**
  - bowl-right: the tip reaches the far rim at z 70-140.
  - bowl-left: a level tip reaches only 50 mm past the centre, and a tip tilted down 30 deg
    barely reaches the centre (0-10 mm).
  - So scoop bowl-left on its near half, or move it about 6 cm closer to the arm.

## 5. Speeds

| move | how | cap | estimate |
|---|---|---|---|
| empty, pan only | `move_joints`, `set_speed(30)` | 30 deg/s | HOME->OVER 1-2 s |
| empty, level line | stream at 30 Hz, trapezoid in s | 20 deg/s on the fastest joint, 60 deg/s^2 | ~4 s |
| loaded, level line | stream | 12 deg/s, 30 deg/s^2 | 5-8 s (PRE_SCOOP_LEFT: lift moves 99.6 deg) |
| loaded sweep | stream (pan only) | 25 deg/s, 30 deg/s^2, slowing to 12 deg/s over the last 25 deg | 125-142 deg in ~6-7 s |

What the sweep caps mean for the food: the tip is 225 mm from the pan axis, so 25 deg/s is
about 0.10 m/s. The accelerations are below 0.2 m/s^2 (~0.02 g), so the food doesn't slosh. The
slow last 25 deg starts where the face can first come into view, and it also makes the stop gentle.

## 6. Hand-off to the scoop motion (`docs/plans/SCOOP_PLAN.md`, other agent)

- **Entry:** `PRE_SCOOP_k`: tip over the bowl centre at z 140, spoon level, roll -2.81, empty,
  reached by a streamed line from `OVER_k`.
- **Exit (CARRY_k):** the scoop should end loaded and level (|tilt| <= 2 deg), with the
  spoon capsule >= 50 mm above the rim and the tip over the bowl. The simplest exit is
  `PRE_SCOOP_k` again. Then this plan streams the level line `CARRY_k -> OVER_k` (= READY_k).
- **Any other exit pose:** it must be level, and a sync line from it to `OVER_k` must pass
  `scripts/plan_home_transit.py`. Add it to `SEGMENTS`.
- **Bowl obstacle:** in the Viam config each bowl is a solid frustum. Any planner-based move
  (`move_tool`) will refuse to put the spoon inside a bowl, so the scoop should be joint-space
  streamed, or the obstacle must be removed or changed for the scoop.
- **Not yet reconciled:** `SCOOP_PLAN.md` didn't exist when this was written. If its entry or exit
  joints differ from the above, it should state them, and they should go through
  `SEGMENTS` here.

## 7. Full feeding flow

### 7.1 States (text diagram)

```
                 [Home]                   [Scoop 1|2]                       [Ready]*
  ANY ──retract──► HOME ──pan──► OVER_k ──line──► PRE_SCOOP_k ──scoop──► CARRY_k ──line──► READY_k (=OVER_k, loaded)
   ▲   (to carry      ▲                          (empty)        (other plan)                      │
   │    posture,      │                                                         [Sweep to person] │ pan only, loaded
   │    level line)   │                                                                           ▼
   │                  │                   no face after search + wait ─────────────────── FACE_START (pan 93, level)
   │                  │                   (pan back to READY_k, keep food)                       │ auto_center
   │                  │                                                                           ▼
   │                  └───── pan ◄── retract to carry (level line) ◄── [Home] / [Next scoop] ── TRACKING (existing _track_loop)
   │                                                                                        │ [Extend] (= per-bite confirm)
   │                                                                                        ▼
   │                                                                                     EXTEND (existing: in -> stop 3 cm ->
   │                                                                                     hold 10 s -> back out) ──► TRACKING
   └── STOP from any state: robot.stop(), mode idle, arm holds where it is. No automatic motion after STOP.
  * "Ready" is a no-op once the scoop has ended at READY_k; it exists for a manual start with a loaded spoon.
```

### 7.2 Buttons (dashboard)

| button | action (`POST /api/cmd`) | precondition | does |
|---|---|---|---|
| Home | `home` (exists; new behaviour) | idle | retract to the carry posture at the current pan (level line), then pan to 0 |
| Scoop bowl 1 | `scoop` `{"bowl": 1}` (bowl-right) | idle | pan to OVER_1 -> line to PRE_SCOOP_1 -> scoop -> line to READY_1 |
| Scoop bowl 2 | `scoop` `{"bowl": 2}` (bowl-left) | idle | the same for bowl 2 |
| Ready | `ready` | idle | retract to the carry posture at the current pan, level (a loaded spoon is fine) |
| Sweep to person | `sweep_to_person` | at the carry posture | streamed pan to FACE_PAN, then `auto_center` |
| Center / Extend / Retreat / STOP | exist | | unchanged (Extend can be pressed while tracking) |
| Feed bowl 1 / 2 | `feed` `{"bowl": n}` | idle | Home (if not there) -> Scoop n -> Sweep to person -> tracking, then **stops and waits for Extend** |
| Teach face pan | `teach_look` (exists) | at FACE_START | stores look = FACE_START, so `_do_search` / `_do_find_face` search around the person (section 7.4) |

Why "Feed" stops before Extend: AGENTS.md section 4 says to confirm before every bite, so the
Extend press is that confirmation. An optional `auto_return_s` (default off) would return the arm
after the back-out.

### 7.3 Joints, speeds and checks per step

| step | waypoints | motion | gate before moving |
|---|---|---|---|
| Home | current -> carry@current pan -> HOME | level line (sync stream, 12 deg/s if possibly loaded), then pan | the preflight check on the line passes (section 7.5); if the spoon tip is below a rim, lift it out first (section 7.4) |
| Scoop k | HOME/any carry -> OVER_k -> PRE_SCOOP_k -> scoop -> CARRY_k -> OVER_k | pan, empty line 20 deg/s, scoop, loaded line 12 deg/s | at the carry posture; the scoop module reports "loaded" (it can't be verified yet: see Q8) |
| Ready | current -> carry@current pan | loaded level line | \|tilt\| <= 5 deg at the start, otherwise `_do_level_spoon` first |
| Sweep to person | READY_k -> FACE_START (pan only) | stream 25 -> 12 deg/s | at the carry posture (+-2 deg); the lips prior is known (FACE_PAN) |
| Track | FACE_START -> `_track_loop()` | existing | `_require_level` (already level); a face fix from `_face_target` |
| Extend | existing | existing | existing (`allow_extend`, `extend_decision`) |
| Return | tracker pose -> carry@current pan -> HOME or OVER_next | finish tracking, then a loaded level line, then pan | as for Home |

### 7.4 Failure handling

- **STOP (any state).** Same as today: `_stop` is set, `robot.stop()`, the worker is joined, the
  mode goes idle. Nothing moves until a new button is pressed. The next Home or Ready starts from
  wherever the arm is:
  - It first runs the preflight on the retract line.
  - If the spoon was stopped inside a bowl (tip z < rim + 50), Home first lifts the tip straight
    up (level IK at the same xy, as `tip_over` in the script) to PRE_SCOOP height.
  - If the preflight still fails, the dashboard says: hand-guide or teleop out, then Home.
- **No face after the sweep.**
  - `auto_center` calls `_do_search`. With look = FACE_START it searches pan 93 +-8/16/24.
  - **Clamp the search to pan <= 105** (limit 110 minus 5). As written, +24 would ask for 117
    deg and fail.
  - If no face is found: hold level at FACE_START, keep perception running, and show
    "no face: waiting".
  - After `face_wait_s` (proposed 20 s), pan back to READY_k, keeping the food.
  - Never Extend without a face; `extend_decision` already enforces this.
- **Face lost while tracking or extending.** Existing behaviour: hold still, or back out on fast
  head motion. No change.
- **Tracking never ends on its own.** Today only STOP ends `_track_loop`. Add a `"finish"`
  value for `_ext_request`: if extended, back out first (as `retreat`), then return. Home and
  "Next scoop" use it, so they don't need a STOP plus re-level.
- **The arm can't follow** (lag > 10 deg) while streaming a transit: reuse the tracker's check
  (`stop()` + `Stopped`).
- **The spoon isn't level on arrival** (gravity sag): after each loaded line, read `joints_fast`.
  If |tilt| > 2 deg, run `_do_level_spoon` (wrist only) before the next step.
- **Safety words, "I'm done", or a timeout:** finish (back out), then Home.
- **When to return home:**
  - the operator presses Home;
  - "I'm done";
  - no face for `face_wait_s` twice in a row;
  - otherwise "Next scoop" goes straight from the tracker to OVER_next with no stop at HOME.

### 7.5 How it plugs into `face_servo_session.py` (proposal; no edits made)

1. **Pose source.** Add a frozen `FeedPoses` dataclass with the joints from section 3 and the
   speeds from section 5. Load it from a new `poses/so101_feed.json`. Don't add keys to
   `so101_face.json`: `Taught.load` does `cls(**json)`, so an unknown key crashes older code
   that reads the same file. Set `taught.home` = HOME with `teach_home` after driving there,
   and `taught.look` = FACE_START with `teach_look`.
2. **A pure checker in `src/`.** Move `check()` / `segment_report()` from the script into a
   pure module, `feeding/transit.py` (FK only, no robot). The session can then preflight every
   line before streaming it. The script becomes a thin CLI (scripts/README: no logic only in
   a script).
3. **New private helpers:**
   - `_stream_line(q_to, max_degs, max_acc)`: a sync trapezoid, `stream_joints` at
     `track_cfg.rate_hz`, `_check_stop()` every tick, the lag check, then a settle on
     `joints_fast`.
   - `_to_carry(pan=None)`.
   - `_pan_to(pan, loaded)`.
4. **ACTIONS:** add `"scoop", "ready", "sweep_to_person", "feed"`. Each one is a `_do_*`
   method run by the existing `_run` worker, so STOP, busy and error handling come for free. Bowl
   1/2 arrives through `self._kw` (as `jog` already does). Change `_do_home` to
   `_to_carry(); _pan_to(HOME_PAN)` instead of one `_move`, which would be an unsynchronised
   multi-joint move.
5. **`_do_sweep_to_person`:**
   - `_pan_to(FACE_PAN, loaded=True)`, then `self.last_move_end = now`, then `_do_auto_center()`.
   - `FACE_PAN` is the last good face heading, clamped to 85-100 deg, defaulting to 93.
   - The hand-off is seamless: `_track_loop` → `LevelTracker.reset(q_meas)` starts from the carry
     posture, inside the comfort box.
6. **`_do_feed`:** `_do_home` if not at the carry posture, then `_do_scoop`, then
   `_do_sweep_to_person`. The latter is already inside the tracking loop, so the user presses
   Extend there (`_ext_request`).
7. **"Next scoop" / "Home" while tracking:** `command()` sees `_tracking_live()`, sets
   `_ext_request = "finish"` and records the follow-up. The loop backs out and returns, then
   `_run` chains `_to_carry()` and the pan.
8. **Tests (FakeRobot):** cover these:
   - the transit lines are level at every streamed setpoint;
   - STOP mid-sweep stops streaming;
   - the search clamps the pan to <= 105;
   - Feed never calls Extend on its own.

## 8. Open questions and risks (verify on the real arm, empty spoon first)

- **Q1. Does `MoveToJointPositions` synchronise joints?** This decides whether `move_joints` is
  safe for multi-joint moves. To test, with an empty spoon over the table:
  - `move_joints` OVER_RIGHT -> PRE_SCOOP_RIGHT with no bowl there;
  - log `joints_fast` at 30 Hz;
  - compute tilt(t) = -(sum) - (-4.83).
  
  SYNC predicts 0 deg; UNSYNC predicts up to -22 deg.
- **Q2. Gravity sag at the stretched poses.** The torque proxy is 0.40 (PRE_SCOOP_RIGHT),
  0.72 (PRE_SCOOP_LEFT) and 0.61 (FACE_NEAR) N m, against 0.07 at carry. Measure the real tip
  height and tilt at PRE_SCOOP_k against FK; the bowl clearance is only 55 mm. The masses are
  guesses, so treat the torques as relative.
- **Q3. Bowl mesh units.** The PLY vertices are in metres (rim r 0.075, h 0.06). If Viam reads
  them as mm, the obstacle is 0.15 mm wide and protects nothing. Check the 3D scene. The mesh is
  also a solid frustum (bottom r 50, rim r 75), which the planner sees as solid.
- **Q4. Camera.** The plan assumes the boresight is tool +z (config: ov th = -90 about z) and a
  +-28 deg usable cone. Check the "face in view from pan ~74" prediction during an empty sweep.
- **Q5. Bowl naming.** "bowl-right" (y 174) and "bowl-left" (y 324) are both on the arm's left.
  Confirm that bowl 1 = bowl-right, and whose left/right is meant.
- **Q6. Behind the arm.** At carry the elbow is at x -30 (behind the table edge, z 177) and
  swings on about a 70 mm radius during pans. Check for the Orin, cables or a wall there.
- **Q7. The person's position.** FACE_PAN 93 and the 266 mm tip-to-lips at FACE_START assume
  lips at (0..40, -490, 270). If the person sits more than about 7 cm closer, FACE_START is
  inside 20 cm. Derive the sweep's end from the last face fix and refuse if it would be < 20 cm.
- **Q8. A "loaded" spoon can't be verified yet.** The plan treats every move after a scoop as
  loaded until Home. A false "scooped" claim is a bug (AGENTS.md section 4): use PEEK_k (camera
  into the bowl and onto the spoon) or a food detector to confirm.
- **Q9. Reach to the 3 cm stop.** At lips z 270 the spoon stops about 6 cm short
  (section 4). This is a seating or prop-position question, not a transit question.
- **Q10. The table model** is an infinite z = 0 plane here (conservative). The real top ends at
  y = -76, so on the person's side a lap or tray could be higher. The sweep keeps >= 154 mm
  above z = 0.

## Files

- `scripts/plan_home_transit.py`: the offline checker (new; no robot calls).
- `docs/plans/HOME_AND_TRANSIT_PLAN.md`: this plan.
