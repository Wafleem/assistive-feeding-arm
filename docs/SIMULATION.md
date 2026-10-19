# Simulation: building everything before the arm and the wheelchair exist

The rule for this project: **no skill goes to hardware until it passes in
simulation through the same `RobotInterface`, the same perception code and the
same safety envelope it will use on the chair.** This document says what we
simulate, with which tools, how far the current setup is from reality, and what
to check when moving to hardware.

What already runs (`make test`, `make demo`, `make eval`):

* `sim/models/xarm6_wheelchair.xml`: an arm with xArm6 link dimensions and joint
  limits, a two-finger parallel gripper (85 mm stroke), a wrist force sensor, a
  wrist RGB-D camera and a wheelchair-mast RGB-D camera, on a wheelchair that
  can sway, next to a seated user model that the arm must never touch.
* Four scenes: `elevator` (10 spring-loaded buttons), `bag` (1.2 kg bag on a
  counter), `door_lever` and `door_knob` (latched, hinged pull door).
* `SimRobot` (MuJoCo) implementing the same API as `ViamRobot`; the executor,
  locator and skills are identical in both.

Baseline (`scripts/eval_sim.py`, 3 seeds each; "world ok" = the scene confirms
it, not the skill's own claim):

| Condition | Elevator (10 buttons) | Bag | Lever door | Knob door | False success claims | User contacts |
|---|---|---|---|---|---|---|
| Oracle detector, estimated hinge | 30/30 | 3/3 | 3/3 (door opened, skill reported only partial/failed) | 0/3 | 0 | 0 |
| Oracle detector, known hinge | 30/30 | 3/3 | 3/3 | 0/3 | 0 | 0 |
| Degraded detector (box jitter, no masks, 10% misses) | 30/30 | 1/3 | 2/3 | 1/3 (not claimed) | 0 | 0 |
| 4 mm wheelchair sway, 2 seeds | 20/20 | 2/2 | 0/2 | 0/2 | 0 | 0 |

Reading it: buttons are solid; bag grasping is sensitive to the handle-centre
estimate when there is no mask (that's what SAM is for); doors depend on the
hinge estimate and need compliance; the knob skill is not working yet. The two
columns that must stay at zero do.

---

## 0. Simulate the flagship fetch first

The flagship demo (PLAN.md §0, "hand me ___") is rehearsed in three layers,
cheapest first. None of them needs the arm.

**(a) Offline perception on recorded frames, from day one.** Record ~50
RealSense RGB-D frames of tabletop scenes (the demo objects, two look-alike
mugs, clutter, the bin) from the wrist and wheelchair viewpoints. Save per
frame: colour PNG, depth as uint16 millimetres (`.npz`, aligned to colour),
intrinsics JSON, and a `labels.json` with the object phrases and hand-drawn
boxes (CVAT, or one SAM click per object). A small replay class that serves
these frames as `RGBDFrame`s lets the *same* `Detector`/`Locator` code run on
them. Metrics per detector (OWLv2, Grounding DINO, YOLO-World, NanoOWL):
recall at IoU 0.5 for the named object, wrong-object rate on the look-alikes,
3D centroid error against a tape-measured position for 10 frames, object
width error against calipers (this drives the 85 mm gripper check), latency
per device (see COMPUTE.md). Store the frames in `data/` (git-ignored) and a
copy on the demo USB stick.

**(b) MuJoCo tabletop with YCB objects.** A `tabletop.xml` scene (to add):
table at seat height in front of the chair, 3-6 objects per episode drawn from
YCB meshes (mustard bottle, mug, banana, can, sponge, box; the same set as the
real props, PROPS.md: they are bought at Target-type stores, so measure the
bought item and adjust or replace the YCB mesh to match), plus a bin. Sources: the YCB benchmark meshes, or the
Google Scanned Objects / YCB conversions to MJCF (e.g. the
`mujoco_scanned_objects` collection); convex-decompose with `obj2mjcf`
(CoACD) so contacts are stable. The wrist and wheelchair cameras render the
scene and feed the **same** perception code; with the oracle detector first,
then the real detectors on textured renders. Success metric per episode:
object lifted > 5 cm and still held 2 s later (and, for "put it in the bin",
its centre inside the bin volume). Target before day 1: **≥ 70% grasp success
over randomised scenes**, reported per object by `eval_sim.py` with zero false
claims and zero user contacts, as for the other tasks.

**(c) One robot API.** Unchanged: the fetch skill talks to `RobotInterface`,
so the same code runs on `SimRobot` and `ViamRobot`.

**(d) Domain randomisation** (§7): object pose/yaw, which objects and how many,
look-alike pairs, table colour, lighting, camera extrinsics, depth noise and
dropouts (black and shiny objects lose depth on a real RealSense: model it, and
choose matte props).

**Stretch: "two brains, same task".** Collect ~50 teleop demos of the fetch
in sim (and a few real ones), train SmolVLA or pi0 through LeRobot, and score
it in the same `eval_sim.py` tabletop task against the modular pipeline.

Not in the repo yet: the `fetch`/`place` intents, the tabletop scene, the
recorded-frame replay class and the PCA grasp. They are first tasks for the
workstreams in WORKSTREAMS.md.

## 1. Robot models

### xArm6 + gripper

| Source | What you get | Use |
|---|---|---|
| **UFACTORY `xarm_ros2`** (github.com/xArm-Developer/xarm_ros2, `xarm_description/`) | Official xacro/URDF for xArm5/6/7, Lite6, xArm Gripper, Bio Gripper, vacuum gripper; meshes (STL); exact link frames and joint limits | **Primary source.** Generate a URDF for `dof:=6 add_gripper:=true`, convert for MuJoCo (below). Also what Gazebo/Isaac import. |
| **MuJoCo Menagerie** (github.com/google-deepmind/mujoco_menagerie) | Hand-tuned MJCF with actuators and collision meshes. Has UFACTORY models (xArm7 with its gripper, Lite 6 at the time of writing). Check whether an xArm6 has been added before converting your own. | Copy its actuator/default/contact settings onto the converted xArm6; use its xArm gripper MJCF (the real gripper is a 4-bar linkage, not two sliders). |
| `pybullet_data` | Ships an `xarm/xarm6_robot.urdf` (check `pybullet_data.getDataPath()`) | Quick kinematics checks in PyBullet only. |
| This repo's placeholder | Capsules with xArm6 dimensions: d1 267 mm, upper arm 289.5 mm (53.5 mm offset), elbow offset 77.5 mm, forearm 342.5 mm, wrist 76 + 97 mm; xArm6 joint limits; gravity compensation on | Kinematics/reach/contact logic now. Replace in M2. |

Converting the official URDF for MuJoCo:

```bash
# in a ROS 2 workspace with xarm_ros2 (or any machine with xacro):
xacro xarm_description/urdf/xarm_device.urdf.xacro dof:=6 add_gripper:=true > xarm6_gripper.urdf
# replace package://xarm_description/meshes/... with relative paths, copy the STLs next to it
python - <<'EOF'
import mujoco
m = mujoco.MjModel.from_xml_path("xarm6_gripper.urdf")   # MuJoCo reads URDF directly
mujoco.mj_saveLastXML("xarm6_gripper.xml", m)             # then hand-edit the MJCF
EOF
```

Then, in the MJCF: add `<position>` actuators (the real arm is position
controlled at 250 Hz), `gravcomp="1"` on the links, a `tcp` site at the
fingertips with +z out of the tool, the `ft_site` force/torque sensors between
flange and gripper, the `wrist` camera, and the collision bits used here
(arm `contype=2 conaffinity=1`: no self-collision, collides with everything
else). Put downloaded meshes under `sim/assets/` (git-ignored) and keep the
names `tcp`, `ft_site`, `wrist`, `scene`, `arm_base`, `joint1..6`, `finger_l`
so `SimRobot` needs no changes. Verify forward kinematics against the xArm
Studio readout for 5 joint configurations (< 1 mm).

### Gripper notes
* Real xArm Gripper: 0-85 mm, up to ~30 N grip (set by `set_gripper_speed` /
  position + current limits). Its fingers move on an arc, so the fingertip
  height changes with opening: model it if grasp depth matters (it does for
  thin handles).
* Two sim settings were necessary for grasps not to fail spuriously, and are
  *physics*, not tuning knobs to fiddle with: elliptic friction cones with
  `impratio=10` plus `noslip_iterations`, and a **slow jaw close**
  (0.05 m/s per finger). Without them a 1.2 kg bag crept out of the grasp
  during the carry, and the first finger to touch knocked it off the counter.

## 2. Simulator choice

| | MuJoCo 3 | Isaac Sim / Isaac Lab | PyBullet | Gazebo (Harmonic) |
|---|---|---|---|---|
| Install | `pip install mujoco`, macOS/Linux/Windows | NVIDIA RTX GPU, Linux/Windows | `pip install pybullet` | ROS 2 stack |
| Contact quality for grasp/press | Good, tunable, deterministic | Good (PhysX) | Fair | Fair (DART/Bullet) |
| Speed | ~1000x realtime here on a laptop CPU; MJX/MuJoCo Warp for GPU batches | Thousands of envs on GPU | Fast | Slow |
| Cameras | Raster RGB, metric depth, segmentation | Ray-traced RGB, realistic depth/noise, Replicator for labelled data | Basic | Plugins |
| Fit here | **Default for everything now** | Perception data + large-scale VLA/RL training later | Quick URDF checks | Only if a ROS stack is needed |

**Recommendation:** MuJoCo for development, CI and skill evaluation (it is
already wired, and it runs natively with the viewer on the team's Mac; Isaac
Sim does not run on macOS at all). Add Isaac Sim (Replicator) when the perception workstream needs
photoreal training images of elevator panels and handles, and Isaac Lab or MJX
when the VLA workstream needs many parallel rollouts. Both import the same
URDF. Benchmarks worth borrowing for VLA evaluation: SIMPLER (sim evaluation
that correlates with real OpenVLA/Octo results), LIBERO, ManiSkill3,
RoboCasa (kitchen scenes with doors and handles).

## 3. The wheelchair as a slightly moving base

`xarm6_wheelchair.xml` puts the arm base on a `wheelchair` body with
`chair_x`, `chair_y` (slide) and `chair_yaw` (hinge) joints driven by stiff
position actuators. `SimRobot(sway=0.004, sway_hz=0.4)` drives them with
incommensurate sines (4 mm in x, 2 mm in y, ~0.07° yaw) to mimic a chair on
pneumatic tyres while the user shifts their weight. Everything the skills see
is in the arm base frame, so sway shows up as the *world* moving, which is
what will happen on the chair.

Experiments to run (`make eval-hard`, `--sway`):
* Sway 0 / 2 / 4 / 8 mm: button pressing still passes at 4 mm because the press
  is guarded and has 2 cm of overtravel (with 1.5 cm it produced a false
  "no resistance" while the button had latched). Doors fail under sway with an
  estimated hinge: another reason for compliant door control.
* Add a pitch joint for parking on a 1:12 ADA ramp (4.8°): checks that
  gravity-aligned "top-down" grasps use the world vertical, not the base z.
  (Today they use base z; on a ramp that matters. Needs an IMU on the chair.)
* Mount compliance: a 6-DoF weld with `solref` softness between chair and arm
  base to model a flexing armrest bracket under a 3 kg load.
* Chair moved between "locate" and "press": re-run the locate step if the base
  moved more than ~5 mm (wheel odometry or the wheelchair camera's view of a
  static landmark tells you).

## 4. Scenes

All dimensions are in the XML comments; geometry is placeholder, dimensions
are chosen from accessibility norms so reach problems show up early.

* **elevator.xml**: wall 0.52 m in front of the arm base; 3x3 keypad + L,
  70 mm pitch, buttons 24 mm diameter, protrude 8 mm, 6 mm travel,
  400 N/m spring (~2.4 N to bottom out; real buttons 2.5-5 N, ADA limit
  22 N). `SimRobot.events` gets `pressed:<label>` when a button passes 4 mm,
  so evaluation can check "that button and no other". Next:
  * labelled textures (generate PNGs of digits, braille dots, a lit ring) so
    real detectors/OCR can be tested on rendered frames;
  * panel variants: vertical row, two columns, recessed/flush buttons,
    touch-sensitive (no travel: force spike only), call buttons (up/down) on
    the lobby wall, panel heights 0.9-1.2 m;
  * "lit" state as an emissive material switched on by the latch.
* **bag.xml**: counter top 0.60 m high, front-right of the chair; bag
  0.12x0.30x0.24 m, 1.2 kg, rigid loop handle 16 mm thick. Next: soft handles
  (`<composite type="cable">` or MuJoCo `flex` for a fabric loop; plastic-bag
  handles are the hard, realistic case), heavier bags (3-5 kg to exercise the
  weight check), handles lying flat, bags on the floor (out of reach by
  design: the right answer is "I can't reach that").
* **door_lever.xml / door_knob.xml**: 0.85 m pull door, hinge on the robot's
  right, handle 0.95 m high (ADA 0.86-1.22 m), lever spring return, knob
  either way. A latch (equality constraint `door_latch`) holds the door until
  the handle is turned past 0.5 rad; `SimRobot._scene_hooks` toggles it.
  Next: door closer (hinge spring + damping), push doors, push bars, left
  hinges, heavier doors (ADA interior doors: ≤ 22 N to open), and the
  question of the chair blocking its own door swing.

Adding a scene: copy one, `<include file="../models/xarm6_wheelchair.xml"/>`,
name target geoms so `ORACLE_QUERIES` in `robot_iface/sim.py` maps them to a
perception query, and add a row to `TASKS` in `scripts/eval_sim.py` with a
world-side success check.

## 5. Simulated RGB-D and the same perception code

`SimRobot.get_frames()` renders, per camera, colour, metric depth (distance
along the optical axis, same convention as RealSense) and a geom-id
segmentation image, and wraps them in the same `RGBDFrame` the real cameras
produce. `camera_pose()` returns OpenCV optical axes (MuJoCo cameras look
down -z with +y up; the flip happens in one place). `Locator` does not know
which robot it runs on.

Make the sim cameras match the hardware you will buy:

| | D435 (wheelchair mast) | D405 (wrist, short range) | Sim now |
|---|---|---|---|
| Depth FOV | 87° x 58° | 87° x 58° | `fovy=58`, square pixels |
| Useful depth | 0.3-3 m | 7-50 cm | perfect, 0.0 = invalid |
| Depth noise | ~2% at 2 m (σ ∝ z²) | sub-mm at 10 cm | `depth_noise=` z² model |
| Resolution | 848x480 / 1280x720 | same | 320x240 (speed; set `image_size`) |

Things to add to the noise model before trusting perception numbers: missing
depth at edges and on shiny metal (elevator panels are brushed steel), IR
shadow on one side of protruding objects, colour/depth misalignment of a few
pixels, rolling exposure flicker under fluorescent lights.

Detectors in sim:
* `SegmentationOracleDetector` (ground truth masks) for skill development;
* `NoisyDetector(oracle)` to rehearse real failure modes (box jitter, no mask,
  misses); `make eval-hard` uses it;
* the real `OwlV2Detector` / `GroundingDinoDetector` on rendered frames once
  scenes have textures; compare their 3D error to the oracle's on the same
  frames (`Locator.locate(...).position` minus `SimRobot.geom_position(...)`).

Hand-eye and extrinsic calibration error: perturb the `wrist` and `scene`
camera poses in the XML by ±5 mm / ±1° (or at load time via
`model.cam_pos`/`cam_quat`) and see which skills still pass. Today, the
wheelchair camera's estimate of a button from ~0.7 m was ~7 mm off and the
wrist refine brought it to ~2 mm; the guarded press tolerates the rest.

## 6. One codebase for sim and real

```
skills / executor / perception ──► RobotInterface ──┬─► SimRobot   (MuJoCo, this repo)
                                                    ├─► ViamRobot  (viam-sdk → xArm6, gripper, RealSense, F/T)
                                                    └─► FakeRobot  (unit tests)
```

Rules that keep them interchangeable:
* SI units and the base frame everywhere above the interface; `viam_convert.py`
  is the only place that knows millimetres and orientation vectors.
* Contact only through `move_linear(max_force=...)`. On Viam this is stepped
  2 mm moves with an F/T read in between (slow but safe); the upgrade is a
  custom Viam module exposing the xArm SDK's force/impedance modes through
  `do_command`.
* Free-space moves through `move_to_pose`, which on Viam is the motion
  service with the keep-out boxes as `WorldState` obstacles. `SimRobot`
  imitates that: a straight Cartesian line if it is clear, otherwise a
  joint-space path (IK with restarts, via the ready pose if needed), both
  dry-run collision-checked against the user with a 4 cm clearance before
  anything moves.
* Contract tests: the tests in `tests/test_skills.py` describe what a backend
  must do (guarded stop, gripper `holding`, tare). Run the same assertions
  against `ViamRobot` on the bench (M4).

Two intermediate steps between MuJoCo and the real arm, both worth doing:
1. **Viam fake components.** A Viam machine config with a `fake` arm
   (`arm-model: xArm6`), `fake` gripper and a `fake`/image-file camera lets the
   `ViamRobot` code path (connection, frame system, motion planner, unit
   conversions) run without hardware. No contact physics, but it catches
   API/frame mistakes.
2. **MuJoCo as a Viam module.** A small Python Viam module implementing the
   Arm, Gripper, Camera and Sensor APIs on top of `SimRobot`. Then the whole
   Viam stack, including the builtin motion planner and the machine's frame
   system, drives the simulated arm. This is the most faithful rehearsal and
   the natural home for "sim in CI with the real config".

## 7. Synthetic data and domain randomisation

Segmentation is free in sim, so every rendered frame is a labelled training
example (boxes, masks, 3D pose, normal). Use it to fine-tune or at least
validate the open-vocabulary detector on *our* targets.

Randomise per episode (MuJoCo: edit `model` arrays before `mj_forward`, or
generate XML variants; Isaac: Replicator randomisers):

| Category | Ranges to start with |
|---|---|
| Lighting | 1-4 lights, intensity ×0.3-2, colour temperature 2700-6500 K, harsh specular on steel |
| Materials | panel steel/plastic/black glass; button colours; door wood/metal/glass; bag paper/plastic/cloth |
| Layout | panel height 0.85-1.25 m, button pitch 50-90 mm, 1-4 columns; handle height 0.86-1.22 m; door width 0.8-0.95 m; hinge left/right |
| Labels | fonts, sizes, raised/flat, braille, lit/unlit, glare |
| Cameras | extrinsics ±5 mm / ±2°, intrinsics ±2%, exposure, motion blur, depth noise and dropouts |
| Physics | button stiffness 200-1500 N/m, travel 2-8 mm; bag 0.3-4 kg; door friction, closer strength; friction ±30% |
| Base | sway 0-8 mm, ramp pitch 0-5° |
| Distractors | other bags, people's hands near the panel, posters with digits |

Store generated data outside git (`data/`, ignored), in COCO format for
detectors and LeRobot / RLDS format for policy learning, with the scene
seed and randomisation parameters saved alongside each episode.

## 8. Evaluating in sim, including VLAs

`scripts/eval_sim.py` metrics, per task:
* **world ok**: the scene says the job is done (button latched and *only* that
  button; bag in the gripper and off the counter; door open > 15°);
* **claimed**: the skill said success;
* **FALSE**: claimed but not done. Must be 0; a non-zero value fails the script;
* **user hit**: any arm contact with the seated-user model. Must be 0;
* **peak N**: the highest contact force seen.

To score a VLA: replace the `Executor(...).run(...)` line with
`run_vla(robot, ctx, policy, text, cameras=["scene"])`. The runner already
counts clamped steps (how often the policy asked for something too fast or
too big) and refuses steps into keep-out zones; report both next to success.
Also log: time to completion, number of confirmations requested, path length.
Sim success is an optimistic bound for VLAs (see SIMPLER for how to make sim
evaluation track real results: matched camera pose, textures and control
frequency). Evaluate with the camera placement the policy was trained with:
most open VLA checkpoints are third-person, i.e. the wheelchair camera, not
the wrist.

## 9. Sim-to-real checklist

Before the first contact on hardware, each line checked and written down:

**Kinematics and frames**
- [ ] FK of the sim model matches the real arm at 5 configurations (< 1 mm).
- [ ] TCP offset of the real gripper measured (fingertip, closed) and set in
      both the sim `tcp` site and the Viam frame system.
- [ ] Base frame on the chair measured; keep-out boxes re-measured with the
      actual user seated, with margins; boxes loaded into Viam as obstacles.
- [ ] Orientation-vector conversion checked against `arm.get_end_position()`
      at 3 poses (tool down, tool forward, tool sideways).

**Cameras**
- [ ] Hand-eye calibration of the wrist camera (e.g. ChArUco, OpenCV
      `calibrateHandEye`), residual < 3 mm.
- [ ] Wheelchair-camera extrinsic to the arm base, residual < 5 mm.
- [ ] Both cameras report OpenCV optical axes in the frame system (or the flip
      is applied once, in `camera_pose`).
- [ ] Depth validity checked on brushed steel and glass at the real distances.

**Force and contact**
- [ ] F/T sensor (or current-based estimate) noise and drift measured; tare
      before every contact phase verified; button threshold set above noise.
- [ ] Guarded `move_linear` stop latency measured (distance travelled after
      the threshold is crossed at 2 cm/s) and < 2 mm.
- [ ] Real button stiffness/travel measured on 3 real elevators; sim updated.
- [ ] Gripper "holding" signal validated (closed on nothing vs on a handle).

**Safety**
- [ ] Hardware e-stop within reach of the user and tested; wireless stop for a carer.
- [ ] xArm collision sensitivity at maximum; reduced-mode speed/joint limits set
      on the controller (not only in this code).
- [ ] Software stop: "stop" by voice halts motion in < 300 ms, measured.
- [ ] Every skill run first in sim with the measured parameters, then on a
      bench mock-up (the PROPS.md arcade panel, bag and cabinet door), then with a
      mannequin in the chair, then with a person.
- [ ] Risk assessment written (ISO 10218-2 / ISO/TS 15066 / ISO 13482 as
      references) and reviewed by someone not on the build team.

**Behaviour**
- [ ] `eval_sim.py` with measured parameters: 0 false claims, 0 user contacts.
- [ ] The same skill test file passes against `ViamRobot` on the bench.
- [ ] What the robot says when it fails is understandable to the user
      ("I couldn't find button 7", not a stack trace).
