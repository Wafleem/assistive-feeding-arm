# Simulation: develop every section before touching the arm

Rule: **nothing approaches the (fake) face on hardware until it has done so in
MuJoCo through the same `RobotInterface`, the same perception code and the same
safety numbers.** Section 7's plan is in `src/assistive_arm/sim/README.md`; this
page is the overview.

## What exists now

* `sim/models/xarm6_table.xml`: an arm with xArm6 link dimensions and joint
  limits (capsule geometry), a two-finger parallel gripper (85 mm), wrist force
  sensor, wrist RGB-D camera, and a fixed RGB-D camera on a post at the table
  corner. Base frame = world, z = 0 at the table top.
* `sim/scenes/feeding.xml`:
  * a plate with grapes, a strawberry, a melon cube and a cracker;
  * a cup with a straw, and a spoon on a rest;
  * a head on a moving base (x/y/z slides + yaw, all actuated) and a `mouth`
    site whose +z is the face normal. **It still has a hinged jaw from the first
    scaffold: TODO replace it with the 3D-printed mouth** (an always-open opening and
    a catch cup, from the same CAD as the real print; see `src/assistive_arm/sim/README.md`);
  * a torso.
* `SimRobot` (`robot_iface/sim.py`):
  * the same API as `ViamRobot`;
  * RGB, metric depth and segmentation from both cameras, and an oracle detector
    with one detection per object instance;
  * scene behaviour plugged in through `hooks`;
  * any arm contact with the head or torso latches a stop;
  * `now()`/`wait()` use simulated time, so trackers and timeouts behave identically.
* `scripts/view_feeding_scene.py` (`make render`, `make viewer`).
* Tests: the scene loads, food stays put, and the head and jaw move; camera specs
  are marked `@todo` until `Locator.locate_all` is implemented.

## To build (section 7, then each section uses it)

| Piece | What | For |
|---|---|---|
| Printed-mouth scene | face plate with an always-open opening + catch cup, replacing the jaw head | everyone |
| `FeedingWorld` hook | scenarios: still / drift / moved away / tracking lost; "delivered" = the bite ends inside the catch cup | 1, 4, 6 |
| Mouth oracle with noise + delay | `SimOracleMouthDetector` (TODO in section 1) | 1, 4 |
| `scripts/eval_feeding.py` | scenarios × foods × seeds → table | everyone |
| Domain randomisation | plate layout, food size/colour, lighting, camera extrinsics ±5 mm/2°, depth noise and dropouts, head start pose | 1, 2, 3 |
| Real models | UFACTORY `xarm_ros2` xArm6 + gripper URDF → MJCF (or MuJoCo Menagerie's UFACTORY models as a reference); keep the site/camera names | 3, 4 |
| Measured props | the printed mouth from its CAD; the plate and foods measured after buying (docs/PROPS.md) | 8 |
| VLA data (stretch) | ~50 scripted/teleop acquisition demos per food in LeRobot format | 3 |

## Metrics (the eval table)

| Metric | Target |
|---|---|
| Bites delivered / attempted, per food | ≥ 80% |
| Bite released inside the opening and landed in the catch cup | ≥ 90% |
| Lateral error from the opening centre at release | ≤ 1 cm |
| Max TCP speed within 15 cm of the mouth | ≤ 3 cm/s |
| Arm contacts with the face plate | **0** |
| Moved-away scenarios that caused a retreat | **100%** |
| Tracking-lost scenarios that paused or retreated | **100%** |
| False "delivered" claims | **0** |

## Simulator choice

MuJoCo: `pip install`, native on the Mac with the viewer, fast, deterministic,
good contact for pinch grasps. (Needed settings already in the model: elliptic
friction cones, `impratio`, `noslip_iterations`, slow gripper closing.) Isaac Sim
does not run on macOS; use it on the Linux PC only if photoreal training images
are needed later.

## Sim-to-real checklist (before the real fake mouth)

- [ ] FK of the sim arm vs the real xArm6 at 5 poses (< 1 mm); real TCP offset of the venue gripper
- [ ] Camera intrinsics/extrinsics calibrated (hand-eye for the wrist, ChArUco for the fixed one)
- [ ] Mouth tracker latency measured end to end and set in `TrackerParams.latency`
- [ ] Stop latency ("stop" spoken → arm halted) measured, < 300 ms
- [ ] Gripper holding signal validated on grapes (Viam gripper position/current)
- [ ] Every sim scenario passes with the measured parameters
- [ ] Real trials: printed mouth only; e-stop tested; a person outside the team reviewed the risk assessment
