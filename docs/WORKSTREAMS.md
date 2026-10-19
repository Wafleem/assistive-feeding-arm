# Workstreams: splitting the work across a team

The scaffold is cut along interfaces so six pieces (plus props and a safety lead) can move in parallel without
waiting on each other. Each workstream **owns** a set of paths, **consumes**
and **provides** a small number of frozen contracts, has a **first task**
it can start today with no hardware and no other stream's output, and a
**done** criterion that is a test or an `eval_sim.py` number, not an opinion.

```
            ┌──────────── A. Voice & Interaction ────────────┐
            │  speech → Intent, confirmations, UI             │
            └───────────────┬────────────────────────────────┘
                            │ Intent (intent/schema.py)
            ┌───────────────▼──────────── C. Skills & Control ───────────┐
            │  planner, executor, skills, safety envelope                 │
            └───────┬─────────────────────────────┬───────────────────────┘
          Target3D  │                             │ RobotInterface
   ┌────────────────▼─────┐        ┌──────────────▼──────────────┐
   │ B. Perception         │        │ D. Simulation & Evaluation  │  E. Hardware & Viam
   │ detectors, 3D locate  │        │ SimRobot, scenes, eval      │  ViamRobot, calibration,
   └───────────────────────┘        └──────────────┬──────────────┘  F/T, chair mount
                                                   │ eval harness
                                    ┌──────────────▼──────────────┐
                                    │ F. Learning / VLA           │
                                    └─────────────────────────────┘
   P. Props & instrumentation: bought or flat-pack props, measured into sim twins (PROPS.md).
   S. Safety lead (part-time, cross-cutting): reviews every change to planning/safety.py,
      skills' force/speed numbers, and owns the risk assessment and test protocol.
```

## Frozen contracts (change only by a PR that every affected stream reviews)

| Contract | File | Used by | What "frozen" means |
|---|---|---|---|
| `Intent`, `Task`, `Intent.json_schema()` | `intent/schema.py` | A → C | Adding a `Task` is fine with a planner entry + tests; renaming fields is not. |
| `RobotInterface`, `MoveResult`, `GripperState`, `Wrench` | `robot_iface/base.py` | C, D, E, F | Signatures and units (SI, base frame, OpenCV camera axes). New *optional* hooks OK. |
| `Detector.detect`, `Locator.locate` → `Target3D` | `perception/detectors.py`, `perception/locate.py`, `perception/types.py` | B → C | `Target3D.position/normal` in base frame, normal points out of the surface. Extras go in `extras`. |
| `SkillResult` statuses + evidence | `skills/base.py` | C → A, D, F | `success` only with world-verifiable evidence. |
| `SafetyConfig` defaults | `planning/safety.py` | everyone | Numbers change only with the safety lead's review. |
| Eval tasks + world checks | `scripts/eval_sim.py` | D → everyone | Tasks can be added; a check can't be loosened. |

Ground rules: `make test` green on every PR; a stream that needs a contract
change opens that PR first and small; nobody else edits your owned paths
without telling you; results go in the PR description as an `eval_sim.py`
table when behaviour changes.

## Priority: the flagship fetch comes first

The flagship demo is open-vocabulary "hand me ___" (PLAN.md §0). Each stream's
**first** task is its piece of the fetch; the wheelchair-skill tasks listed in
the stream sections below come after it.

| Stream | Fetch task (do first) | Done when |
|---|---|---|
| A | Intent LLM → strict JSON `{action, object, attributes, relation, target}` with schema validation + rule fallback; add `fetch` and `place` tasks to `Intent`; "which one?" dialogue resolving "left/right/nearest" | 95% correct JSON on 100 recorded fetch phrases; ambiguous phrases always produce a question |
| B | Recorded-frame replay as `RGBDFrame`s; OWLv2 / GDINO / YOLO-World / NanoOWL + SAM 2 / NanoSAM on ~50 RealSense frames; centroid, PCA axes and width from mask + depth; numbered candidates for the overlay | Accuracy and latency baseline table (COMPUTE.md) by week -2 |
| C | `fetch` skill: top-down grasp along the shortest PCA axis with the 85 mm width check, collision scene (table + other objects as boxes, carried object attached), grip check, lift, handover to the open palm, release on pull or "got it", verify by re-detection; `place` into the bin | ≥ 70% in the sim tabletop eval by week -1, 0 false claims |
| D | `tabletop.xml` with YCB meshes and a bin; fetch/place rows in `eval_sim.py` with world checks (lifted > 5 cm and held 2 s; inside the bin) | Scene randomised over objects, poses, look-alikes |
| E | Jetson setup per COMPUTE.md pre-event checklist; NanoOWL/NanoSAM as Viam vision modules; gripper position/current exposed for the grip check; hand-eye calibration | Live overlay on the real camera; 4 of 5 real picks on day 1 |
| F | ~50 sim teleop fetch demos; SmolVLA / pi0 via LeRobot ("two brains, same task") | Scored in the same tabletop eval |
| S | Handover safety: open-palm target speed limits, keep-out zone from MediaPipe body on the sidecar, release logic that can't drop onto the lap | Reviewed before day 1 |
| P | Buy and measure the fetch objects, bin, shelf riser, calibration board (PROPS.md shopping list) | Every fetch prop measured into its sim twin; near-venue items listed by brand and size |

---

## A. Voice & Interaction

**Owns:** `src/assistive_arm/intent/`, new `src/assistive_arm/speech/`, new `ui/`
**Consumes:** nothing. **Provides:** `Intent` objects, a `confirm(msg) -> bool`
implementation for `SkillContext`, spoken feedback for `SkillResult.message`.

First tasks (no hardware):
1. Push-to-talk ASR: `faster-whisper` (`small.en` or `distil-small.en`) on a
   laptop mic, returns text; measure latency and WER on 50 recorded commands
   from 3 people, including the example phrases.
2. LLM intent: a function `llm_parse(text) -> Intent` that calls Claude (or a
   local model) with `Intent.json_schema()` as the output schema and runs the
   result through `Intent.from_dict`; the rule parser handles "stop" *before*
   any LLM call and is the fallback when the LLM is slow or offline.
3. `confirm()` by voice ("Press button 7? Say yes.") with a timeout that
   returns False, plus a big on-screen yes/no for users who can't speak
   reliably.
4. Grow `tests/test_intent.py` to 100+ phrases (paraphrases, ASR errors like
   "press button heaven", other languages if relevant).

Done: 95% intent accuracy on the recorded set; "stop" end-to-end latency
< 300 ms from end of speech; no un-validated intent can reach the planner
(test). **Can start:** today.

## B. Perception

**Owns:** `src/assistive_arm/perception/`
**Consumes:** `RGBDFrame` + camera poses from any `RobotInterface`.
**Provides:** `Target3D` (point + normal + optional extras such as
`axis`, `hinge_point`, `hinge_side`, `door_width`, which the executor already
passes to the skills).

First tasks:
1. Run `OwlV2Detector` and `GroundingDinoDetector` on photos of real elevator
   panels, bags and door handles (collect ~50 from phones); report recall for
   "elevator button", "shopping bag handle", "door handle", "door".
2. Add SAM 2 mask refinement (`MaskRefiner` after the box detector). The
   degraded-detector eval shows why: without masks the bag succeeds 1/3.
3. Button label reading: crop each button box, read with Florence-2 OCR or a
   small VLM; match to the requested label; handle "L"/"G"/"B1"/arrows.
4. Door geometry: detect the door edge / frame to estimate `hinge_point` and
   `door_width` and put them in `Target3D.extras` (the lever door succeeds with
   the true hinge and stops early with today's estimate).
5. Bag handle axis from the mask (PCA of the handle pixels) into
   `extras["axis"]`.

Done: on rendered sim frames with textures (from D), the real detector
pipeline gets within 1 cm of the oracle `Target3D` for buttons and handles;
`make eval` with the real detector ≥ 90% on the elevator. **Can start:**
today on photos; on sim frames once D lands textures.

## C. Skills & Control

**Owns:** `src/assistive_arm/skills/` (except `vla.py`), `planning/`,
`executor.py`. Safety numbers co-owned with S.
**Consumes:** `Intent`, `Target3D`, `RobotInterface`. **Provides:**
`SkillResult`, the executor.

First tasks:
1. Door v2: compliant turn and swing. Admittance on the F/T reading
   (move along the measured force within a cone around the planned arc) so a
   2-3 cm hinge/spindle error does not end in a force-limit stop. Target: lever
   door ≥ 80% with the *estimated* hinge in `make eval`, still 0 false claims.
2. Knob turning (0/3 today): grasp centring on the knob axis, try both turn
   directions on force limit, then swing.
3. Button press second check: wrist-camera image diff (button lit / moved)
   alongside the force evidence.
4. Multi-step tasks: "take me to floor 3" = press call button, wait for the
   doors, (user drives in), press 3. Planner support for waiting on a
   perception condition with a timeout.
5. Ramp-aware grasps: top-down means gravity-down, not base-z-down, when the
   chair is tilted (needs an IMU reading through `RobotInterface`).

Done: the numbers above in `eval_sim.py`; every new behaviour has a
`FakeRobot` unit test and a sim test. **Can start:** today.

## D. Simulation & Evaluation

**Owns:** `sim/`, `src/assistive_arm/robot_iface/sim.py`, `scripts/`,
`tests/test_sim.py`.
**Consumes:** `RobotInterface` contract. **Provides:** scenes, `SimRobot`,
the eval harness everyone reports with, rendered training data for B and F.

First tasks:
1. Real xArm6 + xArm Gripper model from UFACTORY's `xarm_ros2` URDF (and
   Menagerie's tuning), keeping the site/camera/joint names (see
   SIMULATION.md §1). FK check against published numbers.
2. Labelled textures: generated button-label PNGs, lit state, braille;
   brushed-steel material. Unblocks B's real-detector eval.
3. Domain randomisation hooks (SIMULATION.md §7) behind `SimRobot(randomize=...)`
   and `eval_sim.py --randomize`.
4. Scene variants: door closer, push door, left hinge, panel layouts,
   soft bag handle (`composite type="cable"`), heavy bag.
5. MuJoCo-as-a-Viam-module (SIMULATION.md §6), so the full Viam stack can drive
   the sim arm; shared with E.
6. CI: GitHub Actions running `make test` (MuJoCo works headless with
   `MUJOCO_GL=egl` or `osmesa` on Linux).

Done: real meshes in; `make eval-hard --randomize` runs; CI green. **Can
start:** today.

## E. Hardware & Viam Integration

**Owns:** `src/assistive_arm/robot_iface/viam.py`, `viam_convert.py`, new
`hardware/` (machine config JSON, mount CAD, calibration scripts).
**Consumes:** `RobotInterface` contract. **Provides:** a `ViamRobot` that
passes the same skill tests on a bench arm; measured frames and limits.

First tasks (bench before chair):
1. Viam machine: xArm6 module, xArm Gripper, two RealSense cameras, frame
   system (wrist cam parent = gripper, scene cam parent = world), builtin
   motion. Without the arm: the same config with `fake` components to exercise
   `ViamRobot` (SIMULATION.md §6).
2. Verify `viam_convert.py` against `arm.get_end_position()` at 3 poses; fix
   `ViamRobot.get_frames()` for the installed SDK's `get_images` format.
3. Force sensing: UFACTORY 6-axis F/T sensor via a small Viam sensor module
   (or joint-current estimate); measure noise, drift and guarded-stop
   overshoot at 2 cm/s.
4. Calibration scripts: hand-eye (ChArUco + OpenCV `calibrateHandEye`) and
   wheelchair-camera extrinsic; write results into the frame system.
5. Chair mount: armrest bracket design, measured base frame, re-measured
   keep-out boxes with a person seated (hand to S).
6. A `do_command` path to the xArm SDK's force/impedance mode for C's
   compliant door control.

Done: `tests/test_skills.py`-style assertions pass against `ViamRobot` on a
bench button board; checklist in SIMULATION.md §9 "Kinematics/Cameras/Force"
ticked. **Can start:** today with fake components; with the arm when it
arrives.

## F. Learning / VLA

**Owns:** `src/assistive_arm/skills/vla.py`, new `learning/` (data
collection, training configs), `data/` (git-ignored).
**Consumes:** `RobotInterface`, D's eval harness and rendered data, E's
teleop on the bench. **Provides:** policies behind `VLAPolicy`, scored by
`eval_sim.py`.

First tasks:
1. Zero-shot baseline: run Octo (small) and OpenVLA through `run_vla` on the
   bag and button scenes with the wheelchair camera; report success, clamp
   count, refusals. Expect low numbers; the point is a working pipeline.
2. Data: record scripted-skill rollouts in sim (D's randomised scenes) in
   LeRobot or RLDS format; add a teleop path (SpaceMouse or keyboard) for
   corrections.
3. Fine-tune Octo or pi0 on bag grasping; compare against `grasp_handle`.
4. The hybrid in PLAN.md §3: VLA for the approach, scripted guarded contact;
   measure whether it beats either alone on randomised bags.

Done: a policy that beats the scripted skill on at least one randomised
task in `eval_sim.py` with 0 user contacts and 0 keep-out refusals.
**Can start:** today (zero-shot in the current scenes).

## P. Props & instrumentation

**Owns:** new `props/` (measurements of bought items, CAD for the few custom
flat parts, ESP32 firmware), the sim twins' dimensions in `sim/scenes/`
(shared with D), and the packing list.
**Provides:** props that are either bought ready-made (A) or flat-pack and
table-clamped (B), each reporting its own success as JSON lines, each with a
sim twin built from measurements (PROPS.md).

First tasks: buy the "ahead" items on the PROPS.md shopping list; measure each
one with calipers or a tape measure and hand the numbers to D; assemble and
test the arcade-button panel (USB encoder or ESP32 reporting `pressed:N`);
fit hinges, catch and pull to the cabinet door and bolt up the 2020 frame;
time a full teardown and a 30-minute rebuild at home; write the adapter that
feeds prop events into the dashboard and the eval harness alongside
`SimRobot.events`.

Done: everything packs into the bags in the packing plan; the on-site setup
checklist takes ≤ 30 minutes; every demo prop reports its own success; every
sim twin matches the measured item.

## S. Safety lead (cross-cutting, part-time)

**Owns:** `planning/safety.py` review, `docs/SAFETY.md` (to write), the
test protocol. First tasks: hazard list (pinch points, unexpected motion,
dropped objects onto the user's lap, door swinging into the chair, speech
misrecognition), map each to a mitigation and a test; set the force/speed
numbers against ISO/TS 15066 body-region limits; define the bench →
mannequin → person trial gates. Done: risk assessment reviewed by someone
outside the team before M6.

---

## Who does what, by team size

| Team | Split |
|---|---|
| 3 people | (A + S + P), (B + F), (C + D + E) |
| 4 people | A + S, B, C + D, E + F + P |
| 6 people | one per A-F; the C owner doubles as safety lead; props shared by D and E |
| Weekend hackathon | follow PLAN.md §5: E gets the arm moving, B gets buttons found, A gets voice, C/D keep sim and skills green and do the demo glue |

## Integration points (put them on the calendar)

1. **Week -2:** B's offline perception baseline on recorded frames (Linux and
   Jetson numbers in COMPUTE.md); A's JSON intent on recorded phrases.
2. **Week -1:** C's fetch skill + B's perception in D's tabletop scene, ≥ 70%
   in `eval_sim.py`; E's Jetson passes the pre-event checklist.
3. **Day 1:** hand-eye calibration, live overlay, 4 of 5 real picks (E + B + C).
4. **Day 2:** voice + clarification + handover (A + C + S), unrehearsed
   objects, recorded backup video.
5. **After the event:** wheelchair skills on the same pipeline: button board
   bench test, compliant door control (C + E), F's policy vs the scripted
   skills on randomised scenes; every item in SIMULATION.md §9 signed off by S
   before the chair.
