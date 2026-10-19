# Plan

## 1. System architecture

```
 mic ──► ASR ──► intent ──► task planner ──► executor ───────────────────────────────┐
        (Whisper   (rules,     (plan as data,   │                                      │
         / Vosk)    or LLM →    read back to    │ clear_view → locate → observe →      │
                    validated   the user)       │ locate (close-up) → skill            │
                    Intent)                     ▼                                      ▼
                                    perception (Locator)                    skills (guarded)
                          wheelchair cam ─┐  open-vocab detector         press_button / grasp_handle /
                          wrist cam ──────┼► (+ SAM mask, + OCR) ──►     open_door / release / run_vla
                                          │  depth → 3D point + normal          │
                                          └─ camera pose from frame system      │ safety envelope:
                                                                                │ keep-out, speed, force,
                                                                                │ confirm-before-contact
                                                                                ▼
                                                           RobotInterface (one API)
                                              ┌───────────────────┼───────────────────┐
                                          FakeRobot          SimRobot (MuJoCo)     ViamRobot
                                         (unit tests)      arm, gripper, F/T,     Arm, Gripper, Camera,
                                                           2 RGB-D cams, sway     Sensor, Motion service
```

| Stage | Today (in repo) | Target |
|---|---|---|
| Speech | none (text in) | Whisper `small.en` or `distil-whisper` locally on the chair computer, push-to-talk button on the armrest; Vosk as a CPU fallback. Wake word optional. |
| Intent | `intent/parser.py` rule parser, `Intent.from_dict` validator | LLM (Claude / local Llama) with `Intent.json_schema()` as a tool/JSON schema; rules stay as the offline fallback and for "stop" (never wait on an LLM to stop). |
| Planner | `planning/task_planner.py`: fixed per-task step list | Same shape; add multi-step tasks ("take me to floor 3" = call button, wait, enter, press 3) as sequences of plans. |
| Perception | `perception/locate.py` + sim oracle, OWLv2 / Grounding DINO wrappers | Grounding DINO or OWLv2 boxes → SAM 2 mask → depth → point + support-plane normal. OCR/VLM reads button labels. |
| Skills | 3 hand-written + VLA runner | Add impedance control for handle turns; door hinge estimation; more handles. |
| Robot | `ViamRobot` (untested), `SimRobot`, `FakeRobot` | Viam xArm6 module + gripper + 2 RealSense + F/T sensor module; motion service with keep-out obstacles. |

### Frames and cameras

* **Base frame** = xArm6 base flange: +x forward of the chair, +y toward the user
  (arm on the right armrest), +z up. Everything the skills see is in this frame.
* **Wheelchair camera** (`mount="base"`): a D435/D455 on a mast behind and above
  the arm, looking forward-down. Calibrated once to the arm base (it moves with
  the chair, so the extrinsic does not change). Sees the whole workspace while the
  arm works; used for the coarse fix, for checking the arm stays clear of the user,
  and as the third-person view most VLAs were trained with.
* **Wrist camera** (`mount="wrist"`): a D405/D435 on the gripper. Hand-eye
  calibrated. Used for the close-up refine just before contact.
* **World camera** (`mount="world"`, optional): fixed in the environment for
  lab work and data collection; needs the chair's pose (AprilTag on the chair).

Why both: the wheelchair camera alone was ~7 mm off on a button from ~70 cm in
sim (and its normal estimate is noisy); the wrist camera alone can't find things
outside its narrow view and is blind when the arm is parked. Coarse-then-fine
uses each where it is good. In sim the parked arm's elbow occluded the wheelchair
camera badly enough that a "clear view" move became the first step of every plan.

## 2. Skill library

Every skill: free-space approach → confirm → guarded contact → verify → retract.
Parameters live in dataclasses at the top of each file.

### press_button (`skills/press_button.py`)
1. Target = button face point + panel normal (the normal comes from the
   **panel** plane, not the tiny button).
2. Close the fingers into one blunt pusher. Pre-press pose 8 cm out along the
   normal, tool axis = -normal, tool rolled "camera up"; if that roll needs a
   wrist flip, try rolled 180° (the closed fingers are symmetric).
3. Confirm. Tare the F/T sensor.
4. Guarded straight-line move to 2 cm *past* the expected face at 2 cm/s, stop at 6 N.
5. Verify: stopped on force (else "no resistance") and not more than 12 mm
   early (else "wrong surface": bezel, frame, neighbouring button).
6. Retract to the pre-press pose. Future: watch the button light up with the
   wrist camera as a second check.

### grasp_handle (`skills/grasp.py`), e.g. a shopping bag
1. Top-down approach; tool x along the handle bar so the fingers close across it.
2. Fingertips 2 cm below the seen top of the bar; guarded descent (15 N).
3. Tare, close slowly (jaw speed matters: a fast close knocks free objects away),
   check the jaws stopped on something.
4. Lift 12 cm, **weigh** with the F/T sensor; over 3 kg → put it back.
5. Carry at lift height, then lower to the hand-over pose in front of the user at
   0.05 m/s; check it is still in the hand. Hold until "let go".

### open_door (`skills/door.py`)
1. Target = handle grip point; normal from the **door** plane.
2. Grasp from the front, fingers closing vertically across a lever.
3. Turn: lever = arc about the spindle (axis = door normal, radius ≈ half the
   lever, chosen so the lever goes down); knob = twist about its own axis.
   Guarded at 25 N.
4. Swing: arc about the hinge line (vertical), sign chosen so the handle moves
   toward the robot for "pull", guarded at 30 N, 2° steps, 25° target.
5. Verify the handle is still in the hand. Hold the door (a closer would shut it).

The arc geometry is only as good as the hinge and spindle estimates. Sim
findings to design around: with the true hinge the lever door opens 25°; with
the hinge estimated from handle + door width (2-3 cm off) the guarded swing
stops early or loses the handle (correctly reported as partial/failed, not
success); spindle radius estimates of 0.04-0.06 m work for a 0.06 m lever,
0.065 m and above jam. Real fixes, in order: estimate the hinge from the
detected door edge and the door plane; then make the turn and swing compliant
(xArm impedance/force mode via a custom Viam module, or admittance control on
the F/T reading: move along the measured force within a cone around the arc).

### release, stop
`release` asks, opens, backs off 5 cm. `stop` halts and stays stopped.

### run_vla (`skills/vla.py`)
Any policy with `reset(instruction)` / `act(obs) -> delta pose + gripper`.
Each step is clamped (2 cm, 0.15 rad), checked against the keep-out boxes,
executed as a guarded move, and the episode is step-bounded. One confirmation
before the policy may move.

## 3. Modular pipeline vs end-to-end VLA

| | Modular (detect → locate → scripted skill) | VLA (OpenVLA 7B, Octo, pi0 / pi0.5) |
|---|---|---|
| Novel objects/scenes | Good where open-vocab detection works; each new *kind* of interaction needs a skill | Promising in-distribution; brittle on new embodiments without fine-tuning |
| Data needed | None to start | 50-500 demos per task on *this* arm/camera for useful fine-tunes |
| Precision (a 2 cm button) | mm-level from depth + guarded press | Typically cm-level; fine pushing of small targets is a known weak spot |
| Contact / force | Explicit thresholds, verifiable | No force input in most VLAs; must be wrapped |
| Explainability | Plan can be read back; every step logged | Opaque |
| Compute | CPU-ish (OWLv2 ~1 s/frame) | GPU: OpenVLA 7B ~16 GB bf16 at ~5 Hz; Octo small runs on a laptop GPU |
| Safety argument | Straightforward | Needs the wrapper anyway |

**Recommendation: modular core, VLA as a skill.** The planner owns the task
and the safety envelope; perception and the hand-written skills handle the
three reference tasks; a VLA is invoked only for open-ended "grab X" requests
that the handle-grasp skill doesn't cover, and only through `run_vla`. Ways
to combine, cheapest first:

1. **VLA for the approach, scripted for contact.** Let the policy bring the
   gripper near the object (it is good at the "where and how to approach"
   part), then hand over to a guarded scripted grasp/press.
2. **Scripted for the approach, VLA for manipulation** of awkward objects
   (crumpled bag handles, cups), with the locator's 3D target as a sanity bound.
3. **VLM as the planner/perception helper**: a VLM reads button labels, picks
   which handle, or answers "is the door open?" as a verifier.
4. **Fine-tune** Octo or pi0 on sim rollouts of the scripted skills + teleop
   demos from the real chair, evaluated in `scripts/eval_sim.py` before hardware.

## 4. Milestones

| # | Milestone | Done when | Depends on |
|---|---|---|---|
| M0 | Scaffold (this repo) | 72 offline tests pass; 3 sim skills run end to end | — |
| M1 | Real perception in sim | OWLv2/GDINO + SAM on rendered frames replaces the oracle; `make eval` ≥ 90% on the elevator with real detector + labelled button textures + OCR | M0 |
| M2 | Sim realism | Real xArm6 + gripper meshes; domain randomisation; `eval-hard` (sway + noise) ≥ 90% elevator/bag | M0 |
| M3 | Door v2 | Hinge from door-edge detection; compliant turn/swing; lever ≥ 80% with estimated hinge; knob working | M1 |
| M4 | Bench hardware | xArm6 on a table + Viam: `ViamRobot` passes the same skill tests against a real button board and bag; F/T sensor wired | M0 |
| M5 | Voice | Whisper + LLM intent + spoken confirmations; "stop" latency < 300 ms | M0 |
| M6 | On the chair | Arm on the chair mount, keep-out boxes measured, mannequin trials, risk assessment | M2-M5 |
| M7 | VLA track | Octo/pi0 fine-tuned on sim + teleop data, scored by `eval_sim.py` with zero user contacts | M2 |

## 5. Hackathon-sized MVP (one weekend, 3-5 people)

Goal: **"press button 7" works on a real xArm6 against a printed/3D-printed
button board, by voice, with a confirmation, and the same code passes in sim.**

* Hour 0-2: `make setup && make test` on every laptop; Viam machine with
  xArm6 + gripper + one RealSense (wrist or table-mounted); `make viam-check`.
* Hour 2-8: perception on real frames: OWLv2 "button" boxes + a digit reader
  (Florence-2 OCR or a small VLM on crops); fall back to AprilTag-marked
  buttons if detection is flaky (a legitimate hackathon shortcut, say so).
* Hour 2-8 in parallel: bench F/T. If there is no F/T sensor, guard on joint
  current via a Viam `do_command`, or on a cheap load cell under the button
  board for the demo.
* Hour 8-14: `ViamRobot.move_linear` guarded stepping tuned; press 9 buttons
  in a row; record a video.
* Hour 8-14 in parallel: Whisper push-to-talk + `parse()` + spoken
  confirmation ("Press button 7? Say yes.").
* Hour 14-20: bag grasp as the stretch goal; door only in sim for the demo.
* Demo: live voice → press, plus sim videos of bag and door, plus the
  `eval_sim.py` table and the safety framing slide.

See `docs/WORKSTREAMS.md` for who does what.
