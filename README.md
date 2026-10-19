# physical-ai-hackaton: a voice-driven assistive arm for wheelchair users

A UFACTORY **xArm6** with a parallel-jaw gripper (the xArm Gripper, 0-85 mm),
mounted on a power wheelchair, that does the small reach-and-touch jobs a
seated user often can't:

> "Press button 7 on the elevator." · "Grab the shopping bag." ·
> "Grab the door handle and open it." · "Stop." · "Let go."

Speech becomes a validated intent; open-vocabulary vision finds the target in
3D using a camera **on the wheelchair** plus one **on the wrist**; a small
library of force-guarded skills does the work through Viam on the real arm,
or through MuJoCo before there is an arm. Open VLA models (OpenVLA, Octo, pi0)
plug in as an optional skill that runs inside the same safety envelope.

There is no hardware yet. Everything here runs offline: `make setup && make test`
runs 72 tests, including MuJoCo end-to-end runs where the arm finds and presses an
elevator button, lifts a 1.2 kg bag and hands it over, and opens a lever door.

## User stories

| As a seated user I want to... | so that... | Skill | Status in sim |
|---|---|---|---|
| say "press 7" in an elevator | I don't need to ask a stranger to press my floor | `press_button` | works: 10/10 buttons, with and without 4 mm chair sway |
| say "grab the shopping bag" | I can get my bag from a counter onto my lap | `grasp_handle` | works: grasp, weigh (1.22 kg measured / 1.2 kg true), hand over |
| say "open the door" | I can get through a pull door alone | `open_door` | lever door works when the hinge is known; with an estimated hinge it stops partway (safe, not yet useful); knob: not yet |
| say "stop" at any time | the arm freezes immediately | `stop` | software stop wired; hardware e-stop is mandatory on the real chair |
| say "let go" | I take the item when I'm ready | `release` | works |
| be asked before the arm touches anything | nothing happens that I didn't agree to | `confirm` hook | every contact phase asks; alarm/call buttons ask twice |

## Safety framing

A person sits within arm's reach of this robot, often one who cannot move out
of its way. The design rules, in priority order:

1. **Hardware first.** A hardware e-stop the user can hit (and a wireless one
   for a carer), the xArm's own collision detection set to its most sensitive
   level, reduced-mode joint speed limits, and a gripper force limit. Software
   below is a second layer, never the only one.
2. **Nothing touches the world without a "yes".** Every skill has a free-space
   approach, then asks the user (`SkillContext.confirm`, voice or a big
   button), then does a *guarded* contact move. No answer = no.
3. **Keep-out zones around the user.** Boxes for torso/head, legs and the chair
   (`planning/safety.py`) are checked for every target and every straight-line
   path; the same boxes go to Viam's motion planner as obstacles; in sim the
   arm is also collision-checked against a body model of the user, and any
   contact latches a stop.
4. **Slow near people, slow in contact.** 0.15 m/s free space, 0.05 m/s within
   20 cm of the user, 0.02 m/s for any contact move.
5. **Force limits everywhere.** Contact moves stop at 6 N (button), 15 N (generic
   ceiling), 25 N (handle turn), 30 N (door swing). The real arm needs a wrist
   F/T sensor (or a calibrated joint-current estimate) for this to be real.
6. **Claims need evidence.** A skill reports success only if the world agrees:
   the press met resistance where the button was expected, the gripper stopped
   on something, the load was weighed, the handle is still in the hand after
   the swing. The eval harness counts false success claims separately; the
   target is zero.
7. **The arm parks out of the way** (low, to the right, tool forward) when not
   working, which also keeps the wheelchair camera's view clear.

None of this is a certified safety case. Before anyone sits in the chair: a
risk assessment against ISO 10218 / ISO/TS 15066 (collaborative force and
speed limits) and the assistive-robot standard ISO 13482, and testing with a
mannequin.

## Quick start

```bash
make setup        # .venv with Python 3.12, numpy, mujoco, pytest (uses uv if present)
make test         # 72 tests, ~15 s, offline
make demo         # "press button 7 on the elevator" in MuJoCo, prints every step
make demo SCENE=bag SAY="grab the shopping bag"
make eval         # success-rate table: world-verified success, false claims, user contacts
make viewer       # watch it (macOS: runs under mjpython)
```

## Repo map

```
src/assistive_arm/
  geometry.py            Pose, quaternions, look-at rotations, arc waypoints (SI units, base frame)
  intent/
    schema.py            Intent + Task: the validated command schema (also the LLM's JSON schema)
    parser.py            offline rule-based parser for the example phrases; asks when unsure
  perception/
    types.py             CameraSpec (wrist | base | world mount), RGBDFrame, Detection, Target3D
    detectors.py         Detector protocol; sim oracle; NoisyDetector; OWLv2 + Grounding DINO wrappers
    geometry.py          deprojection, plane fitting
    locate.py            detection + depth -> 3D point + approach normal (from the support surface)
  planning/
    safety.py            keep-out boxes, reach, speed and force limits
    task_planner.py      Intent -> inspectable plan (clear view, locate, observe, locate, skill)
  skills/
    base.py              SkillContext/SkillResult, safe_move, guarded_linear
    press_button.py      approach along the normal, confirm, press to a force threshold
    grasp.py             top-down handle grasp, weigh, carry, hand over; release
    door.py              grasp, turn lever/knob, swing along the hinge arc
    vla.py               run any VLA policy step-by-step inside the safety envelope; OpenVLA adapter
  robot_iface/
    base.py              RobotInterface: the one API every skill uses
    fake.py              FakeRobot: kinematics-free, for unit tests
    sim.py               SimRobot: MuJoCo arm, gripper, F/T sensor, two RGB-D cameras, sway
    viam.py              ViamRobot: xArm6 + gripper + RealSense + F/T through the Viam SDK (untested)
    viam_convert.py      mm/deg and orientation-vector conversions (tested)
  executor.py            runs a plan on any RobotInterface
sim/
  models/xarm6_wheelchair.xml   xArm6-dimension arm + parallel gripper on a swaying wheelchair, seated user
  scenes/{elevator,bag,door_lever,door_knob}.xml
scripts/sim_demo.py      one request end-to-end, optional viewer / camera PNGs
scripts/eval_sim.py      benchmark table over seeds, sway, detector noise
docs/PLAN.md             architecture, skills, modular-vs-VLA, milestones, hackathon MVP
docs/SIMULATION.md       how to build and test all of it before the arm and chair exist
docs/WORKSTREAMS.md      the work split into parallel pieces for a team
tests/                   intent, geometry, safety, skills, executor, MuJoCo end-to-end
```

## Lineage

Lessons carried over from the armfarm pick-and-place hackathon project (xArm6 +
vacuum gripper + wrist D435 + natural-language commands): look before every
move; collision-checked planning with live obstacles; treat a carried object
as part of the arm for collision checks; approach, pause and confirm before
contact; close the loop on the gripper instead of trusting a command; and map
language to a validated schema, never straight to motion. This is a fresh
repo, not a fork.
