# Engineering overview (original plan)

> The original hackathon plan and safety framing. The final build runs on an SO-101 arm; see the [README](../README.md).

## assisted eating with an xArm6

> *We used to aim at armor plates. Now we aim a spoon, gently.*
> — NYU RoboMaster controls leads, "Physical AI for Good"

Many people need help to eat, for example after a stroke, with ALS, MS, a
spinal cord injury or cerebral palsy, or in old age. Being fed by someone else
at every meal costs dignity and a carer's time. This project builds a robot arm
that feeds a person **at their pace, on their words**: "feed me the
strawberries", "a sip of water", "slower", "I'm done".

For the hackathon the arm feeds a **real person** (a team member) in `person` mode:
the spoon stops 3 cm in front of the lips, and the person closes the gap. A 3D-printed
face with an always-open mouth stays available for rehearsals without a person.
The bench rig right now is an SO-101 arm with a spoon fixed in its gripper and a webcam
on top, scooping small food from bowls.
The arm is the venue's UFACTORY **xArm6** with a standard parallel gripper, driven
through the venue's **Viam** machine. A **Jetson AGX Orin** adds the AI compute
that machine may not have.

**Status: scaffold, with perception started.** Mouth tracking for a real face (MediaPipe,
Kalman filter, ready and retreat gates) and bowl finding and tracking are implemented and
tested offline; `scripts/perception_viewer.py` shows them live. The shared core already works: the robot API, a MuJoCo
scene with a moving fake head, the cameras, the safety envelope and the Viam
backend. Each of the eight sections below is a package with:
* a plan README,
* typed interfaces,
* stubs that raise `NotImplementedError("TODO(...)")`,
* spec tests marked as expected failures (`xfail`) until someone implements them.

The previous wheelchair-assist version is at git tag `v0-wheelchair-skills`.

## How it works

```
 "feed me a strawberry" ──► 5 language ──► 6 orchestration (state machine)
                                             │
     ┌───────────────────┬──────────────────┼────────────────────┐
     ▼                   ▼                  ▼                    ▼
 2 food perception   3 grasping &      1 mouth perception    4 feeding motion
 (open-vocab detect,   acquisition       (track, predict,      & safety (approach,
  SAM masks, depth)   (pinch / utensil   ready gate, retreat)   stop 3 cm out, retreat)
                       / cup; open model)
     └───────────────────┴──────── RobotInterface ─────────────┘
                     SimRobot (MuJoCo, 7 sim)  |  ViamRobot (xArm6 + gripper + RealSense)
```

## Safety framing

The arm works at a person's face. These rules don't change, in priority order:

1. **Hardware first:**
   * an e-stop within reach of the user and of a carer;
   * the xArm's collision detection at its most sensitive setting;
   * reduced-mode speed limits set on the arm's controller.
2. **Two delivery modes, never mixed up:**
   * With the printed mouth (the demo), the arm releases the bite just inside the
     opening, into the catch cup.
   * With a real person (future work, and the code's default), the bite stops
     **3 cm in front of the lips** and the person closes the gap; nothing enters the mouth.
3. **Ask before every bite** in demo mode: "Ready for a strawberry?" No answer means no.
4. **Only approach a ready mouth:** still for 0.5 s and tracked with confidence (and
   open, for a real face; the printed one is always open).
5. **Slow, and along the face normal:** at most 2 cm/s on the final approach, and at most
   3 cm/s anywhere within 15 cm of the face.
6. **Leave at once** on any of: head motion, "stop" or "wait", any force over 3 N near
   the face, lost tracking, or a timeout.
7. **A moving keep-out zone** around the head is passed to the motion planner every tick.
8. **Claims need evidence:** a bite counts as delivered only if the world agrees (sim
   metrics, re-detection, gripper state).

This is not a certified safety case. Before a real person, the team needs a
written risk assessment (ISO 10218, ISO/TS 15066 and ISO 13482 as references)
and trials with the printed mouth only.

## Sections (one package, one owner each)

| # | Section | Package | Plan |
|---|---|---|---|
| 1 | Mouth perception: detect, track (Kalman filter), predict, ready gate, retreat triggers | `src/assistive_arm/mouth_perception/` | [README](../src/assistive_arm/mouth_perception/README.md) |
| 2 | Food perception: open-vocabulary detection + masks + depth; VLM; VLA discussion | `src/assistive_arm/food_perception/` | [README](../src/assistive_arm/food_perception/README.md) |
| 3 | Grasping & acquisition: PCA + AnyGrasp/Contact-GraspNet; pinch / utensil / cup | `src/assistive_arm/grasping/` | [README](../src/assistive_arm/grasping/README.md) |
| 4 | Feeding motion & safety: staging, servoed final approach, release into the printed mouth (or standoff for a person), retreat | `src/assistive_arm/feeding/` | [README](../src/assistive_arm/feeding/README.md) |
| 5 | Language layer: speech, LLM router → validated intents, rules, dialogue | `src/assistive_arm/language/` | [README](../src/assistive_arm/language/README.md) |
| 6 | Orchestration: the bite-cycle state machine with timeouts and aborts | `src/assistive_arm/orchestration/` | [README](../src/assistive_arm/orchestration/README.md) |
| 7 | Simulation: feeding scene, head scripts, metrics, eval | `src/assistive_arm/sim/`, `sim/` | [README](../src/assistive_arm/sim/README.md) |
| 8 | Props: 3D-printed mouth with catch cup, plate, foods, utensils (store-bought or flat-pack) | — | [docs/PROPS.md](PROPS.md) |
| 9 | Hard-coded pick and place on the SO-101 bench rig (`pi-arm2`): teach poses with the leader arm, replay them | `src/assistive_arm/pick_place/` | [README](../src/assistive_arm/pick_place/README.md), [docs/SO101_SETUP.md](SO101_SETUP.md) |

**Shared core** (works now, used by every section):
* `geometry.py`
* `robot_iface/`: the Fake, Sim and Viam robots
* `vision/`: camera types, deprojection, detector wrappers, `Locator`
* `motion/`: the safety envelope and guarded move primitives

## Quick start

```bash
make setup        # .venv (Python 3.12) with numpy, mujoco, pytest
make test         # passing tests + "xfail" specs for unimplemented sections
make render       # PNGs of the feeding scene from the wrist and fixed cameras (out/)
make viewer       # interactive MuJoCo viewer
make todo         # list every TODO by section
```

New here? Read [AGENTS.md](../AGENTS.md) (the rules: plans before code, modularity, safety),
then [CONTRIBUTING.md](../CONTRIBUTING.md), [docs/PLAN.md](PLAN.md),
[docs/WORKSTREAMS.md](WORKSTREAMS.md), and your section's README. Every folder has a
README with its plan; `src/assistive_arm/README.md` maps the packages.

## Docs

| Doc | What |
|---|---|
| [AGENTS.md](../AGENTS.md) | how to implement anything here: modularity and import rules, interfaces, safety, tests, git (CLAUDE.md imports it) |
| [docs/PLAN.md](PLAN.md) | architecture, 90-second demo, milestones, risks, team roles |
| [docs/WORKSTREAMS.md](WORKSTREAMS.md) | one stream per section: contracts, first tasks, done-when |
| [docs/SIMULATION.md](SIMULATION.md) | developing everything in MuJoCo first |
| [docs/COMPUTE.md](COMPUTE.md) | what runs on the venue's Viam machine, the AGX Orin and the Mac |
| [docs/PROPS.md](PROPS.md) | the 3D-printed mouth and food props: shopping and packing lists |
