# Plan: assisted eating

## 1. Architecture

```
                         ┌──────────────── Venue Viam machine (Linux PC) ─────────────────────┐
 fixed RGB-D cam ───────►│ viam-server: xArm6 + gripper + cameras, builtin motion service      │──► arm
 (plate + printed mouth) │ 1 mouth perception: ArUco → Kalman tracker → ready gate / retreat   │
 wrist RGB-D cam ───────►│ 6 orchestration state machine, 4 feeding motion & safety            │
                         │ dashboard: camera + overlays, state, transcript, big STOP            │
                         └──────────────────────────────┬─────────────────────────────────────┘
                                   Viam remote part, our own switch + cable
                         ┌──────────────────────────────▼─────────────────────────────────────┐
 mic ───────────────────►│ Jetson AGX Orin 32 GB: the additional AI compute                    │
                         │ 2 food perception: NanoOWL/OWLv2 + NanoSAM/SAM 2 + depth → items    │──► FoodItem[]
                         │ 3 grasp model (AnyGrasp / Contact-GraspNet), optional VLM / VLA      │──► GraspCandidate
                         │ 5 language: faster-whisper → local LLM → FeedIntent (rules first)    │──► FeedIntent
                         └────────────────────────────────────────────────────────────────────┘
 Mac: development + 7 MuJoCo simulation only. Every model is a Viam vision/generic service,
 so moving it between the AGX and the Viam machine is a config change (docs/COMPUTE.md).
```

**The fake mouth** is a 3D-printed face plate with an always-open opening and a
removable catch cup behind it. The arm releases each bite just inside the opening,
and the cup is dumped after every use (docs/PROPS.md). The feeding code keeps a
separate, default `person` mode for later, in which the bite stops 3 cm in front of
the lips (`feeding/README.md`).

**Bite cycle** (section 6):
`IDLE → SELECT_ITEM → ACQUIRE → VERIFY_GRASP → WAIT_READY → APPROACH → DELIVER (release into the opening) → RETREAT → next`.
"Stop" from any state halts the arm, and it retreats if it is near the face.
Every state has a timeout.

**Cameras (an assumption to confirm with the venue):** one fixed RealSense-class
RGB-D camera on a clamp-on post that sees the plate and the face, plus one on the
wrist. If only one camera is available, use it fixed for the mouth (at the end
of the approach the food blocks the wrist camera's view) and add the wrist
camera for food when you have it.

## 2. The 90-second demo

| t | Who | What happens | What it shows |
|---|---|---|---|
| 0:00 | User | "What's on the plate?" → "Grapes, strawberries and melon." | Open-vocabulary perception + voice |
| 0:10 | User | "Feed me a strawberry." The overlay highlights the strawberry; the arm pinches it, checks the grip, lifts | Detection → grasp → closed-loop check |
| 0:25 | Robot | "Ready for a strawberry?" User: "Yes." The arm waits at the staging point until the printed mouth is still and tracked | Confirm before every bite; ready gate |
| 0:35 | Robot | Final approach along the face normal, following small movements of the mouth smoothly; releases the strawberry into the opening and it drops into the catch cup | Kalman prediction + latency compensation (the RoboMaster part) |
| 0:45 | Teammate | Slides or turns the printed mouth mid-approach, and the arm retreats at once | Safety: retreat on motion |
| 0:55 | User | "Slower, and a grape this time." | LLM router, dialogue preferences |
| 1:10 | User | "Stop!" mid-move: the arm freezes. "I'm done.": the arm parks | Safety words override everything |
| 1:20 | Slide | Sim eval table: deliveries, distance at delivery, max speed near the face, 0 contacts | Evidence |

## 3. Milestones

| When | Milestone | Done when |
|---|---|---|
| Week -2 | Offline perception | Mouth tracker running on recorded video of the fake mouth; food detection + masks on ~50 recorded plate frames; latency measured on each device |
| Week -2 | First implementations of the stubs | The KF, ready gate, rule parser, PCA grasp and feeding-pose specs in `make test` pass instead of xfail |
| Week -2 | Printed mouth | Face plate + catch cup designed, printed, test-fitted; the same CAD exported to the sim scene |
| Week -1 | Full bite cycle in sim | "Feed me a grape" end to end in the printed-mouth sim scene, including the moved-away retreat; eval table |
| Week -1 | AGX ready | JetPack, TensorRT engines (NanoOWL/NanoSAM), local LLM, and Viam services on the AGX as a remote part |
| Day 1 | Real arm | Hand-eye calibration; overlay on the live camera; 4 of 5 real grape picks; bites released into the printed mouth |
| Day 2 | Full demo | Voice + confirmations + retreat on head motion; rehearse the 90 s; record a backup video |
| Stretch | Utensil + cup, VLA | Spoon from a rest + a scoop of yogurt; cup with a straw; SmolVLA/pi0 acquisition compared in sim |

## 4. Risks

| Risk | Mitigation |
|---|---|
| The arm hitting the face plate | Speed caps, force limit, moving keep-out zone with a corridor into the opening, confirm each bite, e-stop. The printed-mouth mode is never used with a person; person mode keeps a 3 cm standoff |
| Mouth tracking jitter or latency makes the approach jerky | Kalman filter + latency compensation, rate-limited servoing, stop on stale tracking; measure latency end to end |
| Depth inside the opening reads the catch cup, not the face | ArUco tag on the face plate as the main path; take depth from the plate, never from the hole |
| Bites miss the opening or bounce out | Release only within the opening; a deep catch cup; measure the hit rate in sim and on the prop |
| Soft or wet food slips or bruises | Firm foods (grapes, cheese or melon cubes), low grip force, holding check before the approach |
| Detection fails on shiny plates or black items | Matte plate, colourful foods, depth validity check |
| Venue camera, network or compute differs from the plan | Fixed-camera-only fallback; our own switch and cables; if the AGX fails, run the models on the Viam machine or the Mac (slower) |
| LLM slow, wrong or offline | Safety words handled by rules before any model; per-backend timeouts; local → (optional) cloud API → rules fallback chain; schema validation |
| Spills (cup) | Stretch goal only; lid + straw; never tilt near the face |

## 5. Team roles

| Role | Section(s) | Best fit |
|---|---|---|
| Tracking lead | 1 mouth perception | RoboMaster vision/tracking: Kalman, latency compensation |
| Servo & safety lead | 4 feeding motion & safety (+ safety review of all the numbers) | RoboMaster controls: smooth servoing, limits |
| Perception | 2 food perception (+ VLM) | detection/segmentation, Jetson TensorRT |
| Manipulation | 3 grasping & acquisition (+ VLA stretch) | grasping, Viam motion |
| Interaction | 5 language + dashboard | LLMs, speech, UI |
| Integration | 6 orchestration + 7 sim + 8 props (incl. printing the mouth) | systems, MuJoCo, CAD/printing, hardware bring-up |

With fewer people: (1+4), (2+3), (5+6), sharing 7 and 8. See `docs/WORKSTREAMS.md`.
