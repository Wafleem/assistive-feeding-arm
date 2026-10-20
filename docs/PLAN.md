# Plan: assisted eating

## 1. Architecture

```
                         ┌──────────────── Jetson Orin Nano (at the camera) ─────────────────┐
 fixed RGB-D cam ───────►│ 1 mouth perception: detector → Kalman tracker → ready gate /       │──► MouthState @30 Hz
 (face + plate)          │   head-motion monitor   (+ wake word, safety monitor)              │
                         └────────────────────────────────────────────────────────────────────┘
                         ┌──────────────── Jetson AGX Orin 32 GB (model host) ────────────────┐
 wrist RGB-D cam ───────►│ 2 food perception: NanoOWL/OWLv2 + NanoSAM/SAM 2 + depth → items   │──► FoodItem[]
 mic ───────────────────►│ 5 language: faster-whisper → local LLM → FeedIntent (rules fallback)│──► FeedIntent
                         │ 3 grasp model (AnyGrasp / Contact-GraspNet), optional VLM / VLA     │──► GraspCandidate
                         └────────────────────────────────────────────────────────────────────┘
                         ┌──────────────── Venue Linux PC ────────────────────────────────────┐
                         │ viam-server: xArm6 + gripper + cameras, builtin motion service      │
                         │ 6 orchestration state machine, 4 feeding motion & safety            │──► arm
                         │ dashboard: camera + overlays, state, transcript, big STOP            │
                         └────────────────────────────────────────────────────────────────────┘
 Mac: development + 7 MuJoCo simulation only.  Every model is a Viam vision/generic service,
 so moving it between machines is a config change (docs/COMPUTE.md).
```

**Bite cycle** (section 6):
`IDLE → SELECT_ITEM → ACQUIRE → VERIFY_GRASP → WAIT_READY → APPROACH → DELIVER → WAIT_BITE → RETREAT → next`.
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
| 0:25 | Robot | "Ready for a strawberry?" User: "Yes." The arm waits at the staging point until the mouth is open and still | Confirm before every bite; ready gate |
| 0:35 | Robot | Final approach along the face normal, stopping 3 cm from the lips while following small head motions smoothly | Kalman prediction + latency compensation (the RoboMaster part) |
| 0:45 | Fake mouth | The head turns away mid-approach, and the arm retreats at once | Safety: retreat on head motion |
| 0:55 | User | "Slower, and a grape this time." | LLM router, dialogue preferences |
| 1:10 | User | "Stop!" mid-move: the arm freezes. "I'm done.": the arm parks | Safety words override everything |
| 1:20 | Slide | Sim eval table: deliveries, distance at delivery, max speed near the face, 0 contacts | Evidence |

## 3. Milestones

| When | Milestone | Done when |
|---|---|---|
| Week -2 | Offline perception | Mouth tracker running on recorded video of the fake mouth; food detection + masks on ~50 recorded plate frames; latency measured on each device |
| Week -2 | First implementations of the stubs | The KF, ready gate, rule parser, PCA grasp and feeding-pose specs in `make test` pass instead of xfail |
| Week -1 | Full bite cycle in sim | "Feed me a grape" end to end in `sim/scenes/feeding.xml`, including the turn-away retreat and the never-opens timeout; eval table |
| Week -1 | Jetsons ready | JetPack, TensorRT engines (NanoOWL/NanoSAM), MediaPipe, local LLM, and Viam services on the AGX and the Nano |
| Day 1 | Real arm | Hand-eye calibration; overlay on the live camera; 4 of 5 real grape picks; approach to the fake mouth stopping at 3 cm |
| Day 2 | Full demo | Voice + confirmations + retreat on head motion; rehearse the 90 s; record a backup video |
| Stretch | Utensil + cup, VLA | Spoon from a rest + a scoop of yogurt; cup with a straw; SmolVLA/pi0 acquisition compared in sim |

## 4. Risks

| Risk | Mitigation |
|---|---|
| Anything touching the face | 3 cm standoff, speed caps, force limit, moving keep-out zone, confirm each bite, e-stop; mannequin only |
| Mouth tracking jitter or latency makes the approach jerky | Kalman filter + latency compensation, rate-limited servoing, stop on stale tracking; measure latency end to end |
| MediaPipe doesn't detect the fake mouth | Fiducial (ArUco) detector on the mannequin as the main demo path |
| Soft or wet food slips or bruises | Firm foods (grapes, cheese or melon cubes), low grip force, holding check before the approach |
| Detection fails on shiny plates or black items | Matte plate, colourful foods, depth validity check |
| Venue camera or network differs from the plan | Fallback using only the fixed camera; everything wired, no dependence on venue Wi-Fi |
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
| Integration | 6 orchestration + 7 sim + 8 props | systems, MuJoCo, hardware bring-up |

With fewer people: (1+4), (2+3), (5+6), sharing 7 and 8. See `docs/WORKSTREAMS.md`.
