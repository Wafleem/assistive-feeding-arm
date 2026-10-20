# Workstreams: one per section

Each section is a package with an owner, frozen interfaces, a first task you can
start today without the arm, and a "done when" that is a test or a number.
The plan for each one is in that package's README; this page is the overview
and the rules for working in parallel.

```
   5 language ──FeedIntent──► 6 orchestration ◄── 8 props (real-world success signals)
                                 │
     ┌───────────────┬───────────┼────────────────┐
     ▼               ▼           ▼                ▼
 2 food ──items──► 3 grasping  1 mouth ──MouthState──► 4 feeding motion & safety
                                 │
                 RobotInterface: 7 sim (MuJoCo)  |  ViamRobot (venue xArm6)
```

## Frozen contracts (change only in a small PR reviewed by every stream that uses them)

| Contract | File | Between |
|---|---|---|
| `MouthObservation`, `MouthState`, `MouthDetector` | `mouth_perception/types.py` | 1 → 4, 6, 7 |
| `FoodItem`, `MenuItem` / vocabulary | `food_perception/perceive.py`, `vocabulary.py` | 2 → 3, 5, 6 |
| `GraspCandidate`, `AcquisitionResult` (incl. `bite_offset`) | `grasping/types.py` | 3 → 4, 6 |
| `FeedingConfig`, `DeliveryResult` | `feeding/config.py`, `feeding/types.py` | 4 → 6, 7 (+ safety review) |
| `FeedIntent`, `Action` | `language/schema.py` | 5 → 6 |
| `S`, `TRANSITIONS`, `TIMEOUTS` | `orchestration/states.py` | 6 → everyone |
| `RobotInterface` (+ `now`, `wait`, `update_world`) | `robot_iface/base.py` | shared |
| `SafetyConfig` numbers | `motion/safety.py`, `feeding/config.py` | safety lead reviews |

**Spec tests:** each section's tests in `tests/` are marked `@todo` (xfail while
the code raises `NotImplementedError`). When you implement something, remove the
mark: from then on the test must pass. CI runs everything on each push.

---

## 1. Mouth perception — owner: tracking lead
**Owns** `mouth_perception/`. **First task:** implement `ConstantVelocityKF` and
`MouthTracker` against `tests/test_mouth_perception.py` (pure numpy, no hardware).
**Then:** the ArUco detector on the printed mouth (the demo path); sim detectors once the
printed-mouth scene exists; a Viam vision service on the Viam machine; MediaPipe later.
**Done when:** the printed mouth is tracked within 5 mm at ≥ 30 Hz; moving it by hand
triggers a retreat within 150 ms.

## 2. Food perception — owner: perception
**Owns** `food_perception/`, the detector wrappers in `vision/detectors.py`.
**First task:** `FoodPerception.find/scan` on sim renders with the oracle detector
(`test_find_grapes_in_sim`). **Then:** recorded RealSense plate frames → OWLv2 /
Grounding DINO / NanoOWL + SAM 2 / NanoSAM accuracy and latency table; VLM questions.
**Done when:** ≥ 90% recall on the demo foods, ≤ 1 cm 3D error, ≤ 300 ms on the AGX.

## 3. Grasping & acquisition — owner: manipulation
**Owns** `grasping/`. **First task:** `pca_topdown_grasp` against `tests/test_grasping.py`.
**Then:** `acquire_pinch` on FakeRobot, then on sim grapes; Viam gripper feedback;
AnyGrasp / Contact-GraspNet wrappers; utensil and cup (stretch); VLA (stretch, data plan
in `food_perception/README.md`).
**Done when:** ≥ 80% pinch success on randomised sim plates; 4/5 on the real plate.

## 4. Feeding motion & safety — owner: servo & safety lead
**Owns** `feeding/`, `motion/safety.py`. **First task:** `staging_pose`, `final_pose`,
`check_near_face` against `tests/test_feeding.py`. **Then:** `deliver_bite` with a
scripted tracker on FakeRobot; the moving face zone sent to the planner; all sim
head scenarios; measure stop latency on the real arm.
**Done when:** 0 face-plate contacts; the bite lands in the catch cup; a retreat on every
moved-away scenario; 10/10 on the real printed mouth. Both delivery modes are specified
and reviewed with the safety lead.

## 5. Language — owner: interaction
**Owns** `language/`, the dashboard. **First task:** `FeedIntent` validation + `rules.parse`
for every example in `tests/test_language.py`. **Then:** the provider-agnostic `IntentRouter` (local backend on the AGX by default, optional Claude/OpenAI API backends),
faster-whisper, browser TTS, dialogue preferences.
**Done when:** ≥ 95% correct intents on 100 recorded utterances; "stop" never waits on the LLM.

## 6. Orchestration — owner: integration
**Owns** `orchestration/`, Viam module packaging. **First task:** complete `TRANSITIONS`
(every failure edge) and `FeedingMachine` with fake components. **Then:** run the full
cycle in sim; dashboard; on the Linux PC with viam-server.
**Done when:** "feed me a grape" end to end in sim and on hardware; "stop" from any state < 300 ms.

## 7. Simulation — owner: integration (shared)
**Owns** `sim/`, `src/assistive_arm/sim/`, `robot_iface/sim.py`, `scripts/`.
**First task:** replace the jaw head in `sim/scenes/feeding.xml` with the printed mouth
(face plate + catch cup, from the same CAD as the print); then the `FeedingWorld` hook:
scenarios, "bite in cup", metrics.
**Then:** `scripts/eval_feeding.py`, domain randomisation, real meshes, measured props.
**Done when:** the eval table runs in CI and every section reports its numbers with it.

## 8. Props — owner: integration (shared)
**Owns** `docs/PROPS.md`, `props/` (to create: measurements, CAD, firmware).
**First task:** design and print the fake mouth (face plate with an always-open opening,
removable catch cup, ArUco recess, stand); share the CAD with sim; buy and measure the foods.
**Done when:** everything packs per the packing plan and sets up in ≤ 30 minutes.

## Integration points

1. **Week -2:** sections 1, 3, 4, 5 have their first specs passing; 7 has the head scenarios.
2. **Week -1:** full bite cycle in the printed-mouth sim scene (6 + all); the AGX serves the
   models as Viam services (a remote part of the Viam machine).
3. **Day 1:** real arm: calibration, overlay, picks, approach to the fake mouth.
4. **Day 2:** voice, confirmations, retreat demo, rehearsal, backup video.
