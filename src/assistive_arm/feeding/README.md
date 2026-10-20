# 4. Feeding motion and safety

**Owner:** stream 4 (servoing + safety; pair with the safety lead).
**Goal:** bring the bite to the mouth smoothly, only when it's safe, and leave the
moment anything looks wrong.

## Two delivery modes (`FeedingConfig.target`), to review with the safety lead
| Mode | Used for | Delivery |
|---|---|---|
| `printed_mouth` | the hackathon prop (3D-printed, always open, catch cup behind) | the bite goes `insertion_depth` (~1.5 cm) **into** the opening; the gripper opens; the bite drops into the catch cup, which is emptied after every use |
| `person` (**default**) | a real person, future work | the bite stops `standoff` (3 cm) **in front of** the lips; the person closes the gap; nothing ever enters the mouth |

The demo config must set `printed_mouth` explicitly, and that mode must never be
used with a person.

## Rules in both modes
- Final approach only along the face normal, at ≤ 2 cm/s.
- ≤ 3 cm/s anywhere within 15 cm of the face.
- Confirm before **every** bite (`confirm_each_bite=True`).
- Retreat immediately on any of:
  - the prop or head moving (HeadMotionMonitor);
  - "stop" or "wait";
  - any force above 3 N near the face;
  - stale tracking;
  - a timeout.
- Hardware e-stop within reach; the xArm's collision sensitivity at maximum; reduced mode set on the controller.

## Interfaces
- `FeedingConfig`: every number lives here; the safety lead reviews changes.
- `face_zone`, `check_near_face`, `speed_cap`: the moving keep-out zone around the head.
- `staging_pose`, `final_pose`, `deliver_bite`, `retreat`: each returns a `DeliveryResult` with metrics.

## Plan
1. Compute the staging and final poses from a `MouthState`, per mode, including the bite offset.
2. Moving face keep-out zone:
   - a sphere around the head plus the approach corridor (in printed-mouth mode the
     corridor continues into the opening);
   - sent to the Viam motion planner every tick (`robot.update_world`);
   - the carried item attached to the gripper as part of the arm.
3. Servo loop for the final approach at 20 Hz, using the latency-compensated mouth state.
4. Deliver:
   - `printed_mouth`: release, then check the gripper is empty;
   - `person`: hold and wait for the bite.
5. Retreat along +normal, then park.

## TODO checklist
- [ ] Poses + face-zone checks → `tests/test_feeding.py` (both modes)
- [ ] `deliver_bite` on FakeRobot with a scripted MouthTracker
- [ ] Sim: the FeedingWorld scenarios (still, drift, moved away, tracking lost)
- [ ] Measure the stop latency on the real arm

## Done when
- Sim: 0 face-plate contacts; the bite lands in the catch cup; a retreat on every
  moved-away scenario.
- Real: 10/10 on the printed mouth.
