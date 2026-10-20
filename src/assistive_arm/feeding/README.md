# 4. Feeding motion and safety

**Owner:** stream 4 (servoing + safety; pair with the safety lead).
**Goal:** bring the bite to *just in front of* the lips, smoothly, only when the
person is ready, and leave the moment anything looks wrong.

## Rules that do not change
- The bite stops `standoff` (3 cm) **in front of the lips**. The person closes the gap.
- Final approach only along the face normal, at ≤ 2 cm/s; ≤ 3 cm/s anywhere within 15 cm of the mouth.
- Confirm before **every** bite (`confirm_each_bite=True` for the demo).
- Retreat immediately on: head motion (HeadMotionMonitor), "stop"/"wait", any force
  above 3 N near the face, stale tracking, timeout.
- Hardware e-stop within reach; xArm collision sensitivity at max; reduced mode on the controller.

## Interfaces
- `FeedingConfig` (all the numbers; safety lead reviews changes)
- `face_zone / check_near_face / speed_cap` (moving keep-out around the head)
- `staging_pose / final_pose / deliver_bite / retreat`, returning `DeliveryResult` with metrics

## Plan
1. Geometry: staging and final poses from a `MouthState` (+ the bite offset).
2. Moving face keep-out: sphere around the head + approach corridor; pushed to the
   Viam motion service as an obstacle every tick (`robot.update_world`), carried item
   attached to the gripper.
3. Servo loop for the final approach at 20 Hz on the latency-compensated mouth state
   (this is where the RoboMaster smoothing/latency experience pays off).
4. Bite detection: mouth closes after being open near the bite / pull on the gripper / "got it".
5. Retreat along +normal, then park.

## TODO checklist
- [ ] Poses + face-zone checks → `tests/test_feeding.py`
- [ ] `deliver_bite` on FakeRobot with a scripted MouthTracker
- [ ] Sim: all FeedingWorld scenarios (still, drift, lean-in, turn-away, never-opens)
- [ ] Measured stop latency on the real arm

## Done when
Sim: 0 head contacts, delivery within 1 cm of the standoff point, retreat on every
turn-away, timeout on every never-opens. Real: same on the fake mouth, 10/10.
