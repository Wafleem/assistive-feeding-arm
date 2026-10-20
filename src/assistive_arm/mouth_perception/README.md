# 1. Mouth perception

**Owner:** stream 1 (best fit: the RoboMaster tracking / Kalman people).
**Goal:** know where the mouth is, which way the face points, whether it is open,
and where it *will* be when the arm moves, at 30+ Hz, close to the camera
(Orin Nano).

## Interfaces (`types.py`, frozen once agreed)
- in: RGB-D frames from `RobotInterface.get_frames()` (fixed and/or wrist camera)
- `MouthDetector.observe(robot) -> MouthObservation | None`: one frame, no filtering
- `MouthTracker.state(now) -> MouthState`: filtered, latency-compensated
  (position, velocity, face normal, open/closed, confidence, age)
- `ReadyGate.update(state, now) -> bool`, `HeadMotionMonitor.check(state) -> MotionVerdict`

## Plan
1. **Fake mouth first** (`FiducialMouthDetector`): ArUco tag on the mannequin +
   measured tag→mouth offset; jaw state from the prop's servo (serial) or a second
   tag. Most robust for the demo.
2. **Sim** (`SimOracleMouthDetector`, `CameraSegMouthDetector`): ground truth and the
   camera path on MuJoCo renders, with noise/dropouts/delay knobs.
3. **Real face** (`MediaPipeMouthDetector`): Face Landmarker, inner-lip landmarks,
   mouth aspect ratio, depth deprojection, face-plane normal. The "it works on a
   person" story.
4. **Tracking** (`kalman.py`, `tracker.py`): constant-velocity KF on capture
   timestamps, Mahalanobis gating, latency compensation (`predict_ahead`),
   normal smoothing, open/closed hysteresis.
5. **Gates** (`gates.py`): ready = open + still for N ms; retreat on fast motion,
   drift, turn, stale tracking, mouth closing.
6. Wrap the detector + tracker as a Viam vision service on the Orin Nano.

## TODO checklist
- [ ] KF + tracker + gates (make `tests/test_mouth_perception.py` pass; remove the xfail marks)
- [ ] Sim oracle + camera-seg detectors on `sim/scenes/feeding.xml`
- [ ] Fiducial detector on the real fake mouth
- [ ] MediaPipe detector on a real face (optional for the demo)
- [ ] Measure end-to-end latency and tune `TrackerParams.latency`, `KalmanParams`
- [ ] Viam vision service wrapper on the Orin Nano

## Done when
Sim: tracked mouth within 5 mm (still head) and the retreat fires within 150 ms
of a scripted turn-away. Real: fake mouth tracked at ≥ 30 Hz with ≤ 5 mm jitter.

## Open questions
Wrist vs fixed camera for the mouth? (Fixed sees the face during the whole
approach; wrist gets occluded by the food at the end.) Which mannequin?
