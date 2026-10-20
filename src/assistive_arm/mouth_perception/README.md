# 1. Mouth perception

**Owner:** stream 1 (best fit: the RoboMaster tracking / Kalman people).
**Runs on:** the venue Viam machine (light fiducial tracking next to the camera) or the AGX Orin.
**Goal:** know where the mouth opening is, which way the face points, and where it
*will* be when the arm moves.

## The hackathon target: a 3D-printed mouth
- A printed face plate with an opening (~50 × 40 mm) that is **always open**.
- A removable **catch cup** behind the opening; bites drop in and it is dumped after every use.
- An **ArUco tag** at a known offset from the opening, from the same CAD file.
- A contrasting lip-colour ring around the opening.

So for the demo: `open_ratio` is always 1.0, and "ready" means **still and tracked**.
The prop may be slid or turned by hand to test retreat. Real faces (MediaPipe)
come later, behind the same interface.

## Interfaces (`types.py`, frozen once agreed)
- in: RGB-D frames from `RobotInterface.get_frames()` (fixed and/or wrist camera)
- `MouthDetector.observe(robot) -> MouthObservation | None`: one frame, no filtering
- `MouthTracker.state(now) -> MouthState`: filtered and latency-compensated
  (position, velocity, face normal, open/closed, confidence, age)
- `ReadyGate.update(state, now) -> bool`, `HeadMotionMonitor.check(state) -> MotionVerdict`

## Plan
1. **Fiducial detector** (`FiducialMouthDetector`): ArUco pose × the CAD offset to the
   opening. This is the main demo path.
2. **Tracking** (`kalman.py`, `tracker.py`):
   - constant-velocity Kalman filter on the frames' capture timestamps;
   - Mahalanobis gating to reject outlier detections;
   - latency compensation (`predict_ahead`);
   - smoothing of the face normal.
3. **Gates** (`gates.py`):
   - ready = still and tracked for N ms (the open check stays for real faces);
   - retreat on fast motion, drift, a turn, or stale tracking.
4. **Sim** (`SimOracleMouthDetector`, `CameraSegMouthDetector`): ground truth and the
   vision path on MuJoCo renders, once the sim has the printed-mouth scene (section 7).
5. **Known pitfall:** depth *inside* the opening is the catch cup, several cm behind the
   face. Take depth from the face plate or the tag, never from the hole.
6. **Later:** `MediaPipeMouthDetector` for a real face.
7. Wrap the detector and tracker as a Viam vision service.

## TODO checklist
- [ ] KF + tracker + gates (make `tests/test_mouth_perception.py` pass; remove the `@todo` marks)
- [ ] Fiducial detector on the printed mouth; tag→opening offset stored in `props/fake_mouth/`
- [ ] Sim oracle and camera detectors (after the printed-mouth sim scene exists)
- [ ] Measure end-to-end latency; tune `TrackerParams.latency` and `KalmanParams`
- [ ] Viam vision service wrapper

## Done when
- The printed mouth is tracked at ≥ 30 Hz with ≤ 5 mm jitter.
- Moving the prop by hand triggers a retreat within 150 ms.

## Open questions for review
- Fixed camera only, or wrist camera as well? (The fixed one sees the face for the whole approach.)
- Tag size and where it sits on the face plate.
