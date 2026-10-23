# 1. Mouth perception

**Owner:** stream 1 (best fit: the RoboMaster tracking / Kalman people).
**Runs on:** the venue Viam machine (light fiducial tracking next to the camera) or the AGX Orin.
**Goal:** know where the mouth opening is, which way the face points, and where it
*will* be when the arm moves.

## The target: a real person's face
The demo uses a **real person** (team member), so delivery runs in `person` mode
(`feeding/config.py`, the default): the spoon stops **3 cm in front of the lips** and
the person closes the gap; nothing enters the mouth. Real faces move, close their mouth
and turn away, so the open check and the retreat triggers below all matter.

Camera: the Logitech C270 on the gripper (`webcam-2` on `orin-dev-kit`, RGB only, no
depth). Range comes from the eye-corner span (~90 mm adults, calibrate per person);
an RGB-D camera, when added, is used automatically instead.

## Interfaces (`types.py`, frozen once agreed)
- in: frames from `RobotInterface.get_frames()` (or `robot_iface.viam.ViamCameraRig` on the SO-101 rig)
- `MouthDetector.observe(robot) -> MouthObservation | None`: one frame, no filtering
- `MouthTracker.state(now) -> MouthState`: filtered and latency-compensated
  (position, velocity, face normal, open/closed, confidence, age)
- `ReadyGate.update(state, now) -> bool`, `HeadMotionMonitor.check(state) -> MotionVerdict`

## Status
| Piece | File | Status |
|---|---|---|
| Kalman filter (CV model, Mahalanobis gate, latency look-ahead) | `kalman.py` | **done**, tested |
| Tracker (normal smoothing, open/closed hysteresis, re-acquire after loss) | `tracker.py` | **done**, tested |
| Ready gate / head-motion retreat monitor | `gates.py` | **done**, tested; numbers need safety-lead review |
| Face geometry (lips centre, face normal, mouth ratio, range from eye span or depth) | `face_geometry.py` | **done**, tested on synthetic faces |
| MediaPipe Face Landmarker detector (largest face, `jawOpen` blendshape) | `detectors.py` | **done**; checked against the real model API, not yet on a live face |
| Live viewer | `scripts/perception_viewer.py` | **done** |
| Sim oracle / camera-seg detectors | `detectors.py` | TODO (sim) |
| ArUco detector for a printed face prop | `detectors.py` | optional (rehearsal without a person) |

## Next steps
1. Run `scripts/perception_viewer.py` on a team member: check the mouth position and
   open/closed against reality; tape-measure the face distance and compare.
2. Per-person calibration: `calibrate_eye_span` at a measured distance; record the
   person's "closed" and "open wide" mouth ratios into `FaceGeometryParams`.
3. Checkerboard intrinsics for the C270 (the viewer assumes fx = 780 px at 640×480).
4. Measure end-to-end latency (capture → arm); set `TrackerParams.latency`; fit
   `KalmanParams.sigma_accel` from recorded head motion.
5. Review every number in `gates.py` with the safety lead before the arm approaches a face.
6. Run the detector as a Viam vision service on the Orin (docs/COMPUTE.md).

## Done when
- A person's mouth is tracked at ≥ 20 Hz on the Orin with ≤ 1 cm jitter at 40 cm.
- Turning away, closing the mouth or leaning out triggers a retreat within 150 ms.

## Open questions for review
- Wrist camera only, or also a fixed camera? The wrist camera loses the face when the
  spoon is close; a fixed camera would keep watching during the final approach.
- Monocular range error (~5-10%) vs. adding a depth camera for the face.
