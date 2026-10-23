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
The prop may be slid or turned by hand to test retreat. Calibrated real-face 3D
tracking remains a later step; the RGB preview described below is implemented
independently of that interface.

## Current vision-only scope: a real person's lips

`landmarks_2d.py` implements `MediaPipeMouthLandmarker` for the wrist-camera
preview. It accepts uint8 HWC **RGB** arrays from the camera reader and a supplied
capture timestamp. It does not connect to or command the arm.

```python
from assistive_arm.mouth_perception.landmarks_2d import (
    MediaPipeMouthLandmarker, MouthLandmarkerConfig,
)

with MediaPipeMouthLandmarker(MouthLandmarkerConfig(
    model_path="data/models/face_landmarker.task",
)) as detector:
    mouths = detector.detect_rgb(rgb, timestamp_s=capture_timestamp_s)
```

The local `.task` model is required; constructing the wrapper never downloads it.
MediaPipe is imported only when constructing the wrapper. It uses the CPU
delegate and IMAGE mode, so each image is independent and no monotonic processing
timestamp is substituted for a capture timestamp. Jetson inference and dependency
compatibility must be validated on the actual installation.

Each `MouthLandmarks2D` contains ordered outer/inner lip pixel contours, an xyxy
pixel bounding box, the midpoint between inner upper/lower lips, and a raw
aperture ratio. The ratio is pixel separation of landmarks 13/14 divided by the
distance between corners 61/291. It is **not** calibrated openness or a probability.
No face returns an empty tuple; incomplete, nonfinite or subpixel-width lip
geometry is omitted. A face's index is its order within a frame, not a stable ID.

These outputs have no measured depth, metric pose, face normal, tracking,
readiness or feeding decision. They do not populate the existing `MouthObservation`
3D contract. The remaining tracker/gate and calibrated 3D detector TODOs below
are unchanged.

Validated offline with a fake MediaPipe boundary: pixel geometry on non-square
images, RGB channel preservation, multiple/no faces, invalid outputs, and task
cleanup. See [MediaPipe's Tasks guide](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker/python)
and [lip topology](https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/python/solutions/face_mesh_connections.py).

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
- [x] RGB-only MediaPipe lip geometry for a real-person camera preview, with offline tests
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
