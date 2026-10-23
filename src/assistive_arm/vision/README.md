# Shared core: `vision`: cameras, deprojection, detectors, 3D locating

**Owner:** food perception (stream 2), shared with mouth perception (stream 1).

| File | What | Status |
|---|---|---|
| `types.py` | `CameraIntrinsics`, `CameraSpec` (mount: wrist / base / world), `RGBDFrame`, `Detection`, `Target3D` | contract |
| `geometry.py` | deproject / project, mask → points, plane fit | works, tested |
| `detectors.py` | `Detector` protocol; `SegmentationOracleDetector` (sim), `NoisyDetector`; `OwlV2Detector`, `GroundingDinoDetector` (lazy imports, untested) | partly works |
| `rfdetr.py` | `RFDETRSpoonDetector.detect_rgb`: local RF-DETR Nano, RGB image → spoon boxes | offline adapter tests; deployment validation required |
| `locate.py` | `Locator.locate` (best detection → 3D point + normal, normal from a support surface) works; `Locator.locate_all` (every instance) is a **TODO** | partly works |

## Rules
- The 3D locator's detectors take an `RGBDFrame` and a text query. The RGB-only
  RF-DETR backend takes a uint8 RGB array and returns `Detection`s for the spoon
  preview; it requires neither invented depth nor camera intrinsics. Neither
  detector interface knows about robots.
- Anything that needs a 3D pose goes through `Locator`, using the robot's `camera_pose`.
- Heavy model packages (torch, transformers, TensorRT) are imported lazily, inside the
  wrapper, and declared as optional extras.

## RGB-only spoon preview

The first live wrist-camera milestone uses `RFDETRConfig(weights_path=...)` and
`RFDETRSpoonDetector.detect_rgb(rgb)` on the Jetson. The checkpoint must already
exist on disk; loading occurs on the first frame. Defaults are CUDA and a 0.5
score threshold. RF-DETR receives RGB without a BGR channel swap, and its own
class metadata determines which detections are spoons. COCO's RF-DETR IDs must
not be interpreted with a YOLO class list.

The backend returns only clipped 2D boxes and model scores. It neither estimates
a grasp point or metric depth nor commands the arm. Live results still need
validation with the actual spoon, camera angles and lighting. Camera capture and
preview composition are supplied by the robot interface and orchestration.

## TODO
- [ ] `Locator.locate_all` (needed by food perception)
- [ ] `MaskRefiner` protocol + SAM 2 / NanoSAM wrappers
- [ ] NanoOWL wrapper for the AGX Orin
- [ ] Viam vision-service adapters (a detector served from the AGX, called from the Viam machine)
