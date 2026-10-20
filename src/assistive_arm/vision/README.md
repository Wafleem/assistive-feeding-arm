# Shared core: `vision`: cameras, deprojection, detectors, 3D locating

**Owner:** food perception (stream 2), shared with mouth perception (stream 1).

| File | What | Status |
|---|---|---|
| `types.py` | `CameraIntrinsics`, `CameraSpec` (mount: wrist / base / world), `RGBDFrame`, `Detection`, `Target3D` | contract |
| `geometry.py` | deproject / project, mask → points, plane fit | works, tested |
| `detectors.py` | `Detector` protocol; `SegmentationOracleDetector` (sim), `NoisyDetector`; `OwlV2Detector`, `GroundingDinoDetector` (lazy imports, untested) | partly works |
| `locate.py` | `Locator.locate` (best detection → 3D point + normal, normal from a support surface) works; `Locator.locate_all` (every instance) is a **TODO** | partly works |

## Rules
- Detectors take an `RGBDFrame` and a text query, and return `Detection`s. They don't know about robots.
- Anything that needs a 3D pose goes through `Locator`, using the robot's `camera_pose`.
- Heavy model packages (torch, transformers, TensorRT) are imported lazily, inside the
  wrapper, and declared as optional extras.

## TODO
- [ ] `Locator.locate_all` (needed by food perception)
- [ ] `MaskRefiner` protocol + SAM 2 / NanoSAM wrappers
- [ ] NanoOWL wrapper for the AGX Orin
- [ ] Viam vision-service adapters (a detector served from the AGX, called from the Viam machine)
