# Spoon vision: parallel work plan

Status: original planning record, superseded for the first live milestone by
[JETSON_VISION.md](JETSON_VISION.md). The user selected the moving wrist camera,
RF-DETR spoon detection, MediaPipe real-person mouth landmarks, and Jetson compute;
the additive RGB-only camera interface was approved on 2026-10-23. Overhead
localization and grasp execution remain later work. The historical plan below
does not override that narrower, vision-only implementation.
Prepared 2026-10-23 for the current SO-101 setup.

## Scope and current setup

The immediate milestone is to locate a spoon and its handle while the movement
owner calibrates and operates the arm independently. Perception produces
observations; the movement owner decides whether and how to execute a grasp.

Confirmed in this session:

- Repository: `Wafleem/open-vocab-helper-arm`, starting at `a87f3ed`.
- Viam machine: `pi-arm2`, Raspberry Pi 5, SO-101 follower/leader arms.
- Resources: `follower-arm`, `follower-gripper`, and `arm-camera`.
- `arm-camera` is an icSpring RGB USB camera configured at 640 x 480, 30 fps.
  The user reports that it is mounted to the left of the gripper and moves with it.
- The team has an AGX Orin, a clear work area, and plans a fixed overhead camera.
  The overhead camera's model, connection, resolution, and depth capability are
  not yet confirmed. Its availability is not assumed for today's first task.
- The teammate owns active arm movement and calibration during vision development.

The repository's xArm6, 85 mm gripper, force sensing, and RGB-D assumptions are
not measurements of this SO-101 setup. They must be reconciled by the relevant
owners before integrated execution.

## Ownership

| Area | Vision owner | Movement/calibration owner |
|---|---|---|
| Camera images | Read existing streams, decode, replay, display overlays | Coordinate any hardware/configuration changes |
| Spoon perception | Mask, bowl/handle distinction, candidate handle point, image direction, quality | Review whether the proposed handle region is mechanically usable |
| Calibration | Camera intrinsics, distortion, image reference observations | Arm calibration, robot frame, tool center, measured jaw geometry |
| Camera-to-arm mapping | Jointly validate fixed-camera mapping and later wrist hand-eye calibration | Supply robot reference measurements and time-matched poses |
| Grasping and motion | Supply observations and visual verification evidence | Reachability, grasp orientation, clearance, trajectories, gripper commands, execution |
| Integration | Agree a typed observation handoff | Explicitly enable and consume it when ready |

During parallel work the vision process must not call movement, gripper, stop,
torque, tare, homing, or arm calibration methods. It must not stop the arm on
viewer exit. Read through the existing camera service instead of competing for
its USB device. Do not restart viam-server or change machine configuration while
the teammate is calibrating. A vision result is never an instruction to move.

## Existing code to reuse and gaps to resolve

| Existing interface | Useful part | Gap for this milestone |
|---|---|---|
| `vision.detectors.Detector` / `vision.types.Detection` | Detector abstraction, label, score, box, mask | Input currently requires `RGBDFrame`; handle geometry and capture metadata need a reviewed typed handoff |
| `vision.types.RGBDFrame` | Aligned RGB/depth frame with timestamp | Requires depth and intrinsics; the current webcam supplies neither calibrated intrinsics nor depth |
| `robot_iface.viam.ViamRobot.get_frames` | Centralizes Viam camera access | Unconditionally selects a depth image, assumes depth units, and does not propagate capture time into the returned frame |
| `vision.locate.Locator` | Uses camera pose to locate objects in the base frame | Uses depth, and `locate_all` is still a stub; the food README incorrectly calls that method working |
| `food_perception.FoodItem` | Existing 3D item handoff | A spoon handle is not simply the whole object's center; metric points are unavailable until calibrated |
| `grasping.GraspCandidate` | Existing grasp pose/width handoff | Belongs to manipulation; perception should not claim a reachable grasp from image geometry alone |

Keep these interfaces unchanged in this planning change. Follow `AGENTS.md`:
RGB-only support and any new cross-section observation contract need a separate,
small contract PR reviewed by consumers. Keep SDK imports in `robot_iface/viam.py`
and OpenCV/model imports lazy behind optional extras. Do not insert fabricated
depth or invented intrinsics to satisfy the existing RGB-D locator.

## First implementation target: offline spoon overlay

Proposed task for review: a spoon-specific detector/handle estimator in the
perception section, using recorded RGB images and no robot connection.

1. Use the actual spoon against the clear area's matte background, with diffuse
   light. Collect positive and negative examples at several positions/angles,
   including blur, reflections, partial occlusion, and no spoon.
2. Start with OpenCV foreground/contour processing inside a defined pickup area.
   This is a controlled-scene baseline, not a general spoon recognizer. Compute
   shape/width along the outline to distinguish the bowl from the handle.
3. Mark a candidate inside the handle, its local axis, and its visible width.
   Do not use the spoon bounding-box center as the grasp point. Principal-axis
   orientation alone cannot distinguish the bowl end from the handle end.
4. Display mask, bowl/handle overlay, candidate point, direction, and quality.
   If the handle is occluded, the image is blurred, or multiple candidates are
   ambiguous, output an explicit invalid/ambiguous result.
5. Use held-out images for evaluation. If reflections defeat the baseline,
   compare a small segmentation/keypoint model on those same examples; use cloud
   GPU credits for training only if needed. A generic object detector can locate
   a spoon box but does not establish a handle grasp.

Recorded frames and model weights belong in ignored `data/`, never in the PR.
Core tests must remain offline without GPUs, cameras, credentials, or robot access.
Use small synthetic fixtures and explicitly approved repository-safe fixtures.

## Handoff proposal for review

Use a typed observation wrapper around detection results, not an unstructured
dictionary or an overloaded `FoodItem`. The following describes semantics only;
it does not add or freeze a Python API.

| Information | Meaning |
|---|---|
| Status/reason | Valid 2D observation, not found, ambiguous, occluded, blurred, or stale |
| Camera / frame identity | Camera name, frame sequence, image dimensions; points refer to this exact image |
| Capture time / clock | Exposure timestamp with clock domain, plus local receive time; receive time alone is not exposure time |
| Detection | Spoon label, box, mask, score; heuristic quality must not be presented as a calibrated probability |
| Handle geometry | Candidate `(u, v)` pixels, directed handle axis from bowl toward handle tip, local width in pixels |
| Calibration identity | Which camera mounting/intrinsic calibration produced any metric estimate |
| Optional metric estimate | Handle position/axis in the agreed robot base frame, uncertainty, and source (`depth` or `plane_projected`); absent when unavailable |

Images use `u` right and `v` down. In-process geometry follows repository SI units:
metres, radians, seconds; any quaternion is `(w, x, y, z)`. Only the existing Viam
conversion boundary uses millimetres/degrees. Never put pixel coordinates in a
field described as a base-frame position.

The movement owner chooses finger closing direction, approach pose, jaw opening,
and reachable orientation using measured SO-101 geometry. Agree frame origin,
axis signs, grasp-point definition, maximum observation age, uncertainty limits,
and clock synchronization before observations can be used for execution. These
are integration decisions, not new hardcoded safety limits in perception.

## Camera and compute sequence

1. **Now:** develop the RGB detector and overlay on the Mac from saved images.
   After the camera adapter/contract review, add a read-only live preview. While
   the wrist moves, its pixel observations remain image-relative only.
2. **Overhead camera:** use it as the primary source for tabletop localization.
   Calibrate distortion and the fixed camera-to-base relationship. For a known
   table/handle height, use ray-plane intersection or a calibrated homography;
   label the output as a plane-based estimate, not measured depth. Validate on
   separate measured points and account for a spoon rest or raised handle.
3. **AGX Orin:** run perception behind the same detector interface, packaged as a
   Viam vision/generic service as required by the repo. Keep the Pi as the arm's
   hardware owner and motion/orchestration host. Use wired local networking and
   measure capture-to-result latency; cloud inference is not required for MVP.
4. **Wrist camera later:** calibrate its rigid transform relative to the gripper.
   Metric wrist observations require the arm pose at the image's capture time.
   Do not combine an old image with the latest arm pose or apply the overhead
   homography to wrist images. If synchronization is unavailable, request a
   stationary capture window coordinated by the movement owner.

Vision can continue on saved frames while the teammate changes calibration.
Changing either camera's mounting invalidates that camera's extrinsic mapping;
moving the arm base also invalidates the overhead-to-base mapping.

## Milestones and proposed acceptance criteria

| Milestone | Evidence before proceeding |
|---|---|
| Offline 2D perception | Overlay reviewed on at least 20 held-out positive and 20 negative/challenging images; target at least 90% correct handle localization on visible, unambiguous positives; zero accepted grasp locations on empty-table examples |
| Read-only live preview | Teammate can move/calibrate without any commands or configuration changes from vision; stale/blurred/occluded frames are flagged and prior targets expire |
| Metric overhead localization | Held-out point error measured across the pickup region and shown smaller than the agreed grasp-clearance budget; no uncalibrated output is marked usable for motion |
| Orin deployment | Record actual device/software versions, p50/p95 latency, frame age, and detection quality; establish a measured budget with the movement owner |
| Integrated pickup | Movement owner enables one supervised trial, then randomized placements; report pickup, slip, and false-success counts separately |

These are proposed criteria, not measured results or permission to run the arm.
The initial deliverable is the vision overlay and a reviewable handoff. Grasp
execution, feeding, mouth tracking, VLA training, and general cluttered-scene
grasping are outside this first vision milestone.

## Next review decisions

- Confirm the overhead camera and how frames reach the Pi/Orin.
- Agree RGB-only frame support and the typed observation wrapper in a contract PR.
- Agree calibration ownership and the robot base/tool reference used at integration.
- Review the offline spoon detector task above before implementation.
