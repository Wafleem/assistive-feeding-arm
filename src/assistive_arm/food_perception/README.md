# 2. Food perception

**Owner:** stream 2. **Runs on:** AGX Orin (NanoOWL/NanoSAM) or the Linux PC (full models).
**Goal:** "where are the strawberries?" → 3D items with their points, ranked by how
easy they are to pick, from an open vocabulary.

## Interfaces
- `vocabulary.py` (done, data): menu items, synonyms, acquisition strategy, grip force, feasibility
- `FoodPerception.find(robot, name) -> list[FoodItem]` (TODO), `scan(robot)` (TODO)
- `FoodItem`: name, 3D position, 3D points (for grasp planning), score, camera, attributes
- `VisualQA.ask(rgb, question)` for attributes/state (TODO; `NoVLM` default)
- Shared infra that already works: `vision.Locator.locate_all` (detection + depth → 3D
  points per instance), `vision.detectors` (sim oracle, OWLv2, Grounding DINO wrappers)

## Bowls (implemented: `bowls.py`)
The current rig scoops small food from **bowls** with a spoon fixed in the gripper, so
the first job is "where is each bowl, and which one is it?" while bowls may be moved:
- detection: YOLOv8 COCO already knows `bowl` and `cup` (the Orin's `spoon-detector`
  Viam service, or `vision.detectors.UltralyticsDetector` locally); YOLO-World / OWLv2 for
  anything COCO lacks ("small white ramekin");
- 3D from one RGB camera: the box centre's ray meets the plane `table_z + rim_height`
  (depth is used instead when the camera has it); radius from the box width;
- `BowlTracker`: stable ids while bowls slide, dropped after 10 missed frames.
Next: what is in each bowl (crop → open-vocab query or VLM), per-bowl rim heights, and
the scoop target inside the bowl (food centroid, not the rim centre).

## Plan
1. Offline first: ~50 recorded RealSense frames of the real plate (grapes,
   strawberries, melon cubes, crackers, cup) → measure OWLv2 / Grounding DINO /
   NanoOWL recall per item and latency per device.
2. Boxes → masks with SAM 2 (Linux) / NanoSAM (AGX); masks → points → `FoodItem`.
3. Rank by isolation (finger clearance at 85 mm open) and distance to the plate centre.
4. VLM (small Qwen-VL / SmolVLM / Florence-2 class, quantised) only at decision points:
   "is the spoon loaded?", "which bowl is the soup?", "what's on the plate?".
5. Wrap detector and segmenter as Viam vision services.

## The VLA question (honest)
A VLA (SmolVLA / pi0 / OpenVLA via LeRobot) could do **bite acquisition end to end**
(especially scooping, where scripted motions are brittle). It is **not** a good fit for
the face approach: no force input, no guarantees, cm-level precision. Plan: modular
pipeline for the MVP; VLA as a stretch for acquisition only, run through
`grasping/vla.py` (clamped steps, keep-out checks). Data plan: ~50 teleop demos per
food in sim + ~20 real, LeRobot format, fine-tune on the Linux PC, score in the same
sim eval as the scripted skill.

## TODO checklist
- [x] Bowl finder + tracker (`bowls.py`, `tests/test_bowls.py`); live view in `scripts/perception_viewer.py`
- [ ] `FoodPerception.find/scan` on sim renders (oracle detector) → `tests/test_food_perception.py`
- [ ] Recorded-frame replay + accuracy/latency table (docs/COMPUTE.md)
- [ ] SAM 2 / NanoSAM mask refiner; NanoOWL wrapper on the AGX
- [ ] VLM attribute questions
- [ ] Viam vision service wrappers

## Done when
≥ 90% recall on the demo foods on recorded frames, 3D centre error ≤ 1 cm, ≤ 300 ms per frame on the AGX.
