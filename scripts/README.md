# Scripts

| Script | What | Status |
|---|---|---|
| `perception_viewer.py` | live overlay: the person's mouth (MediaPipe: lips, open/closed, distance, readiness) and bowls (YOLOv8, ids, 3D). `--source viam` (C270 on orin-dev-kit), `local:0` or `file:img.jpg` | works (offline-checked; live face test pending) |
| `fetch_models.sh` | download model files into `data/models/` (MediaPipe face landmarker) | works |
| `view_feeding_scene.py` | render the wrist and fixed cameras to PNGs (`make render`) or open the MuJoCo viewer (`make viewer`) | works |
| `eval_feeding.py` | the sim eval table: scenarios × foods × seeds → delivered, lateral error, speed near the face, contacts, correct retreats | TODO (section 7) |
| `record_frames.py` | record RealSense RGB-D frames + intrinsics for offline perception work | TODO (sections 1, 2) |
| `calibrate_hand_eye.py` | ChArUco hand-eye calibration for the wrist camera; extrinsic for the fixed camera | TODO (hardware) |

| `so101_teach.py` | teach named poses on pi-arm2 with the leader arm (teleop running) → `poses/<task>.json` | TODO (section 9) |
| `so101_pick_place.py` | replay a taught pick-and-place N times on the follower; per-step log | TODO (section 9) |

Scripts are thin: argument parsing plus calls into `assistive_arm`. No logic lives only in a script.
