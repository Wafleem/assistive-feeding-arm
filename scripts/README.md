# Scripts

| Script | What | Status |
|---|---|---|
| `view_feeding_scene.py` | render the wrist and fixed cameras to PNGs (`make render`) or open the MuJoCo viewer (`make viewer`) | works |
| `eval_feeding.py` | the sim eval table: scenarios × foods × seeds → delivered, lateral error, speed near the face, contacts, correct retreats | TODO (section 7) |
| `record_frames.py` | record RealSense RGB-D frames + intrinsics for offline perception work | TODO (sections 1, 2) |
| `calibrate_hand_eye.py` | ChArUco hand-eye calibration for the wrist camera; extrinsic for the fixed camera | TODO (hardware) |

Scripts are thin: argument parsing plus calls into `assistive_arm`. No logic lives only in a script.
