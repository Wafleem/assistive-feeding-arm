# Scripts

| Script | What | Status |
|---|---|---|
| `view_feeding_scene.py` | render the wrist and fixed cameras to PNGs (`make render`) or open the MuJoCo viewer (`make viewer`) | works |
| `eval_feeding.py` | the sim eval table: scenarios × foods × seeds → delivered, lateral error, speed near the face, contacts, correct retreats | TODO (section 7) |
| `record_frames.py` | record RealSense RGB-D frames + intrinsics for offline perception work | TODO (sections 1, 2) |
| `calibrate_hand_eye.py` | ChArUco hand-eye calibration for the wrist camera; extrinsic for the fixed camera | TODO (hardware) |

Scripts are thin: argument parsing plus calls into `assistive_arm`. No logic lives only in a script.

## Observation-only arm camera preview

With the vision dependencies and local model files on the Jetson, use the camera
that is now directly attached by USB:

```bash
python -m assistive_arm.orchestration.vision_preview \
  --source usb \
  --camera arm-camera --spoon-weights /path/to/rf-detr-nano.pth \
  --mouth-model /path/to/face_landmarker.task --device cuda --port 8765
```

USB uses `/dev/v4l/by-id/usb-icSpring_icspring_camera-video-index0` by default;
override it with `--video-device /dev/v4l/by-id/...` if the device differs. This
source needs no Viam credentials. The adapter continuously reads the camera and
keeps the latest frame only. If using a camera hosted by Viam instead, set
`--source viam` (the CLI default) and configure the Viam connection environment.

`python scripts/vision_preview.py` accepts the same arguments. This acquires RGB
images only and draws spoon boxes and lip contours. It does not move the robot,
record frames or estimate calibrated 3D positions. The interface listens on
Jetson localhost; from your computer forward it with
`ssh -N -L 8765:127.0.0.1:8765 jetson-sentry-wifi`, then open
`http://127.0.0.1:8765`.

Add `--smoke-frames 3` to process three frames, print only shape/count/timing and
error summaries, and disconnect. Smoke counts describe the processed image;
the state is `stale` if model warmup/inference exceeded the maximum receive age.
`--threshold` controls RF-DETR confidence
(default 0.5); `--max-age` controls preview age since receipt (default 3 seconds).
The latter does not measure source exposure age. Ctrl-C closes this preview's
camera connection and MediaPipe instance without sending hardware stop commands.
