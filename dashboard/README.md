# Dashboard: face-servo debugging (SO-101 rig)

A browser page for stepping through "sweep to the person, find the face, centre on it,
extend toward it" one command at a time, with the gripper camera overlay and every number
the controller uses.

| Step | Button | What happens |
|---|---|---|
| 1 | Teach look | (teleop on) records the follower's joints: turned toward the person, face in view, far away |
| 2 | Teach near | (teleop on) records the end of the approach path (spoon well short of the face) |
| 3 | Sweep to person | stops teleop, moves to the look pose at the servo speed cap |
| 4 | Find face | pans ±8/16/24° around the look pose until MediaPipe sees a face |
| 5 | Calibrate | 4° probe moves of the base pan and wrist flex: measures how the lips move in the image (px/deg) |
| 6 | Center on face | closes the loop: steers pan + wrist flex until the lips sit on the spoon's aim point |
| 7 | Extend toward face | moves along look → near in 5% steps while centred, still and facing; stops at the stop distance |
| | Retreat / STOP | back to the look pose / stop now (Space or Esc also stop) |

Safety (all in `feeding/face_servo.py: FaceServoConfig`; review with the safety lead):
- the arm only moves on the taught look→near line plus clamped pan/flex offsets (no IK near a face);
- small steps (≤ 3° per centring step, 5% of the path per approach step) at ≤ 20 °/s,
  because the SO-101's Stop cannot interrupt a joint move in progress;
- **stop distance 20 cm (spoon tip to lips) for first runs**, hard floor 3 cm;
- retreat if the face is lost (> 0.4 s), the head moves fast, or turns away during the approach;
- Extend needs the readiness box ticked; every request needs the server's token.

## Run (on the Orin, next to the camera)

```bash
cd ~/assistive-arm-servo && .venv/bin/python dashboard/server.py --local
```
Then open the printed URL (`http://rm-sentry.local:8090/?token=...`). `--local` talks to
viam-server on localhost:8080, so the venue Wi-Fi only carries the page and the video.
Credentials: `.env` (`VIAM_API_KEY`, `VIAM_API_KEY_ID`). Poses and calibration are
saved in `poses/so101_face.json`.

Logic lives in `src/assistive_arm/orchestration/face_servo_session.py` (sequencing) and
`src/assistive_arm/feeding/face_servo.py` (decisions); tests: `tests/test_face_servo*.py`.
