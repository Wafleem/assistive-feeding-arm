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

## Run (on your laptop)

```bash
dashboard/run_mac.sh
```
It opens an SSH tunnel over the Jetson's USB link for the **camera only** (plug a USB-C
cable into the Orin's device-mode port, next to the 40-pin header; the Orin is then
`192.168.55.1`), and starts the server on `localhost:8090`. Arm and teleop commands
go through the normal Viam cloud connection (`VIAM_ADDRESS` in `.env`). Open the printed
`http://localhost:8090/?token=...`.

On the Orin itself instead: `.venv/bin/python dashboard/server.py --local --host 0.0.0.0`
(needs mediapipe, opencv, viam-sdk installed there).

Logic lives in `src/assistive_arm/orchestration/face_servo_session.py` (sequencing) and
`src/assistive_arm/feeding/face_servo.py` (decisions); tests: `tests/test_face_servo*.py`.
