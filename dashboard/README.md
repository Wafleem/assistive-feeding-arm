# Dashboard: face-servo debugging (SO-101 rig)

## Human-interface preview

The separate `static/language.html` page reuses the Viam hackathon's Ultraviolet
styling, text/voice command card, hold-to-talk and keyboard microphone controls,
voice selection and spoken acknowledgments. It does not change `server.py`, the
original `static/index.html`, controller code, robot configuration or calibration.

```bash
python3 dashboard/preview.py
```

Open `http://127.0.0.1:8091/`. The preview server serves only an explicit list of
UI assets. It rejects POST requests and exposes no robot API, camera stream,
credentials or proxy. **Send is disabled**, and typing/checking makes no robot
requests. Browser speech services may use the internet after microphone/voice
activation; typing and translation work locally. No real microphone or hardware
was used in UI checks.

### Live natural-language feeding (`/language.html?token=...`)
Served by the real dashboard (link at the top of the main page). With the token the Language page
is **live**: the bowl selectors start from the last **Identify bowls** result (CLIP on the Orin),
"I want some cereal" / "give me M&Ms" + **Send** runs `scoop_food` (the bowl that holds that food,
the scoop refitted to it), then waits for "ready" (or the button) before `feed` brings the spoon to
the mouth. Single commands ("go home", "level the spoon", ...) are sent as they are. "Stop",
"wait", "no" and "I'm done" send STOP immediately. Opened without a token (or via `preview.py`) it
stays the offline preview below.

The **Language** view defaults to Bowl 1: cereal and Bowl 2: M&Ms. The selectors
can swap those assignments or mark a bowl empty. Requests such as “grab me some
cereal,” “could you bring me some cereal, please?” and “I'd like some M&Ms”
(also spoken “m and ms”) select their labeled bowl. Translation runs locally
and accepts conversational greetings, polite wording and several request verbs.
It requires one complete request; mixed foods, extra task clauses and unsupported
quantities or qualifiers need clarification. Labels are
local to the page and do not imply a physical position or saved calibration.
Missing/duplicate food labels require clarification. Editing text, changing bowl
labels or starting another recording clears the previous review.

Two-bowl requests produce `kind: feeding_request_preview` JSON with food, logical
source bowl, one bite and scoop → confirm readiness → mouth approach → return home.
This is a human-interface specification, **not an executable command supported by
main's controller**. No generic scoop/feed action is substituted for a named bowl.
Existing individual dashboard commands (home, scoop, stop, etc.) can be previewed
as their current JSON. Stop/wait in this page cannot halt the physical robot.

The **Cameras & controls** view shows `static/face-servo.html`, a separate offline
copy of main's dashboard with both video panels and all original manual controls.
It uses video placeholders, disables commands and makes no API/video requests.
The real dashboard and its live startup script remain exactly as on main.

Offline UI checks: `node --test dashboard/test/*.test.mjs`.

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
| 0 | Level spoon first | slow wrist-only move to the saved spoon level; Center / Auto center / Extend refuse to start more than 5° off level |
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

## Depth + mouth only (no arm)

`dashboard/run_depth_view.sh` serves the dashboard's depth view (depth over the picture, lips,
distances) on http://localhost:8091/ without the arm: the full dashboard needs `follower-arm`.
It runs `dashboard/depth_view.py` (only the camera rig, MediaPipe and `make_depth_render`) **on the
Jetson**, so the raw depth (~650 kB a frame) stays on the Jetson and only the finished picture
crosses the tunnel; fetched from a laptop, the depth filled the Orin's Wi-Fi uplink and froze the
live dashboard's video and depth. It fetches at most 3 frames/s (`--max-fps`), restarts with the
tunnel, and Ctrl-C stops it on the Jetson. Over the USB cable by default; on a shared network
`JETSON=ultraviolet@<jetson-ip> dashboard/run_depth_view.sh`. Read-only: it sends no commands.

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
