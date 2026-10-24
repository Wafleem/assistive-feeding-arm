# SO-101 setup on Viam (`pi-arm2`, then `orin-dev-kit`)

The bench rig the team develops on before the venue xArm6: an SO-101 **follower** arm
that does the work, and an SO-101 **leader** arm you move by hand to drive it. Both
run on a Raspberry Pi Viam machine, `pi-arm2`.

## Machine

| Item | Value |
|---|---|
| Viam org / location / machine | `3percentclub` / Physical AI for Good / `pi-arm2` (part `pi-arm2-main`, linux/arm64) |
| Module | `devrel:so101-arm` (models: arm, gripper, calibration, discovery, teleop) |
| Follower | `follower-arm` + `follower-gripper`, port `/dev/serial/by-id/usb-1a86_USB_Single_Serial_5C4C124377-if00`, `calibration_file: so101_follower_calibration.json`, frame at the world origin |
| Leader | `leader-arm` + `leader-gripper`, port `/dev/serial/by-id/usb-1a86_USB_Single_Serial_5C4C124416-if00`, `calibration_file: so101_leader_calibration.json`, frame at y = −600 mm |
| Calibration sensors | `follower-calibration`, `leader-calibration` (same ports and files as their arms) |
| Teleop | `teleop` (`devrel:so101:teleop`): leader → follower, grippers mirrored, 100 Hz, `auto_start: false` |
| Other | `arm-camera` (USB webcam, 640×480), `table` obstacle (the 72.6 × 56.5 cm marble table, 30 mm thick, top at z = 0), `spoon` collision box on the gripper (see below), discovery services |

Calibration files live in the module's data directory on the Pi
(`/root/.viam/module-data/<part-id>/devrel_so101-arm/`), not in this repo.

## Moving to `orin-dev-kit` (2026-10-23)
The hardware moved to the Jetson Orin dev kit (machine `orin-dev-kit`, part id
`af8aae81-2f68-46bb-a2a0-fdc12ed08ad0`). It has the same components, ports (by-id) and
names. Leader, follower and teleop are enabled (teleop `auto_start` off); both calibration
sensors are disabled until someone recalibrates. Snapshots: `viam/machines/`.

What happened at the move, and what fixed it:
- **Servo `voltage` errors** on all six follower servos: the wrong power adapter came along
  with the move. With the flag set, the servos refuse register reads, so the arm fell back to
  *default* calibration. Fixed by swapping to the correct adapter (and rebooting the Orin;
  viam-agent brings viam-server back by itself, and the arms reconnect within a minute or two).
- **No calibration file on the Orin:** not needed. `save_calibration` also writes the
  calibration into the servos, and with correct power the arm reads it back from there
  ("No calibration file loaded, attempting to read from servo registers"). It matched
  `viam/calibration/so101_follower_calibration.json` exactly. The file is still the more
  reliable source; see `viam/README.md` to install it.
- `reload_calibration` resolves a relative `calibration_file` against the module's working
  directory, not its data folder, so it can't find a relative file that startup loads fine.
  After installing a file, restart the arm component (toggle `disabled`, or edit its config)
  instead of using `reload_calibration`.
- The leader also read its calibration from its servos on the Orin, matching
  `viam/calibration/so101_leader_calibration.json` exactly.
- `pi-arm2` still has the arm components configured, with no arms attached.

## Teleop
- Start: DoCommand `{"command": "start"}` on `teleop`. Stop: `{"command": "stop"}`.
  Status: `{"command": "status"}` (running, cycles, errors, streaming).
- On start the follower moves to the leader's pose at normal speed, then tracks it in
  real time. **Clear the follower's workspace first.**
- While teleop runs, the leader's torque is off, so it can be moved by hand; it comes back on at stop.
- Stop teleop before running any scripted motion on the follower.

## Calibration (redo after rebuilding an arm, or if joints read wrong)
1. Enable `<arm>-calibration`, run `{"command": "start"}` (torque off: support the arm).
2. Pose the arm at the **zero pose**: pan centred, arm extended forward, forearm and
   gripper level. Both arms must use the same zero pose, or teleop mirrors at an offset.
   Then `{"command": "set_homing"}`.
3. `{"command": "start_range_recording"}`, move every joint (and the gripper) to both
   limits, then `{"command": "stop_range_recording"}`.
4. `{"command": "save_calibration"}` (writes the servos and the file; torque back on).
5. On the arm: `{"command": "reload_calibration"}`, then `{"command": "get_calibration"}` and
   compare with the saved values.

## Lessons from the first bring-up (2026-10-23)
- **USB re-enumeration:** replugging changes `/dev/ttyACM*` numbers. Use the
  `/dev/serial/by-id/...` paths (already done).
- **Stale serial handles:** after a replug, the module keeps the old handle and every
  call fails with `write failed: input/output error`. `reinitialize` doesn't fix it.
  Disable **every** component on that port (arm, gripper, calibration sensor), then
  re-enable them.
- **Calibration-file race:** an arm restarted right as its file was being written logs
  "Failed to load calibration … no such file" and silently falls back to the servo
  registers. Always `reload_calibration` + `get_calibration` after calibrating.
- Leader and follower homing offsets differ (each arm is homed on its own); that's expected.

## Wrist camera (`arm-camera`) frame: estimate, to be verified

A Hiwonder USB camera module (640×480, manual-focus M12 lens) is on an L-bracket bolted
to the side of the gripper's servo housing, after the wrist-roll joint. It moves with the
whole gripper (photos in the team chat, 2026-10-23).

| Quantity | Estimate | How it was obtained |
|---|---|---|
| Parent frame | `follower-arm` (the tool flange) | mounted after wrist roll |
| Viewing direction | parallel to the gripper approach axis (tool +Z) | photo from in front of the gripper |
| Sideways offset | ~55 mm from the gripper centreline, along the jaw servo's axis (tool ±Y: **sign unknown**) | scaled from the gripper body width in the photos |
| Offset along the gripper | ~35 mm forward of the flange (jaw tips are 99.9 mm) | photos |
| Height offset | ~0 mm (level with the jaw servo axis) | photos |
| Rotation about the lens axis | **unknown** (image "up" vs tool axes) | needs a live frame |

Verification (with focus set and the camera streaming):
1. Pose the gripper pointing down at the table ~20 cm up, with an object to the arm's left.
   Which side of the image the object appears on fixes the ±Y sign and the image rotation.
2. Add the frame (`parent: follower-arm`, translation `(0, ±55, 35)` mm, orientation
   `ov_degrees (0, 0, 1, θ)`), then check it in the 3D SCENE tab against the real arm.
3. Refine with a hand-eye check: touch a marked point with the spoon tip, compare it with
   where the camera says it is, and adjust; later, a ChArUco hand-eye calibration.
Also needed: camera intrinsics (checkerboard capture), which the webcam config doesn't have yet.

Focus: twist the M12 lens until text ~20–30 cm away is sharp (the plate/food distance).

## Gripper camera now: Logitech C270 (`webcam-2`)
The icSpring wrist module (`arm-camera`) shows up on USB and opens, but sends **zero
frames** in every mode. That held on the Orin, outside Viam, after a USB reset and after a
power cycle, so treat it as a hardware fault. It was replaced by a **Logitech C270** taped
on top of the gripper: centred, lens forward along the spoon (photos IMG_3077-3079,
2026-10-23). The spoon stays taped in the fixed jaw.

Frame of `webcam-2` (parent `follower-arm`; tool axes: +Z out along the spoon, +X down,
+Y to the gripper's left), estimated from the photos with the C270's 73 mm width as scale:

| | Value | Meaning |
|---|---|---|
| translation | (x −21, y −8, z 90) mm | ~28 mm above the spoon handle, ~8 mm right of centre (the lens is off-centre in the C270), lens ~5-10 mm behind the jaw tips |
| orientation | `ov_degrees (0, 0, 1, −90)` | optical axis along the spoon; image "up" = gripper top |
| known error | ~6° yaw (top view) and a few degrees of pitch | refine with a hand-eye check |

The live image confirms that orientation: it is upright, and the spoon bowl shows at bottom-left.
Intrinsics are not calibrated yet; code assumes fx = fy = 780 px at 640×480 (C270, ~60° diagonal).

Viewing detections:
- **Viam app → CONTROL → `detections-view`**: the C270 feed with YOLOv8 boxes (bowl, cup,
  spoon, person, ...), from `spoon-detector` running on the Orin.
- **`scripts/perception_viewer.py`**: the same feed plus the person's face (lips, mouth
  open/closed, distance, readiness) and bowls with ids and 3D positions. Needs
  `VIAM_ADDRESS`, `VIAM_API_KEY`, `VIAM_API_KEY_ID` in `.env` (Viam app → CONNECT → API keys).

## Collision geometry: `table`, `spoon` and the bowls (motion planner only)
`table` and `spoon` are `viam:obstacle-pack:obstacle` boxes; the bowls are
`viam:obstacle-pack:obstacle-from-mesh` meshes. All show in the Viam app's 3D SCENE tab.

| Component | Parent | Geometry | Why |
|---|---|---|---|
| `table` | `world`, (x 287.2, y 227, z −15) | box x 726 × y 565 × z 30 mm | the marble table (measured 2026-10-24), 72.6 × 56.5 cm: its long side runs straight ahead of the arm (+x, the way the arm points at zero joints). The back of the base is 5 cm from the near edge (x −75.8; the back of the module's base box is x −25.8), and the base's right side is flush with the right edge (y −55.5, half the base box's 111 mm width; the left edge is at y +509.5). Top at z = 0, the bottom of the SO-101 base (the module's base origin; the shoulder pan axis is 62.4 mm up) |
| `spoon` | `follower-gripper`, z +61.5 | box x 50 × y 45 × z 143 mm | the spoon sticks out **123 mm** past the jaw tips (measured 2026-10-23); the box runs from 10 mm behind the tips to 10 mm past the spoon tip |
| `bowl-right` | `world`, (x 309.2, y 94.5, z 0) | mesh `bowl-150x60.ply` | paper bowl, 150 mm across the rim, modelled 60 mm tall (measured ~70 mm, lowered 15%; the 100 mm bottom is a guess). The empty bowl nearest the arm: its rim is 75 mm from the right edge and 310 mm from the near edge (2026-10-24) |
| `bowl-left` | `world`, (x 234.2, y 259.5, z 0) | mesh `bowl-150x60.ply` | the same bowl (the cereal bowl): its rim is 240 mm from the right edge and 235 mm from the near edge (2026-10-24). A third bowl, farther back, isn't modelled yet |

- `follower-gripper`'s frame is the module's TCP, the point between the jaw tips, with the arm
  tool's axes (+Z along the spoon). So the spoon box starts at the tips, and it moves with the
  arm: the planner treats it as part of the arm and keeps it off the table.
- The 50 × 45 mm cross-section is a generous guess (bowl orientation and the fixed jaw's offset
  weren't measured). Check it against the real spoon in the 3D SCENE tab and shrink it if
  close approaches get refused.
- If the follower sits on a riser, lower `table` by the riser height. If the arm or table moves,
  re-measure: with the near edge at `x_near` (= −25.8 − gap behind the base) and the right edge at
  `y_right` (= −55.5 when the base's side is flush), the centre is `x_near + length / 2`,
  `y_right + width / 2`.
- Past the table's edges the planner lets the arm reach below z = 0, as the real arm can.
- The bowl mesh is only the sloped wall: no top (so planned moves can put the spoon inside)
  and no bottom (the table is the bottom). Viam treats meshes as surfaces, not solids, so the
  spoon can't pass through the wall but can sit inside. Touching the bottom means touching the
  table, so a scooping move has to allow `spoon` against `table` in its collision specification.
- The mesh is `viam/meshes/bowl-150x60.ply`, written by `viam/meshes/bowl_ply.py` (rim
  diameter, bottom diameter, height in mm; PLY units are metres). The Orin reads it from
  `/home/ultraviolet/bowl-150x60.ply`; to install or update it:
  `viam machines part cp --part af8aae81-2f68-46bb-a2a0-fdc12ed08ad0 viam/meshes/bowl-150x60.ply machine:/home/ultraviolet/bowl-150x60.ply`,
  then restart both bowl components (the mesh is read once, at construction).
- Moving a bowl: change its frame `translation` (bowl centre, on the table top). A frame at
  `x = x_near + d_near + 75`, `y = y_right + d_right + 75` puts the rim `d_near` from the near
  edge and `d_right` from the right edge (now `x_near` = −75.8, `y_right` = −55.5).
- **Only motion-service moves are checked.** Teleop and direct `MoveToJointPositions` ignore
  these boxes.
- The planner ignores any contact that already exists when a move starts. That's why the base's
  collision box (which dips ~15 mm below z = 0) and the spoon/gripper overlap cause no errors,
  but it also means: if the spoon starts a move touching the table, the table isn't checked for
  that move.

## Lessons from the camera debugging (2026-10-23)
- **A module download blocks every config change.** While the 2.9 GB YOLOv8 package was
  downloading, the Orin applied no new config. Removing the module didn't help; only a
  viam-server restart did (the download then resumed from where it stopped).
- **Modules can inherit an open camera.** A module process started while a webcam is
  open inherits its device file (seen with `find-webcams`, then YOLOv8). When the camera
  is then reconfigured (even just a `frame` change), the new instance fails with "no
  queryable drivers ... busy". Fix: restart the module that holds it (`RestartModule`),
  or restart viam-server. Check with `sudo fuser -v /dev/video*`.
- **Debug hardware outside Viam.** SSH in (`ultraviolet@rm-sentry.local`, key
  `~/.ssh/id_ed25519_jetson`) and test the camera directly with `lsusb`,
  `/dev/v4l/by-id`, `dmesg` and a one-frame OpenCV read. That separates "dead camera"
  from "bad config".
- **Venue Wi-Fi:** 300-600 ms pings and failed WebRTC connections made live commands
  (`DoCommand`, `GetImages`) time out, even while the part showed online. Config still got
  through, which is why teleop now uses `auto_start: true`.

## Network: give the Orin the laptop's internet over the USB cable (2026-10-23)
At its spot the Orin's Wi-Fi was at −80 dBm (6 Mbit/s uplink, 300–600 ms pings), so
viam-server kept dropping off the cloud: 32 of 45 WebRTC connections failed in 10 minutes,
and teammates couldn't connect from app.viam.com. Routing its traffic through a laptop on
the Jetson's USB link fixed it (about 22 ms pings, 6 of 7 connections succeeding, no
log-upload errors).

1. USB-C cable from the laptop to the Orin's **device-mode port** (next to the 40-pin
   header). The Orin is `192.168.55.1`, the laptop `192.168.55.100`.
2. On the laptop (macOS, Wi-Fi on `en0`), once per laptop boot. You run this yourself; it needs your password:
   ```bash
   sudo sysctl -w net.inet.ip.forwarding=1 && echo "nat on en0 from 192.168.55.0/24 to any -> (en0) static-port" | sudo pfctl -a com.apple/jetson-share -f - && sudo pfctl -E
   ```
   Undo: `sudo pfctl -a com.apple/jetson-share -F all && sudo sysctl -w net.inet.ip.forwarding=0`
3. On the Orin, once per Orin boot (it doesn't survive a reboot, on purpose):
   ```bash
   sudo setsid nohup ~/assistive-arm-servo/scripts/usb_uplink_watchdog.sh >/dev/null 2>&1 </dev/null &
   ```
   Every 10 s it checks 1.1.1.1 through the laptop. While that works, the default route goes
   via the laptop (metric 50) with DNS 1.1.1.1/8.8.8.8. After 3 failures (cable out, laptop
   asleep, sharing off), it removes that route and the Orin's own Wi-Fi (metric 600) takes over.
   Its switches are logged under `usb-uplink` in the system log.
   Stop it with `sudo pkill -f usb_uplink_watchdog.sh` (run from a separate shell, not over
   an `ssh host '...'` line that contains the same text).

`static-port` matters: without it, macOS's NAT randomises source ports, Viam's network check
reports a "hard" NAT, and the app labels the machine **Degraded** (UDP/STUN). The direct
WebRTC paths get weaker and connections fall back to TURN relays.

Teammates still SSH in over Wi-Fi as usual (`ssh ultraviolet@10.55.13.77` or `rm-sentry.local`):
the watchdog adds a source rule (table 100) so anything addressed to the Wi-Fi IP answers over Wi-Fi.

Keep the laptop awake and on the cable while others rely on the machine. SSH to the Orin
over the cable with `HostKeyAlias=rm-sentry.local`, because every Jetson uses `192.168.55.1`.

## Face servo: "zero IK solutions" errors (2026-10-23), found and fixed
**Symptom:** Auto center worked for small corrections, then failed with `move to position failed
with the caller-supplied pose_cloud: ... zero IK solutions produced, goal positions appears
to be physically unreachable`. This happened when the person shifted sideways: the aim error
jumped to 14-15° and the next 8° turn failed. Small turns also never quite converged (3-5° left over).

**Cause:** the SO-101 has 5 joints and **no wrist yaw**. Its tool axis always lies in the
vertical plane through the base axis, so aiming left or right *requires* the base pan, which
carries the whole gripper sideways (about 2.8 cm for 8° at a 20 cm reach). The servo asked
the planner to turn the tool sideways while holding its position within ±8 mm. That pose doesn't
exist for this arm, so the IK was right to find no solution.

**Fix** (`feeding/face_servo.py: aim_pan_tilt`): every target is reachable by construction.
The tool pose is swung about the base axis toward the face's azimuth (≤ 8°/step), then tilted
inside the arm's plane toward the lips (≤ 8°/step); extend steps stay in that plane. A failed
plan is retried once with a 3× wider pose cloud, and the servo stops only after 3 failures in a
row. Every planner call (inputs, target, joints before/after, result, error, timing) is appended
to `data/logs/face_servo-YYYYMMDD.jsonl` (git-ignored). The test fake now rejects
out-of-plane targets the way the real arm does.

## Face servo: smooth continuous tracking (2026-10-23)
**Symptom:** centring was jerky and didn't settle on the mouth. The plan log showed why:
every tight `MoveToPosition` timed out after about 1 s with no solution, and the wider retry
"succeeded" in 25 ms without moving (the arm was already inside the looser goal). So each step
was a 1 s stall followed by nothing, with occasional real moves.

**Now** (`feeding/face_track.py: LevelTracker`): a 20 Hz loop computes the next joint setpoint
from local forward kinematics (`robot_iface/so101_kinematics.py`, from `Arm.GetKinematics`,
checked to 0.2 mm / 0.3° against the arm). Each tick it reduces the lips' sideways and
vertical miss from the spoon line, the spoon's tilt (level = 0) and any reach change, using
pan, lift, elbow and flex (wrist roll is held). It uses a soft deadband (no hunting),
per-joint velocity limits (20–30 °/s) and an acceleration limit (80 °/s²). Setpoints are
streamed (`streamed: true`, the teleop path) over the direct cable connection. While tracking,
the camera pose comes from the commanded joints at each frame's time (minus 0.12 s servo lag),
so moving doesn't skew the face estimate. The loop stops if the real arm falls more than 10°
behind its commands. Extend still uses the planner and stays off by default.

## Depth Anything 3 corrects the face range (2026-10-23)
The `depth-anything-3` module (`modules/depth-anything-3`, merged from `core/depth-anything-3`)
serves `webcam-2-depth`: metric depth aligned to `webcam-2`, at about 5 Hz on the Orin's GPU.
That's too slow to drive the 20 Hz face loop, so the dashboard uses it to calibrate instead:
a background thread reads the face's depth (forehead, nose bridge, cheeks; never the mouth)
about 4×/s and learns this person's eye-corner span (`mouth_perception/depth_range.py`, a
running median of plausible samples). The fast pipeline then ranges the face with that span,
so distance, and with it the lips' height, are metric. That was the up/down offset seen
when the 90 mm average span didn't fit the person. Both cameras use fx = 772.5 px (hfov 45°).
The dashboard shows the depth distance next to the fast estimate and embeds the module's
own preview page (tunnelled to `localhost:8766` by `dashboard/run_mac.sh`).

**2026-10-24: Depth Pro, and depth + mouth in one view.** Up close DA3's scale is wrong: on a
face ~0.23 m away (eye-corner span) it read 0.59 m; Depth Pro read 0.21 m and MoGe-2 0.45 m.
`webcam-2-depth` now runs Depth Pro (`model: depth-pro`), about 1 frame/s instead of ~5; DA3
stays available as `model: da3-metric`. The dashboard's depth view now draws the depth over
the depth camera's own colour frame, with the lips that MediaPipe found on that same frame and
their distance measured on the lip surface (never inside the open mouth), so depth and mouth
line up exactly. Caveat: Depth Pro and MoGe-2 both estimate the C270's focal at ~1008 px, not
the 772.5 px used here; if they're right, all distances are ~23 % short. Calibrate it.

## Scoop from a bowl (2026-10-23)
Dashboard → **Scoop**: teach the bowl once, then **Plan scoop** (dry run) and **Scoop and come back**.
1. Put the bowl **28-34 cm from the arm's pan axis** (about 32-38 cm in front of the base's
   origin), any direction the arm can turn to. Weigh it down or tape it: the spoon slides along its bottom.
2. **Hand-guide on**, put the spoon tip on the bowl's inside bottom centre, enter the rim radius
   and depth, **Teach bowl**, **Hand-guide off** (saved in `poses/so101_face.json`).
3. Move the arm back up (Start centered or teleop), **Level spoon first**, then **Plan scoop**:
   the log says the dip it found and the duration, or why it refuses (nothing moves).
4. **Scoop and come back**: over the bowl, dip to 6 mm above the bottom, turning tip-down, then
   turn back up to level while sliding toward the far wall, lift, and return to the start joints.
   Streamed at 30 Hz like the tracker; STOP works at any time, and it stops itself if the arm
   falls more than 10° behind.

Why not a 90° rotation: with the spoon tip 13.3 cm out of the gripper, the SO-101 can tip it only
~45-65° near the table while keeping its elbow bent one way. The planner tries 90° first and
uses the deepest dip that fits. Details and checks: `src/assistive_arm/grasping/README.md`.

