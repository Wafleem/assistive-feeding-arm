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
| Other | `arm-camera` (USB webcam, 640×480), `table` obstacle (2 × 2 m, 30 mm thick, top at z = 0), discovery services |

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
