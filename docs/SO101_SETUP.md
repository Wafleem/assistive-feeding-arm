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
names; the leader, both calibration sensors and teleop are **disabled** until the leader is
plugged in. Snapshots: `viam/machines/`.

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
