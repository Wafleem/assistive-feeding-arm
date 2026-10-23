# Viam machine configs and calibrations

| Path | What |
|---|---|
| `machines/pi-arm2.json` | the Raspberry Pi rig as calibrated on 2026-10-23 (leader + follower + teleop + webcam) |
| `machines/orin-dev-kit.json` | the same SO-101 setup on the Jetson Orin dev kit (leader + teleop on), plus webcam-2 and its Depth Anything 3 depth camera and range sensor (`modules/depth-anything-3`) |
| `meshes/bowl-150x60.ply`, `meshes/bowl_ply.py` | the open-top bowl mesh used by `bowl-right` / `bowl-left` on orin-dev-kit, and the script that writes it (see `docs/SO101_SETUP.md`, Collision geometry) |
| `calibration/so101_follower_calibration.json` | follower calibration, read back from the arm after calibrating |
| `calibration/so101_leader_calibration.json` | leader calibration, read back from the arm after calibrating |

These are reference snapshots, not live config: the live config is in the Viam app.
Re-snapshot after any change you want to keep (the Viam MCP `read_machine_config` or the app's JSON view).

## Putting a calibration file on a new machine
The arm reads `calibration_file` from the module's data directory on that machine:
`/root/.viam/module-data/<part-id>/devrel_so101-arm/<file>` (`<part-id>` is the machine
part id; for orin-dev-kit it's `af8aae81-2f68-46bb-a2a0-fdc12ed08ad0`). Copy the JSON there,
then send `{"command": "reload_calibration"}` to the arm and check it with
`{"command": "get_calibration"}`. A calibration is only valid for the physical arm it was
recorded on: re-run calibration after rebuilding an arm.
