# Viam module: `depth-anything-3`

Turns any RGB camera into a metric depth camera with a point cloud and a lidar-style range
sensor, using [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3)
(DA3METRIC-LARGE, Apache-2.0). Runs on the AGX Orin's GPU (docs/COMPUTE.md: models are services).

| Model | API | What it serves |
|---|---|---|
| `assistive-arm:depth-anything-3:camera` | camera | GetImages: `color` (JPEG), `depth` (Viam raw depth, mm), `depth_color` (JPEG, near = red). GetPointCloud: PCD in **mm**, camera frame. DoCommand `{"command": "ranges"}` / `{}` (status) |
| `assistive-arm:depth-anything-3:range` | sensor | the lidar readings below, from the camera's newest frame (no extra inference) |

Range readings (metres, degrees; `None` = no return): `distance_m` (median straight ahead,
the "rangefinder"), `nearest_m` + `nearest_bearing_deg` / `nearest_elevation_deg`,
`scan_ranges_m` (a 2D scan across the horizontal field of view, left to right, with
`scan_angle_min_deg` and `scan_angle_increment_deg`), `valid_fraction`, `age_s`.
Ranges are along each pixel's ray, like a lidar, not z-depth. Bearing is + right of the
optical axis, elevation + up.

## Configure

```json
{
  "modules": [{ "type": "local", "name": "depth-anything-3",
                "executable_path": "/home/ultraviolet/depth-anything-3/run.sh" }],
  "components": [
    { "name": "webcam-2-depth", "api": "rdk:component:camera", "model": "assistive-arm:depth-anything-3:camera",
      "attributes": { "source": "webcam-2", "hfov_deg": 45,
                      "weights": "/home/ultraviolet/depth-anything-3/weights/DA3METRIC-LARGE" },
      "frame": { "parent": "<same parent as webcam-2>", "translation": "<same as webcam-2>",
                 "orientation": "<same as webcam-2>" } },
    { "name": "webcam-2-range", "api": "rdk:component:sensor", "model": "assistive-arm:depth-anything-3:range",
      "attributes": { "camera": "webcam-2-depth" } }
  ]
}
```

Give the depth camera **the same frame as its source**: the point cloud is in the source
camera's optical frame (+z out of the lens, +x right, +y down).

| Attribute | Default | |
|---|---|---|
| `source` | required | the RGB camera |
| `weights` | `depth-anything/DA3METRIC-LARGE` | HF repo id or local dir; metric models only (DA3METRIC, DA3NESTED) |
| `device` | `auto` | `auto` / `cuda` / `mps` / `cpu`; `auto` checks that CUDA really runs |
| `process_res` | 504 | DA3 input size (longest side); lower = faster, coarser |
| `produce_hz` | 2 | background inference rate (upper bound) |
| `stale_after_s` | 5 | older depth is refused with an error, never served |
| `fx` `fy` (`cx` `cy`) / `hfov_deg` | – | intrinsics: these, else the source's properties, else an error |
| `stride` | 2 | point-cloud pixel subsampling |
| `min_range_m` / `max_range_m` | 0.1 / 10 | returns outside are dropped |
| `center_roi` | 0.1 | fraction of the image used for `distance_m` |
| `scan_beams` / `scan_band` | 64 / 0.2 | beams, and the fraction of the image height the scan looks through |
| `min_support_px` | 25 | a return needs this many pixels at or closer than it (rejects flying pixels) |
| `preview_port` | 0 (off) | serve the preview page on `127.0.0.1:<port>` (below) |
| `vision_status_url` | off | e.g. `http://127.0.0.1:8765/status.json`: also show how far away vision_preview's mouths and spoons are |

## Preview page

With `preview_port` set (the Orin uses 8766), the module serves a read-only page from the
frames it already has: colour and depth side by side, metres under the cursor, the
rangefinder box, the nearest return, and the scan seen from above. It binds to localhost
only; open `http://127.0.0.1:8766` on the Orin, or tunnel it:

```bash
ssh -N -L 8766:127.0.0.1:8766 ultraviolet@<orin>   # then open http://localhost:8766
```

Frames are pushed down one Server-Sent Events connection, newest only, so a slow link lowers
the frame rate instead of adding delay; the page blanks and says "stale" when frames stop.
The venue Wi-Fi (-82 dBm at the Orin, ~1 Mbit/s, 7 s ping on 2026-10-23) still only manages a
frame every ~15 s: use the Orin's screen, Ethernet, or the USB-C device-mode link
(`ssh ultraviolet@192.168.55.1` from a computer plugged into the Orin's USB-C port).

**Accuracy.** Metric depth from one image is an estimate. It scales with the focal length,
so a 5 % focal error is a 5 % distance error: calibrate the camera, or at least check
`hfov_deg` (webcam-2 is a Logitech C270: 55° diagonal, so about 45° horizontal at 640x480). Treat it as
a perception aid, never as a safety sensor (AGENTS.md §4).

## Install on a machine

`setup.sh` (the module's `first_run`) builds `viam-env/`. On the AGX Orin, start from a venv
that already has torch: `DA3_BASE_VENV=~/assistive-arm-vision/.venv ./setup.sh`. PyPI's
cu13x torch runs on the Orin's GPU: its sm_80 kernels run on sm_87, though torch warns
that 8.7 isn't an official target. Download the weights beside it with `./fetch_weights.sh`
(resumable; the venue Wi-Fi drops connections) and set `weights` to that directory.

viam-server stops a module with SIGTERM and then closes its connection; the Python SDK waits
for that, so `kill` on its own doesn't restart the module. To load new code, change the
module's config (for example its `env`) and viam-server restarts it.

## Develop

```bash
python3.12 -m venv .venv && .venv/bin/pip install torch torchvision -r requirements.txt pytest pytest-asyncio
.venv/bin/python -m pytest   # offline: fake source camera, fake model
```
