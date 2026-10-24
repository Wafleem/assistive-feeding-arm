# Viam module: `depth-anything-3`

Turns any RGB camera into a metric depth camera with a point cloud and a lidar-style range
sensor. Two models: [Depth Pro](https://github.com/apple/ml-depth-pro) (Apple, the default) and
[Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3) (DA3METRIC-LARGE, Apache-2.0).
Runs on the AGX Orin's GPU (docs/COMPUTE.md: models are services).

Which model (measured on the Orin 2026-10-24, one close-up face ~0.23 m away by its eye-corner
span, same focal for all): **Depth Pro 0.21 m**, MoGe-2 0.45 m, DA3 0.59 m. DA3's scale is only
trustworthy at room scale (0.5 m+, like its benchmarks) but it is ~6x faster: about 0.15 s per
frame vs ~1.1 s for Depth Pro. The module was first built around DA3, hence its name.

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
                      "weights": "/home/ultraviolet/depth-anything-3/weights/depth_pro.pt" },
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
| `model` | `depth-pro` | `depth-pro` (accurate close up, ~1 Hz) or `da3-metric` (~6 Hz, trust from ~0.5 m) |
| `weights` | the model's own | Depth Pro: HF repo `apple/DepthPro` or a local `.pt`; DA3: HF repo id or local dir (metric models only) |
| `device` | `auto` | `auto` / `cuda` / `mps` / `cpu`. `auto` with a CUDA build of torch requires a working GPU: if it is down, the module reports an error and retries every 30 s instead of crawling on the CPU (on the Orin ~8 min per frame, starving teleop). Use `cpu` to force the CPU |
| `process_res` | 504 | DA3 only: input size (longest side); lower = faster, coarser. Depth Pro always uses 1536 |
| `produce_hz` | 2 | background inference rate (upper bound) |
| `stale_after_s` | 5 | older depth is refused with an error, never served |
| `fx` `fy` (`cx` `cy`) / `hfov_deg` | – | intrinsics: these, else the source's properties, else Depth Pro's own focal estimate (DA3: an error) |
| `stride` | 2 | point-cloud pixel subsampling |
| `min_range_m` / `max_range_m` | 0.1 / 10 | returns outside are dropped |
| `center_roi` | 0.1 | fraction of the image used for `distance_m` |
| `scan_beams` / `scan_band` | 64 / 0.2 | beams, and the fraction of the image height the scan looks through |
| `min_support_px` | 25 | a return needs this many pixels at or closer than it (rejects flying pixels) |
| `preview_port` | 0 (off) | serve the preview page on `127.0.0.1:<port>` (below) |
| `vision_status_url` | off | e.g. `http://127.0.0.1:8765/status.json`: also show how far away vision_preview's mouths and spoons are |

## Cloud GPU (the default)

`compute: cloud` (default) sends each frame (a ~35 kB JPEG) to `cloud/depth_server.py` on a cloud
GPU and gets depth back (~100 kB, uint16 mm). The Orin runs no model then, which keeps it cool.
While the cloud is unreachable the local model takes over (`cloud_fallback`, rechecked every
30 s); `compute: local` runs only on this machine. Status shows where each frame ran (`device`)
and the round trip (`timing`).

| Attribute | Default | |
|---|---|---|
| `compute` | `cloud` | `cloud` or `local` (cloud serves Depth Pro only) |
| `cloud_url` | `http://127.0.0.1:8800` | the server, through an SSH tunnel |
| `cloud_timeout_s` | 3 | per frame; a timeout counts as the cloud being down |
| `cloud_fallback` | true | run the local model while the cloud is down |

Set up (NVIDIA Brev, PyTorch template, one L4/A10/L40S; stop it when idle, it bills by the hour):
1. On the instance: copy `cloud/` there and run `bash cloud/setup_instance.sh` (installs Depth Pro,
   fetches the weights, starts the server on the instance's 127.0.0.1:8800).
2. On the Orin, keep the tunnel up with `cloud/depth-cloud-tunnel.service` (systemd, as `ultraviolet`;
   starts at boot, restarts itself). It uses a dedicated key, `~/.ssh/brev_depth` on the Orin, whose
   public half is in the instance's `authorized_keys` as
   `restrict,port-forwarding,permitopen="127.0.0.1:8800" ssh-ed25519 ...`, through Brev's SSH gateway
   (`global.prd.ga.run.brev.nvidia.com`, the instance's port from `~/.brev/ssh_config` after
   `brev refresh`; it can change when the instance is recreated). The conference network blocks the
   instance's direct SSH port, and the server is never exposed publicly (Brev's public URLs need a
   browser login, which a program can't do). Check: `curl http://127.0.0.1:8800/health` on the Orin.
3. Switch the camera over: `compute: cloud` (and `produce_hz: 3`); back with `compute: local`.

Measured 2026-10-24, Orin at the venue → Brev H200 (Nebius, Iowa), 640x480 frames: **3.2 fps**,
latency median 288 ms (p90 367) = 103 ms GPU + ~183 ms network; same depth as on the Orin. On the
Orin's own GPU: ~0.8 fps, ~1.1-1.6 s.

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
(Depth Pro, 1.9 GB) or `./fetch_weights.sh da3-metric` (resumable; the venue Wi-Fi drops
connections) and point `weights` at them.

**Focal length.** webcam-2 is configured with `hfov_deg: 45` (772.5 px, from the C270's 55°
diagonal spec). Depth Pro and MoGe-2 both estimate ~1008 px for it; if they're right, every
distance is ~23 % short. Calibrate the camera (checkerboard) before trusting absolute distances.

viam-server stops a module with SIGTERM and then closes its connection; the Python SDK waits
for that, so `kill` on its own doesn't restart the module. To load new code, change the
module's config (for example its `env`) and viam-server restarts it.

## Develop

```bash
python3.12 -m venv .venv && .venv/bin/pip install torch torchvision -r requirements.txt pytest pytest-asyncio
.venv/bin/python -m pytest   # offline: fake source camera, fake model
```
