# Jetson spoon and mouth preview

This milestone reads the arm-mounted camera and runs RF-DETR Nano for spoon
boxes and MediaPipe Face Landmarker for real-person lip contours. Outputs are
image pixels and a raw mouth aperture ratio. They do not authorize a grasp or
feeding movement, and do not estimate metric depth or feeding readiness.

The user approved the additive RGB camera interface on 2026-10-23; see
[RGB_ONLY_CAMERA_PROPOSAL.md](RGB_ONLY_CAMERA_PROPOSAL.md). Existing RGB-D and
motion interfaces retain their behavior. The contract is reviewed separately
from the implementation.

## Runtime

- Host: `ssh jetson-sentry-wifi`, AGX Orin 32 GB.
- Root: `/home/ultraviolet/assistive-arm-vision`.
- Python: isolated `.venv`, Python 3.12; Ubuntu 24.04, L4T R39.2.1.
- Candidate GPU packages: torch 2.13.0+cu132, torchvision 0.28.0+cu132 from
  `https://download.pytorch.org/whl/cu132`. Verify a real CUDA operation before
  claiming GPU inference works; see `deploy/jetson-cu132-constraints.txt`.
- Models: RF-DETR 1.11.2 and MediaPipe 1.1.1. Viam SDK 0.85.0.
- Physical USB source: `/dev/v4l/by-id/usb-icSpring_icspring_camera-video-index0`,
  640 × 480 at a requested 30 fps. The camera was physically moved to the Jetson;
  a direct capture succeeded on 2026-10-23. `UsbCameraReader` implements the same
  RGB-only `RobotInterface` and continuously drains capture to keep the latest frame.
- Viam source: `orin-dev-kit` / `arm-camera`; added by the team after relocating
  the camera. `GetImages` successfully returned a 640 × 480 JPEG on 2026-10-23.
- Viam authentication uses a dedicated machine-scoped Operator key stored in
  `vision.env` with mode 0600, inside a private directory. Operator is not a
  camera-only permission; the application enforces read-only behavior. The
  initial `pi-arm2` key cannot access `orin-dev-kit`; its separate credential
  approval is pending.

Viam on the Jetson now owns the USB camera, so the preview must use the Viam
source. No Viam configuration or server restart is required. The old `pi-arm2`
camera resource reports disconnected because the camera is no longer plugged
into that Pi. The team also configured a YOLO service; this preview does not
alter it.

The direct USB adapter remains available with `ASSIST_VISION_SOURCE=usb`, but
only when Viam or another application is not using the device. That path does
not load credentials or access any arm serial devices. The default Viam source
loads the private `vision.env` credentials.

## Start and view

To reproduce the environment on this JetPack/CUDA version, from the deployed
checkout (with the virtual environment at `../.venv`):

```sh
python3 -m venv ../.venv
../.venv/bin/python -m pip install \
  'torch==2.13.0+cu132' 'torchvision==0.28.0+cu132' \
  --index-url https://download.pytorch.org/whl/cu132
../.venv/bin/python -m pip install -r deploy/jetson-vision-requirements.txt \
  -c deploy/jetson-cu132-constraints.txt
../.venv/bin/python -m pip install --no-deps -e .
../.venv/bin/python -m pip check
python3 deploy/fetch_vision_models.py ../data/models
```

`fetch_vision_models.py` retrieves the two official model files and verifies
their recorded hashes before making them available to inference. Keep
`VIAM_ADDRESS`, `VIAM_API_KEY_ID`, and `VIAM_API_KEY` in the private `vision.env`,
not in tracked configuration. The CUDA wheel choice is specific to the observed
L4T R39.2.1 / CUDA 13.2 environment.

From the deployed checkout on the Jetson:

```sh
sh deploy/run-vision.sh --smoke-frames 3
sh deploy/run-vision.sh --port 8765
```

From the Mac, forward the local preview port:

```sh
ssh -N -L 127.0.0.1:8765:127.0.0.1:8765 jetson-sentry-wifi
```

Open `http://127.0.0.1:8765`. The server binds to the Jetson's loopback address;
the SSH tunnel provides access. Green boxes are spoons; orange contours are lips.
`/status.json` reports observations, processing durations and timestamp provenance.
No raw frames, face images or videos are saved by the preview.

Ctrl-C or terminating this preview only closes its own model/camera connections.
It never calls an arm, gripper or robot-stop API. Both camera readers explicitly
reject those operations locally; the Viam reader also disables SDK sessions.

## Interpretation and troubleshooting

- The preview shows age since the camera adapter received a frame, not measured
  exposure age. For USB, capture timestamp/clock remain unknown. For Viam, its
  capture timestamp and clock domain are reported separately.
- Images and observations expire after three seconds since receipt. Camera or
  inference failures clear the prior overlay. The browser also hides stale
  imagery if the connection stops responding.
- A successful empty-scene inference verifies the software path, not spoon or
  mouth accuracy. Positive checks require a visible spoon and a visible face.
- A spoon box is not a handle grasp point. MediaPipe's relative landmark depth is
  not used as a physical distance.
- If Viam reports the webcam is disconnected, check the camera USB connection and
  its live view in Viam. Do not restart the arm server during someone else's
  calibration session.
- This is a standalone preview; it does not add a vision service or overlay to
  Viam's CONTROL tab. Viam service packaging remains a separate integration task.

## Verification

The offline suite covers model output filtering, lip geometry, RGB preservation,
timestamp provenance, camera-only behavior and stale/error preview handling.
Run `make test` with the development extra. Real model weights, credentials and
hardware are not required by those tests.

Deployment measurements and live verification will be recorded here after the
Jetson installation and camera checks finish.
