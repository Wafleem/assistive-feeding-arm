# RGB-only camera contract proposal

Status: approved by the user on 2026-10-23 and implemented in
[contract PR #1](https://github.com/Wafleem/open-vocab-helper-arm/pull/1).
The camera was subsequently moved from the Pi to the Jetson; the local USB
adapter uses this same approved RGB contract. See [JETSON_VISION.md](JETSON_VISION.md)
for the current deployment.

Requested deployment: RF-DETR spoon boxes and MediaPipe real-person lip landmarks
on the Jetson, consuming `pi-arm2` / `arm-camera`. Vision only; no arm, gripper,
calibration, stop, or motion calls. The current webcam has no depth or calibrated
intrinsics, so the existing `RGBDFrame` cannot represent it accurately.

## Proposed additive interface

Add `RGBFrame` in `vision/types.py`:

```python
@dataclass
class RGBFrame:
    camera: str
    rgb: np.ndarray  # H x W x 3, uint8, RGB
    received_at_s: float  # the consuming robot adapter's monotonic clock
    captured_at_s: float | None = None
    capture_clock: str | None = None  # e.g. "unix"; absent if unknown
```

Add a nonabstract `RobotInterface.get_rgb_frames(names=None)` returning
`dict[str, RGBFrame]`. Its default projects color from existing RGB-D captures;
existing robot implementations keep working. Missing/unknown capture timestamps
stay explicit. Receiving a frame is not proof that the camera just exposed it.

Implement a separate camera-only `RobotInterface` subclass inside
`robot_iface/viam.py`. It creates only a Viam camera client and reads `GetImages`.
It rejects all unsupported arm/gripper/force/pose operations locally, including
`stop()`. Closing the connection never sends a robot-stop command. No changes
are needed to existing `ViamRobot` motion behavior.

## Scope and review

- Existing RGB-D, 3D mouth, food, grasp and motion contracts stay unchanged.
- RF-DETR outputs the existing pixel-space `Detection` values.
- MediaPipe's separate 2D result contains lip points, image center and raw
  aperture ratio. It is not a metric `MouthObservation` or a feeding-ready signal.
- The preview composes those outputs; old overlays expire on missing frames.
- Consumers to review the additive camera contract: shared robot interface,
  vision, mouth perception (stream 1), food perception (stream 2), and simulation.
- Per `AGENTS.md` section 3, put the additive contract and its specs in a separate
  small PR before the implementation PR. No shared contract has been edited by
  this proposal.

Offline specs will check RGB projection, explicit timestamp provenance, unchanged
RGB-D behavior, and that the camera-only adapter cannot send movement commands.
