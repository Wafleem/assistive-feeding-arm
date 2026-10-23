# Shared core: `robot_iface`: one robot API, three backends

**Owner:** integration (changes reviewed by every section; this is the most-used contract).

| File | What | Status |
|---|---|---|
| `base.py` | `RobotInterface` (abstract), `MoveResult`, `GripperState`, `Wrench` | contract |
| `fake.py` | `FakeRobot`: no physics; straight-line motion in 1 mm steps, spring "surfaces", graspable points, a virtual clock | works (from the first scaffold) |
| `sim.py` | `SimRobot`: MuJoCo arm, gripper, F/T sensor, wrist + fixed RGB-D cameras, oracle detector, scene `hooks`, simulated clock | works (from the first scaffold) |
| `viam.py` | `ViamRobot`: the venue xArm6 + gripper + cameras through the Viam SDK; keep-out zones as motion-planner obstacles | written, **untested on hardware** |
| `viam_so101.py` | `SO101ViamRobot`: the SO-101 follower + gripper on `pi-arm2` (5 joints, joint-space first, teleop status) | **stub** (section 9) |
| `viam_convert.py` | metres/quaternions ↔ Viam millimetres/orientation vectors | works, tested |

## The contract (what every backend must do)
- **Units and frames:**
  - everything is in the **arm base frame**, in metres, seconds and newtons;
  - camera poses use OpenCV optical axes (+z forward, +y down).
- `move_to_pose(pose, speed)`: a free-space move, collision-checked by the backend.
- `move_linear(pose, speed, max_force)`: a straight line. With `max_force` set it is a
  **guarded move** and stops with `reason="force_limit"`. This is the *only* way to touch anything.
- `open_gripper` / `close_gripper` → `GripperState(width, holding)`: closed-loop, `holding` = it stopped on something.
- `get_wrench` / `tare_wrench`: external force on the tool.
- `cameras`, `get_frames`, `camera_pose`.
- `now` / `wait`: the robot's clock (simulated time in sim). Trackers and timeouts must use it.
- `update_world(zones)`: moving obstacles (the tracked head) for the planner.
- `stop`: halt now. This is not a substitute for the hardware e-stop.

## RGB-only local camera preview

When the wrist camera is plugged directly into the Jetson and no Viam camera
resource owns it, use `UsbCameraReader(UsbCameraConfig(...))`. The default device
is `/dev/v4l/by-id/usb-icSpring_icspring_camera-video-index0`, named `arm-camera`,
requesting 640 x 480 at 30 fps and a one-frame driver buffer. The frozen config
also sets frame wait, maximum receive age, and shutdown wait limits.

The capture worker continuously reads while the model runs and retains only the
newest frame. Each successful `get_rgb_frames()` returns a newer frame than its
previous result. Camera failure clears cached frames; missing or old frames
produce a bounded timeout. A failed capture releases its device and retries
after the configurable one-second reconnect delay, so replugging can restore
the preview. The error clears only after a new frame arrives. Shutdown interrupts
the retry delay. `received_at_s` records read completion, not poll time;
exposure time and its clock remain unknown. Requested V4L2 settings may be
negotiated or ignored, so actual latency must be measured on the device.

This adapter has no Viam or arm connection. Movement, force, pose, depth,
calibration, and `stop()` operations raise locally. Closing only asks the capture
worker to release its camera. If a native driver read remains blocked past the
configured close timeout, shutdown reports that release is pending; it does not
release the same native capture concurrently with its read. Start it only when
another camera process is not using the device. OpenCV is imported lazily and is
provided by the optional live-vision dependency set.

## TODO
- [ ] `ViamRobot` against the real venue machine: check the SDK signatures, the
  `get_images` format, and gripper position/current for `holding`.
- [ ] Attach the carried item to the gripper in `ViamRobot._obstacles` (`TODO(feeding)`).
- [ ] Remove the one layering exception: `viam.py` imports `motion.safety` for the
  keep-out zones; it should receive zones through `update_world` instead.
- [ ] Contract tests: run one shared set of assertions against Fake, Sim and (on the bench) Viam.
