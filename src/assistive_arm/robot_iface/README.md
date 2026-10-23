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

## TODO
- [ ] `ViamRobot` against the real venue machine: check the SDK signatures, the
  `get_images` format, and gripper position/current for `holding`.
- [ ] Attach the carried item to the gripper in `ViamRobot._obstacles` (`TODO(feeding)`).
- [ ] Remove the one layering exception: `viam.py` imports `motion.safety` for the
  keep-out zones; it should receive zones through `update_world` instead.
- [ ] Contract tests: run one shared set of assertions against Fake, Sim and (on the bench) Viam.
