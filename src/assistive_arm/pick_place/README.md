# 9. Hard-coded pick and place (SO-101 bring-up)

**Owner:** stream 9 (manipulation + integration).
**Robot:** the `pi-arm2` Viam machine: the SO-101 `follower-arm` + `follower-gripper` (5 joints
+ gripper), taught with the `leader-arm` through the `teleop` service. Setup is in
[`docs/SO101_SETUP.md`](../../../docs/SO101_SETUP.md).
**Goal:** a repeatable pick-and-place with **no perception**. Taught poses, played back
in a fixed sequence, with a grip check and safe aborts. It's the first real motion on
hardware, and the skeleton that perception (sections 2-3) plugs into later.

## Approach: teach, then replay
1. **Teach** with the leader arm. Teleop mirrors the leader onto the follower; at each
   key pose, read the follower's joint angles and save them under a name
   (`home`, `pre_pick`, `pick`, `lift`, `pre_place`, `place`, `retreat`). The poses are
   stored in `poses/<task>.json`, which is versioned in git. Manual mode on the follower
   (`enter_manual_mode`) is the alternative to the leader.
2. **Replay** with teleop **stopped**: joint-space moves between the taught poses
   (`MoveToJointPositions`), with the gripper opening and closing at the marked steps.
3. **Check** after the grasp: the gripper reports it is holding something (`grab()` returns
   true / position not fully closed). If not: open, go back to `pre_pick`, stop with a reason.

Why joints, not Cartesian poses: the SO-101 is 5-DOF, so most 6-DOF poses are unreachable
exactly, and taught joint angles replay exactly. Cartesian `MoveToPosition` (with the
module's orientation cone) is the later step, for picking at positions perception provides.

## The sequence (data in `sequence.py`, `PICK_PLACE_STEPS`)
`home → open → pre_pick → pick → close → verify_grip → lift → pre_place → place → open → retreat → home`

## Interfaces
- `types.py`: `TaughtPose` (name, joint angles in **degrees**, gripper %), `PoseLibrary`,
  `Step`, `StepResult`, `RunResult`
- `config.py`: `PickPlaceConfig` (component names, speeds, the grip check, timeouts)
- `teach.py`: `record_pose`, `save_library`, `load_library` (stubs)
- `sequence.py`: `PICK_PLACE_STEPS` (data), `run_pick_and_place` (stub)
- Robot access goes through the proposed joint-space methods on `RobotInterface`
  (see "Contract proposal" below) and the SO-101 Viam adapter
  `robot_iface/viam_so101.py` (stub).

## Contract proposal (needs review before anyone implements it; AGENTS.md §3)
`RobotInterface` today is Cartesian-only. Proposed **optional** additions, defaulting to
`NotImplementedError` so the existing backends are unaffected:
- `get_joint_positions() -> list[float]` (degrees, follower joint order)
- `move_to_joint_positions(joints_deg, speed_deg_s=None) -> MoveResult`
- `move_through_joint_positions(list_of_joints_deg, speed_deg_s=None) -> MoveResult`
- `set_gripper_percent(pct)` or reuse `open_gripper` / `close_gripper`
Units are degrees here, because that's what the SO-101 module and the Viam arm API use for
joints; it's the one documented exception to SI, at this boundary only.

## Safety
- Teleop must be **stopped** before replay. The sequence checks `teleop` status and refuses if it's running.
- Speed: start at `speed_deg_s=30` (the module default is 50); the first run on a new task at 20.
- The workspace is clear, a hand is near the follower's power switch (the SO-101 has no e-stop),
  and the `table` obstacle is in the frame system.
- Every step has a timeout; any failure → `open` gripper only if nothing is held over
  something fragile, then `retreat` → `home`, then stop with a reason.

## TODO checklist
- [ ] Review the contract proposal; add the joint methods to `RobotInterface` (a separate PR)
- [ ] `viam_so101.py`: the SO-101 adapter (arm + gripper + teleop status)
- [ ] `teach.py`: record from the follower while teleop runs; save/load JSON
- [ ] `scripts/so101_teach.py`: interactive teaching CLI ("name this pose:")
- [ ] `sequence.py`: `run_pick_and_place` on FakeRobot first, then on pi-arm2 at low speed
- [ ] `scripts/so101_pick_place.py`: run a taught task N times; log success/failure per step
- [ ] Simulated SO-101 (`devrel:so101:simulated` + simulated gripper) machine for dry runs

## Done when
10 consecutive pick-and-place cycles of one object between two fixed spots on pi-arm2,
with the grip check catching a deliberately empty grasp every time.
