# 7. Simulation

**Owner:** stream 7. **Runs on:** the Mac (MuJoCo native) and CI.
**Goal:** every section is developed and scored here before it touches the arm.

## What exists
- `sim/scenes/feeding.xml`: the table-mounted xArm6-dimension arm + gripper, a plate
  (grapes, strawberry, melon cube, cracker), a cup with a straw, a spoon on a rest,
  and a **fake head on a moving base** (x/y/z slides + yaw, actuated) with an
  opening **jaw** and a `mouth` site (+z = face normal).
- `robot_iface/sim.py` `SimRobot`: same API as the real robot; wrist + fixed RGB-D
  cameras with depth and segmentation; per-instance oracle detector; `hooks` for
  scene behaviour; arm contact with head/torso geoms latches a stop.
- `scripts/view_feeding_scene.py`: renders both cameras / opens the viewer.

## Plan (`feeding_world.py`)
1. Head scripts: still, drift, lean-in, turn-away (must cause a retreat), never-opens (must time out).
2. Jaw: opens after a delay; "bite taken" when food is held within 2 cm of the mouth while open.
3. Metrics: distance and lateral error at delivery, max speed within 15 cm of the
   face, head contacts (must be 0), correct aborts per scenario, success per food.
4. `scripts/eval_feeding.py` (TODO): table over scenarios × foods × seeds; CI runs a short version.
5. Domain randomisation: plate layout, food sizes, lighting, camera extrinsics, depth noise, head pose.
6. Swap in real xArm6/gripper meshes (UFACTORY xarm_ros2 URDF / MuJoCo Menagerie);
   measure the real props and update the scene.

## TODO checklist
- [ ] FeedingWorld hook + metrics
- [ ] eval_feeding.py and the CI job
- [ ] Domain randomisation
- [ ] Real meshes, measured props
