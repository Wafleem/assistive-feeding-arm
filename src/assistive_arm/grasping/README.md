# 3. Grasping and bite acquisition

**Owner:** stream 3. **Runs on:** PCA on any CPU; AnyGrasp / Contact-GraspNet on the Linux PC or AGX.
**Goal:** get one bite into the gripper (or onto a utensil) and *know* it's there.

## Interfaces
- `GraspModel.propose(points, rgbd, mask) -> list[GraspCandidate]`
- `pca_topdown_grasp(points, support_z) -> GraspCandidate | None` (heuristic fallback)
- `acquire_pinch / acquire_with_utensil / acquire_cup -> AcquisitionResult`
  (`bite_offset` tells feeding where the edible part is relative to the TCP)
- `vla.run_vla(robot, ctx, policy, instruction)`: works today with `ScriptedPolicy`
- `scoop.plan_scoop(q_start, BowlTarget, ScoopConfig, level_pitch_deg) -> ScoopPlan`: the SO-101
  bench rig's scripted spoon scoop (spoon taped in the jaw), planned and checked offline, streamed by
  `orchestration/face_servo_session.py` (dashboard: Teach bowl, Plan scoop, Scoop)

## Feasibility with a parallel-jaw gripper (85 mm)
| Food | Strategy | Verdict |
|---|---|---|
| grape, cheese cube, firm strawberry | pinch from the top | **MVP** |
| melon cube | pinch, low force | ok, may slip when wet |
| cracker lying flat | pinch | hard: stand it in a rack or use cracker cubes |
| yogurt / pudding | spoon from a fixed rest + scoop | stretch |
| water | cup side grasp, straw to the mouth | stretch (spill risk near the face) |
| soup | — | skip |

## Plan
1. PCA top-down grasp on the item's points; check width vs the gripper.
2. `acquire_pinch`: staging above, guarded descent (≤ 5 N on the plate), close gently
   (`MenuItem.max_grip_force`), check width (not fully closed = holding), lift,
   re-detect that the item left the plate.
3. Open models (AnyGrasp / Contact-GraspNet) as alternatives; compare in sim.
4. Utensil: spoon in a fixed, measured rest; scripted scoop; "spoon loaded?" via the VLM.
5. Cup: side grasp, keep upright, present the straw.
6. Stretch: VLA for acquisition (see food_perception/README.md).

## SO-101 scripted scoop (`scoop.py`, implemented)
No trained model: no public scooping policy fits this rig. The SO-101 SmolVLA/ACT checkpoints
on the Hugging Face Hub are pick-and-place; scooping research (CARBS, VAPORS) uses two Franka
arms and a pusher. A learned scoop needs demos recorded on this arm (leader-arm teleop, LeRobot
format) and a SmolVLA/ACT fine-tune: the stretch item below.

The motion, in the vertical plane through the pan axis and the bowl (no wrist yaw):
over the bowl → dip to 6 mm above the bottom, turning tip-down → **scoop**: turn back up to level
(+3°) while the tip slides along the bottom toward the far wall, hollow leading → lift → back to
the start joints. Every setpoint is solved with local IK (wrist roll held) and checked before
anything moves: IK error, joint limits, no elbow flip, the spoon (with its width) inside the bowl
below the rim and clear of the table. Each phase starts and ends at rest; tip, rotation and joint
speeds are capped. A plan that fails a check is refused with the reason.

What the kinematics allow (spoon tip 21 cm from the tool frame, still a TODO to measure):
- **No 90° dip.** Near the table a 21 cm spoon tips at most ~45-50° with the elbow kept bent one
  way (steeper needs the elbow to swing through straight mid-scoop). `dip_deg` (default 90) is
  tried first, then 5° less at a time; the plan reports the dip it used.
- **Bowl 34-38 cm from the pan axis** (the pan axis is 39 mm in front of the base origin). Closer,
  a level spoon can't get over it; farther, it can't tip down at the bottom.
- The dip point moves toward the bowl's centre as the dip gets shallower, so the handle clears
  the near rim.

## TODO checklist
- [ ] `pca_topdown_grasp` → `tests/test_grasping.py`
- [ ] `acquire_pinch` on FakeRobot, then on grapes in the MuJoCo feeding scene
- [ ] Gripper feedback from Viam (position/current) for the holding check
- [ ] AnyGrasp / Contact-GraspNet wrappers
- [x] SO-101 scripted scoop from a taught bowl (`scoop.py`, `tests/test_scoop*.py`)
- [ ] Measure `spoon_tip_from_tool` (it decides the reachable dip and bowl distance)
- [ ] Bowl from `food_perception` (rim centre − depth) instead of only a taught one
- [ ] "Spoon loaded?" check after the lift (VLM on the gripper camera)
- [ ] Learned scoop: record leader-arm demos, fine-tune SmolVLA/ACT, run it through `run_vla`
- [ ] Utensil and cup strategies (stretch)

## Done when
≥ 80% pinch success on grapes/cubes over randomised sim plates; 4/5 on the real plate.
