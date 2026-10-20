# 3. Grasping and bite acquisition

**Owner:** stream 3. **Runs on:** PCA on any CPU; AnyGrasp / Contact-GraspNet on the Linux PC or AGX.
**Goal:** get one bite into the gripper (or onto a utensil) and *know* it's there.

## Interfaces
- `GraspModel.propose(points, rgbd, mask) -> list[GraspCandidate]`
- `pca_topdown_grasp(points, support_z) -> GraspCandidate | None` (heuristic fallback)
- `acquire_pinch / acquire_with_utensil / acquire_cup -> AcquisitionResult`
  (`bite_offset` tells feeding where the edible part is relative to the TCP)
- `vla.run_vla(robot, ctx, policy, instruction)`: works today with `ScriptedPolicy`

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

## TODO checklist
- [ ] `pca_topdown_grasp` → `tests/test_grasping.py`
- [ ] `acquire_pinch` on FakeRobot, then on grapes in the MuJoCo feeding scene
- [ ] Gripper feedback from Viam (position/current) for the holding check
- [ ] AnyGrasp / Contact-GraspNet wrappers
- [ ] Utensil and cup strategies (stretch)

## Done when
≥ 80% pinch success on grapes/cubes over randomised sim plates; 4/5 on the real plate.
