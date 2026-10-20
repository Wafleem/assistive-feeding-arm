# MuJoCo assets

| Path | What |
|---|---|
| `models/xarm6_table.xml` | an arm with xArm6 link dimensions and joint limits (capsule geometry), an 85 mm parallel gripper, a wrist F/T sensor, a wrist RGB-D camera, and a fixed RGB-D camera on a post; table-mounted, base frame = world, z = 0 at the table top |
| `scenes/feeding.xml` | the arm, a plate of finger foods, a cup with a straw, a spoon on a rest, a torso and a head on an actuated base. **The head still has a hinged jaw; TODO: replace it with the 3D-printed always-open mouth + catch cup** |

The behaviour and metrics for these scenes, and the plan, are in
[`src/assistive_arm/sim/README.md`](../src/assistive_arm/sim/README.md); the overview is in
[`docs/SIMULATION.md`](../docs/SIMULATION.md).

## Conventions
- **Geom names are the oracle detector's vocabulary:** `grape_1` is an instance of "grape",
  and `melon_cube_1` of "melon cube".
- **Geoms that belong to the person** start with `user_`, `head_` or `face`. Any arm contact
  with them latches a stop.
- **Real props:** the sim twin comes from the same CAD file (printed parts) or from
  measurements of the bought item (docs/PROPS.md).
- **Downloaded meshes** go in `sim/assets/` (git-ignored); commit the download instructions, not the files.
