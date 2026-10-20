# 7. Simulation

**Owner:** stream 7. **Runs on:** the Mac (MuJoCo native) and CI.
**Goal:** every section is developed and scored here before it touches the arm.

## What exists (from the first scaffold; review before building on it)
- `sim/scenes/feeding.xml`:
  - the table-mounted xArm6-sized arm and gripper;
  - a plate with grapes, a strawberry, a melon cube and a cracker;
  - a cup with a straw, and a spoon on a rest;
  - a head on a moving base (x/y/z slides + yaw) with a `mouth` site. **It still has a
    hinged jaw, which no longer matches the prop.**
- `robot_iface/sim.py` `SimRobot`:
  - the same API as the real robot;
  - wrist and fixed RGB-D cameras;
  - an oracle detector;
  - `hooks` for scene behaviour;
  - any arm contact with the head stops the arm.
- `scripts/view_feeding_scene.py` (`make render`, `make viewer`).

## Plan (`feeding_world.py`)
1. **Scene change:** replace the jaw head with the printed mouth:
   - a face plate built from the same CAD as the real print;
   - an always-open ~50 × 40 mm opening;
   - a catch cup behind it;
   - a `mouth` site at the opening centre and a `mouth_catch` volume.
2. **Head scripts:**
   - still and drift;
   - moved away: slid or turned during the approach; MUST retreat;
   - tracking lost: MUST pause or retreat.
3. **Delivered** = the bite's centre ends inside the catch-cup volume after release.
4. **Metrics:**
   - delivered per food;
   - lateral error at the opening;
   - max speed within 15 cm of the face;
   - face-plate contacts (must be 0);
   - correct retreats.
5. `scripts/eval_feeding.py` (TODO): a table over scenarios × foods × seeds; CI runs a short version.
6. **Domain randomisation:**
   - plate layout and food sizes;
   - lighting;
   - camera extrinsics and depth noise;
   - the head's start pose.
7. Real xArm6 and gripper meshes; measured props.

## TODO checklist
- [ ] Printed-mouth scene (replaces the jaw head) + its tests
- [ ] FeedingWorld hook + metrics
- [ ] eval_feeding.py and the CI job
- [ ] Domain randomisation
- [ ] Real meshes, measured props
