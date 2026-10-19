# Props and their simulation twins

Every prop the robot works on is built twice: once physically and once in
MuJoCo, from the **same CAD model**, and both report success in the **same
format**. The sim eval and the live demo then measure the same thing.

## Principles

1. **One CAD model, three exports.** Design in Fusion, Onshape or OpenSCAD
   (OpenSCAD files diff well in git; put sources in `props/<name>/`). Export:
   * STL for 3D printing,
   * DXF for laser cutting,
   * STL/OBJ meshes for MuJoCo (convex-decompose with `obj2mjcf`/CoACD), with the
     joints, springs and sensors added in a `sim/scenes/<name>.xml` twin.
   Change a dimension in CAD, re-export both. Never hand-edit the sim
   geometry to "match" a prop.
2. **Props report their own success.** An ESP32 (or Arduino) in each prop reads
   its sensors and streams JSON lines over USB serial or Wi-Fi (MQTT or a
   WebSocket to the dashboard), for example:
   ```json
   {"prop": "elevator", "event": "pressed", "button": "7", "t": 12.48}
   {"prop": "door", "angle": 47.0, "handle_turned": true, "t": 30.02}
   {"prop": "bag", "lifted": true, "load_g": 1180, "t": 41.7}
   ```
   The sim twin emits the same events: `SimRobot.events` already produces
   `pressed:<label>` and `door:unlatched`, and `SimRobot.scene_state()` gives
   joint values such as the door angle. `scripts/eval_sim.py` and the dashboard
   should read both sources through one small adapter, so "world ok" means the
   same thing in sim and in the room. On the real props this is the ground truth
   that the robot's own claims are checked against (zero false success claims).

## The props

| Prop | Build | Instrumentation | Sim twin | In repo |
|---|---|---|---|---|
| **Elevator panel** | Laser-cut 3-6 mm plywood or acrylic face, engraved numbers; real 24-30 mm momentary arcade buttons with LEDs (or elevator-style metal buttons); stand at ~90-120 cm with a tilt option | ESP32 reads the buttons, lights the LED, reports `pressed:N` | Button bodies on spring-loaded slide joints with a pressed threshold | `sim/scenes/elevator.xml` (24 mm buttons, 70 mm pitch: set these to the CAD values) |
| **Door with lever handle** | ~40x60 cm leaf, 12 mm plywood/MDF, in an 80/20 aluminium extrusion frame (or 2x2 lumber); real hinges, a **real hardware-store lever handle**, magnetic catch, optional self-closing spring | Potentiometer or magnetic encoder (e.g. AS5600) on the hinge for the angle; microswitch for "handle turned" | Hinge joint with damping + spring; revolute handle with a return spring; latch constraint released past a handle-angle threshold | `sim/scenes/door_lever.xml` is a full-size 0.85 m door: add a `door_prop.xml` at the prop's size from CAD |
| **Knob plate** (stretch) | Swappable door plate with a round knob | Same switch | Revolute knob, either direction | `sim/scenes/door_knob.xml` |
| **Drawer** | Laser-cut/printed box on ball-bearing slides with a D-handle | Linear potentiometer or time-of-flight sensor (VL53L0X) for extension | Slide joint with friction | to add |
| **Shopping-bag station** | Real reusable bag with stiff rounded handles and known weights inside; PETG-printed hook/clamps on extrusion | Optional load cell (HX711) for "lifted" | Box with a rigid handle loop (cloth is hard to simulate; test the real bag on hardware) | `sim/scenes/bag.xml` |
| **Light switch / call button** | Real rocker or large accessibility button on a printed plate | Microcontroller input | Hinged rocker or a button joint like the panel's | to add |
| **Fetch objects** | A bought YCB-style set: mustard bottle, mug (x2 for the "which one?" demo), banana, can, sponge, box; **matte and non-black** (RealSense depth), all **< 80 mm wide** (85 mm gripper) | Re-detection by the camera (object gone / in the bin); optional load cell under the bin | YCB meshes (SIMULATION.md §0) | tabletop scene to add |
| **Bin** | Printed or bought, matte, known size | Load cell or ToF for "object inside" | Box receptacle; success = object centre inside | to add |
| **Wheelchair mount** | Clamp-on 80/20 post with a printed (PETG) or machined xArm base adapter; demo it table-mounted at chair height | — | Fixed or slightly moving base link (`chair_x/y/yaw` joints in `xarm6_wheelchair.xml`) | in the model |
| **Table and shelves** | 2-3 heights, laser-cut or extrusion | — | Boxes at the measured heights | counter in `bag.xml` |
| **Calibration board** | ChArUco printed and mounted flat on acrylic | — | Textured plane (for testing the calibration code in sim) | to add |

## Methods, fastest to sturdiest

1. **Foam board + hot glue** mock-ups: check reach and camera views in an hour,
   before cutting anything.
2. **Laser cutting** for flat parts (panel faces, drawer boxes, plates).
3. **3D printing**: PLA for appearance, **PETG for anything loaded** (hooks,
   clamps, the base adapter).
4. **80/20 extrusion** for any frame that takes arm forces (door frame, mount).
5. **Off-the-shelf hardware wherever the robot touches it**: real buttons, a
   real lever handle, real hinges and slides. The point is to test against
   what exists in buildings.

## Design numbers

| | Value | Why |
|---|---|---|
| Button size | ≥ 19 mm across | ADA minimum for operable parts |
| Controls height | ~38-122 cm | seated reach range |
| Lever handle height | ~86-122 cm | ADA hardware height |
| Force to open | aim for ≤ ~20 N | accessible interior doors; also inside our 30 N door-swing limit |
| Finger clearance behind handles | 20-30 mm | room for the gripper fingers |
| Finish | matte paint, colour contrast | RealSense depth and detection |

Match the sim parameters to the built props: measure button travel and
force, door torque to open, handle return spring, and bag weight, then put the
numbers into the XML twins.

## Rough cost and time

| Prop | Cost | Time |
|---|---|---|
| Elevator panel | $40-60 | one evening |
| Door | $80-150 | a weekend |
| Drawer, bag stand, switch | $20-40 each | an evening each |
| Object set | ~$30 | shopping trip |
