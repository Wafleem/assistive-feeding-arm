# Props: buy it or pack it flat, and give each one a sim twin

Everything the arm manipulates has to get to the venue in suitcases or a car
trunk, or be bought near the venue. So every prop is one of two kinds:

* **A: bought ready-made** at a big-box or online store (Target, Walmart, Home
  Depot / Lowe's, IKEA, Amazon), used as-is or with small changes. It can be
  bought near the venue or shipped there. **Prefer A.**
* **B: knock-down / flat-pack.** It breaks down into parts that fit in a
  carry-on or checked suitcase (or a car trunk). It has no glue joints, goes
  back together on site in **30 minutes or less** with hand tools (hex keys, a
  screwdriver), and **clamps to the venue table** instead of standing on the
  floor.

Anything custom (laser-cut or 3D-printed) stays small and flat.

## Principles

1. **One model per prop, real dimensions.**
   * *Bought items (A):* measure the item after buying it (calipers for
     handles and objects, tape measure for panels) and put the numbers into
     its CAD/MJCF twin. Re-measure if the venue copy is a different brand or
     size.
   * *Custom parts (B):* design once in CAD (OpenSCAD, Fusion or Onshape;
     sources in `props/<name>/`) and export the same model to STL (print),
     DXF (laser cut) and mesh/MJCF (MuJoCo).
   * Never hand-edit the sim geometry to "look right".
2. **Props report their own success.** An ESP32 (or a USB HID encoder) reads
   the prop's sensor and streams JSON lines over USB serial or Wi-Fi:
   ```json
   {"prop": "elevator", "event": "pressed", "button": "7", "t": 12.48}
   {"prop": "door", "angle": 47.0, "t": 30.02}
   {"prop": "drawer", "open_mm": 120, "t": 51.3}
   ```
   The sim twin reports the same events. `SimRobot.events` already gives
   `pressed:<label>` and `door:unlatched`, and `SimRobot.scene_state()` gives
   joint values such as the door angle. One small adapter should feed both
   sources to the dashboard and to `scripts/eval_sim.py`, so "world ok" means
   the same thing in sim and in the room. All prop electronics are low-voltage
   (USB 5 V). **Nothing is wired to mains**, including the light switch.

## The props

| Prop | Kind | What to use | Self-reporting sensor | Sim twin | Status |
|---|---|---|---|---|---|
| **Fetch objects** | A | Grocery and household items: blue water bottle, 2 identical mugs, mustard bottle, soup can, banana, sponge, small cracker or tea box. **Matte, not black, each < 80 mm** across where it will be gripped (the gripper opens 85 mm). | Camera re-detection (the object is gone or in the bin) | YCB meshes where they match; otherwise measure the item and use a box/cylinder primitive | Tabletop scene to add (SIMULATION.md §0) |
| **Bin** | A | Small open wastebasket or storage bin, matte, light colour, wide mouth (≥ 25 cm) | Optional time-of-flight (ToF) sensor taped inside the rim ("object inside") | Box receptacle at the measured size; success = object centre inside it | to add |
| **Shelf height** | A | Kitchen shelf riser or stackable cabinet shelf, used on the venue table to give a second height | — | Box at the measured height | to add |
| **Shopping bag** | A | Reusable shopping bag with **stiff** handles (rigid or webbing that holds its loop shape), with soup cans inside as known weights | None needed: the arm weighs the load with its F/T reading. Optional load cell is skipped for the MVP. | Box with a rigid handle loop, mass = the measured bag + cans. Cloth is hard to simulate, so validate the real bag on hardware. | `sim/scenes/bag.xml`: set its dimensions and mass from the bought bag |
| **Drawer** | A | Small plastic desktop drawer unit (2-3 drawers), used as bought. If the pull is a recessed lip the fingers can't get behind, screw on a cabinet D-pull with 20-30 mm clearance. | ToF sensor (VL53L0X) at the back of the drawer measuring how far it is open | Slide joint with friction; travel and pull size measured | to add |
| **Light switch / call button** | A | Real rocker switch on a wall plate in a plastic single-gang box screwed to a small flat board, **or** a large adaptive/accessibility button (USB) | Switch contacts wired to an ESP32 input at 5 V. No mains, ever. | Hinged rocker or a slide-joint button like the panel's | to add |
| **Elevator panel** | B, mostly A parts | Arcade button kit (24-30 mm LED buttons + USB encoder board). The faceplate is either a bought plastic cutting board / project-box lid drilled with a step bit (A), or one small laser-cut acrylic/plywood plate with engraved numbers (B, flat). It slides into a **clamp-on stand** made from the same 2020 extrusion kit as the door frame. Electronics live in a small project box. | The USB encoder appears as a gamepad (no firmware), or an ESP32 reads the buttons and lights the LEDs; reports `pressed:N` | Spring-loaded slide-joint buttons with a pressed threshold; plate size, button pitch and travel measured | `sim/scenes/elevator.xml`: set its button size, pitch and height to the built panel |
| **Door** (MVP: grab and swing) | B | A **cabinet door bought on its own** (IKEA-style kitchen front or a hardware-store cabinet door, about 40x60 cm, 16-19 mm thick) on surface-mount cabinet hinges, with a **magnetic catch** and a bought D-pull or bar pull with 20-30 mm finger clearance. It hangs on a small **2020 aluminium extrusion frame** (lengths ≤ 60 cm, corner brackets, T-nuts) that **clamps to the table edge**. Hinges stay screwed to the door; the frame bolts together with a hex key. | Magnetic angle encoder (AS5600) or a potentiometer at the hinge, read by an ESP32: `{"prop": "door", "angle": ...}` | Hinge joint with damping, plus a weak catch modelled as the existing latch constraint; dimensions measured from the bought door | `sim/scenes/door_lever.xml` is a full-size 0.85 m door with a turning lever: add `door_cabinet.xml` at the bought door's size |
| **Door with a turning lever** | ⚠ fails A/B as specified, see Flags | | | | stretch / sim-only |
| **Knob plate** | ⚠ skip for the MVP | | | `sim/scenes/door_knob.xml` stays sim-only | sim-only |
| **Calibration board** | A | ChArUco pattern printed at a print shop and mounted on foam board, or a ready-made calibration board; carried flat in a laptop sleeve | — | Textured plane | to add |
| **Arm mount** | B (not manipulated, but must travel) | Aluminium base plate drilled for the xArm6 bolt pattern, with 4 heavy-duty C-clamps to the table. See Flags. | — | Fixed base link at the measured table height (the sway joints stay at zero for the table demo) | in the model |

## Flags: things that don't meet A or B as specified

1. **A door lever that actually turns and releases a latch.** Real door
   levers (passage sets) need a door about 35-45 mm thick with a large bored
   hole for the latch. A cabinet door is too thin, and a thick panel is heavy
   and needs boring. **MVP:** the cabinet door with a magnetic catch and a
   fixed pull, which tests grasp + swing along the hinge arc. The handle-turn
   part of `open_door` stays a sim skill. **Stretch (B, heavier):** a ~35 mm
   laminated panel, bored at home with a hole-saw kit, fitted with a
   hardware-store passage lever. It is about 4-6 kg at 40x60 cm (estimate;
   weigh it) and fits a checked suitcase.
2. **Knob plate:** skip for the MVP (the knob skill isn't working in sim yet either).
3. **Arm mount on an unknown venue table.** The xArm6 (about 12 kg, plus the
   loads when it moves) needs a rigid, heavy table. Bring the plate and
   clamps, and test the table for wobble on arrival. **Fallback (A):** buy a
   sturdy folding work table near the venue and clamp to it. This is the one
   floor-standing item, and it is bought, not carried. The robot arm and its
   controller travel in their own case; check the airline's checked-bag
   weight limit (often 23 kg) before planning to fly with them.
4. **The wheelchair itself:** skip for the MVP. Demo the arm table-mounted at
   seat height. The sim keeps the wheelchair model.
5. **Soft bag handles:** fine on hardware, poor in sim. Keep the bag's sim twin
   rigid and judge the bag skill on the real bag.

## Shopping list

Prices are **rough estimates** for planning only. No SKUs on purpose:
**verify the item, size and price in the store or online before buying.**
"Ahead" = buy before the event (to measure for the sim twins and test).
"Near venue" = buy on arrival (perishable, bulky, or cheap and easy to find).
For anything bought twice (ahead and again near the venue), buy the same
brand and size so the measured sim twin still matches.

| Item | For | Where (store: department) | Approx. price (estimate) | When |
|---|---|---|---|---|
| Blue reusable water bottle, ≤ 80 mm diameter, matte if possible | fetch | Target: kitchen & dining / drinkware | $10-20 | Ahead (measure), carry |
| 2 identical mugs, light matte colour | fetch "which one?" | Target / IKEA: kitchen & dining | $5-15 for two | Ahead, carry (wrap them) |
| Mustard squeeze bottle, soup can, cracker or tea box, sponge | fetch, bag weights | Target / Walmart: grocery, cleaning supplies | $10-15 | Near venue (same brands as the ones measured) |
| Bananas | fetch | any grocery: produce | $1-2 | Near venue |
| Small wastebasket or storage bin | bin | Target: home organisation / bath | $8-15 | Near venue (bulky); measure the one you buy |
| Shelf riser | second height | Target / IKEA: kitchen organisation | $10-20 | Either |
| Reusable shopping bag with stiff handles | bag | Target: checkout lanes / home | $1-5 | Ahead (measure) and a spare near venue |
| Small plastic drawer unit | drawer | Target / Walmart: storage & organisation | $15-30 | Near venue (bulky); measure on arrival or buy the same model ahead |
| Cabinet D-pull (if the drawer lip is ungrippable) | drawer | Home Depot / Lowe's: cabinet hardware | $3-8 | Ahead |
| Cabinet door ~40x60 cm (IKEA-style front or hardware-store cabinet door) | door | IKEA: kitchen fronts; Home Depot: cabinet doors | $20-50 | Ahead (measure, fit hinges) |
| Surface-mount cabinet hinges (pair), magnetic catch, bar or D-pull | door | Home Depot / Lowe's: cabinet hardware | $10-20 | Ahead |
| 2020 aluminium extrusion kit (≤ 60 cm lengths), corner brackets, T-nuts, M5 screws | door frame + panel stand | Amazon: industrial / 3D-printer frame parts | $30-60 | Ahead |
| Arcade button kit (LED buttons + USB encoder), 10+ buttons | elevator panel | Amazon: arcade / game-controller parts | $20-35 | Ahead |
| Plastic cutting board or project box (faceplate), small project box (electronics) | elevator panel | Target: kitchen; Home Depot: electrical; or Amazon | $5-15 | Ahead |
| Rocker switch + wall plate + plastic single-gang box | light switch | Home Depot / Lowe's: electrical | $5-10 | Ahead |
| ESP32 dev boards (x3), AS5600 encoder, VL53L0X ToF sensors, jumper wires | sensors | Amazon / Micro Center: electronics | $25-45 | Ahead |
| Heavy-duty C-clamps (x6), spring clamps (x4) | arm mount, frames | Home Depot / Lowe's: hand tools / clamps | $30-60 | Ahead, or near venue if flying |
| Aluminium plate for the arm base, cut and drilled | arm mount | metal supplier / online cut-to-size | $30-80 | Ahead |
| ChArUco board print + foam board | calibration | print shop; Target / office store: foam board | $10-20 | Ahead |
| Folding work table (fallback) | arm mount | Home Depot / Lowe's: work benches | $60-150 | Near venue, only if the venue table fails |

## Packing plan

| Bag | Contents | Largest piece (approx.) |
|---|---|---|
| **Carry-on 1: electronics** | RealSense camera(s) + cables, Jetson(s) + power supplies, ESP32s and sensors (already wired, in labelled zip bags), arcade-button panel electronics in their project box, USB hub, spare USB-C/USB-A/micro cables, laptop(s), USB stick with weights/engines/backup video, multimeter | Jetson AGX dev kit, ~11x11x7 cm |
| **Checked suitcase 1: door and frames** | Cabinet door with hinges and pull already attached, extrusion lengths (≤ 60 cm), corner brackets, T-nuts and screws in a labelled parts box, elevator faceplate, arm base plate, calibration board (flat, between clothes/foam) | Cabinet door, ~60x40x2 cm, ~3-5 kg (weigh it) |
| **Checked suitcase 2 (or trunk): objects and tools** | Mugs and water bottle (wrapped), shopping bag, drawer pull, switch board, tool kit, clamps | C-clamps, ~2 kg together per pair |
| **Arm case** | xArm6, controller, gripper, cables (original packaging or a hard case) | the arm: check the weight and airline limits |
| **Bought near venue** | Grocery items, bananas, bin, drawer unit, (fallback) folding table | — |

**Pre-assemble at home:** hinges and pull on the door; the door frame as two
pre-bolted L-shaped halves; the elevator panel wired into its faceplate and
project box and tested; ESP32 firmware flashed and each sensor tested; the
switch mounted on its board.

**Assemble on site:** bolt the frame halves together and clamp them to the
table; hang the door (hinge screws); slide the faceplate into its stand and
clamp it; plug the sensors into the USB hub; place the objects.

**Tool kit:** metric hex-key set (M3-M6), Phillips and flat screwdrivers, a
small adjustable wrench, 6 heavy-duty C-clamps + 4 spring clamps, zip ties,
gaffer tape, double-sided foam tape, spare M5 screws and T-nuts, spare hinge
screws, USB hub (powered), spare cables, a power strip, a tape measure,
calipers, a marker and labels, a multimeter.

### 30-minute setup checklist

| Min | Step |
|---|---|
| 0-5 | Check the table for wobble; clamp the arm base plate (or start the fallback-table plan) |
| 5-10 | Bolt the door frame halves together, clamp them to the table edge, hang the door; check the catch holds and the encoder reads 0° |
| 10-15 | Clamp the elevator-panel stand; slide in the faceplate; press each button once and watch the `pressed:N` events |
| 15-20 | Place the drawer unit, switch board, shelf riser and bin; tape the ToF sensors; check each prop's JSON on the dashboard |
| 20-25 | Lay out the fetch objects; check RealSense depth on each (no holes on the bottle or mugs) |
| 25-30 | Hand-eye calibration check with the ChArUco board (residual in tolerance), then one dry-run fetch at reduced speed |
