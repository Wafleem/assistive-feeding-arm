# Props: the fake mouth, the meal, and their sim twins

Same rules as before: every prop is either **(A) bought ready-made** (Target,
Walmart, Home Depot/Lowe's, IKEA, Amazon), used as-is or with small changes,
or **(B) flat-pack**: fits in a suitcase, has no glue joints, clamps to the
venue table and goes together in ≤ 30 minutes with hand tools. Anything custom
(laser-cut or 3D-printed) stays small and flat.

For bought items, the **sim twin comes from measuring the item** (calipers,
tape measure, kitchen scale) and updating `sim/scenes/feeding.xml`. Props
**report their own state** where it matters (JSON lines over USB serial from an
ESP32), in the same form as the sim's events, so the dashboard and the eval
compare like with like:

```json
{"prop": "mouth", "event": "bite_in_cup", "t": 15.1}
```

(The printed mouth needs no electronics; the catch-cup sensor is optional.)

## The props

| Prop | Kind | What to use | Sensor / signal | Sim twin | Status |
|---|---|---|---|---|---|
| **Fake mouth** | B (3D-printed, flat parts) | A **3D-printed face plate** (PLA/PETG, matte, light colour) with an **always-open opening** ~50 × 40 mm, printed in parts that fit the print bed and bolt together. Behind it, a **removable catch cup** (printed, or a small bought plastic cup) that bites drop into; **dump it after every use**. An ArUco tag (~30-40 mm, paper in a recess) at a known offset from the opening, and a contrasting lip-colour ring around the opening. Clamps to the table at mouth height (~35-40 cm above the table) on a printed or 2020-extrusion stand. | Optional: a ToF or break-beam sensor in the catch cup on an ESP32 reports `bite_in_cup`; otherwise the camera/gripper check is enough | `sim/scenes/feeding.xml` still has a jaw head: **TODO** replace it with this face plate + catch cup from the same CAD (sim/README.md) | to design + print |
| **Moving the mouth** (for retreat tests) | — | Slide or turn the printed mouth's stand by hand during an approach; optionally a lazy-susan or slider base for repeatable moves | — | head base x/y/z + yaw actuators | — |
| **Plate / tray** | A | Matte, light-coloured plate (not shiny white porcelain if depth struggles), non-slip tray mat | — | `plate` | measure |
| **Foods** | A, near venue | Grapes (firm), strawberries (firm, hulled), melon or cheese cubes (~2 cm), small cracker cubes; yogurt cup for the scoop stretch | camera re-detection | `grape_*`, `strawberry_*`, `melon_cube_*`, `cracker_*` | measure on arrival |
| **Cup with straw** (stretch) | A | Lidded kids' cup or tumbler with a straw, 60-80 mm diameter (fits the 85 mm gripper) | — | `cup`, `cup_straw` | measure |
| **Utensil rest + spoon** (stretch) | A + small B | A large-handled adaptive spoon (or a normal spoon with foam grip tube) in a fixed, clamped rest so its pose is known | — | `utensil_rest`, `spoon_*` | measure |
| **Camera post** | B | Clamp-on light stand / monitor arm / 2020 extrusion post for the fixed RealSense | — | `scene` camera in the model | measure pose |
| **ChArUco board** | A | Printed at a print shop, on foam board | — | — | — |

## Shopping list (estimates only; verify in store or online, no SKUs)

| Item | Where (store: department) | Approx. price | When |
|---|---|---|---|
| PLA/PETG filament (matte, light colour), M3/M4 screws for the face-plate parts | Amazon / Micro Center: 3D printing | $20-30 | ahead (print and test-fit) |
| Small plastic cups for the catch cup (spares) | Target: party supplies | $3-5 | either |
| (optional) ESP32 + ToF or IR break-beam for the catch cup | Amazon / Micro Center: electronics | $10-20 | ahead |
| Matte plate, non-slip mat | Target / IKEA: kitchen & dining | $5-15 | ahead (measure) |
| Lidded cup with straw | Target: kitchen / kids' cups | $5-12 | ahead (measure) |
| Adaptive spoon or foam grip tubing | Amazon / pharmacy: daily living aids | $8-20 | ahead |
| Grapes, strawberries, melon, cheese, crackers, yogurt | any grocery | $15-25 | **near venue** (same kinds as measured) |
| Clamp-on light stand or monitor arm | Amazon / Home Depot: clamps, AV mounts | $20-40 | ahead |
| C-clamps, spring clamps | Home Depot: hand tools | $15-30 | ahead |
| Paper towels, wipes, food-safe gloves, small trash bags | Target: household | $10 | near venue |

## Packing and setup

* **Carry-on (electronics):** the AGX Orin + power, RealSense camera(s) + cables,
  network switch + cables, USB hub, USB stick (optional catch-cup ESP32).
* **Checked bag / trunk:** the printed mouth (face-plate parts flat, catch cups, stand), camera post, clamps, plate, cup,
  spoon, ChArUco board (flat), tool kit (hex keys, screwdrivers, zip ties, gaffer
  tape, spare screws, multimeter).
* **Buy near venue:** the food, wipes, gloves.

**30-minute setup:** clamp the camera post (5) → bolt the face plate together, clamp the
printed mouth at mouth height, empty catch cup in place, check the tag is detected (5) →
plate, cut food, cup, spoon rest (5) →
calibration check with the ChArUco board (10) → one dry run with no food at reduced
speed (5).

**Food safety and cleanup:** gloves for handling, keep food covered until the demo, **empty
and wipe the catch cup after every bite**, discard the food; nothing is eaten by a person
from the robot at the hackathon.
