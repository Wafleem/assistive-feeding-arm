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
{"prop": "mouth", "jaw": 0.8, "t": 12.4}
{"prop": "mouth", "event": "bite_taken", "t": 15.1}
```

## The props

| Prop | Kind | What to use | Sensor / signal | Sim twin | Status |
|---|---|---|---|---|---|
| **Fake mouth** | A + small B | Mannequin head or wig/display head (beauty-supply, Amazon), **or** a printed face on foam board. Cut a mouth opening ~5×3 cm. Optional **servo jaw**: a hobby servo + a flat printed lower-lip plate inside the opening, so the "open/closed" ready gate can be tested. ArUco tag (~30 mm) on the forehead; matte paint; red lip ring for colour fallback. | ESP32 drives the servo on command or a scripted schedule and reports `jaw` and `bite_taken` (a light gate or switch inside the mouth) | `head_base` + `jaw` in feeding.xml; `mouth` site at the measured lips centre | sim done; prop TODO |
| **Moving base for the head** (stretch) | B | Head on a camera slider or a small pan servo base, so turn-away is repeatable | ESP32 scripts the motion | head_x/y/z/yaw actuators | TODO |
| **Plate / tray** | A | Matte, light-coloured plate (not shiny white porcelain if depth struggles), non-slip tray mat | — | `plate` | measure |
| **Foods** | A, near venue | Grapes (firm), strawberries (firm, hulled), melon or cheese cubes (~2 cm), small cracker cubes; yogurt cup for the scoop stretch | camera re-detection | `grape_*`, `strawberry_*`, `melon_cube_*`, `cracker_*` | measure on arrival |
| **Cup with straw** (stretch) | A | Lidded kids' cup or tumbler with a straw, 60-80 mm diameter (fits the 85 mm gripper) | — | `cup`, `cup_straw` | measure |
| **Utensil rest + spoon** (stretch) | A + small B | A large-handled adaptive spoon (or a normal spoon with foam grip tube) in a fixed, clamped rest so its pose is known | — | `utensil_rest`, `spoon_*` | measure |
| **Camera post** | B | Clamp-on light stand / monitor arm / 2020 extrusion post for the fixed RealSense | — | `scene` camera in the model | measure pose |
| **ChArUco board** | A | Printed at a print shop, on foam board | — | — | — |

## Shopping list (estimates only; verify in store or online, no SKUs)

| Item | Where (store: department) | Approx. price | When |
|---|---|---|---|
| Mannequin / display head | Amazon or beauty supply: wig heads | $15-40 | ahead (measure, modify) |
| Hobby servo, ESP32, micro switch or IR break-beam, jumper wires | Amazon / Micro Center: electronics | $20-35 | ahead |
| Matte plate, non-slip mat | Target / IKEA: kitchen & dining | $5-15 | ahead (measure) |
| Lidded cup with straw | Target: kitchen / kids' cups | $5-12 | ahead (measure) |
| Adaptive spoon or foam grip tubing | Amazon / pharmacy: daily living aids | $8-20 | ahead |
| Grapes, strawberries, melon, cheese, crackers, yogurt | any grocery | $15-25 | **near venue** (same kinds as measured) |
| Clamp-on light stand or monitor arm | Amazon / Home Depot: clamps, AV mounts | $20-40 | ahead |
| C-clamps, spring clamps | Home Depot: hand tools | $15-30 | ahead |
| Paper towels, wipes, food-safe gloves, small trash bags | Target: household | $10 | near venue |

## Packing and setup

* **Carry-on (electronics):** Jetsons + power, RealSense camera(s) + cables, ESP32 +
  servo (pre-wired into the head), USB hub, cables, USB stick.
* **Checked bag / trunk:** mannequin head (wrapped), camera post, clamps, plate, cup,
  spoon, ChArUco board (flat), tool kit (hex keys, screwdrivers, zip ties, gaffer
  tape, spare screws, multimeter).
* **Buy near venue:** the food, wipes, gloves.

**30-minute setup:** clamp the camera post (5) → place and clamp the head, plug in the
ESP32, check the jaw moves and reports (5) → plate, cut food, cup, spoon rest (5) →
calibration check with the ChArUco board (10) → one dry run with no food at reduced
speed (5).

**Food safety:** gloves for handling, keep food covered until the demo, discard after,
nothing is eaten by a person from the robot at the hackathon.
