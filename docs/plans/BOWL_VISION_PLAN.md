# Bowl vision: where each bowl is, and what is in it (plan)

Status: **plan + offline prototype**. Nothing here commanded the arm or changed the Viam config.
Evidence comes from SO-101 FK (`robot_iface/so101_kinematics.py`), a synthetic error budget
(`scripts/bowl_rim_fit_experimental.py --selftest`), and one real gripper-camera frame of Bowl 1
grabbed read-only from the dashboard stream during a teammate's scoop run on 2026-10-24.

Units below: mm and deg in the arm base frame, as in `poses/so101_bowl*.json`.

## 0. Recommendation (build this first)

1. **Label the bowls by content with CLIP (few-shot enrolled), not with a big VLM.** It is the piece
   that makes "feed me M&Ms" work, it works with today's calibrated bowls, and it is cheap:
   CLIP ViT-B/32 took 25 ms on the Mac CPU and put 98 % on "cereal" vs 0.5 % on "M&M's" for
   the real Bowl 1 frame. Enroll 5-10 crops per food once per session (at the survey pose), then
   nearest-neighbour, with zero-shot prompts as the fallback for new foods. Use a small VLM
   (SmolVLM-256M: 0.4 s, answered "Cereal.") only as a tie-breaker and for "empty?".
   Note: the cereal is **Froot Loops** (multi-coloured), so "colourful = M&Ms" colour heuristics
   are unsafe, and COCO YOLO calls the loops **"donut"**.
2. **Localise each bowl from a dedicated survey pose with a rim-circle fit, not the box centre.**
   Detect the bowl (YOLO "bowl" box from the Orin's `spoon-detector`, OWLv2/SAM as backups),
   take edge pixels inside the box, back-project them onto horizontal planes, and fit the circle
   whose radius matches the known rim (~75 mm) and whose plane is the roundest (this also
   measures the rim height). The current `BowlFinder` (box-centre ray onto z = 40 mm) is off by
   ~30 mm at a reachable survey pose; the rim fit is < 1 mm with a perfect camera model.
3. **Track bowls as deltas from the spoon-tip calibration.** The error that dominates is the
   camera mount (hand-measured (-21, -8, 90) mm, -90 deg): 1 deg / 3 mm of mount error gives
   6-13 mm centre error, outside the 1 cm budget. Taking one survey view of each spoon-calibrated
   bowl and applying only the **change** since then cancels it (1-2 mm, 95th pct; table 1.3). So:
   calibrate Bowl 2 with the spoon tip like Bowl 1, and let vision move those calibrations.
   A proper hand-eye calibration (ArUco on the table, 6 poses) is the follow-up that removes
   the need for reference views and also helps the face servo.
4. Keep the scoop exactly as it is: vision only rewrites the dict `_fit_scoop_to_bowl` already
   consumes (`centre_mm`, `radius_mm`, `rim_z_mm`, `floor_z_mm`).
5. **No VLA on the critical path** (section 5).

Not recommended for the centre: Depth Anything 3 (monocular metric depth, roughly 5-10 % of
range, i.e. 1-2 cm at 22 cm; fine as a sanity check), and box centres from any detector.

## 1. What was measured

### 1.1 Camera geometry (FK + the Viam camera frame, C270 fx = 772.5, HFOV 45 deg)
- The gripper camera looks along the spoon. During the recorded scoop it is 3-8 cm from the
  food (it sees only cereal, frame below), and at home / the face "look" pose it sees the
  ceiling or the room. **No existing pose gives a bowl-survey view**; one must be taught.
- FK search (200 k random joint sets, rim centre inside the inner image): the farthest view of
  Bowl 1 is about **22 cm** away, looking ~26 deg down, e.g. `[-51, -92, 25, 93, 0]` (Bowl 1),
  `[-24, -98, 29, 92, 0]` or `[-13, -85, 24, 91, 0]` (Bowl 2). These are home-like (lift folded
  back) with the wrist flexed ~93 deg: short, low-risk moves from home. Start from these when
  teaching; check them in the Viam scene first.
- At 22 cm a 15 cm rim is a ~540 x 210 px ellipse: it just fits across; the bowl's outside wall
  does not, and a bowl slid 2 cm toward the arm has its near rim cut off the bottom of the
  image. Both bowls in one view is not reachable (needs ~50 cm range). So: one survey view per
  bowl, optionally 2-3 views per bowl pooled (section 3.2), or a wider-FOV camera.
- The dashboard stream draws a status bar over rows ~455-480; use raw `webcam-2` frames
  (`ViamCameraRig.get_frames`) for perception, not `/stream.mjpg`.

### 1.2 Error budget for the rim-circle fit (synthetic, 300 trials, bowl moved up to +-3 cm)
`.venv/bin/python scripts/bowl_rim_fit_experimental.py --selftest`; centre error, median / 95th pct, mm:

| Error sources | rim z known | rim z 5 mm off | rim z fitted |
|---|---|---|---|
| 2 px edge noise only | 0.2 / 0.7 | 7.1 / 8.1 | 0.4 / 1.3 |
| + camera mount 1 deg / 3 mm | 5.9 / 12.7 | 7.0 / 17.2 | 3.5 / 7.9 |
| + camera mount 2 deg / 5 mm | 9.6 / 23.8 | 10.6 / 23.9 | 6.7 / 14.5 |
| mount 1 deg / 3 mm, reference-bowl bias removed | 0.6 / 1.7 | 1.0 / 2.3 | 1.0 / 3.7 |

Readings: at a 26 deg grazing view, a rim-height error of h moves the centre by h / tan 26 deg
(~2x h), so **either fit the rim height or know it to 2 mm**; the mount calibration dominates
everything else; the reference-bowl trick brings it inside the 1 cm budget with margin.

Image mode on a synthetic render with marble-like line clutter: with a bowl box, centre within
0.6 mm and rim z 94.2 (truth 94.0); **without a box, RANSAC locked onto clutter** (8-14 cm off).
The detector box (or mask) is required, and real marble veins plus the bowls' printed pattern
will be harder than the synthetic case.

### 1.3 Detectors and labellers on the real Bowl 1 close-up (MacBook M5, scratch venv)
| Model | Time | Result on the Froot Loops close-up |
|---|---|---|
| YOLOv8s COCO (torch CPU, 640) | 51 ms | 8 x "donut" (0.44-0.81), no "bowl" (too close to see one) |
| YOLO-World v2-s, vocab bowl/cereal/candy/froot loops/m&m | 43 ms | nothing >= 0.10 |
| CLIP ViT-B/32 OpenAI, 6 prompts (CPU) | 25 ms | cereal prompts 0.977, M&M's 0.005, colourful candy 0.011, empty 0.002 |
| SmolVLM-256M-Instruct, fp16 on MPS | ~0.4 s / answer | "Cereal." / "Cereal" / fill: "Half full." (unverified) |

No M&Ms frame exists yet, so the M&Ms side is untested: that is the first thing to record.

### 1.4 Facts in the repo that matter
- `poses/so101_bowl.json`: Bowl 1 centre (222, 229) by spoon tip vs (229.6, 216.5) with the scene's
  spoon geometry (14 mm apart), rim z 94, floor z 62. A 5-6 cm bowl on a z = 0 table would have
  its rim near 55-60 mm: **either the table is not at z = 0 in this base frame or the spoon/FK
  model is ~35 mm off in z.** Settle this before trusting any table-plane geometry (the rim-z
  fit in the prototype measures it from images; touch the bare table with the spoon tip too).
- `poses/so101_bowl2.json` (309.2, 94.5) and `SCOOP_PLAN.md` bowl-right (316.2, 173.8) disagree:
  bowl positions already drift between files, which is the case for measuring them.
- `food_perception/bowls.py` has `BowlFinder` + `BowlTracker` (box-centre ray onto
  `rim_height = 0.04`, radius from box width). Keep its tracker and detector plumbing; replace
  `locate()` with the rim fit and the default plane with the calibrated rim z.
- `.env` has `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` entries but they are empty, and neither SDK is
  in the venv; torch/ultralytics/open_clip/transformers are not in the venv either (only
  opencv, mediapipe, numpy, PIL, viam).

## 2. Options compared

### 2.1 Bowl localisation (centre to ~1 cm, rim radius, rim height)
| Method | Runs where | Latency | Centre accuracy (expected) | Effort | Verdict |
|---|---|---|---|---|---|
| YOLO "bowl" box centre -> ray onto a plane (current `bowls.py`) | Orin (`spoon-detector`, exists) | ~30 ms + Viam RTT | ~30 mm at the survey pose (box centre is not the rim centre; clipped boxes) | done | detection/ROI only |
| **YOLO box -> edges -> rim-circle fit on planes (rim z fitted)** | Mac or Orin, numpy + OpenCV | < 20 ms fit | 1-4 mm with the reference-bowl bias; 4-13 mm with a raw 1 deg mount | prototype exists | **build first** |
| OWLv2 / Grounding DINO box (open vocab) | Orin (or Mac) | ~0.2-1 s (estimate) | same as YOLO box; better recall for odd containers | wrappers exist in `vision/detectors.py` | backup detector |
| YOLO-World | Orin / Mac | ~40 ms | open-vocab boxes; found nothing on the close-up | wrapper exists | try at the survey pose |
| SAM 2-tiny / EfficientSAM / FastSAM mask from the box -> mask contour as the rim edge | Orin (GPU) | ~50-150 ms (estimate) | cleaner edges than Canny on marble; top contour = far rim | medium | add if edges are too cluttered |
| Depth Anything 3 (`webcam-2-depth`, ~5 Hz) median depth | Orin (exists) | 200 ms | ~1-2 cm at 22 cm (5-10 % of range) | done (`depth_first`) | sanity check only |
| ArUco / AprilTag on each bowl or under it | Mac, OpenCV (cv2 5.0 has aruco) | < 10 ms | ~2-3 mm, identity for free | low (print, tape) | fallback if the rim fit fails on real frames; use one tag for hand-eye calibration either way |

### 2.2 What is in each bowl
| Method | Runs where | Latency | Accuracy (expected) | Effort | Verdict |
|---|---|---|---|---|---|
| Colour histogram (HSV) of the bowl interior | anywhere | < 5 ms | poor: Froot Loops and M&Ms are both multi-coloured | low | no |
| **CLIP zero-shot (ViT-B/32 or MobileCLIP) on the inner-rim crop** | Mac CPU / Orin | 25 ms CPU (measured) | good for cereal vs candy (98 % on the real frame); weaker on "almost empty" | low | baseline |
| **CLIP few-shot: 5-10 enrolled crops per food, cosine kNN** | Mac / Orin | 25 ms + ~0 | best for the fixed menu; robust to lighting/angle seen at enrollment | low | **build first** |
| SmolVLM-256M / 500M | Mac MPS / Orin | 0.4 s (256M, measured) | right on the one frame; small VLMs hallucinate fill levels | low-medium | tie-breaker, "empty?" |
| Florence-2-base (0.23 B) caption / OD | Mac / Orin | ~0.3-0.6 s (estimate) | good captions; needs `trust_remote_code` | medium | optional |
| Moondream2 (1.9 B), PaliGemma-3B, Qwen2.5-VL-3B (4-bit) | Orin GPU / Mac (MLX) | ~1-3 s (estimate) | strong open-set, follows JSON prompts; 2-7 GB weights | medium | if open-set foods matter |
| Claude / GPT vision API on the bowl crop | cloud | ~1.5-4 s incl. network (estimate) | strongest open-set and "how full?" | low (needs a key in `.env` + SDK as an optional extra) | optional; crop to the bowl so no person leaves the rig |

How to prompt (any VLM): closed set + an escape, strict output, one crop per bowl:
`"This is the inside of a bowl seen from above. Which food is in it? Answer with exactly one
word from: cereal, mms, empty, other. Then, on a second line, one of: full, half, almost_empty,
empty."` Parse strictly; anything else -> `None` (`vlm.yes_no` style), and the caller asks the
user. For CLIP, average several prompts per class ("a photo of a bowl of Froot Loops cereal",
"a bowl of colourful cereal rings", "a bowl of M&M's chocolate candies", "an empty paper bowl").

How often: once per survey per bowl (so once per scoop at most), cached by bowl track id;
re-label only if the bowl was re-acquired (moved > 3 cm or lost) or the user asks.

## 3. Algorithms

### 3.1 Survey one bowl (single view)
1. Move to the bowl's taught survey pose (home -> survey is a short move); wait for the
   joints to settle; read joints; grab 2-3 raw frames; camera pose =
   `fk(q) @ T_cam` (or `ViamCameraRig.camera_pose`, same thing via the frame system).
2. Detect: YOLO `bowl` (and `cup`) via `ViamVisionDetector`; pick the box nearest the
   predicted projection of the last known centre. Gate: score >= 0.35, box intersects the
   predicted rim ellipse.
3. Rim fit (prototype `fit_rim` + RANSAC): Canny edges inside the box (+20 px) -> back-project
   onto z = rim prior -> RANSAC circle with radius 75 +- 15 mm, inlier band = 2 px -> refine
   by scanning z (roundest circle with r near the prior) -> centre (x, y), rim z, r, rms.
   Reject if |r - 75| > 15 mm, rms > 4 mm, fitted rim z more than 10 mm from the calibrated
   one, or fewer than ~90 deg of rim arc covered.
4. Debias: `centre = spoon_calibrated_centre + (fit_now - fit_at_calibration)`, both taken
   at the same survey pose (store `fit_at_calibration` when the bowl is spoon-calibrated).
5. Contents: warp the inner-rim ellipse (shrunk 15 %) to a square crop -> CLIP embedding ->
   kNN against enrolled crops (+ zero-shot) -> label, margin.

### 3.2 Several views, tracking, re-detection
- Pool rim points, not centres: back-projected edge points from 2-3 views (e.g. survey pose
  +-8 deg of pan) go into one circle fit, so a near rim clipped in one view is seen in another.
- Per bowl, keep a small filter like the mouth tracker but static: position-only Kalman with
  zero velocity (or a running weighted mean) with sigma_meas from the fit rms, Mahalanobis gate;
  `BowlTracker` from `bowls.py` keeps the ids.
- The camera cannot watch the bowls while it feeds (it faces the person), so "dynamic" here
  means **re-survey before every scoop** (2-3 s on the way from home), not continuous tracking.
  - shift < 5 mm: keep the old estimate (noise);
  - 5-30 mm: update the target, log it;
  - > 30 mm or not found: re-survey both bowls, re-label, and confirm with the user.
- Also flag a bump when the scoop's servo load spikes or the in-dip close-up (the frame
  above) shows no food in the centre of the image; re-survey next time.

## 4. Integration plan

### 4.1 Modules (section 2 + orchestration; respects `tests/test_architecture.py`)
```
src/assistive_arm/food_perception/
  rim_fit.py        NEW  numpy-only: backproject, circle_fit, fit_rim (z scan), ransac_rim  (from the prototype)
  contents.py       NEW  ContentClassifier Protocol; ClipContents (zero-shot + enrolled kNN, lazy torch/open_clip);
                         VLMContents(VisualQA) adapter; ContentLabel(name, score, margin, fill)
  bowl_survey.py    NEW  BowlSurvey(detector, classifier, params).survey(robot, bowl_id, q_survey) -> BowlEstimate;
                         pools views, applies the reference-bowl debias, rejects bad fits
  bowls.py          EDIT locate(): rim fit instead of box centre; BowlParams.rim_z from calibration, not 0.04
  vlm.py            EDIT SmolVLMQA (and optional AnthropicQA behind ASSIST_* env) implementing VisualQA
  vocabulary.py     CONTRACT (separate PR): MenuItem("cereal", ("froot loops", "loops", "cheerios"), "utensil_scoop", ...),
                         MenuItem("m&ms", ("m and ms", "m&m", "mms", "candy", "chocolate"), "utensil_scoop", ...)
src/assistive_arm/orchestration/face_servo_session.py
                    EDIT action "survey_bowls"; _bowl_for(item) -> the BOWL_CAL-style dict; scoop(item=...)
poses/so101_bowl_survey.json   NEW taught survey joints per bowl + reference fits (fit_at_calibration)
poses/so101_bowls_seen.json    NEW last accepted estimates + labels + time (a cache, like retry-cached targets)
pyproject.toml      EDIT optional extra "food-vision": open_clip_torch, torch (and transformers for SmolVLM)
tests/test_rim_fit.py, tests/test_bowl_survey.py  NEW offline: synthetic rims, fake detector, fake classifier
```
Models stay services: CLIP can later be a Viam vision service (`GetClassifications`) on the Orin,
next to `spoon-detector`; moving it is a config change (docs/COMPUTE.md). Start on the Mac,
where the dashboard already runs, to avoid touching the Jetson.

### 4.2 Data structures
```python
@dataclass(frozen=True)
class BowlEstimate:            # food_perception (new types.py: contract PR) or bowl_survey.py
    bowl_id: str               # "bowl1" / "bowl2": the spoon-calibrated identity it is anchored to
    centre: np.ndarray         # (2,) m, rim centre, base frame (debiased)
    rim_z: float               # m
    rim_radius: float          # m (outer rim, ~0.075)
    contents: str | None       # canonical menu name, None = unsure
    contents_score: float
    fill: str | None           # "full" | "half" | "almost_empty" | "empty" | None
    fit_rms: float; n_views: int; t: float
    source: str                # "vision" | "calibration" (fallback)
```
Orchestration converts it into the dict the scoop already uses, keeping the per-bowl
offsets from the spoon calibration (so only the centre moves):
`centre_mm = 1000 * centre`, `radius_mm = cal.radius_mm` (47: what the spoon tip can reach, not
the 75 mm rim), `rim_z_mm = cal.rim_z_mm + (rim_z_now - rim_z_at_cal)`, `floor_z_mm` likewise.
`grasping.scoop.BowlTarget(bottom, rim_radius, depth)` gets the same numbers for the planned scoop.

### 4.3 "Feed me M&Ms"
1. `language.rules.parse` -> `FeedIntent(FEED, item="m&ms")` (synonyms via `vocabulary.canonical`).
2. Orchestration: latest `BowlEstimate`s younger than N minutes? else `survey_bowls`.
3. Pick the bowl whose `contents == "m&ms"` with margin >= threshold; exactly one must match.
4. Re-survey that bowl (one view) on the way: update the centre, confirm the label.
5. Say/show "Scooping M&Ms from the right bowl. OK?" (existing confirm-before-bite rule), then the
   scoop with the updated dict, then the usual Feed flow.

### 4.4 Failure handling
| Situation | Action |
|---|---|
| Bowl not detected / fit rejected at its survey pose | try the +-8 deg pan views; still nothing -> use the last accepted estimate if < 5 min old and the user confirms, else the spoon calibration (`so101_bowl*.json`) and say so |
| Fit moved the bowl > 30 mm | re-survey both, ask "Did a bowl move?" before scooping |
| Label unsure (margin low) or two bowls get the same label | ask: "Is the right bowl the M&Ms?"; remember the answer for the session (bowl id -> contents) |
| Requested food in no bowl / bowl "empty" | say so; offer the other bowl |
| Vision unavailable (no torch, model error, Viam vision down) | fall back to the calibrated bowls and a fixed assignment (bowl1 = cereal, bowl2 = M&Ms) from config |
| Person in the frame | only bowl crops are labelled or stored; nothing leaves the Mac unless a cloud backend is explicitly enabled |
Safety words, confirm-before-bite and all motion checks stay as they are; perception can only
choose among targets the scoop already accepts (inside `_fit_scoop_to_bowl`'s checks).

## 5. VLA angle (short)
- **SmolVLA (450 M, LeRobot)** is the only one that fits naturally: it was built for SO-100/101,
  runs on the Orin AGX with action chunking, and fine-tunes from ~50 demos per task. It would
  replace the *scoop motion*, not the "which bowl" decision (language grounding of "the bowl with
  M&Ms" from ~50 demos is weak; perception does that reliably).
- **pi0 / pi0-FAST (3 B)**: better generalisation, but fine-tuning wants a large GPU and inference on
  the Orin is slow; **OpenVLA (7 B)**: too big/slow, 7-DOF EEF deltas; **Octo**: dated, weaker.
- What it would take: the joint logger records joints only (`data/demos/scoop-*.jsonl`); add
  synchronized gripper-camera frames (and ideally a fixed scene camera) at 30 Hz, record ~50
  hand-guided scoops per bowl with bowls moved around, fine-tune SmolVLA on a cloud GPU, run it
  through `grasping/vla.py` (clamped steps, keep-outs). A day of work plus data; **stretch only**.
  The recorded-scoop replay + vision-updated bowl centre gets the demo there sooner.

## 6. Build and test plan
1. **Offline now:** `scripts/bowl_rim_fit_experimental.py --selftest` (done). Port `fit_rim` and the
   RANSAC into `food_perception/rim_fit.py` with offline tests (synthetic rims, clutter, clipped arcs).
2. **Settle the heights:** spoon tip on the bare table next to each bowl -> table z in this base frame;
   compare with rim z 94 / floor 62.
3. **Teach survey poses** (with the team driving, Viam scene checked): start from the FK suggestions
   in 1.1; the whole rim ellipse in view, ~20-25 cm away. Save to `poses/so101_bowl_survey.json`.
4. **Calibrate Bowl 2 by spoon tip** like Bowl 1; at each bowl's calibration also store
   `fit_at_calibration` from its survey pose (the debias reference).
5. **Record a small dataset** (raw `webcam-2` frames + joints): per bowl ~20 survey frames with the bowl
   at taped positions (+-2 cm, measured with a ruler), both foods, plus empty/almost-empty bowls and
   the bowls swapped. Store under `data/` (git-ignored), bowl-only crops for anything shared.
6. **Evaluate localisation** on it: centre error vs ruler/spoon truth; target <= 10 mm at the 95th pct,
   rim z within 3 mm. If edges are too cluttered: SAM mask contour, then ArUco fallback.
7. **Evaluate contents:** CLIP zero-shot vs few-shot (leave-one-out) vs SmolVLM; target 100 % on
   cereal/M&Ms, >= 90 % on empty-vs-not; log latencies on the Mac (and the Orin if moved).
8. **Wire `survey_bowls`** into the session behind a dashboard button: dry-run first (frames from the
   current pose, no motion), then with motion when the team runs it.
9. **Wire "feed me M&Ms"** (language rules + `vocabulary` contract PR) and the failure table; offline
   tests with fake detector/classifier for every row.
10. **Hand-eye calibration** (ArUco on the table, corners touched with the spoon tip, 6+ poses,
    solvePnP + least squares on `T_cam`) to drop the reference-view dependence; recheck table 1.2.

## 7. Risks
- **Camera mount and intrinsics are uncalibrated** (fx 772.5 assumed): the main error source;
  mitigated by the reference-bowl debias, fixed properly by step 10 + a checkerboard intrinsic run.
- **Rim/table height inconsistency** (rim 94 mm in this frame): wrong planes cost ~2x the error in
  centre at 26 deg; fit the rim z, and resolve step 2.
- **Edge clutter**: marble veins and the bowls' printed pattern; the synthetic test shows the box gate
  is essential. SAM / ArUco are the fallbacks.
- **Field of view**: the C270 cannot see a whole bowl from reachable poses; a moved bowl may leave
  the survey view (then: extra pan views, or re-teach). A wider-FOV webcam would help a lot.
- **SO-101 repeatability** (backlash, gravity sag at the survey pose) breaks the debias if the arm does
  not return to the same pose; measure repeatability by surveying an unmoved bowl 10x.
- **Label confusion**: COCO sees Froot Loops as donuts; CLIP zero-shot untested on M&Ms; small-VLM
  fill levels are unreliable. Few-shot enrollment and asking the user cover it.
- **Latency**: a survey adds ~3-5 s per scoop (move + settle + Viam round trips); acceptable once per
  bite, but cache labels.
- **Privacy**: frames often contain the person; only bowl crops are kept or sent anywhere.

## Files
- `scripts/bowl_rim_fit_experimental.py` (new, experimental): rim-circle fit, synthetic error budget
  (`--selftest`), image mode (`--image f.jpg --joints ... --box ...`). numpy + OpenCV only; read-only.
- The CLIP / YOLO / SmolVLM numbers in 1.3 came from a throw-away venv (`uv venv` + `torch torchvision
  open_clip_torch ultralytics transformers accelerate num2words`, plus `git+https://github.com/ultralytics/CLIP.git`
  for YOLO-World; ~1.5 GB incl. weights), not the project venv.
