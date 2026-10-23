# AGENTS.md: how to work in this repo (humans and AI coding agents)

This project is an **assisted-eating arm**: the venue's xArm6 driven through Viam,
with a Jetson AGX Orin for extra AI compute, feeding a **3D-printed, always-open
fake mouth** in the demo. It is **scaffolded**:
- every section has a plan README, typed interfaces, stubs that raise
  `NotImplementedError("TODO(<section>)")`, and spec tests marked `@todo`;
- the team reviews the plans before anything is implemented.

Read this file before changing anything. `CLAUDE.md` imports it, so Claude Code reads it too.

## 0. The prime directive: plans before code
- **Don't implement what wasn't asked for.** If you were asked to plan, write or
  update the plan (the section README, docstrings, `TODO(...)`, `@todo` spec
  tests). Leave the code as stubs.
- **Implement one TODO at a time,** and only after its plan has been reviewed. Never
  "while I'm here" another section's TODOs.
- **When a plan and the code disagree, the plan wins** until the team changes the
  plan. Update the README in the same PR as the code.

## 1. Repo map

```
AGENTS.md / CLAUDE.md     these rules
README.md                 concept, safety framing, section map, quick start
docs/                     PLAN, WORKSTREAMS, SIMULATION, COMPUTE, PROPS (docs/README.md indexes them)
sim/                      MuJoCo models and scenes (assets)
scripts/                  thin CLIs (render the scene, eval, recording, calibration)
tests/                    offline tests; @todo specs per section
src/assistive_arm/
  geometry.py  robot_iface/  vision/  motion/            SHARED CORE
  mouth_perception/  food_perception/  grasping/          SECTIONS 1-3
  feeding/  language/  orchestration/  sim/               SECTIONS 4-7
  pick_place/                                             SECTION 9 (SO-101 bench rig)
```
Section 8 (props) lives in `docs/PROPS.md` (and a future `props/` folder for CAD and firmware).

## 2. Modularity rules (enforced by `tests/test_architecture.py`)

### 2.1 Layers and allowed imports

```
            orchestration (6)            ← composes sections; may import any section package
                  │
  mouth(1) food(2) grasping(3) feeding(4) language(5) sim(7)   ← sections
                  │   may import: the shared core, and ONLY the contract modules of
                  │   other sections (types.py, schema.py, config.py, vocabulary.py, states.py)
                  ▼
            motion  →  robot_iface  →  vision  →  geometry      ← shared core
```

| Package | May import from `assistive_arm` |
|---|---|
| `geometry` | nothing |
| `vision` | `geometry` (+ `robot_iface` for type hints only, under `TYPE_CHECKING`) |
| `robot_iface` | `geometry`, `vision`, and, as a documented exception to remove, `motion.safety` in `viam.py` |
| `motion` | `geometry`, `robot_iface`, `vision` |
| a section (1-5, 7, 9) | the shared core + other sections' **contract modules** only |
| `orchestration` | everything |

- No section imports another section's implementation files (`kalman.py`, `pca.py`,
  `approach.py`, ...). If you need behaviour from another section, it arrives
  **injected** (a constructor argument typed with that section's Protocol), and
  `orchestration` wires it up.
- No import cycles. If you need one, a contract is in the wrong place: move the
  type into a `types.py`.

### 2.2 What every section package contains
| File | Role |
|---|---|
| `README.md` | the plan: goal, owner, interfaces, numbered plan, TODO checklist, done-when, open questions |
| `types.py` / `schema.py` / `config.py` / `states.py` | **contracts**: dataclasses, Protocols, enums, config. Frozen once reviewed (§3) |
| implementation modules | one concern per file, behind the contracts |
| `__init__.py` | the public API (`__all__`). Other code imports from here or from the contract modules |

### 2.3 Interfaces
- Data crosses boundaries as **dataclasses** (`MouthState`, `FoodItem`,
  `GraspCandidate`, `FeedIntent`, `DeliveryResult`, ...). Plain dicts don't.
- Swappable implementations sit behind a **`typing.Protocol`** (`MouthDetector`,
  `Detector`, `GraspModel`, `LLMBackend`, `VisualQA`, `SpeechToText`). New backends are
  new classes, not `if backend == ...` branches.
- **Configuration is a frozen dataclass** with documented defaults (`FeedingConfig`,
  `SafetyConfig`, `TrackerParams`, ...):
  - no magic numbers inside functions;
  - environment variables only for deployment choices and secrets (`ASSIST_*`, API keys).
- **Hardware independence:**
  - anything that moves or senses goes through `RobotInterface`;
  - only `robot_iface/viam.py` imports `viam`;
  - only `robot_iface/sim.py` imports `mujoco`;
  - the same section code runs on `FakeRobot`, `SimRobot` and `ViamRobot`.
- **Heavy or optional dependencies** (torch, transformers, TensorRT, mediapipe, opencv,
  anthropic, openai, viam-sdk):
  - import them **lazily** inside the wrapper class;
  - declare them as an optional extra in `pyproject.toml`;
  - the core install and the tests never need them.
- **Models are services:** each model (detector, segmenter, grasp model, LLM, VLM) is
  wrapped so it can run as a Viam vision/generic service. Moving it between the AGX
  Orin and the Viam machine must be a config change (docs/COMPUTE.md).
- **Time:** use `robot.now()` / `robot.wait()`, never `time.time()` / `time.sleep()` in
  section code, so sim and real behave the same.

### 2.4 Conventions (from `geometry.py`)
- **Units:** SI everywhere (m, rad, s, N). Only `viam_convert.py` knows millimetres and degrees.
- **Frames:**
  - every pose is in the **arm base frame**: z = 0 at the table top;
  - quaternions are (w, x, y, z);
  - the tool's +z points out of the gripper (the approach direction);
  - camera poses use OpenCV optical axes;
  - the face normal points **out** of the face.

## 3. Changing a contract
Contract modules (`types.py`, `schema.py`, `config.py`, `states.py`,
`robot_iface/base.py`, `vision/types.py`, `motion/safety.py`):
1. Open a **separate, small PR** that only changes the contract and its stubs/specs.
2. Tag every section that imports it (the table in `docs/WORKSTREAMS.md`).
3. Additive changes (a new optional field with a default, a new Protocol) are cheap.
   Renames and removals need agreement from every consumer.

## 4. Safety rules (never weaken these without the safety lead)
- The **hardware e-stop** and the xArm's own collision detection and reduced mode are
  the first layer. Software is the second.
- **Delivery modes** (`feeding/config.py`):
  - `printed_mouth` (the demo) releases the bite just inside the printed opening;
  - `person` (the **default**) stops 3 cm in front of the lips, and nothing ever enters
    a person's mouth;
  - never make `printed_mouth` the default, and never use it near a person.
- **Before a bite:**
  - confirm before every bite;
  - approach only when the mouth is ready (still and tracked; also open, for a real face);
  - speed caps near the face;
  - a force limit near the face.
- **Retreat immediately** on motion of the head or prop, "stop"/"wait", force, stale
  tracking, or a timeout.
- **Safety words** ("stop", "wait", "no", "I'm done") are handled by the rule layer
  **before** any model, and nothing can override them.
- Every safety number lives in a config dataclass; changing one needs the safety lead's
  review (see the PR checklist).
- A skill reports success only with evidence the world agrees with (gripper state,
  re-detection, the sim metrics). False success claims are bugs.

## 5. Implementing a TODO (the workflow)
1. Read the section README and the contract modules it uses. Run `make todo` to see the open TODOs.
2. If the plan is unclear or wrong, **fix the plan first** (README + docstrings) and get it reviewed.
3. Implement behind the existing interface. Keep the `TODO(...)` text accurate for whatever remains.
4. Remove the `@todo` mark from the spec tests you've satisfied; add tests for new behaviour
   (`FakeRobot` for logic, `SimRobot("feeding")` for physics/vision).
5. Tick the README checklist. `make test` must be green (no network, keys, GPU or arm).
6. Open a PR with the template filled in. Paste the sim eval table if behaviour changed.

## 6. Tests
- `make test`: everything offline. Spec tests for stubs are `@todo` (xfail on
  `NotImplementedError`); a spec that *passes* means its mark must be removed.
- `tests/test_architecture.py` enforces §2.1. If it fails, fix the import, not the test.
- Never write tests that need network access, API keys, GPUs or hardware.

## 7. Secrets, data, assets
- Keys only in environment variables or a local `.env` (git-ignored; copy `.env.example`).
  Never commit, print or log a key.
- Consumer subscriptions (ChatGPT, Claude.ai) are not API access. Never drive them programmatically.
- Model weights, engines, recorded frames, videos: `data/` or shared storage (git-ignored).
- Downloaded meshes: `sim/assets/` (git-ignored). Commit how to fetch them.

## 8. Git
- **Branches:** `s<N>/<topic>` (for example `s1/kalman-tracker`); `core/<topic>` for shared core; `docs/<topic>`.
- **Commits and PRs:** small, one concern each; imperative subject lines.
- **No `Co-Authored-By` trailers.**
- **Never force-push `main`.**
- **CI** (`.github/workflows/ci.yml`) must be green before merging.

## 9. Notes for AI coding agents
- Stay inside the section you were asked about. Touching a contract or another section
  means stopping and asking.
- Prefer editing the plan (README, docstrings, specs) over writing code unless you were
  explicitly asked to implement a specific TODO.
- Match the surrounding style: dataclasses, Protocols, short docstrings with a `PLAN`
  block, `TODO(<section>):` markers, SI units.
- Report what you changed, which TODOs remain, and which tests went from `@todo` to passing.
