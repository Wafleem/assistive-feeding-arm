# Working on this repo

## First 15 minutes

```bash
git clone git@github.com:Wafleem/assistive-feeding-arm.git && cd assistive-feeding-arm
make setup      # .venv with Python 3.12 + numpy, mujoco, pytest (uses uv if installed)
make test       # passing tests + xfail specs for sections not implemented yet
make render     # what the cameras see in the MuJoCo feeding scene (out/)
make todo       # every TODO, by section
```

Then read **`AGENTS.md`** (the rules) → `README.md` → `docs/PLAN.md` →
`docs/WORKSTREAMS.md` → your section's `src/assistive_arm/<section>/README.md`.

## Implementing a section

1. Pick a `TODO(<section>)` from `make todo` (or your section README's checklist).
2. Implement it behind the existing interface (`types.py`); don't change a frozen
   contract without a separate small PR (see WORKSTREAMS.md).
3. Remove the `@todo` mark from the matching spec test in `tests/`; it must now pass.
4. Add tests for anything new; use `FakeRobot` for logic and `SimRobot("feeding")` for physics/vision.

## How we work

* **One section per branch:** `s<N>/<short-topic>`, e.g.
  `s1/kalman-tracker`, `s3/pca-grasp`. Open a PR into `main`;
  the PR template has the checklist.
* **`make test` green on every PR.** CI runs the same tests on GitHub.
* **Frozen contracts** (`docs/WORKSTREAMS.md`): `intent/schema.py`,
  every section's `types.py`/`schema.py`/`states.py`, `robot_iface/base.py`,
  and the safety numbers in `motion/safety.py` and `feeding/config.py`. Changing one
  is its own small PR, reviewed by every stream that uses it.
* **Safety numbers** (speeds, forces, keep-out boxes) change only with the
  safety lead's review.
* **Behaviour changes come with numbers:** once `scripts/eval_feeding.py` exists,
  paste its table into the PR. Head contacts and false "delivered" claims must stay at 0.
* **Own your paths:** each stream owns the directories listed in
  WORKSTREAMS.md; tell the owner before editing theirs.
* **Track work as issues** with the "Workstream task" template, labelled by
  stream, with a "done when" that is a test or a number.

## Never commit

* Credentials: Viam and LLM API keys go in environment variables or a local `.env`
  (git-ignored; copy `.env.example`). Never commit, print or log a key.
* Model weights, TensorRT engines, recorded RealSense frames, videos: keep them
  in `data/` (git-ignored) or shared storage, and note where in the PR.
* Downloaded robot meshes: `sim/assets/` (git-ignored). Commit the script or
  instructions that fetch them instead.

## Layout cheat sheet

| I work on... | Start in |
|---|---|
| 1 mouth tracking | `src/assistive_arm/mouth_perception/`, `tests/test_mouth_perception.py` |
| 2 food detection | `src/assistive_arm/food_perception/`, `src/assistive_arm/vision/` |
| 3 grasping | `src/assistive_arm/grasping/`, `tests/test_grasping.py` |
| 4 feeding motion & safety | `src/assistive_arm/feeding/`, `src/assistive_arm/motion/` |
| 5 language | `src/assistive_arm/language/`, `tests/test_language.py` |
| 6 orchestration | `src/assistive_arm/orchestration/` |
| 7 simulation | `sim/`, `src/assistive_arm/sim/`, `src/assistive_arm/robot_iface/sim.py`, `scripts/` |
| 8 props | `docs/PROPS.md`, `props/` (to create) |
| the real arm through Viam | `src/assistive_arm/robot_iface/viam.py`, `viam_convert.py` |
