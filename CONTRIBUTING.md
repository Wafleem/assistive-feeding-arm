# Working on this repo

## First 15 minutes

```bash
git clone <this repo> && cd physical-ai-hackaton
make setup      # .venv with Python 3.12 + numpy, mujoco, pytest (uses uv if installed)
make test       # ~70 tests, ~15 s, no robot or GPU needed
make demo       # "press button 7 on the elevator" in MuJoCo, every step printed
make viewer     # watch it (macOS runs the viewer under mjpython automatically)
```

Then read, in order: `README.md` → `docs/PLAN.md` §0 (the flagship demo) →
`docs/WORKSTREAMS.md` (find your stream and its first task) → whichever of
`docs/SIMULATION.md`, `docs/COMPUTE.md`, `docs/PROPS.md` your stream uses.

Python 3.12 is the tested version. If `make setup` fails, install
[uv](https://docs.astral.sh/uv/) and run it again, or create a Python 3.12
venv yourself and `pip install -e ".[dev]"`.

## How we work

* **One workstream per branch:** `ws-<letter>/<short-topic>`, e.g.
  `ws-b/owlv2-recorded-frames`, `ws-c/fetch-skill`. Open a PR into `main`;
  the PR template has the checklist.
* **`make test` green on every PR.** CI runs the same tests on GitHub.
* **Frozen contracts** (`docs/WORKSTREAMS.md`): `intent/schema.py`,
  `robot_iface/base.py`, `perception/types.py`, `skills/base.py`,
  `planning/safety.py`, the world checks in `scripts/eval_sim.py`. Changing one
  is its own small PR, reviewed by every stream that uses it.
* **Safety numbers** (speeds, forces, keep-out boxes) change only with the
  safety lead's review.
* **Behaviour changes come with numbers:** paste the `make eval` table into
  the PR. "FALSE" (false success claims) and "user hit" must stay at 0.
* **Own your paths:** each stream owns the directories listed in
  WORKSTREAMS.md; tell the owner before editing theirs.
* **Track work as issues** with the "Workstream task" template, labelled by
  stream, with a "done when" that is a test or a number.

## Never commit

* Credentials: Viam API keys go in environment variables (`VIAM_ADDRESS`,
  `VIAM_API_KEY`, `VIAM_API_KEY_ID`) or a local `.env` (git-ignored).
* Model weights, TensorRT engines, recorded RealSense frames, videos: keep them
  in `data/` (git-ignored) or shared storage, and note where in the PR.
* Downloaded robot meshes: `sim/assets/` (git-ignored). Commit the script or
  instructions that fetch them instead.

## Layout cheat sheet

| I work on... | Start in |
|---|---|
| speech, intent, confirmations | `src/assistive_arm/intent/`, `tests/test_intent.py` |
| detection, masks, 3D | `src/assistive_arm/perception/` |
| skills, planner, safety | `src/assistive_arm/skills/`, `planning/`, `executor.py`, `tests/test_skills.py` |
| simulation, scenes, eval | `sim/`, `src/assistive_arm/robot_iface/sim.py`, `scripts/`, `tests/test_sim.py` |
| real arm through Viam | `src/assistive_arm/robot_iface/viam.py`, `viam_convert.py` |
| VLA policies | `src/assistive_arm/skills/vla.py` |
| props | `docs/PROPS.md`, `props/` (to create) |
