# Tests

`make test` runs everything offline: no arm, no network, no API keys.

| File | Covers | State |
|---|---|---|
| `test_geometry.py` | poses, rotations, Viam orientation vectors, deprojection | passing |
| `test_safety.py` | the static safety envelope | passing |
| `test_sim.py` | the feeding scene and SimRobot | passing + `@todo` specs |
| `test_architecture.py` | the import rules from AGENTS.md | passing |
| `test_mouth_perception.py`, `test_food_perception.py`, `test_grasping.py`, `test_feeding.py`, `test_language.py`, `test_orchestration.py` | one file per section | mostly `@todo` specs |

## `@todo` specs
`conftest.py` defines `todo = xfail(raises=NotImplementedError)`. A spec test describes
what a stub must do. While the stub raises `NotImplementedError`, it shows as `x`
(expected failure). When you implement the stub, **remove the `@todo` mark**: from then
on the test must pass, and CI enforces it.

## Writing tests
- Logic: use `FakeRobot` (fast, no physics) and plain numbers.
- Physics and vision: use `SimRobot("feeding")`. Camera tests use the `robot` fixture,
  which skips cleanly where there's no OpenGL.
- Never depend on network access, API keys, GPUs or hardware. Those belong in scripts or
  bench checklists, not tests.
