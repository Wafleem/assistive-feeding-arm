# Shared core: `motion`: safety envelope and guarded move primitives

**Owner:** the servo & safety lead (section 4). Every change to a number here needs the safety lead's review.

| File | What | Status |
|---|---|---|
| `safety.py` | `SafetyConfig` (speeds, forces, reach, table height), keep-out zones (`Box`, `Sphere`), `SafetyEnvelope` (point/path checks, speed limit near the user, moving zones via `with_zones`) | works, tested |
| `primitives.py` | `SkillContext` (safety + `confirm` + `log`), `SkillResult`, `safe_move`, `guarded_linear`, `guarded_waypoints`, `safe_move_first` | works (from the first scaffold) |
| `smoothing.py` | `SmoothingConfig`, `min_jerk` / `point_to_point` (rest-to-rest with zero speed and acceleration at the ends), `resample` (uneven recordings onto an even grid), `gaussian_smooth` (zero-phase, ends held), `limit_speed` (stretch time under speed and acceleration caps), `ease_in_out` (playback starts and ends at rest): pure numpy, for streamed servo setpoints | works, tested (`tests/test_motion_smoothing.py`); caps need the safety lead's review |

## Rules
- Sections move the arm only through these primitives (or `RobotInterface` calls they
  wrap): checked targets and paths, speed limits, force limits.
- `SkillContext.confirm` must return False on timeout. No answer means no.
- The face-specific moving zone and the delivery modes live in `feeding/`, not here.

## TODO
- [ ] Review all default numbers against the measured venue layout and ISO/TS 15066 (safety lead).
