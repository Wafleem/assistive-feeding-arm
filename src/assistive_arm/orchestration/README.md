# 6. Orchestration (task state machine)

**Owner:** stream 6 (integration lead). **Runs on:** the venue Linux PC, next to viam-server.
**Goal:** run the meal: one bite cycle at a time, every state with a timeout and an abort path.

```
IDLE → SELECT_ITEM → ACQUIRE → VERIFY_GRASP → WAIT_READY → APPROACH → DELIVER → WAIT_BITE → RETREAT → IDLE/next
  any state --"stop"--> ABORT (halt; retreat if near the face)      "wait" → PAUSED → "resume"
```

## Interfaces
- `states.py`: `S`, `TRANSITIONS`, `TIMEOUTS` (data: drawable, testable)
- `FeedingMachine.handle(intent)`, `.step()`; transition log with reasons for the dashboard
- Consumes: language (`FeedIntent`), food_perception, grasping, mouth_perception, feeding

## Plan
1. Complete the transition table, including every failure edge.
2. Implement `step()` with injected components (fakes in tests, sim, then Viam).
3. Dashboard: camera feed with overlays (items, mouth, state), transition log, big STOP.
4. Viam: run as a Viam module/process on the Linux PC; models as vision services on the Jetsons.

## TODO checklist
- [ ] Transition table complete + `tests/test_orchestration.py`
- [ ] `FeedingMachine` with fake components; then end-to-end in the MuJoCo feeding scene
- [ ] Dashboard (camera, overlays, state, STOP)
- [ ] Viam module packaging

## Done when
A full "feed me a grape" cycle runs in sim and on hardware; "stop" from any state halts within 300 ms.

## Implemented separately: observation-only vision preview

`vision_preview.py` composes a camera-only Viam or local USB reader, RF-DETR spoon detector
and MediaPipe real-face lip landmarker. It does not construct `FeedingMachine`
or issue robot actions. `--source usb` selects the camera directly attached to
the Jetson without Viam credentials; `--source viam` preserves remote acquisition.
One worker acquires and processes one frame at a time,
then replaces an in-memory JPEG/status snapshot; frames are not recorded.

The HTTP server binds only to `127.0.0.1` and exposes GET `/`, `/status.json` and
`/frame.jpg?revision=N`. A failed camera/model call clears the previous image.
Snapshots expire by age since receipt (default 3 seconds), and the browser has
its own expiry timer for network/server failures. This is not a claim about
exposure age: source capture time and its clock are reported separately.

Shutdown stops the preview worker, closes MediaPipe and disconnects its camera
client. It sends no arm/gripper/stop commands. See `scripts/README.md` for CLI use.
