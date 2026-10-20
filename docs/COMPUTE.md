# Compute: what runs where

## Hardware (assumed; confirm with the venue)

| Machine | From | Role |
|---|---|---|
| UFACTORY xArm6 + parallel gripper | venue | the arm, through Viam |
| The venue's Viam machine (a normal Linux PC) | venue | `viam-server`, cameras, motion service, orchestration, feeding motion & safety, mouth tracking, dashboard |
| Jetson AGX Orin 32 GB | us | **additional AI compute** the Viam machine may not have: food detection + segmentation, grasp model, local LLM, speech, optional VLM/VLA |
| RealSense-class RGB-D camera(s) | **assumption: confirm** | one fixed (sees the plate and the printed mouth), ideally one on the wrist |
| Mac (M-series) | us | development and MuJoCo simulation only |

The AGX joins as a **remote part of the same Viam machine**. Each model is wrapped as a
Viam vision or generic service, so moving a model between the AGX and the Viam machine
is a config change, not a code change.

## Placement (to review)

| Stage | Where | Why |
|---|---|---|
| Cameras, `viam-server`, motion service | Viam machine | next to the arm; that's what the venue provides |
| Mouth tracking: ArUco on the printed mouth + Kalman tracker + gates | **Viam machine** | light (CPU), and it's the safety-critical loop, so keep it next to the camera and the arm and off the network hop. Move it to the AGX only if the PC is too slow. |
| Food detection (NanoOWL; OWLv2 / Grounding DINO fallback) + masks (NanoSAM; SAM 2) | **AGX Orin** | GPU, TensorRT, 32 GB |
| Grasp model (AnyGrasp / Contact-GraspNet) | **AGX Orin** | CUDA |
| Intent LLM: a local small quantised model (Ollama / llama.cpp) + faster-whisper | **AGX Orin** (default backend) | private and offline; optional Claude / OpenAI **API** backends as fallbacks (API keys, not subscriptions; see `language/README.md`) |
| VLM attribute questions (small Qwen-VL / SmolVLM class) | AGX Orin | only at decision points |
| VLA (stretch: SmolVLA / pi0) | AGX inference; training elsewhere | treat published latency figures as self-reported until measured |
| Orchestration, feeding servo loop, dashboard | Viam machine | next to the arm |
| Simulation, development | Mac | MuJoCo native; no CUDA/TensorRT there |

**Network:** the AGX and the Viam machine on our own switch with a cable (no venue Wi-Fi).
Frames for the food models cross the network only at decision points (selecting an
item), never inside the servo loop.

**If the AGX is unavailable:** run OWLv2 / Grounding DINO on the Viam machine's GPU
(if it has one) or on the CPU at low rate, and run the LLM on the Mac or through a cloud API.
The demo still works, just slower.

**If the venue also provides a Jetson Orin Nano or a GPU:** a later option is to move
mouth tracking there; it isn't needed for the printed mouth.

## Pre-event checklist

- [ ] JetPack flashed on the AGX; JetPack/CUDA/TensorRT/model versions written down here
- [ ] TensorRT engines built in advance (NanoOWL, NanoSAM; 10-30 min each)
- [ ] The AGX added as a remote part of a test Viam machine; each model reachable as a Viam service
- [ ] Local LLM + faster-whisper on the AGX, offline. If using cloud fallbacks, API keys
      go in `.env` on the machine that runs the router (never in git).
- [ ] ArUco tracking of the printed mouth running at ≥ 30 fps on a laptop-class CPU (stand-in for the venue PC)
- [ ] A USB stick with weights, engines, recorded frames and the backup video
- [ ] Our own switch and cables; power for the AGX

## Latency budget (fill in on hardware)

| Step | Target | Measured |
|---|---|---|
| Camera capture → MouthState (Viam machine) | < 50 ms | |
| MouthState → arm command takes effect | < 80 ms | |
| Total used by `TrackerParams.latency` | ~100-130 ms | |
| Food detection + mask, one frame (AGX, incl. network) | < 300 ms | |
| Speech end → validated intent (AGX) | < 1.5 s | |
| "stop" spoken → arm halted | < 300 ms | |
