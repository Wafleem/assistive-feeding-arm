# Compute: what runs where

## Hardware (assumed; confirm with the venue)

| Machine | From | Role |
|---|---|---|
| UFACTORY xArm6 + parallel gripper | venue | the arm, through Viam |
| Linux PC | venue | `viam-server`, motion, orchestration, feeding motion & safety, dashboard |
| Jetson AGX Orin 32 GB | us | model host: food detection + segmentation, grasp model, local LLM, speech, optional VLM/VLA |
| Jetson Orin Nano 8 GB dev kit | us | dedicated face/mouth tracker + safety monitor at high frame rate, next to the fixed camera |
| RealSense-class RGB-D camera(s) | **assumption: confirm** | one fixed (face + plate), ideally one on the wrist |
| Mac (M-series) | us | development and MuJoCo simulation only |

Every model is wrapped as a **Viam vision or generic service**, and the
computers are parts of one Viam machine. Moving a model between the AGX, the
Nano and the PC is then a config change.

## Placement

| Stage | Where | Why |
|---|---|---|
| Mouth detection + Kalman tracker + ready gate + head-motion monitor | **Orin Nano** (camera plugged in) | short, predictable latency; it's the safety-critical loop; MediaPipe/ArUco are light |
| Food detection (NanoOWL; OWLv2/GDINO fallback) + masks (NanoSAM; SAM 2) | **AGX Orin** | TensorRT engines, 32 GB |
| Grasp model (AnyGrasp / Contact-GraspNet) | AGX or Linux PC (if it has a GPU) | CUDA |
| Local LLM (small quantised, Ollama / llama.cpp) + faster-whisper | **AGX Orin** | runs offline, no venue-network dependency |
| VLM attribute questions (small Qwen-VL / SmolVLM class) | AGX Orin | only at decision points |
| VLA (stretch: SmolVLA / pi0) | AGX inference; training elsewhere | treat any published latency figures as self-reported until measured |
| viam-server, motion service, orchestration, feeding servo loop, dashboard | **venue Linux PC** | next to the arm |
| Simulation, development | Mac | MuJoCo native; no CUDA/TensorRT there |

**If the Orin Nano is unavailable:** run the mouth tracker on the AGX with its
own camera process at top priority. **If the AGX is unavailable:** the Nano
runs NanoOWL + NanoSAM and the tracker at a lower rate, and the LLM runs on the
Mac or the Linux PC.

## Pre-event checklist

- [ ] JetPack flashed on both Jetsons; versions of JetPack/CUDA/TensorRT/models written down here
- [ ] TensorRT engines built in advance (NanoOWL, NanoSAM; 10-30 min each)
- [ ] librealsense with aligned depth on the Nano (and AGX if it has a camera)
- [ ] MediaPipe Face Landmarker and the ArUco detector running on the Nano at ≥ 30 fps
- [ ] Local LLM + faster-whisper on the AGX, offline
- [ ] Each model reachable as a Viam service from a test machine config
- [ ] A USB stick with weights, engines, recorded frames, the backup video
- [ ] Our own network switch/router and cables (no venue Wi-Fi dependency)
- [ ] Power supplies for both Jetsons

## Latency budget (fill in on hardware)

| Step | Target | Measured |
|---|---|---|
| Camera capture → MouthState (Nano) | < 50 ms | |
| MouthState → arm command takes effect (PC + controller) | < 80 ms | |
| Total used by `TrackerParams.latency` | ~100-130 ms | |
| Food detection + mask, one frame (AGX) | < 300 ms | |
| Speech end → validated intent | < 1.5 s | |
| "stop" spoken → arm halted | < 300 ms | |
