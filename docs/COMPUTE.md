# Compute: what runs where

Four kinds of computer, each doing what it is good at. Every model sits behind
a Viam vision service (or a generic service) module, and the computers run as
**parts of one Viam machine**, so moving a model from one box to another is a
config change, not a code change.

## The split

### Jetson AGX Orin 32 GB (if available): the on-wheelchair brain
* `viam-server` with the xArm6, gripper and RealSense components (librealsense
  with aligned depth).
* Detection + masks: **NanoOWL + NanoSAM** as TensorRT engines (NVIDIA
  describes them as real-time on Orin).
* Intent: a local quantised ~3B LLM (Ollama or llama.cpp) emitting the
  validated JSON intent.
* Speech: faster-whisper.
* Optional VLA: SmolVLA. One project's self-reported figures (not
  independently verified): about 0.26 s per inference on AGX Orin and about
  0.57 s on Orin Nano; pi0 about 0.9 s on AGX and about 2 s on Nano.
  OpenVLA 7B is too heavy for live use on the chair.

### Jetson Orin Nano 8 GB: safety and handover sidecar
* MediaPipe Hands / Pose: the open palm for the handover target, and a
  keep-out zone around the user's body that the motion plan respects.
* Wake word.
* Fallback NanoOWL if the AGX is busy or absent.

### Linux GPU box: heavy, accurate, and the fallback for everything
* Full Grounding DINO / OWLv2 + SAM 2 (the accuracy baseline for the week -2
  perception milestone).
* AnyGrasp / Contact-GraspNet (grasp v2).
* VLA fine-tuning (LeRobot: SmolVLA, pi0), dataset recording and storage.
* Isaac Sim if photoreal synthetic data is needed (it needs an NVIDIA GPU).

### This Mac (Apple M5, 24 GB): development and simulation
* MuJoCo natively, with the viewer (`make viewer`, under `mjpython`).
* MLX for the intent LLM (7-8B at 4-bit) and mlx-whisper.
* PyTorch on `mps` for OWLv2 / Grounding DINO / SAM 2 on recorded frames:
  slower than CUDA, and some ops fall back to the CPU (set
  `PYTORCH_ENABLE_MPS_FALLBACK=1`).
* The dashboard and the sim screen for the presentation.
* **Not here:** no CUDA or TensorRT, so no NanoOWL / NanoSAM engines, no
  AnyGrasp, no VLA training. Isaac Sim doesn't run on macOS, so MuJoCo is the
  simulator (see SIMULATION.md §2).

## Without the AGX Orin

* The LLM and speech move to the Mac (MLX) or the Linux box.
* The Orin Nano runs NanoOWL + NanoSAM next to the camera, plus its sidecar
  jobs; heavy models fall back to the Linux box over the network.
* The arm still needs `viam-server` close to it: on the Orin Nano, or on the
  Mac/Linux box tethered to the arm's controller.

## Stage-by-stage placement

| Stage | With AGX | Without AGX | Dev (Mac) |
|---|---|---|---|
| viam-server + arm/gripper/RealSense | AGX | Orin Nano or laptop | fake components or `SimRobot` |
| Speech (faster-whisper) | AGX | Mac (mlx-whisper) / Linux | Mac |
| Intent LLM | AGX (3B, quantised) | Mac (MLX 7-8B 4-bit) / Linux | Mac |
| Detection | AGX NanoOWL | Orin Nano NanoOWL; Linux GDINO/OWLv2 | Mac `mps` on recorded frames |
| Masks | AGX NanoSAM | Orin Nano NanoSAM; Linux SAM 2 | Mac `mps` |
| Grasp v1 (PCA) | AGX (numpy) | anywhere | Mac |
| Grasp v2 (AnyGrasp) | Linux | Linux | — |
| Hands / body keep-out | Orin Nano | Orin Nano | Mac (MediaPipe runs on CPU) |
| VLA inference (optional) | AGX SmolVLA | Linux | — |
| VLA training, datasets | Linux | Linux | — |
| Simulation, dashboard | Mac | Mac | Mac |

## Pre-event checklist

- [ ] Flash JetPack on each Jetson and build the TensorRT engines beforehand
      (10-30 min each); keep the build commands in the repo.
- [ ] librealsense with aligned depth working on the Jetson (verify the
      colour-depth alignment on a known object).
- [ ] `viam-server` with the xArm driver tested on the Jetson, end to end
      (`make viam-check`).
- [ ] Pin JetPack / CUDA / TensorRT / model versions; write them down here.
- [ ] A USB stick with all model weights, built engines, the recorded frames
      and the backup video.
- [ ] Power: plan 19 V for the Jetson if it is mounted on the chair; know the
      battery budget for the demo length.
- [ ] Network: the parts of the Viam machine must reach each other without
      venue Wi-Fi (bring a router or use wired links).

## Latency budget to measure (fill in on real hardware)

| Step | Target | AGX | Orin Nano | Linux | Mac |
|---|---|---|---|---|---|
| Speech → text (end of utterance) | < 0.5 s | | | | |
| Text → validated intent | < 1 s | | | | |
| Detection + mask, one frame | < 0.3 s | | | | |
| Mask → grasp pose | < 0.1 s | | | | |
| "stop" → arm halted | < 0.3 s | | | | |
