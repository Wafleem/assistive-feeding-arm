#!/usr/bin/env bash
# Download the model files perception needs into data/models/ (git-ignored).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/models
[ -f data/models/face_landmarker.task ] || curl -fsSL -o data/models/face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
ls -la data/models
