#!/bin/sh
# Local preview process only. This does not start or restart viam-server.
set -eu
VISION_ROOT=${ASSIST_VISION_ROOT:-"$HOME/assistive-arm-vision"}
# Viam owns the current camera. Direct USB is available when the device is free.
VISION_SOURCE=${ASSIST_VISION_SOURCE:-viam}
if [ "$VISION_SOURCE" = viam ]; then
  set -a
  . "$VISION_ROOT/vision.env"
  set +a
fi
export HF_HOME="$VISION_ROOT/data/huggingface"
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export RUST_LOG=error
exec "$VISION_ROOT/.venv/bin/python" -m assistive_arm.orchestration.vision_preview \
  --camera "${ASSIST_VISION_CAMERA:-arm-camera}" \
  --source "$VISION_SOURCE" \
  --spoon-weights "$VISION_ROOT/data/models/rf-detr-nano.pth" \
  --mouth-model "$VISION_ROOT/data/models/face_landmarker.task" \
  --device cuda \
  "$@"
