#!/bin/bash
# Resumable download of DA3METRIC-LARGE: curl -C - resumes, and a transfer slower than
# 10 kB/s for 30 s counts as failed and is retried.
mkdir -p "$(dirname "$0")/weights/DA3METRIC-LARGE" && cd "$(dirname "$0")/weights/DA3METRIC-LARGE"
base=https://huggingface.co/depth-anything/DA3METRIC-LARGE/resolve/main
for f in config.json model.safetensors; do
  until curl -fsSL -C - --speed-limit 10000 --speed-time 30 --retry 1000 --retry-all-errors --retry-delay 2 -o "$f" "$base/$f"; do sleep 2; done
done
echo WEIGHTS_DONE
