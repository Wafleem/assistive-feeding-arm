#!/bin/bash
# Resumable weight downloads into weights/: curl -C - resumes, and a transfer slower than
# 10 kB/s for 30 s counts as failed and is retried (the venue Wi-Fi drops connections).
#   ./fetch_weights.sh            Depth Pro (the default model): weights/depth_pro.pt, 1.9 GB
#   ./fetch_weights.sh da3-metric DA3METRIC-LARGE: weights/DA3METRIC-LARGE/, 1.3 GB
cd "$(dirname "$0")" && mkdir -p weights && cd weights
get() {  # url file
  until curl -fsSL -C - --speed-limit 10000 --speed-time 30 --retry 1000 --retry-all-errors --retry-delay 2 -o "$2" "$1"; do sleep 2; done
}
case "${1:-depth-pro}" in
  depth-pro) get https://huggingface.co/apple/DepthPro/resolve/main/depth_pro.pt depth_pro.pt ;;
  da3-metric)
    mkdir -p DA3METRIC-LARGE
    for f in config.json model.safetensors; do
      get "https://huggingface.co/depth-anything/DA3METRIC-LARGE/resolve/main/$f" "DA3METRIC-LARGE/$f"
    done ;;
  *) echo "usage: $0 [depth-pro|da3-metric]" >&2; exit 2 ;;
esac
echo WEIGHTS_DONE
