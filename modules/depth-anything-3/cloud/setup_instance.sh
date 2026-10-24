#!/bin/bash
# Run on the cloud GPU instance (Brev VM mode: Python + CUDA): installs Depth Pro in its own
# venv, fetches its weights, and starts depth_server.py on 127.0.0.1:8800 in the background.
# Safe to re-run.
#   bash setup_instance.sh            # then: tail -f ~/depth-cloud/server.log
set -euo pipefail
DEPTH_PRO_REF=9e65e4dbe9568d23c546fcec53302b10445e109e  # same pin as ../setup.sh
D=~/depth-cloud
mkdir -p "$D" && cp "$(dirname "$0")/depth_server.py" "$D/"
cd "$D"
# A venv: Ubuntu's own Python refuses system-wide pip installs (PEP 668).
[ -x venv/bin/python ] || python3 -m venv venv || { sudo apt-get install -y -q python3-venv && python3 -m venv venv; }
PY=venv/bin/python
"$PY" -m pip install -q --upgrade pip
"$PY" -c "import torch" 2>/dev/null || "$PY" -m pip install -q torch torchvision
"$PY" -c "import torch; assert torch.cuda.is_available(), 'no CUDA GPU here'; print('torch', torch.__version__, torch.cuda.get_device_name(0))"
"$PY" -m pip install -q timm pillow_heif "numpy<2" pillow
"$PY" -m pip install -q --no-deps "https://github.com/apple/ml-depth-pro/archive/${DEPTH_PRO_REF}.zip"
[ -s depth_pro.pt ] || until curl -fsSL -C - --retry 100 --retry-all-errors -o depth_pro.pt \
  https://huggingface.co/apple/DepthPro/resolve/main/depth_pro.pt; do sleep 2; done
pkill -f "depth_server.py --weights" || true
nohup "$PY" depth_server.py --weights depth_pro.pt --port 8800 > server.log 2>&1 &
for i in $(seq 1 90); do grep -q "depth server on" server.log && break; sleep 2; done
tail -2 server.log
curl -s http://127.0.0.1:8800/health; echo
