#!/bin/bash
# Run on the cloud GPU instance (Brev VM mode: Python + CUDA): installs Depth Pro in its own
# venv, fetches its weights, and starts depth_server.py on 127.0.0.1:8800 in the background.
# Safe to re-run.
#   bash setup_instance.sh            # logs: journalctl -u depth-server -f
set -euo pipefail
DEPTH_PRO_REF=9e65e4dbe9568d23c546fcec53302b10445e109e  # same pin as ../setup.sh
D=~/depth-cloud
HERE="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$D" && cp "$HERE/depth_server.py" "$D/"
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
# A systemd service (depth-server.service), so the server also comes back after the instance is
# stopped and started; Brev's VMs don't let the ubuntu user use cron.
pkill -f "depth_server.py --weights" || true
sudo install -m 644 "$HERE/depth-server.service" /etc/systemd/system/depth-server.service
sudo systemctl daemon-reload
sudo systemctl enable depth-server
sudo systemctl restart depth-server
for i in $(seq 1 90); do curl -s --max-time 2 http://127.0.0.1:8800/health && break; sleep 2; done; echo
systemctl is-active depth-server
