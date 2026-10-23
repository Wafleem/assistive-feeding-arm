#!/usr/bin/env bash
# Builds viam-env/ next to this script. viam-server runs it once as the module's first_run
# (meta.json); run.sh also calls it if the env is missing. Safe to re-run.
#
# Environment (set in the module's "env" in the machine config):
#   DA3_TORCH_INDEX_URL  pip index for torch/torchvision. PyPI's aarch64 CUDA wheels have no
#                        kernels for the Jetson Orin (sm_87), so they run on the CPU there.
#                        Point this at NVIDIA's Jetson wheels for your JetPack to use the GPU,
#                        or at https://download.pytorch.org/whl/cpu for a smaller CPU-only install.
#   DA3_PYTHON           interpreter to build the venv from (default python3, needs >= 3.10)
#   DA3_BASE_VENV        start from a copy of this existing venv and keep its torch (skips a
#                        multi-GB download). On the AGX Orin, PyPI's cu13x torch runs on the GPU:
#                        its sm_80 kernels are binary-compatible with the Orin's sm_87, although
#                        torch warns that 8.7 is not an officially supported target.
set -euo pipefail
cd "$(dirname "$0")"

# Depth Anything 3, pinned. Installed without its dependencies: those are for exporting
# GLB/COLMAP/videos (open3d, pycolmap, xformers, ...); requirements.txt has what inference needs.
DA3_REF=3d835ec1a5802d64a8b8b15f817a1ab54809bfe4
VENV=viam-env

[ -f "$VENV/.installed" ] && exit 0
rm -rf "$VENV"
# Always `python -m pip`: a copied venv's bin/pip still points at the original venv.
PIP=("$VENV/bin/python" -m pip)
if [ -n "${DA3_BASE_VENV:-}" ]; then
  cp -a "$DA3_BASE_VENV" "$VENV"
else
  "${DA3_PYTHON:-python3}" -m venv "$VENV"
  "${PIP[@]}" install --upgrade pip
  if [ -n "${DA3_TORCH_INDEX_URL:-}" ]; then
    "${PIP[@]}" install torch torchvision --index-url "$DA3_TORCH_INDEX_URL"
  else
    "${PIP[@]}" install torch torchvision
  fi
fi
"${PIP[@]}" install -r requirements.txt
# hatch-vcs can't read a version from a source archive; tell it one.
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 "${PIP[@]}" install --no-deps \
  "https://github.com/ByteDance-Seed/Depth-Anything-3/archive/${DA3_REF}.zip"
"$VENV/bin/python" -c "import torch, depth_anything_3; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
touch "$VENV/.installed"
