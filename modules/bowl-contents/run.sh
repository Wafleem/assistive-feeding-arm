#!/usr/bin/env bash
# Module entrypoint for viam-server. Reuses the depth module's Python (GPU torch + transformers
# + the Viam SDK are already there): nothing is installed for this module. Override with
# BOWL_CONTENTS_PYTHON if that environment moves.
set -euo pipefail
cd "$(dirname "$0")"
PY="${BOWL_CONTENTS_PYTHON:-/home/ultraviolet/depth-anything-3/viam-env/bin/python}"
exec "$PY" -m bowl_contents "$@"
