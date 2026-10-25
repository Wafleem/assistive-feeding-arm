#!/usr/bin/env bash
# Depth + mouth view without the arm (dashboard/depth_view.py), on this laptop.
# Keeps an SSH tunnel to the Jetson's viam-server up (restarted if it drops) and serves the
# page on http://localhost:8091/. Reach the Jetson over its USB link (default) or a shared
# network:  JETSON=ultraviolet@192.168.1.252 dashboard/run_depth_view.sh
# Needs .env (VIAM_API_KEY / VIAM_API_KEY_ID), an SSH key the Jetson accepts, and
# data/models/face_landmarker.task (copy it from the Orin's data/models).
set -euo pipefail
cd "$(dirname "$0")/.."
JETSON="${JETSON:-ultraviolet@192.168.55.1}"
KEY="${JETSON_KEY:-$HOME/.ssh/id_ed25519}"
HOSTKEY_ALIAS="${JETSON_HOSTKEY_ALIAS:-rm-sentry.local}"  # 192.168.55.1 is shared by every Jetson
PORT="${TUNNEL_PORT:-18080}"
LOG="${TMPDIR:-/tmp}/depth_view_tunnel.log"
(
  while true; do  # an SSH tunnel can drop after a while; bring it straight back
    ssh -N -o BatchMode=yes -o ExitOnForwardFailure=yes -o ServerAliveInterval=5 -o ServerAliveCountMax=6 \
        -o HostKeyAlias="$HOSTKEY_ALIAS" -i "$KEY" -L "$PORT:localhost:8080" "$JETSON" 2>>"$LOG" || true
    echo "$(date +%T) tunnel to $JETSON exited; restarting" >>"$LOG"
    sleep 1
  done
) &
TUNNEL_PID=$!
trap 'kill $TUNNEL_PID 2>/dev/null; pkill -P $TUNNEL_PID 2>/dev/null || true' EXIT
for _ in $(seq 1 20); do nc -z localhost "$PORT" 2>/dev/null && break; sleep 0.5; done
.venv/bin/python dashboard/depth_view.py --camera-address "localhost:$PORT" "$@"
