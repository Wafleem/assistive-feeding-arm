#!/usr/bin/env bash
# Run the dashboard on this laptop. The camera video comes over the Jetson's USB link
# (an SSH tunnel to viam-server); arm and teleop commands use the normal Viam cloud connection.
# Needs: the USB-C cable in the Orin's device-mode port (next to the 40-pin header), .env with
# VIAM_ADDRESS / VIAM_API_KEY / VIAM_API_KEY_ID, and an SSH key for the Orin.
set -euo pipefail
cd "$(dirname "$0")/.."
JETSON="${JETSON:-ultraviolet@192.168.55.1}"
KEY="${JETSON_KEY:-$HOME/.ssh/id_ed25519_jetson}"
HOSTKEY_ALIAS="${JETSON_HOSTKEY_ALIAS:-rm-sentry.local}"  # 192.168.55.1 is shared by every Jetson
PORT="${TUNNEL_PORT:-18080}"
if ! nc -z localhost "$PORT" 2>/dev/null; then
  ssh -f -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=10 -o HostKeyAlias="$HOSTKEY_ALIAS" \
      -i "$KEY" -L "$PORT:localhost:8080" "$JETSON"
  echo "camera tunnel: localhost:$PORT -> $JETSON:8080"
fi
if ! nc -z localhost 8766 2>/dev/null; then  # the depth module's preview page (bound to the Orin's localhost)
  ssh -f -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=10 -o HostKeyAlias="$HOSTKEY_ALIAS" \
      -i "$KEY" -L "8766:127.0.0.1:8766" "$JETSON" && echo "depth preview tunnel: localhost:8766" || true
fi
# Everything (arm, cameras, vision services) goes direct over the USB link: the cloud (WebRTC)
# connection over venue Wi-Fi stalled the whole dashboard while it retried (video froze for
# seconds, joint reads timed out). Set VIAM_VIA_CLOUD=1 to use the cloud link instead.
if [ -z "${VIAM_VIA_CLOUD:-}" ]; then export VIAM_ADDRESS="localhost:$PORT"; fi
exec .venv/bin/python dashboard/server.py --camera-address "localhost:$PORT" "$@"
