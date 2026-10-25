#!/usr/bin/env bash
# Keep the dashboard's tunnel to the Orin up (run on the Mac):
#   localhost:18080 -> Orin viam-server :8080, localhost:8766 -> depth preview :8766
# Reconnects within a second or two if it drops; tries the USB link first, then Wi-Fi.
#   nohup scripts/tunnel_keeper.sh >/tmp/tunnel_keeper.log 2>&1 &     stop: pkill -f tunnel_keeper
KEY="${JETSON_KEY:-$HOME/.ssh/id_ed25519_jetson}"
HOSTS=(${JETSON_HOSTS:-192.168.55.1 rm-sentry.local})
while true; do
  for h in "${HOSTS[@]}"; do
    if ssh -o ConnectTimeout=4 -o BatchMode=yes -o HostKeyAlias=rm-sentry.local -i "$KEY" "ultraviolet@$h" true 2>/dev/null; then
      echo "$(date +%T) tunnel via $h"
      ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=3 -o ServerAliveCountMax=3 \
          -o HostKeyAlias=rm-sentry.local -i "$KEY" \
          -L 18080:localhost:8080 -L 8766:127.0.0.1:8766 "ultraviolet@$h"
      echo "$(date +%T) tunnel via $h dropped"
      break
    fi
  done
  sleep 1
done
