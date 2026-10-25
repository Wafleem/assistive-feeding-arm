#!/usr/bin/env bash
# Depth + mouth view without the arm (dashboard/depth_view.py), opened on this laptop.
# The viewer runs ON the Jetson (it reads depth from viam-server on the Jetson's localhost), and
# only its finished picture comes over an SSH tunnel to http://localhost:8091/. Fetching the raw
# depth (~650 kB a frame) from the laptop filled the Orin's Wi-Fi uplink and froze the live
# dashboard. The tunnel is restarted if it drops; Ctrl-C stops the viewer on the Jetson too.
# Reach the Jetson over its USB link (default) or the network:
#   JETSON=ultraviolet@10.17.103.124 dashboard/run_depth_view.sh
# Needs .env (VIAM_API_KEY / VIAM_API_KEY_ID: sent over SSH's stdin, never stored on the Jetson),
# an SSH key the Jetson accepts, and the repo + a venv with mediapipe/opencv/viam on the Jetson.
set -euo pipefail
cd "$(dirname "$0")/.."
JETSON="${JETSON:-ultraviolet@192.168.55.1}"
KEY="${JETSON_KEY:-$HOME/.ssh/id_ed25519}"
HOSTKEY_ALIAS="${JETSON_HOSTKEY_ALIAS:-rm-sentry.local}"  # 192.168.55.1 is shared by every Jetson
PORT="${VIEW_PORT:-8091}"
REPO="${JETSON_REPO:-/home/ultraviolet/open-vocab-helper-arm}"
PY="${JETSON_PYTHON:-/home/ultraviolet/assistive-arm-vision/.venv/bin/python}"
MODEL="${JETSON_FACE_MODEL:-/home/ultraviolet/assistive-arm-vision/data/models/face_landmarker.task}"
LOG="${TMPDIR:-/tmp}/depth_view_tunnel.log"
EXTRA=""; [ $# -gt 0 ] && EXTRA="$(printf '%q ' "$@")"  # extra depth_view.py options, e.g. --max-fps 2
SSH=(ssh -o BatchMode=yes -o ServerAliveInterval=5 -o ServerAliveCountMax=6 -o HostKeyAlias="$HOSTKEY_ALIAS" -i "$KEY")
# "[d]ashboard" and "$v" keep pkill from matching the remote shell's own command line
STOP="pkill -f '[d]ashboard/depth_view.py' || true"
# On the Jetson: stop an earlier viewer, read the keys (up to a blank line), start the viewer, and
# stop it again as soon as stdin closes: this script exiting or the SSH session dropping.
REMOTE="$STOP; sleep 0.5; while IFS='=' read -r k val && [ -n \"\$k\" ]; do export \"\$k=\$val\"; done; \
cd $REPO && v=depth_view && { $PY dashboard/\$v.py --camera-address localhost:8080 --model $MODEL --port $PORT $EXTRA </dev/null & }; \
p=\$!; cat >/dev/null; kill \$p"
(
  while true; do  # an SSH tunnel can drop after a while; bring it (and the viewer) straight back
    # keys, a blank line, then a newline every 5 s: stdin stays open while this runs, and once ssh is
    # gone the next write fails, so the pipeline ends and the loop reconnects
    { grep -E '^VIAM_API_KEY(_ID)?=' .env; echo; while sleep 5; do echo; done; } | "${SSH[@]}" -o ExitOnForwardFailure=yes \
        -L "$PORT:127.0.0.1:$PORT" "$JETSON" "$REMOTE" 2>>"$LOG" || true
    echo "$(date +%T) viewer on $JETSON exited; restarting" >>"$LOG"
    sleep 1
  done
) &
LOOP_PID=$!
cleanup() {
  trap - EXIT INT TERM
  pkill -P $LOOP_PID 2>/dev/null; kill $LOOP_PID 2>/dev/null  # ssh's stdin closes: the Jetson stops the viewer
  "${SSH[@]}" -o ConnectTimeout=5 "$JETSON" "$STOP" 2>/dev/null || true  # and make sure
}
trap cleanup EXIT
trap 'exit 130' INT TERM
echo "depth view: http://localhost:$PORT/  (running on $JETSON; Ctrl-C stops it)"
wait $LOOP_PID
