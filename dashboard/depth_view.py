#!/usr/bin/env python3
"""Depth + mouth view without the arm: the dashboard's own depth overlay, nothing else.

    dashboard/run_depth_view.sh            # on the laptop: runs this on the Jetson, tunnels the page
    .venv/bin/python dashboard/depth_view.py --camera-address localhost:18080

The full dashboard needs follower-arm; this uses only its camera rig, MediaPipe detector and
``make_depth_render``, so the picture is exactly the dashboard's depth view (depth from
``webcam-2-depth``: the cloud GPU or the Orin, whichever the camera is set to). Read-only: no
arm, no commands. Serves http://127.0.0.1:<port>/ (an MJPEG stream and a status line).

Each depth fetch is ~650 kB (the raw 640x480 depth map), so run it on the Jetson
(run_depth_view.sh does): fetched over a laptop's tunnel it fills the Orin's Wi-Fi uplink and the
live dashboard's video and depth stall behind it. It fetches at most ``--max-fps`` times a second
(the depth camera makes ~3 new frames/s; faster only re-fetches the same frame).
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import types
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "dashboard"))

from server import load_dotenv, make_depth_render  # noqa: E402  (the dashboard's renderer)

STATUS_WINDOW_S = 5.0  # frames/s is averaged over this window

PAGE = b"""<!doctype html><meta charset=utf-8><title>Depth + mouth</title>
<style>body{background:#111827;color:#e5e7eb;font:15px system-ui;margin:16px}img{max-width:100%;border:1px solid #475569}</style>
<h2>Depth + mouth (webcam-2-depth, the dashboard's overlay)</h2>
<p id=s>connecting...</p><img src="/depth.mjpg" alt="depth and mouth">
<script>setInterval(async()=>{try{const s=await (await fetch('/status.json')).json();
document.getElementById('s').textContent='frames/s '+s.fps+' | frames '+s.n+' | face '+(s.face?'yes':'no')+
(s.err?' | last error: '+s.err:'')}catch(e){}},1000)</script>"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera-address", default="localhost:18080", help="viam-server, e.g. an SSH tunnel to the Jetson")
    ap.add_argument("--camera", default="webcam-2-depth")
    ap.add_argument("--model", default=str(ROOT / "data/models/face_landmarker.task"))
    ap.add_argument("--port", type=int, default=8091)
    ap.add_argument("--max-fps", type=float, default=3.0, help="depth fetches per second at most (the camera's produce_hz)")
    args = ap.parse_args()
    period = 1.0 / args.max_fps

    load_dotenv(ROOT / ".env")
    from assistive_arm.mouth_perception import MediaPipeMouthDetector
    from assistive_arm.mouth_perception import face_geometry as fg
    from assistive_arm.robot_iface.viam import ViamCameraRig

    rig = ViamCameraRig((args.camera,), address=args.camera_address)
    det = MediaPipeMouthDetector(camera=args.camera, model_path=args.model)
    render = make_depth_render()
    session = types.SimpleNamespace(depth_status=args.camera, det=types.SimpleNamespace(params=fg.FaceGeometryParams()))
    state = {"jpg": None, "n": 0, "t": [], "face": False, "err": None}
    lock = threading.Lock()

    def loop() -> None:
        while True:
            t0 = time.monotonic()
            try:
                frame = rig.get_frames([args.camera])[args.camera]  # reconnects after repeated failures
                found = det.landmarks(frame.rgb, time.monotonic())
                jpg = render(frame, None if found is None else found[0], session)
                now = time.monotonic()
                with lock:
                    state.update(jpg=jpg, n=state["n"] + 1, face=found is not None, err=None)
                    state["t"] = [x for x in state["t"] if now - x < STATUS_WINDOW_S] + [now]
            except Exception as e:  # noqa: BLE001 - keep serving; the status line shows the error
                with lock:
                    state["err"] = f"{type(e).__name__}: {e}"[:200]
                time.sleep(1)
            time.sleep(max(0.0, period - (time.monotonic() - t0)))

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_a) -> None:
            pass

        def send(self, body: bytes, ctype: str) -> None:
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path == "/":
                return self.send(PAGE, "text/html; charset=utf-8")
            if self.path == "/status.json":
                with lock:
                    s = {"n": state["n"], "fps": round(len(state["t"]) / STATUS_WINDOW_S, 1),
                         "face": state["face"], "err": state["err"]}
                return self.send(json.dumps(s).encode(), "application/json")
            if self.path != "/depth.mjpg":
                self.send_response(404)
                return self.end_headers()
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            last = None
            try:
                while True:
                    with lock:
                        jpg = state["jpg"]
                    if jpg and jpg is not last:
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                         + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
                        last = jpg
                    time.sleep(0.05)
            except (BrokenPipeError, ConnectionResetError):
                return

    threading.Thread(target=loop, daemon=True).start()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    server.daemon_threads = True
    print(f"depth view: http://localhost:{args.port}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
