#!/usr/bin/env python3
"""Face-servo debugging dashboard for the SO-101 rig. Run it ON THE ORIN (next to the camera):

    .venv/bin/python dashboard/server.py --local          # viam-server on localhost:8080
    open http://rm-sentry.local:8090/?token=<printed at start>

Thin web layer over ``assistive_arm.orchestration.face_servo_session``:
  GET  /                 the page (dashboard/static/index.html)
  GET  /stream.mjpg      the gripper camera with the overlay (face, spoon aim point, error)
  GET  /api/state        mode, reason, joints, face numbers, taught poses, log
  POST /api/cmd          {"action": "sweep" | "find_face" | "calibrate" | "center" | "extend" |
                          "retreat" | "stop" | "teach_look" | "teach_near" | "teleop_start" |
                          "teleop_stop" | "set_params", ...}
Every request needs the token printed at start-up (anyone on the venue Wi-Fi could
otherwise move the arm). Credentials come from .env (VIAM_API_KEY, VIAM_API_KEY_ID).
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from assistive_arm.feeding.face_servo import FaceServoConfig, target_pixel  # noqa: E402
from assistive_arm.orchestration.face_servo_session import FaceServoSession  # noqa: E402

FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152,
             148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10]
LIPS_INNER = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78]


def load_dotenv(path: Path) -> None:
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                if v.strip():
                    os.environ.setdefault(k.strip(), v.strip())


def make_render():
    import cv2

    def render(frame, session) -> bytes:
        bgr = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
        h, w = bgr.shape[:2]
        snap = session.latest()
        view = snap.get("view")
        lm = getattr(session.det, "last_landmarks", None)
        if lm is not None:
            px = (lm[:, :2] * [w, h]).astype(int)
            cv2.polylines(bgr, [px[FACE_OVAL]], False, (210, 210, 210), 1, cv2.LINE_AA)
            st = snap.get("state")
            col = (0, 220, 0) if (st is not None and st.is_open) else (0, 150, 255)
            cv2.polylines(bgr, [px[LIPS_INNER]], True, col, 2, cv2.LINE_AA)
        if view is not None:
            tgt = target_pixel(view, session.cfg).astype(int)
            lips = np.asarray(view.lips_px).astype(int)
            ok = np.max(np.abs(lips - tgt)) <= session.cfg.center_tol_px
            cv2.drawMarker(bgr, tuple(int(v) for v in tgt), (255, 255, 0), cv2.MARKER_CROSS, 30, 2)
            cv2.circle(bgr, tuple(int(v) for v in tgt), int(session.cfg.center_tol_px), (255, 255, 0), 1, cv2.LINE_AA)
            cv2.arrowedLine(bgr, tuple(int(v) for v in lips), tuple(int(v) for v in tgt),
                            (0, 255, 0) if ok else (0, 0, 255), 2, cv2.LINE_AA, tipLength=0.15)
            cv2.circle(bgr, tuple(int(v) for v in lips), 5, (0, 0, 255), -1)
        else:  # still show where the spoon points at a nominal 50 cm
            cx, cy = w // 2, h // 2
            cv2.drawMarker(bgr, (cx, cy), (160, 160, 160), cv2.MARKER_CROSS, 20, 1)
        label = f"{session.mode.upper()}  {session.reason}"[:90]
        cv2.rectangle(bgr, (0, h - 24), (w, h), (0, 0, 0), -1)
        cv2.putText(bgr, label, (8, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 75])
        return buf.tobytes() if ok else b""

    return render


class Handler(BaseHTTPRequestHandler):
    session: FaceServoSession
    token: str

    def log_message(self, *a):  # quiet
        pass

    def _authorised(self) -> bool:
        q = parse_qs(urlparse(self.path).query)
        tok = self.headers.get("X-Token") or (q.get("token") or [""])[0]
        if secrets.compare_digest(tok, self.token):
            return True
        self.send_response(403)
        self.end_headers()
        self.wfile.write(b"forbidden: open the URL with ?token=... printed by the server")
        return False

    def _json(self, obj, code=200):
        body = json.dumps(obj, default=str).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if not self._authorised():
            return
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            body = (ROOT / "dashboard/static/index.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif path == "/api/state":
            self._json(self.session.state())
        elif path == "/stream.mjpg":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                last = None
                while True:
                    jpg = self.session.jpeg()
                    if jpg and jpg is not last:
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                         + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
                        last = jpg
                    time.sleep(0.03)
            except (BrokenPipeError, ConnectionResetError):
                return
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if not self._authorised():
            return
        if urlparse(self.path).path != "/api/cmd":
            self.send_response(404)
            self.end_headers()
            return
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            action = req.pop("action")
            self._json(self.session.command(action, **req))
        except Exception as e:
            self._json({"ok": False, "error": str(e)}, 400)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--local", action="store_true", help="connect to viam-server on localhost:8080 (run on the Orin)")
    ap.add_argument("--camera", default="webcam-2")
    ap.add_argument("--poses", default=str(ROOT / "poses/so101_face.json"))
    ap.add_argument("--model", default=str(ROOT / "data/models/face_landmarker.task"))
    ap.add_argument("--stop-cm", type=float, default=20.0, help="spoon tip to lips stop distance (first runs: 20)")
    ap.add_argument("--speed", type=float, default=20.0, help="arm speed cap while servoing, deg/s (min 12)")
    ap.add_argument("--token", default="", help="fixed access token (default: random)")
    args = ap.parse_args()

    load_dotenv(ROOT / ".env")
    if args.local:
        os.environ["VIAM_ADDRESS"] = "localhost:8080"
    from assistive_arm.mouth_perception import MediaPipeMouthDetector
    from assistive_arm.robot_iface.viam_so101 import SO101ViamRobot

    robot = SO101ViamRobot(cameras=(args.camera,))
    det = MediaPipeMouthDetector(camera=args.camera, model_path=args.model)
    cfg = FaceServoConfig(stop_tip_distance=args.stop_cm / 100, joint_speed_degs=args.speed)
    session = FaceServoSession(robot, det, args.camera, args.poses, cfg, render=make_render())
    session.start()

    Handler.session = session
    Handler.token = args.token or secrets.token_urlsafe(8)
    server = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    print(f"dashboard: http://<this-machine>:{args.port}/?token={Handler.token}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        session.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
