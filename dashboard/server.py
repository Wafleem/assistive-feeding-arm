#!/usr/bin/env python3
"""Face-servo debugging dashboard for the SO-101 rig.

    dashboard/run_mac.sh      # on the Mac: video over the Jetson USB cable, arm via the Viam cloud
    .venv/bin/python dashboard/server.py --local --host 0.0.0.0   # or on the Orin itself

Thin web layer over ``assistive_arm.orchestration.face_servo_session``:
  GET  /                 the page (dashboard/static/index.html)
  GET  /stream.mjpg      the gripper camera with the overlay (face, spoon aim point, error)
  GET  /api/state        mode, reason, joints, face numbers, taught poses, log
  POST /api/cmd          {"action": "sweep" | "find_face" | "calibrate" | "center" | "extend" |
                          "retreat" | "stop" | "teach_look" | "teach_near" | "teleop_start" |
                          "teleop_stop" | "set_params" | "teach_bowl" | "plan_scoop" | "scoop", ...}
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
        # Bowl openings (fixed bowls from the Viam scene, projected through the camera pose) + CLIP labels
        try:
            overlays = session.bowl_overlays()
        except Exception:
            overlays = []
        for o in overlays:
            col = {"cereal": (0, 200, 255), "mms": (255, 0, 200), "empty": (160, 160, 160)}.get(o["food"], (0, 255, 255))
            pts = o["outline"][np.all(np.isfinite(o["outline"]), axis=1)].astype(np.int32)
            if len(pts) > 2:
                cv2.polylines(bgr, [pts], True, col, 2, cv2.LINE_AA)
            x0, y0, x1, y1 = (int(round(v)) for v in o["box"])
            cv2.rectangle(bgr, (max(0, x0), max(0, y0)), (min(w - 1, x1), min(h - 1, y1)), col, 1)
            ty = max(16, min(h - 32, y0 + 18))
            cv2.putText(bgr, o["text"], (max(4, x0 + 4), ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(bgr, o["text"], (max(4, x0 + 4), ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1, cv2.LINE_AA)
        label = f"{session.mode.upper()}  {session.reason}"[:90]
        cv2.rectangle(bgr, (0, h - 24), (w, h), (0, 0, 0), -1)
        cv2.putText(bgr, label, (8, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 75])
        return buf.tobytes() if ok else b""

    return render


def make_depth_render(near_m: float = 0.25, far_m: float = 2.5):
    """Depth as colour (near = red, far = blue), the face box with its measured distance."""
    import cv2
    from assistive_arm.mouth_perception import face_geometry as fg

    def render(frame, landmarks, session) -> bytes:
        d = frame.depth
        norm = np.clip((d - near_m) / (far_m - near_m), 0, 1)
        img = cv2.applyColorMap((255 * (1 - norm)).astype(np.uint8), cv2.COLORMAP_JET)
        img[d <= 0] = 0
        h, w = img.shape[:2]
        if landmarks is not None:
            px = fg.landmarks_to_pixels(landmarks, frame.intrinsics)
            x0, y0 = px[:, :2].min(axis=0).astype(int)
            x1, y1 = px[:, :2].max(axis=0).astype(int)
            cv2.rectangle(img, (x0, y0), (x1, y1), (255, 255, 255), 2)
            z = fg.face_depth_from_image(px, d, getattr(session.det, "params", fg.FaceGeometryParams()))
            lip = tuple(int(v) for v in px[13, :2])
            cv2.circle(img, lip, 5, (255, 255, 255), -1)
            if z is not None:
                cv2.putText(img, f"face {z * 100:.0f} cm", (x0, max(18, y0 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (255, 255, 255), 2, cv2.LINE_AA)
        cy, cx = h // 2, w // 2
        c = d[cy - 4:cy + 5, cx - 4:cx + 5]
        c = c[c > 0]
        if c.size:
            cv2.drawMarker(img, (cx, cy), (255, 255, 255), cv2.MARKER_CROSS, 16, 1)
            cv2.putText(img, f"{np.median(c) * 100:.0f} cm", (cx + 10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1, cv2.LINE_AA)
        cv2.rectangle(img, (0, h - 22), (w, h), (0, 0, 0), -1)
        cv2.putText(img, f"depth {near_m:.2f} m (red) .. {far_m:.1f} m (blue)   {session.depth_status}"[:80],
                    (8, h - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 75])
        return buf.tobytes() if ok else b""

    return render


class Handler(BaseHTTPRequestHandler):
    session: FaceServoSession
    token: str

    def log_message(self, *a):  # quiet
        pass

    def handle(self):  # a browser closing a stream or poll is normal: no tracebacks
        try:
            super().handle()
        except (BrokenPipeError, ConnectionResetError):
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

    # The natural-language page and its assets (static files, no secrets: served without the token;
    # every robot call they make still needs it, like the main page).
    LANGUAGE_ASSETS = {"/language.css": "text/css", "/language-app.js": "text/javascript",
                       "/language-parser.mjs": "text/javascript"}

    def _static(self, name: str, mime: str) -> None:
        body = (ROOT / "dashboard/static" / name).read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        if path in self.LANGUAGE_ASSETS:
            return self._static(path.lstrip("/"), self.LANGUAGE_ASSETS[path])
        if not self._authorised():
            return
        if path == "/language.html":
            return self._static("language.html", "text/html; charset=utf-8")
        if path in ("/", "/index.html"):
            body = (ROOT / "dashboard/static/index.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif path == "/api/state":
            self._json(self.session.state())
        elif path in ("/stream.mjpg", "/depth.mjpg"):
            source = self.session.jpeg if path == "/stream.mjpg" else self.session.depth_jpeg
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                last = None
                while True:
                    jpg = source()
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
    ap.add_argument("--camera-address", default=os.environ.get("VIAM_CAMERA_ADDRESS", ""),
                    help="camera-only connection, e.g. localhost:18080 (SSH tunnel over the Jetson USB link)")
    ap.add_argument("--host", default="127.0.0.1", help="interface to serve on (0.0.0.0 to share on the network)")
    ap.add_argument("--camera", default="webcam-2")
    ap.add_argument("--poses", default=str(ROOT / "poses/so101_face.json"))
    ap.add_argument("--model", default=str(ROOT / "data/models/face_landmarker.task"))
    ap.add_argument("--stop-cm", type=float, default=20.0, help="spoon tip to lips stop distance (first runs: 20)")
    ap.add_argument("--speed", type=float, default=20.0, help="arm speed cap while servoing, deg/s (min 12)")
    ap.add_argument("--token", default="", help="fixed access token (default: random)")
    ap.add_argument("--allow-extend", action="store_true", help="enable Extend toward the face (off by default)")
    ap.add_argument("--depth-camera", default="webcam-2-depth",
                    help="Depth Anything 3 camera that corrects the face range ('' = off)")
    args = ap.parse_args()

    load_dotenv(ROOT / ".env")
    if args.local:
        os.environ["VIAM_ADDRESS"] = "localhost:8080"
    from assistive_arm.mouth_perception import MediaPipeMouthDetector
    from assistive_arm.robot_iface.viam_so101 import SO101ViamRobot

    cams = (args.camera,) + ((args.depth_camera,) if args.depth_camera else ())
    for attempt in range(1, 7):  # the cloud link can take a few tries (WebRTC through NAT)
        try:
            try:
                robot = SO101ViamRobot(cameras=cams, camera_address=args.camera_address or None)
            except Exception as e:
                if not args.depth_camera or "depth" not in str(e).lower() and args.depth_camera not in str(e):
                    raise
                print(f"depth camera {args.depth_camera!r} unavailable ({e}); running without depth", flush=True)
                args.depth_camera = ""
                robot = SO101ViamRobot(cameras=(args.camera,), camera_address=args.camera_address or None)
            break
        except Exception as e:
            print(f"connect attempt {attempt}/6 failed: {type(e).__name__} {e}", flush=True)
            if attempt == 6:
                raise
            time.sleep(min(2 * attempt, 8))
    det = MediaPipeMouthDetector(camera=args.camera, model_path=args.model)
    cfg = FaceServoConfig(stop_tip_distance=args.stop_cm / 100, joint_speed_degs=args.speed)
    depth_det = MediaPipeMouthDetector(camera=args.depth_camera, model_path=args.model) if args.depth_camera else None
    session = FaceServoSession(robot, det, args.camera, args.poses, cfg, render=make_render(),
                               allow_extend=args.allow_extend, log_dir=ROOT / "data/logs",
                               depth_camera=args.depth_camera or None, depth_detector=depth_det,
                               depth_render=make_depth_render() if args.depth_camera else None)
    session.start()

    Handler.session = session
    Handler.token = args.token or secrets.token_urlsafe(8)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    shown = "localhost" if args.host in ("127.0.0.1", "localhost") else "<this-machine>"
    print(f"dashboard: http://{shown}:{args.port}/?token={Handler.token}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        session.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
