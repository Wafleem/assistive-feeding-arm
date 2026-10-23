"""The preview server, with fake frames and a fake vision_preview status endpoint (offline)."""

import base64
import io
import json
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler

import numpy as np
import pytest
from PIL import Image

from da3_depth.camera import DepthFrame
from da3_depth.config import CameraConfig
from da3_depth.geometry import Intrinsics, RangeParams, range_readings
from da3_depth.preview import GRID_STEP, DepthPreview, LocalHTTPServer, depth_grid, nearest_px, range_at

K = Intrinsics.from_hfov(90.0, 64, 48)
VISION = {"state": "live", "image_shape": [96, 128, 3],  # twice our size: pixels must be scaled
          "mouths": [{"center_px": [64, 48], "aperture_ratio": 0.3}],
          "spoons": [{"bbox_px": [100, 40, 120, 56], "score": 0.9}]}


def jpeg(w=64, h=48):
    buf = io.BytesIO()
    Image.fromarray(np.full((h, w, 3), 90, np.uint8)).save(buf, format="JPEG")
    return buf.getvalue()


def frame(seq=1, age=0.0):
    d = np.full((48, 64), 2.0, np.float32)
    d[20:28, 50:60] = 0.5  # the spoon's region (vision pixels / 2)
    return DepthFrame(jpeg(), d, K, range_readings(d, K, RangeParams()), time.monotonic() - age, 120.0, seq)


@pytest.fixture
def vision_url():
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            body = json.dumps(VISION).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    srv = LocalHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}/status.json"
    srv.shutdown()


class Box:
    """A frame slot the test can swap, like the camera's newest frame."""

    def __init__(self, f):
        self.f = f


def serve(box, url=""):
    cfg = CameraConfig(source="webcam-2", stale_after_s=1.0)
    return DepthPreview(0, url, lambda: box.f, lambda: cfg, lambda: {"status": "running on cpu"}).start()


def events(port, n, path="/events?scale=2", timeout=10):
    """The first n (event, data) pairs from the stream."""
    out = []
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=timeout) as r:
        assert r.headers["Content-Type"] == "text/event-stream"
        name = None
        for raw in r:
            line = raw.decode().rstrip("\n")
            if line.startswith("event: "):
                name = line[7:]
            elif line.startswith("data: "):
                out.append((name, json.loads(line[6:])))
                if len(out) == n:
                    return out


def test_grid_and_sampling_helpers():
    f = frame()
    grid = np.frombuffer(depth_grid(f.depth), "<u2")
    assert grid.size == (64 // GRID_STEP) * (48 // GRID_STEP) and set(np.unique(grid)) <= {500, 2000}
    assert range_at(f, K.cx, K.cy) == pytest.approx(2.0, abs=1e-3)
    assert range_at(f, 55, 24) == pytest.approx(0.5 * np.sqrt(1 + (23 / 32) ** 2), abs=0.01)
    assert range_at(f, -50, 10) is None
    u, v = nearest_px(f)
    assert 50 <= u <= 60 and 20 <= v <= 28  # the red dot lands on the nearest object


def test_stream_pushes_new_frames_with_images(vision_url):
    box = Box(frame(seq=7))
    p = serve(box, vision_url)
    try:
        threading.Timer(0.3, lambda: setattr(box, "f", frame(seq=8))).start()
        (e1, a), (e2, b) = events(p.port, 2)
        assert (e1, a["seq"], e2, b["seq"]) == ("frame", 7, "frame", 8)
        assert a["state"] == "live" and a["module"]["status"] == "running on cpu"
        assert base64.b64decode(a["color"])[:2] == b"\xff\xd8" and base64.b64decode(a["depth"])[:2] == b"\xff\xd8"
        grid = np.frombuffer(base64.b64decode(a["grid"]), "<u2")
        assert grid.size == a["grid_w"] * a["grid_h"] and a["grid_step"] == GRID_STEP
        mouth, spoon = a["targets"]
        assert (mouth["kind"], mouth["u"], mouth["v"]) == ("mouth", 32.0, 24.0)
        assert mouth["range_m"] == pytest.approx(2.0, abs=1e-3)
        assert spoon["kind"] == "spoon" and spoon["range_m"] < 0.7 and spoon["bbox_px"] == [50, 20, 60, 28]
        assert b"Depth preview" in urllib.request.urlopen(f"http://127.0.0.1:{p.port}/", timeout=5).read()
    finally:
        p.stop()


def test_stale_depth_is_announced_not_shown():
    p = serve(Box(frame(age=5.0)))
    try:
        (name, st), = events(p.port, 1)
        assert name == "state" and st["state"] == "stale" and "color" not in st and "readings" not in st
        st = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{p.port}/status.json", timeout=5).read())
        assert st["state"] == "stale"
    finally:
        p.stop()


def test_vision_preview_is_optional():
    p = serve(Box(frame()), "http://127.0.0.1:9/nothing")
    try:
        (name, st), = events(p.port, 1)
        assert name == "frame" and st["targets"] == [] and "not reachable" in st["vision"]
    finally:
        p.stop()
    p = serve(Box(frame()))  # the default: off
    try:
        (name, st), = events(p.port, 1)
        assert name == "frame" and st["targets"] == [] and st["vision"] is None
    finally:
        p.stop()
