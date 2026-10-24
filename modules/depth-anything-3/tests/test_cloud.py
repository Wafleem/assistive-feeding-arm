"""The cloud backend against a fake depth server, and its local fallback (offline)."""

import json
import threading
import zlib
from http.server import BaseHTTPRequestHandler

import numpy as np
import pytest

from da3_depth.estimator import CloudEstimator
from da3_depth.geometry import Intrinsics
from da3_depth.preview import LocalHTTPServer

K = Intrinsics(500.0, 500.0, 32.0, 24.0, 64, 48)
RGB = np.zeros((48, 64, 3), np.uint8)


@pytest.fixture
def server():
    seen = {}

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def send(self, body, headers=()):
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            for k, v in headers:
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self.send(json.dumps({"ok": True, "device": "cuda", "gpu": "NVIDIA L4"}).encode())

        def do_POST(self):
            jpeg = self.rfile.read(int(self.headers["Content-Length"]))
            seen["jpeg"], seen["focal"] = jpeg, self.headers.get("X-Focal-Px")
            mm = np.full((48, 64), 210, "<u2")
            self.send(zlib.compress(mm.tobytes()), [("X-Width", "64"), ("X-Height", "48"),
                                                     ("X-Focal-Px", seen["focal"] or "1008"), ("X-Infer-Ms", "95.0")])

    srv = LocalHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}", seen
    srv.shutdown()


class Local:
    device, estimates_focal, focal_px = "cuda", True, None
    loads = 0

    def load(self):
        Local.loads += 1

    def predict(self, rgb, k):
        self.focal_px = 777.0
        return np.full(rgb.shape[:2], 0.5, np.float32)


def test_cloud_round_trip(server):
    url, seen = server
    c = CloudEstimator(url, 2.0)
    c.load()
    assert c.device == "cloud (NVIDIA L4)"
    d = c.predict(RGB, K)
    assert d.shape == (48, 64) and d[0, 0] == pytest.approx(0.21)  # millimetres back to metres
    assert seen["jpeg"][:2] == b"\xff\xd8" and float(seen["focal"]) == 500.0
    assert c.focal_px == 500.0 and c.timing["where"] == "cloud" and c.timing["server_ms"] == 95.0
    c.predict(RGB, None)
    assert seen["focal"] is None and c.focal_px == 1008.0  # no intrinsics: the server's own estimate


def test_unreachable_cloud_falls_back_locally_then_retries(server, monkeypatch):
    url, _ = server
    Local.loads = 0
    c = CloudEstimator("http://127.0.0.1:9", 0.5, Local())  # nothing listens on port 9
    c.load()
    assert c.device.startswith("local cuda (cloud down") and Local.loads == 1
    assert c.predict(RGB, K)[0, 0] == pytest.approx(0.5) and c.timing["where"] == "local"
    c.url = url  # the tunnel comes back; after the retry period the cloud is used again
    monkeypatch.setattr(c, "_cloud_down_until", 0.0)
    assert c.predict(RGB, K)[0, 0] == pytest.approx(0.21) and c.timing["where"] == "cloud"
    assert Local.loads == 1  # the local model was loaded once, only when needed


def test_without_fallback_a_dead_cloud_is_an_error():
    with pytest.raises(RuntimeError, match="unreachable"):
        CloudEstimator("http://127.0.0.1:9", 0.5, None).load()
