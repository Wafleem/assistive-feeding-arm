"""Depth Pro over HTTP, for a cloud GPU (Brev). The Orin's depth module calls it through an SSH tunnel.

    python depth_server.py --weights depth_pro.pt [--port 8800]

POST /depth   body: one JPEG. Optional header X-Focal-Px: fx at the JPEG's size (else Depth Pro
              estimates it). Answer: zlib-compressed little-endian uint16 millimetres, row-major,
              with headers X-Width, X-Height, X-Focal-Px (the focal used), X-Infer-Ms.
GET  /health  {"ok": true, "device": ..., "gpu": ...}

Binds to 127.0.0.1 only: reach it through `ssh -L`, never expose the port publicly. One frame
at a time (a lock), so a second client waits instead of competing for the GPU.
"""

from __future__ import annotations

import argparse
import dataclasses
import io
import json
import socketserver
import threading
import time
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
from PIL import Image

MAX_BODY = 8 * 1024 * 1024  # a 640x480 JPEG is ~40 kB; refuse anything absurd


class Server(ThreadingHTTPServer):
    daemon_threads = True

    def server_bind(self) -> None:  # skip HTTPServer's reverse DNS lookup of its own address
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = self.server_address[:2]


def load(weights: str):
    import torch
    import depth_pro
    from depth_pro.depth_pro import DEFAULT_MONODEPTH_CONFIG_DICT

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = dataclasses.replace(DEFAULT_MONODEPTH_CONFIG_DICT, checkpoint_uri=weights)
    model, transform = depth_pro.create_model_and_transforms(
        cfg, device=torch.device(device), precision=torch.half if device == "cuda" else torch.float32)
    gpu = torch.cuda.get_device_name(0) if device == "cuda" else None
    return model.eval(), transform, torch, device, gpu


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--weights", required=True)
    ap.add_argument("--port", type=int, default=8800)
    args = ap.parse_args()
    model, transform, torch, device, gpu = load(args.weights)
    lock = threading.Lock()

    def infer(jpeg: bytes, focal: float | None) -> tuple[np.ndarray, float, float]:
        rgb = np.asarray(Image.open(io.BytesIO(jpeg)).convert("RGB"))
        f_px = None if focal is None else torch.tensor(focal, device=device)
        with lock, torch.no_grad():
            t = time.perf_counter()
            out = model.infer(transform(rgb), f_px=f_px)
            depth = out["depth"].float().cpu().numpy()
            ms = (time.perf_counter() - t) * 1000.0
        return depth, float(out["focallength_px"]), ms

    infer(_blank_jpeg(), None)  # warm-up: the first call compiles CUDA kernels

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_a) -> None:
            pass

        def reply(self, code: int, body: bytes, ctype: str, headers: dict | None = None) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            for k, v in (headers or {}).items():
                self.send_header(k, str(v))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path == "/health":
                return self.reply(200, json.dumps({"ok": True, "device": device, "gpu": gpu}).encode(), "application/json")
            self.reply(404, b"not found", "text/plain")

        def do_POST(self) -> None:
            if self.path != "/depth":
                return self.reply(404, b"not found", "text/plain")
            n = int(self.headers.get("Content-Length", 0))
            if not 0 < n <= MAX_BODY:
                return self.reply(413, b"send one JPEG", "text/plain")
            jpeg = self.rfile.read(n)
            focal = self.headers.get("X-Focal-Px")
            try:
                depth, f_used, ms = infer(jpeg, float(focal) if focal else None)
            except Exception as e:  # noqa: BLE001 - a bad frame must not kill the server
                return self.reply(400, f"{type(e).__name__}: {e}".encode(), "text/plain")
            mm = np.clip(np.where(np.isfinite(depth) & (depth > 0), np.round(depth * 1000.0), 0), 0, 65535)
            body = zlib.compress(mm.astype("<u2").tobytes(), 1)
            h, w = depth.shape
            self.reply(200, body, "application/octet-stream",
                       {"X-Width": w, "X-Height": h, "X-Focal-Px": f"{f_used:.2f}", "X-Infer-Ms": f"{ms:.1f}"})

    srv = Server(("127.0.0.1", args.port), Handler)
    print(f"depth server on 127.0.0.1:{args.port} ({device}, {gpu})", flush=True)
    srv.serve_forever()


def _blank_jpeg() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (640, 480), (128, 128, 128)).save(buf, format="JPEG")
    return buf.getvalue()


if __name__ == "__main__":
    main()
