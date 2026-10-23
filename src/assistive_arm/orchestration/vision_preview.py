"""Read-only wrist-camera preview: RF-DETR spoon boxes and MediaPipe lips.

PLAN: acquire one RGB frame, infer serially, publish a single in-memory snapshot.
There is no frame queue, disk recording, robot pose query or movement command.
The HTTP interface only reads snapshots and expires them by receiver-side age.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import dataclass, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
import json
from math import isfinite
import signal
import threading
from typing import TYPE_CHECKING, Callable
from urllib.parse import parse_qs, urlsplit

import numpy as np

from ..mouth_perception.landmarks_2d import MouthLandmarks2D
from ..vision.types import Detection

if TYPE_CHECKING:
    from ..robot_iface.base import RobotInterface
    from ..mouth_perception.landmarks_2d import MediaPipeMouthLandmarker
    from ..vision.rfdetr import RFDETRSpoonDetector


@dataclass(frozen=True)
class PreviewConfig:
    """Preview display policy; age uses the camera reader's monotonic clock."""

    camera: str = "arm-camera"
    port: int = 8765
    max_received_age_s: float = 3.0
    poll_interval_s: float = 0.1
    retry_delay_s: float = 1.0
    jpeg_quality: int = 85
    line_width_px: int = 2

    def __post_init__(self) -> None:
        if not self.camera.strip():
            raise ValueError("camera must be nonempty")
        if not 0 <= self.port <= 65535:
            raise ValueError("port must be between 0 and 65535")
        for name in ("max_received_age_s", "poll_interval_s", "retry_delay_s"):
            value = getattr(self, name)
            if not isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not 1 <= self.jpeg_quality <= 95 or self.line_width_px < 1:
            raise ValueError("invalid JPEG quality or overlay line width")


@dataclass(frozen=True)
class PreviewSnapshot:
    """One atomic inference result. JPEG bytes and observations stay in memory."""

    camera: str
    revision: int = 0
    state: str = "starting"
    received_at_s: float | None = None
    captured_at_s: float | None = None
    capture_clock: str | None = None
    image_shape: tuple[int, int, int] | None = None
    spoons: tuple[Detection, ...] = ()
    mouths: tuple[MouthLandmarks2D, ...] = ()
    capture_ms: float | None = None
    spoon_ms: float | None = None
    mouth_ms: float | None = None
    render_ms: float | None = None
    jpeg: bytes | None = None
    error: str | None = None


@dataclass(frozen=True)
class PreviewView:
    snapshot: PreviewSnapshot
    state: str
    received_age_s: float | None
    expires_in_s: float

    @property
    def jpeg(self) -> bytes | None:
        return self.snapshot.jpeg if self.state == "live" else None

    def status(self) -> dict:
        """JSON boundary; stale results do not expose old detections as current."""
        snap = self.snapshot
        live = self.state == "live"
        return {
            "revision": snap.revision, "state": self.state, "camera": snap.camera,
            "image_shape": snap.image_shape,
            "received_age_s": self.received_age_s, "expires_in_s": self.expires_in_s,
            "captured_at_s": snap.captured_at_s, "capture_clock": snap.capture_clock,
            "age_basis": "receiver monotonic; exposure age is not established",
            "timings_ms": {
                "capture": snap.capture_ms, "spoon": snap.spoon_ms,
                "mouth": snap.mouth_ms, "render": snap.render_ms,
            },
            "spoons": [{"bbox_px": item.bbox, "score": item.score} for item in snap.spoons] if live else [],
            "mouths": [{"center_px": item.center_px, "aperture_ratio": item.aperture_ratio}
                       for item in snap.mouths] if live else [],
            "error": snap.error,
        }


class PreviewStore:
    def __init__(self, config: PreviewConfig):
        self.config = config
        self._lock = threading.Lock()
        self._snapshot = PreviewSnapshot(camera=config.camera)

    def publish(self, snapshot: PreviewSnapshot) -> PreviewSnapshot:
        with self._lock:
            self._snapshot = replace(snapshot, revision=self._snapshot.revision + 1)
            return self._snapshot

    def view(self, now_s: float) -> PreviewView:
        with self._lock:
            snap = self._snapshot
        age = None if snap.received_at_s is None else now_s - snap.received_at_s
        state = snap.state
        if state == "live" and (
            age is None or not isfinite(age) or age < 0
            or age >= self.config.max_received_age_s or snap.jpeg is None
        ):
            state = "stale"
        ttl = self.config.max_received_age_s - age if state == "live" and age is not None else 0.0
        return PreviewView(snap, state, age, ttl)


def render_preview(
    rgb: np.ndarray, spoons: tuple[Detection, ...], mouths: tuple[MouthLandmarks2D, ...],
    config: PreviewConfig,
) -> bytes:
    """Draw source-pixel geometry without mutating the inference input."""
    from PIL import Image, ImageDraw

    image = Image.fromarray(rgb).copy()
    draw = ImageDraw.Draw(image)
    for spoon in spoons:
        draw.rectangle(spoon.bbox, outline="#77ff77", width=config.line_width_px)
        draw.text((spoon.bbox[0], max(0, spoon.bbox[1] - 12)),
                  f"spoon {spoon.score:.2f}", fill="#77ff77")
    for mouth in mouths:
        for contour in (mouth.outer_lip_px, mouth.inner_lip_px):
            if contour:
                draw.line((*contour, contour[0]), fill="#ffbd5b", width=config.line_width_px)
        x, y = mouth.center_px
        draw.line((x - 3, y, x + 3, y), fill="#ffbd5b", width=config.line_width_px)
        draw.line((x, y - 3, x, y + 3), fill="#ffbd5b", width=config.line_width_px)
    output = BytesIO()
    image.save(output, format="JPEG", quality=config.jpeg_quality)
    return output.getvalue()


class PreviewWorker:
    def __init__(
        self, reader: RobotInterface, spoon: RFDETRSpoonDetector,
        mouth: MediaPipeMouthLandmarker, store: PreviewStore,
    ):
        self.reader, self.spoon, self.mouth, self.store = reader, spoon, mouth, store
        self.config = store.config

    def capture_once(self) -> PreviewSnapshot:
        """Publish a fresh result or clear all imagery on capture/model failure."""
        phase = "camera"
        try:
            started = self.reader.now()
            frame = self.reader.get_rgb_frames([self.config.camera])[self.config.camera]
            captured = self.reader.now()
            if not isfinite(frame.received_at_s) or frame.received_at_s > captured:
                raise ValueError("invalid receiver timestamp")
            phase = "spoon detector"
            spoons = tuple(self.spoon.detect_rgb(frame.rgb))
            spoon_done = self.reader.now()
            phase = "mouth detector"
            mouths = tuple(self.mouth.detect_rgb(frame.rgb, timestamp_s=frame.captured_at_s))
            mouth_done = self.reader.now()
            phase = "preview rendering"
            jpeg = render_preview(frame.rgb, spoons, mouths, self.config)
            done = self.reader.now()
            snapshot = PreviewSnapshot(
                camera=self.config.camera, state="live", received_at_s=frame.received_at_s,
                captured_at_s=frame.captured_at_s, capture_clock=frame.capture_clock,
                image_shape=tuple(int(x) for x in frame.rgb.shape), spoons=spoons, mouths=mouths,
                capture_ms=(captured - started) * 1000, spoon_ms=(spoon_done - captured) * 1000,
                mouth_ms=(mouth_done - spoon_done) * 1000, render_ms=(done - mouth_done) * 1000,
                jpeg=jpeg,
            )
        except Exception as exc:
            # SDK errors can contain connection details. Report only phase/type.
            snapshot = PreviewSnapshot(
                camera=self.config.camera, state="error", error=f"{phase} failed ({type(exc).__name__})",
            )
        return self.store.publish(snapshot)

    def run(self, stop_event: threading.Event) -> None:
        while not stop_event.is_set():
            snapshot = self.capture_once()
            if not stop_event.is_set():
                delay = self.config.retry_delay_s if snapshot.state == "error" else self.config.poll_interval_s
                self.reader.wait(delay)


_PAGE = b'''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Arm camera vision</title><style>
body{font:16px system-ui;margin:24px;background:#111827;color:#e5e7eb;max-width:1000px}
img{max-width:100%;height:auto;border:1px solid #475569}pre{white-space:pre-wrap}
.hint{color:#cbd5e1}#state{font-weight:700}</style>
<h1>Arm camera vision</h1><p class="hint">Green: spoon. Orange: lips. Observation only.</p>
<p id="state">Connecting...</p><img id="frame" alt="Annotated arm camera" hidden>
<p id="counts"></p>
<p class="hint">Age is measured since the Jetson received the frame. Source capture time is listed separately.</p>
<details><summary>Frame details</summary><pre id="details"></pre></details><script>
const img=document.getElementById('frame'), state=document.getElementById('state');
const counts=document.getElementById('counts');
const details=document.getElementById('details'); let expiry=0, objectURL=null;
function hide(message){img.hidden=true;img.removeAttribute('src');expiry=0;counts.textContent='';
 if(objectURL){URL.revokeObjectURL(objectURL);objectURL=null;}if(message)state.textContent=message;}
setInterval(()=>{if(expiry && performance.now()>=expiry)hide('Stale: waiting for a fresh frame');},100);
async function poll(){try{
 const response=await fetch('/status.json',{cache:'no-store',signal:AbortSignal.timeout(1500)});
 if(!response.ok)throw Error('status');const status=await response.json();
 details.textContent=JSON.stringify(status,null,2);state.textContent=status.state;
 if(status.state!=='live'){hide(status.error||status.state);return;}
 const requestStart=performance.now();
 const image=await fetch('/frame.jpg?revision='+status.revision,{cache:'no-store',signal:AbortSignal.timeout(1500)});
 if(image.status===409)return;if(!image.ok)throw Error('frame');
 const deadline=requestStart+Number(image.headers.get('X-Frame-TTL-Seconds'))*1000;
 const blob=await image.blob();if(performance.now()>=deadline){hide('Stale');return;}
 if(objectURL)URL.revokeObjectURL(objectURL);objectURL=URL.createObjectURL(blob);
 img.src=objectURL;img.hidden=false;expiry=deadline;
 counts.textContent='Spoons: '+status.spoons.length+' | Mouths: '+status.mouths.length+
  ' | Frame age: '+Math.round(status.received_age_s*1000)+' ms';
 }catch(error){hide('Disconnected: waiting for camera preview');}
 finally{setTimeout(poll,250);}}
poll();</script></html>'''


def make_http_server(store: PreviewStore, now: Callable[[], float]) -> ThreadingHTTPServer:
    """Create a localhost-only GET server. No POST or hardware control routes."""
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args) -> None:
            pass

        def send_bytes(self, code, body, content_type, ttl=None):
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store, max-age=0")
            self.send_header("X-Content-Type-Options", "nosniff")
            if ttl is not None:
                self.send_header("X-Frame-TTL-Seconds", str(ttl))
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            url = urlsplit(self.path)
            if url.path == "/":
                self.send_bytes(200, _PAGE, "text/html; charset=utf-8")
                return
            view = store.view(now())
            if url.path == "/status.json":
                self.send_bytes(200, json.dumps(view.status(), allow_nan=False).encode(), "application/json")
            elif url.path == "/frame.jpg":
                if view.jpeg is None:
                    self.send_bytes(503, b"No fresh frame", "text/plain")
                elif parse_qs(url.query).get("revision") != [str(view.snapshot.revision)]:
                    self.send_bytes(409, b"Frame updated; refresh status", "text/plain")
                else:
                    self.send_bytes(200, view.jpeg, "image/jpeg", view.expires_in_s)
            else:
                self.send_bytes(404, b"Not found", "text/plain")

    server = ThreadingHTTPServer(("127.0.0.1", store.config.port), Handler)
    server.daemon_threads = True
    return server


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("viam", "usb"), default="viam",
                        help="Read from a Viam camera or a USB camera attached to this computer")
    parser.add_argument("--video-device", default="/dev/v4l/by-id/usb-icSpring_icspring_camera-video-index0",
                        help="Stable video device path used with --source usb")
    parser.add_argument("--camera", default="arm-camera")
    parser.add_argument("--spoon-weights", required=True)
    parser.add_argument("--mouth-model", required=True)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-age", type=float, default=3.0, help="Maximum age since receiver got frame, seconds")
    parser.add_argument("--smoke-frames", type=int, default=0, help="Process N frames and print numeric summaries, then exit")
    args = parser.parse_args(argv)
    if args.smoke_frames < 0:
        parser.error("--smoke-frames must be nonnegative")
    config = PreviewConfig(camera=args.camera, port=args.port, max_received_age_s=args.max_age)

    from ..mouth_perception.landmarks_2d import MediaPipeMouthLandmarker, MouthLandmarkerConfig
    from ..vision.rfdetr import RFDETRConfig, RFDETRSpoonDetector

    spoon = RFDETRSpoonDetector(RFDETRConfig(args.spoon_weights, device=args.device, threshold=args.threshold))
    with ExitStack() as stack:
        mouth = stack.enter_context(MediaPipeMouthLandmarker(MouthLandmarkerConfig(args.mouth_model)))
        if args.source == "usb":
            from ..robot_iface.usb_camera import UsbCameraConfig, UsbCameraReader

            reader = stack.enter_context(UsbCameraReader(UsbCameraConfig(
                device_path=args.video_device, camera_name=config.camera,
            )))
        else:
            from ..robot_iface.viam import ViamCameraReader

            reader = stack.enter_context(ViamCameraReader(cameras={config.camera: config.camera}))
        store = PreviewStore(config)
        worker = PreviewWorker(reader, spoon, mouth, store)
        if args.smoke_frames:
            failed = False
            for index in range(args.smoke_frames):
                snap = worker.capture_once()
                failed |= snap.state == "error"
                view = store.view(reader.now())
                print(json.dumps({
                    "frame": index + 1, "state": view.state, "shape": snap.image_shape,
                    "received_age_s": view.received_age_s,
                    "spoons": len(snap.spoons), "mouths": len(snap.mouths),
                    "timings_ms": view.status()["timings_ms"], "error": snap.error,
                }), flush=True)
            return int(failed)

        server = make_http_server(store, reader.now)
        stop_event = threading.Event()
        thread = threading.Thread(target=worker.run, args=(stop_event,), name="vision-preview", daemon=True)
        previous_sigterm = signal.getsignal(signal.SIGTERM)

        def terminate(_signal, _frame):
            raise KeyboardInterrupt

        signal.signal(signal.SIGTERM, terminate)
        thread.start()
        print(f"Vision preview: http://127.0.0.1:{server.server_port}", flush=True)
        try:
            server.serve_forever(poll_interval=0.25)
        except KeyboardInterrupt:
            pass
        finally:
            stop_event.set()
            server.server_close()
            thread.join()
            signal.signal(signal.SIGTERM, previous_sigterm)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
