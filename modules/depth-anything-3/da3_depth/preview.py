"""Read-only depth preview page, served from inside the module (it already holds every frame).

PLAN: localhost only, GET only, no hardware control; view it on the Orin or through an SSH
tunnel. Built for a bad link (the venue Wi-Fi has 1-7 s round trips): each new frame is
pushed down one long-lived Server-Sent Events connection, so there are no per-frame round
trips. The sender only ever sends the newest frame and the socket's send buffer is small,
so a slow link lowers the frame rate instead of building up delay. The page blanks and
says "stale" when frames stop. Optionally (``vision_status_url``) it also shows how far
away vision_preview's mouths and spoons are according to the depth.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import socket
import socketserver
import threading
import time
import urllib.request
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from math import hypot, tan
from typing import TYPE_CHECKING, Callable, Optional
from urllib.parse import parse_qs, urlsplit

import numpy as np
from PIL import Image

from .geometry import colorize

if TYPE_CHECKING:
    from .camera import DepthFrame
    from .config import CameraConfig

LOGGER = logging.getLogger(__name__)
GRID_STEP = 8  # hover grid = every 8th pixel: 80x60 for 640x480, ~10 kB per frame
TARGET_WINDOW_PX = 7  # depth at a mouth/spoon = median over this window around its pixel
VISION_STATUS_TIMEOUT_S = 0.3
VISION_POLL_S = 1.0  # vision_preview's status is fetched at most this often
JPEG_QUALITY = 75
DEFAULT_SCALE = 2  # images are sent at 1/scale resolution (overlays stay in full-res pixels)
SEND_BUFFER_BYTES = 64 * 1024  # small, so frames are dropped here rather than queued in the kernel
POLL_S = 0.05  # how often a stream checks for a new frame
KEEPALIVE_S = 5.0


class LocalHTTPServer(ThreadingHTTPServer):
    """HTTPServer.server_bind does a reverse DNS lookup of its own address, which can hang for
    tens of seconds on a network without working DNS; it's only used for a log string."""

    daemon_threads = True

    def server_bind(self) -> None:
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = self.server_address[:2]


@dataclass(frozen=True)
class Target:
    kind: str  # "mouth" or "spoon"
    u: float
    v: float
    range_m: Optional[float]
    extra: dict


def depth_grid(depth_m: np.ndarray, step: int = GRID_STEP) -> bytes:
    """Little-endian uint16 millimetres, every ``step``-th pixel, row-major (0 = invalid)."""
    d = depth_m[step // 2 :: step, step // 2 :: step]
    mm = np.where(np.isfinite(d) & (d > 0), np.round(d * 1000.0), 0)
    return np.clip(mm, 0, 65535).astype("<u2").tobytes()


def range_at(frame: "DepthFrame", u: float, v: float, window: int = TARGET_WINDOW_PX) -> Optional[float]:
    """Median range (m, along the ray) around pixel (u, v); None if nothing valid there."""
    k, d = frame.intrinsics, frame.depth
    h, w = d.shape
    r = window // 2
    ui, vi = int(round(u)), int(round(v))
    if not (0 <= ui < w and 0 <= vi < h):
        return None
    patch = d[max(0, vi - r) : min(h, vi + r + 1), max(0, ui - r) : min(w, ui + r + 1)]
    z = patch[np.isfinite(patch) & (patch > 0)]
    if z.size == 0:
        return None
    x, y = (u - k.cx) / k.fx, (v - k.cy) / k.fy
    return round(float(np.median(z)) * (1.0 + x * x + y * y) ** 0.5, 3)


def targets_from_vision(status: dict, frame: "DepthFrame") -> list[Target]:
    """Mouths and spoons from vision_preview's /status.json, with depth from our frame.
    Its pixels are scaled if its image size differs from ours."""
    if status.get("state") != "live":
        return []
    shape = status.get("image_shape") or [frame.intrinsics.height, frame.intrinsics.width]
    sx, sy = frame.intrinsics.width / shape[1], frame.intrinsics.height / shape[0]
    out = []
    for m in status.get("mouths", []):
        u, v = m["center_px"][0] * sx, m["center_px"][1] * sy
        out.append(Target("mouth", u, v, range_at(frame, u, v), {"aperture_ratio": m.get("aperture_ratio")}))
    for s in status.get("spoons", []):
        x0, y0, x1, y1 = s["bbox_px"]
        u, v = (x0 + x1) / 2 * sx, (y0 + y1) / 2 * sy
        out.append(Target("spoon", u, v, range_at(frame, u, v),
                          {"score": s.get("score"), "bbox_px": [x0 * sx, y0 * sy, x1 * sx, y1 * sy]}))
    return out


def nearest_px(frame: "DepthFrame") -> Optional[list[float]]:
    """Pixel of the nearest return, from its bearing/elevation (inverse of range_readings)."""
    r, k = frame.readings, frame.intrinsics
    if r.nearest_bearing_deg is None or r.nearest_elevation_deg is None:
        return None
    x = tan(np.radians(r.nearest_bearing_deg))
    y = -tan(np.radians(r.nearest_elevation_deg)) * hypot(1.0, x)
    return [round(k.cx + k.fx * x, 1), round(k.cy + k.fy * y, 1)]


def _jpeg(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=JPEG_QUALITY)
    return base64.b64encode(buf.getvalue()).decode("ascii")


class DepthPreview:
    def __init__(self, port: int, vision_status_url: str, get_frame: Callable[[], Optional["DepthFrame"]],
                 get_config: Callable[[], "CameraConfig"], get_status: Callable[[], dict]):
        self.vision_status_url = vision_status_url
        self.get_frame, self.get_config, self.get_status = get_frame, get_config, get_status
        self._stopped = threading.Event()
        self._lock = threading.Lock()
        self._vision: tuple[float, Optional[dict], Optional[str]] = (-1e9, None, None)
        self._encoded: dict = {}  # (seq, scale) -> images: encoded once, shared by all viewers
        self._server = LocalHTTPServer(("127.0.0.1", port), self._handler())
        self._thread = threading.Thread(target=self._server.serve_forever, kwargs={"poll_interval": 0.25},
                                        name="da3-preview", daemon=True)

    @property
    def port(self) -> int:
        return self._server.server_port

    def start(self) -> "DepthPreview":
        self._thread.start()
        LOGGER.info("depth preview on http://127.0.0.1:%d (localhost only)", self.port)
        return self

    def stop(self) -> None:
        self._stopped.set()  # ends the open streams
        self._server.shutdown()
        self._server.server_close()

    def _fresh(self) -> tuple[Optional["DepthFrame"], str, Optional[float]]:
        frame = self.get_frame()
        if frame is None:
            return None, "starting", None
        age = time.monotonic() - frame.captured_at
        return (frame, "live", age) if age <= self.get_config().stale_after_s else (None, "stale", age)

    def _vision_status(self) -> tuple[Optional[dict], Optional[str]]:
        if not self.vision_status_url:
            return None, None
        with self._lock:
            fetched_at, status, note = self._vision
        if time.monotonic() - fetched_at < VISION_POLL_S:
            return status, note
        try:
            with urllib.request.urlopen(self.vision_status_url, timeout=VISION_STATUS_TIMEOUT_S) as resp:
                status, note = json.load(resp), None
            if status.get("state") != "live":
                note = f"vision_preview {status.get('state')}" + (f": {status['error']}" if status.get("error") else "")
        except Exception as e:  # noqa: BLE001 - optional: the page works without it
            status, note = None, f"vision_preview not reachable ({type(e).__name__})"
        with self._lock:
            self._vision = (time.monotonic(), status, note)
        return status, note

    def status(self, frame: Optional["DepthFrame"] = None) -> dict:
        """Everything about the newest (or the given) frame except the images."""
        if frame is None:
            frame, state, age = self._fresh()
        else:
            state, age = "live", time.monotonic() - frame.captured_at
        out = {"state": state, "age_s": None if age is None else round(age, 3), "module": self.get_status()}
        if frame is None:
            return out
        k, cfg = frame.intrinsics, self.get_config()
        vision, vision_note = self._vision_status()
        targets = targets_from_vision(vision, frame) if vision else []
        out.update({
            "seq": frame.seq, "infer_ms": round(frame.infer_ms, 1),
            "intrinsics": {"fx": k.fx, "fy": k.fy, "cx": k.cx, "cy": k.cy, "width": k.width, "height": k.height},
            "readings": frame.readings.to_dict(), "nearest_px": nearest_px(frame),
            "center_roi": cfg.ranges.center_roi, "scan_band": cfg.ranges.scan_band,
            "min_range_m": cfg.ranges.min_range, "max_range_m": cfg.ranges.max_range,
            "targets": [{"kind": t.kind, "u": t.u, "v": t.v, "range_m": t.range_m, **t.extra} for t in targets],
            "vision": vision_note,
        })
        return out

    def _images(self, frame: "DepthFrame", scale: int) -> dict:
        """Base64 JPEGs at 1/scale and the hover grid; encoded once per frame and scale."""
        key = (frame.seq, scale)
        with self._lock:
            if key in self._encoded:
                return self._encoded[key]
        cfg = self.get_config()
        k = frame.intrinsics
        w, h = max(1, k.width // scale), max(1, k.height // scale)
        color = Image.open(io.BytesIO(frame.color_jpeg))
        color.draft("RGB", (w, h))  # JPEG decodes straight to a smaller size
        color = color.convert("RGB").resize((w, h))
        depth = Image.fromarray(colorize(frame.depth[::scale, ::scale], cfg.ranges.min_range, cfg.ranges.max_range))
        g = frame.depth[GRID_STEP // 2 :: GRID_STEP, GRID_STEP // 2 :: GRID_STEP]
        images = {"color": _jpeg(color), "depth": _jpeg(depth.resize((w, h))),
                  "grid": base64.b64encode(depth_grid(frame.depth)).decode("ascii"),
                  "grid_w": g.shape[1], "grid_h": g.shape[0], "grid_step": GRID_STEP}
        with self._lock:
            self._encoded = {kk: v for kk, v in self._encoded.items() if kk[0] == frame.seq}
            self._encoded[key] = images
        return images

    def event(self, scale: int) -> tuple[Optional[int], dict]:
        """(seq, payload with images) for a fresh newest frame, else (None, status)."""
        frame, state, _ = self._fresh()
        if frame is None:
            return None, self.status()
        return frame.seq, {**self.status(frame), **self._images(frame, scale)}

    def _handler(self):
        preview = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args) -> None:
                pass

            def send(self, code: int, body: bytes, content_type: str) -> None:
                self.send_response(code)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store, max-age=0")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.end_headers()
                try:
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def do_GET(self) -> None:
                url = urlsplit(self.path)
                if url.path == "/":
                    return self.send(200, PAGE.encode(), "text/html; charset=utf-8")
                if url.path == "/status.json":
                    return self.send(200, json.dumps(preview.status(), allow_nan=False).encode(), "application/json")
                if url.path == "/events":
                    return self.stream(url)
                self.send(404, b"Not found", "text/plain")

            def stream(self, url) -> None:
                raw = parse_qs(url.query).get("scale", [""])[0]
                scale = min(max(int(raw), 1), 4) if raw.isdigit() else DEFAULT_SCALE
                self.connection.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, SEND_BUFFER_BYTES)
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                last_seq, last_state, last_write = None, None, time.monotonic()
                try:
                    while not preview._stopped.is_set():
                        now = time.monotonic()
                        frame = preview.get_frame()
                        if frame is not None and frame.seq != last_seq or frame is None and last_state is None:
                            seq, payload = preview.event(scale)
                            if seq is not None:
                                # Blocks while the link is slow; frames that arrive meanwhile are skipped.
                                self.wfile.write(b"event: frame\ndata: " + json.dumps(payload, allow_nan=False).encode() + b"\n\n")
                                last_seq, last_state, last_write = seq, "live", now
                            elif payload["state"] != last_state:
                                self.wfile.write(b"event: state\ndata: " + json.dumps(payload, allow_nan=False).encode() + b"\n\n")
                                last_state, last_write = payload["state"], now
                        if now - last_write > KEEPALIVE_S:
                            # Also re-announces staleness: a stuck camera stops producing new seqs.
                            seq, payload = preview.event(scale)
                            if seq is None:
                                self.wfile.write(b"event: state\ndata: " + json.dumps(payload, allow_nan=False).encode() + b"\n\n")
                                last_state = payload["state"]
                            else:
                                self.wfile.write(b": keepalive\n\n")
                            last_write = now
                        self.wfile.flush()
                        time.sleep(POLL_S)
                except OSError:
                    pass  # the viewer went away (BrokenPipe/ConnectionReset); EventSource reconnects

        return Handler


PAGE = r"""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Depth preview</title><style>
body{font:15px system-ui;margin:16px;background:#111827;color:#e5e7eb}
h1{font-size:20px;margin:0 0 4px}.dim{color:#94a3b8}#state{font-weight:700}
.row{display:flex;flex-wrap:wrap;gap:12px;margin-top:10px}.panel{flex:1 1 360px;min-width:0}
canvas{width:100%;height:auto;border:1px solid #475569;display:block;cursor:crosshair;background:#0b1220}
#scan{max-width:520px}.stats{display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:6px 14px;margin-top:10px}
.stat b{display:block;font-size:22px;font-variant-numeric:tabular-nums}.stat span{color:#94a3b8;font-size:12px}
ul{padding-left:18px;margin:6px 0}select{background:#1f2937;color:#e5e7eb;border:1px solid #475569}
</style>
<h1>Depth preview</h1>
<div class="dim">Observation only. Single-camera depth is an estimate: use it to judge, not to stop the arm.</div>
<p><span id="state">Connecting...</span> <span class="dim" id="meta"></span>
 <label class="dim">&nbsp; image size <select id="scale"><option value="1">full</option><option value="2" selected>half</option><option value="4">quarter</option></select></label></p>
<div class="stats">
 <div class="stat"><span>Straight ahead (white box)</span><b id="dist">&ndash;</b></div>
 <div class="stat"><span>Nearest return (red dot)</span><b id="near">&ndash;</b></div>
 <div class="stat"><span>Under cursor</span><b id="hover">&ndash;</b></div>
 <div class="stat"><span>Frames received / s</span><b id="fps">&ndash;</b></div>
</div>
<div class="row">
 <div class="panel"><div class="dim">Colour</div><canvas id="color" width="640" height="480"></canvas></div>
 <div class="panel"><div class="dim">Depth (near = red, far = blue, black = no depth)</div><canvas id="depth" width="640" height="480"></canvas></div>
</div>
<div class="row">
 <div class="panel"><div class="dim">Scan from above (camera at the bottom, rings every 1 m): the band between the dashed lines</div><canvas id="scan" width="520" height="300"></canvas></div>
 <div class="panel" id="tpanel" hidden><div class="dim">Mouths and spoons (vision_preview, depth from this frame)</div><ul id="targets"></ul><div class="dim" id="vision"></div></div>
</div>
<script>
const $=id=>document.getElementById(id);const C={color:$('color'),depth:$('depth')};
let S=null,grid=null,imgs={},live=false,lastFrame=0,arrivals=[],es=null;
const m=v=>v==null?'–':v.toFixed(2)+' m';
function blank(msg){live=false;$('state').textContent=msg;$('dist').textContent=$('near').textContent='–';
 for(const c of Object.values(C))c.getContext('2d').clearRect(0,0,c.width,c.height);}
function draw(){const k=S.intrinsics,r=S.readings;
 for(const name of ['color','depth']){const c=C[name],g=c.getContext('2d');
  if(c.width!==k.width||c.height!==k.height){c.width=k.width;c.height=k.height;}
  g.drawImage(imgs[name],0,0,k.width,k.height);g.lineWidth=2;g.strokeStyle='#e5e7eb';
  const bw=k.width*S.center_roi,bh=k.height*S.center_roi;g.strokeRect(k.cx-bw/2,k.cy-bh/2,bw,bh);
  const band=k.height*S.scan_band/2;g.setLineDash([6,6]);g.beginPath();g.moveTo(0,k.cy-band);g.lineTo(k.width,k.cy-band);
  g.moveTo(0,k.cy+band);g.lineTo(k.width,k.cy+band);g.stroke();g.setLineDash([]);
  g.font='bold 16px system-ui';g.fillStyle='#fff';g.fillText(m(r.distance_m),k.cx+bw/2+6,k.cy+5);
  if(S.nearest_px){const[u,v]=S.nearest_px;g.fillStyle='#ff4d6d';g.beginPath();g.arc(u,v,7,0,7);g.fill();g.fillText(m(r.nearest_m),u+10,v-8);}
  for(const t of S.targets){const col=t.kind==='mouth'?'#ffbd5b':'#77ff77';g.strokeStyle=g.fillStyle=col;
   if(t.bbox_px){const[a,b,c2,d]=t.bbox_px;g.strokeRect(a,b,c2-a,d-b);}
   g.beginPath();g.moveTo(t.u-8,t.v);g.lineTo(t.u+8,t.v);g.moveTo(t.u,t.v-8);g.lineTo(t.u,t.v+8);g.stroke();
   g.fillText(t.kind+' '+m(t.range_m),t.u+10,t.v+20);}}
 const c=$('scan'),g=c.getContext('2d'),W=c.width,H=c.height,max=S.max_range_m,s=(H-20)/max;
 g.fillStyle='#0b1220';g.fillRect(0,0,W,H);g.strokeStyle='#334155';g.fillStyle='#94a3b8';g.font='12px system-ui';
 for(let d=1;d<=max;d++){g.beginPath();g.arc(W/2,H-10,d*s,Math.PI,2*Math.PI);g.stroke();g.fillText(d+' m',W/2+d*s-24,H-12);}
 g.fillStyle='#38bdf8';r.scan_ranges_m.forEach((d,i)=>{if(d==null)return;const b=(r.scan_angle_min_deg+i*r.scan_angle_increment_deg)*Math.PI/180;
  const x=W/2+Math.sin(b)*Math.min(d,max)*s,y=H-10-Math.cos(b)*Math.min(d,max)*s;g.fillRect(x-2,y-2,4,4);});
 g.fillStyle='#e5e7eb';g.beginPath();g.moveTo(W/2,H-14);g.lineTo(W/2-6,H-2);g.lineTo(W/2+6,H-2);g.fill();}
function hover(ev,c){if(!live||!grid)return;const k=S.intrinsics,rect=c.getBoundingClientRect(),st=S.grid_step;
 const u=(ev.clientX-rect.left)/rect.width*k.width,v=(ev.clientY-rect.top)/rect.height*k.height;
 const gx=Math.min(S.grid_w-1,Math.max(0,Math.floor(u/st))),gy=Math.min(S.grid_h-1,Math.max(0,Math.floor(v/st)));
 const z=grid[gy*S.grid_w+gx];if(!z){$('hover').textContent='no depth';return;}
 const x=(u-k.cx)/k.fx,y=(v-k.cy)/k.fy;$('hover').textContent=(z/1000*Math.sqrt(1+x*x+y*y)).toFixed(2)+' m';}
for(const c of Object.values(C)){c.addEventListener('mousemove',e=>hover(e,c));c.addEventListener('mouseleave',()=>$('hover').textContent='–');}
function decode(b64){return new Promise((ok,bad)=>{const i=new Image();i.onload=()=>ok(i);i.onerror=bad;i.src='data:image/jpeg;base64,'+b64;});}
function meta(st){const mod=st.module||{};return (mod.status||'')+(mod.device?' on '+mod.device:'')+(mod.timing&&mod.timing.total_ms?' ('+mod.timing.total_ms+' ms)':'')+(mod.last_error?' | '+mod.last_error:'');}
function connect(){if(es)es.close();es=new EventSource('/events?scale='+$('scale').value);$('state').textContent='Connecting...';
 es.addEventListener('frame',async e=>{const st=JSON.parse(e.data);
  const[ci,di]=await Promise.all([decode(st.color),decode(st.depth)]);
  const b=atob(st.grid),u8=new Uint8Array(b.length);for(let i=0;i<b.length;i++)u8[i]=b.charCodeAt(i);
  imgs={color:ci,depth:di};grid=new Uint16Array(u8.buffer);S=st;live=true;
  const now=performance.now();lastFrame=now;arrivals=arrivals.filter(t=>now-t<5000);arrivals.push(now);
  $('state').textContent='live';$('meta').textContent=(st.module.model||'')+' | frame '+st.seq+' | age when sent '+Math.round(st.age_s*1000)+' ms | inference '+st.infer_ms+' ms | '+meta(st);
  $('dist').textContent=m(st.readings.distance_m);$('near').textContent=m(st.readings.nearest_m);
  $('tpanel').hidden=!st.vision&&!st.targets.length;
  $('targets').innerHTML=st.targets.length?st.targets.map(t=>'<li>'+t.kind+': <b>'+m(t.range_m)+'</b></li>').join(''):'<li class="dim">none</li>';
  $('vision').textContent=st.vision||'';draw();});
 es.addEventListener('state',e=>{const st=JSON.parse(e.data);
  blank(st.state==='stale'?'Stale: newest depth is '+st.age_s+' s old':'Waiting for depth ('+st.state+')');$('meta').textContent=meta(st);});
 es.onerror=()=>{if(live)blank('Reconnecting...');};}
setInterval(()=>{const now=performance.now();arrivals=arrivals.filter(t=>now-t<5000);
 $('fps').textContent=live?(arrivals.length/5).toFixed(1):'–';
 if(live&&now-lastFrame>5000)blank('Stale: no frame for '+Math.round((now-lastFrame)/1000)+' s (slow link?)');},500);
$('scale').onchange=connect;connect();
</script></html>"""
