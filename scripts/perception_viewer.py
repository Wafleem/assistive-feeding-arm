#!/usr/bin/env python3
"""Live view of what perception sees: the person's mouth (MediaPipe) and the bowls.

Sources:
  --source viam          frames from the Viam camera (default webcam-2 on orin-dev-kit),
                         bowls from the Orin's YOLOv8 vision service (spoon-detector).
                         Needs VIAM_ADDRESS / VIAM_API_KEY / VIAM_API_KEY_ID (env or .env).
  --source local:0       a webcam on this computer (no Viam); bowls only with --yolo.
  --source file:img.jpg  one still image (add --out to save the overlay instead of a window).

Overlay: lips + face outline, mouth OPEN/closed and ratio, distance, the ready gate's
reason, the mouth position in the base frame; every bowl with its track id, radius and
base-frame position. Press q to quit.

Install: pip install -e ".[face,viam]" (+ ".[yolo]" for --yolo). Model file:
data/models/face_landmarker.task (scripts/fetch_models.sh).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from assistive_arm.food_perception import BowlFinder, BowlParams, BowlTracker  # noqa: E402
from assistive_arm.geometry import Pose  # noqa: E402
from assistive_arm.mouth_perception import MediaPipeMouthDetector, MouthTracker, ReadyGate  # noqa: E402
from assistive_arm.mouth_perception import face_geometry as fg  # noqa: E402
from assistive_arm.vision.types import CameraIntrinsics, RGBDFrame  # noqa: E402

FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152,
             148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10]
LIPS_INNER = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78]


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            if v.strip():
                os.environ.setdefault(k.strip(), v.strip())


class LocalRig:
    """A webcam or image file on this computer, as a camera-only robot (camera frame = base frame)."""

    def __init__(self, source: str, name: str, fx: float):
        import cv2

        self.cv2, self.name = cv2, name
        self.still = None
        if source.startswith("file:"):
            self.still = cv2.cvtColor(cv2.imread(source[5:]), cv2.COLOR_BGR2RGB)
        else:
            self.cap = cv2.VideoCapture(int(source.split(":", 1)[1]))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.fx = fx
        self.pose_error = "local camera: positions are in the camera frame"

    def now(self):
        return time.monotonic()

    def get_frames(self, names=None):
        if self.still is not None:
            rgb = self.still
        else:
            ok, bgr = self.cap.read()
            if not ok:
                return {}
            rgb = self.cv2.cvtColor(bgr, self.cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        k = CameraIntrinsics(self.fx * w / 640, self.fx * w / 640, w / 2, h / 2, w, h)
        return {self.name: RGBDFrame(self.name, rgb, np.zeros((h, w), np.float32), k, timestamp=self.now())}

    def camera_pose(self, name):
        return Pose()


class OneFrame:
    """Wraps a rig so every detector in one loop iteration sees the same frame and pose
    (one camera grab and one frame-system query per loop, not one per detector)."""

    def __init__(self, rig, frames):
        self.rig, self.frames, self._pose = rig, frames, {}

    def now(self):
        return self.rig.now()

    def get_frames(self, names=None):
        return {k: v for k, v in self.frames.items() if names is None or k in names}

    def camera_pose(self, name):
        if name not in self._pose:
            self._pose[name] = self.rig.camera_pose(name)
        return self._pose[name]


def mm(p) -> str:
    return "(" + ", ".join(f"{v * 1000:.0f}" for v in p) + ") mm"


def draw(cv2, bgr, det, obs, state, gate_ok, gate, bowls, fps, note):
    h, w = bgr.shape[:2]
    lm = det.last_landmarks if det is not None else None
    if lm is not None:
        px = (lm[:, :2] * [w, h]).astype(int)
        cv2.polylines(bgr, [px[FACE_OVAL]], False, (200, 200, 200), 1, cv2.LINE_AA)
        colour = (0, 220, 0) if (state is not None and state.is_open) else (0, 140, 255)
        cv2.polylines(bgr, [px[LIPS_INNER]], True, colour, 2, cv2.LINE_AA)
        c = ((px[fg.UPPER_INNER_LIP] + px[fg.LOWER_INNER_LIP]) / 2).astype(int)
        cv2.drawMarker(bgr, tuple(int(v) for v in c), (0, 0, 255), cv2.MARKER_CROSS, 18, 2)
    lines = []
    if obs is not None and state is not None:
        m = det.last
        lines += [
            f"mouth {'OPEN' if state.is_open else 'closed'}  ratio {state.open_ratio:.2f}  conf {state.confidence:.2f}",
            f"distance {m.distance * 100:.0f} cm ({m.distance_source})  facing {m.frontalness:.2f}",
            f"mouth at {mm(state.position)}  speed {state.speed * 100:.1f} cm/s",
        ]
    else:
        lines.append("no face" if det is not None else "face detection off")
    lines.append(f"READY ({gate.reason})" if gate_ok else f"not ready: {gate.reason}")
    for b in bowls:
        x0, y0, x1, y1 = (int(v) for v in b.bbox)
        cv2.rectangle(bgr, (x0, y0), (x1, y1), (255, 160, 0), 2)
        cv2.putText(bgr, f"{b.label} #{b.id} r{b.radius * 100:.0f}cm", (x0, max(14, y0 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 160, 0), 1, cv2.LINE_AA)
        lines.append(f"{b.label} #{b.id}: {mm(b.position)} ({b.source}, {b.score:.2f})")
    lines.append(f"{fps:.1f} fps" + (f"  |  {note}" if note else ""))
    panel_w = max(cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0] for t in lines) + 16
    shade = bgr.copy()
    cv2.rectangle(shade, (0, 0), (panel_w, 10 + 18 * len(lines)), (0, 0, 0), -1)
    bgr = cv2.addWeighted(shade, 0.55, bgr, 0.45, 0)
    for i, text in enumerate(lines):
        cv2.putText(bgr, text, (8, 18 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return bgr


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="viam", help="viam | local:N | file:PATH")
    ap.add_argument("--camera", default="webcam-2")
    ap.add_argument("--bowl-service", default="spoon-detector", help="Viam vision service for bowls ('' = off)")
    ap.add_argument("--yolo", default="", help="run ultralytics locally instead, e.g. yolov8s.pt or yolov8s-worldv2.pt")
    ap.add_argument("--labels", default="bowl,cup", help="detector labels that count as bowls")
    ap.add_argument("--rim-height", type=float, default=0.04, help="m above the table (monocular bowl range)")
    ap.add_argument("--no-face", action="store_true")
    ap.add_argument("--model", default=str(ROOT / "data/models/face_landmarker.task"))
    ap.add_argument("--fx", type=float, default=780.0, help="focal length in px at 640x480 (local sources)")
    ap.add_argument("--out", default="", help="save the overlay to this file and exit (no window)")
    args = ap.parse_args()

    import cv2

    load_dotenv(ROOT / ".env")
    if args.source == "viam":
        from assistive_arm.robot_iface.viam import ViamCameraRig, ViamVisionDetector

        rig = ViamCameraRig(cameras=(args.camera,))
        bowl_det = None
        if args.bowl_service:
            try:
                bowl_det = ViamVisionDetector(rig, args.bowl_service, args.camera)
            except Exception as e:  # service not running: keep going with the face only
                print(f"bowls off: vision service {args.bowl_service!r} not available ({e})", file=sys.stderr)
    else:
        rig = LocalRig(args.source, args.camera, args.fx)
        bowl_det = None
    if args.yolo:
        from assistive_arm.vision.detectors import UltralyticsDetector

        bowl_det = UltralyticsDetector(args.yolo)

    det = None if args.no_face else MediaPipeMouthDetector(camera=args.camera, model_path=args.model)
    tracker, gate = MouthTracker(), ReadyGate()
    finder = BowlFinder(bowl_det, args.camera, BowlParams(labels=tuple(args.labels.split(",")),
                                                          rim_height=args.rim_height)) if bowl_det else None
    bowl_tracker = BowlTracker()
    last, fps = time.monotonic(), 0.0
    while True:
        frames = rig.get_frames([args.camera])
        frame = frames.get(args.camera)
        if frame is None:
            print("no frame", file=sys.stderr)
            time.sleep(0.2)
            continue
        one = OneFrame(rig, frames)
        obs = det.observe(one) if det else None
        tracker.update(obs)
        now = rig.now()
        state = tracker.state(now)
        gate_ok = gate.update(state, now)
        bowls = bowl_tracker.update(finder.find(one)) if finder else []
        t = time.monotonic()
        fps = 0.8 * fps + 0.2 / max(t - last, 1e-3)
        last = t
        note = getattr(rig, "pose_error", None)
        note = "no camera pose: camera-frame positions" if note else ""
        bgr = draw(cv2, cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR), det, obs, state, gate_ok, gate, bowls, fps, note)
        if args.out:
            cv2.imwrite(args.out, bgr)
            print(f"wrote {args.out}")
            return 0
        cv2.imshow("assistive-arm perception", bgr)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
