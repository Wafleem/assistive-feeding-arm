"""Find and track the bowls (food containers) on the table, as they move.

Detection: any ``Detector`` (vision/detectors.py). The default is YOLOv8 COCO,
which already knows "bowl" and "cup"; YOLO-World / OWLv2 for anything else
("ramekin", "small white dish").

3D from one RGB camera (the C270 has no depth): a bowl rests on the table, so
its rim centre lies on the plane z = table_z + rim_height. The ray through the
box's centre pixel, intersected with that plane, gives its position; the box
width at that range gives its radius. With a depth camera, the median depth of
the box centre is used instead (``depth_first=True``).

``BowlTracker`` keeps ids stable across frames (nearest-neighbour association
within ``gate`` metres, exponential smoothing, dropped after ``max_missed``
frames), so "the left bowl" stays the same bowl while someone slides it.

TODO(food_perception): what is in each bowl (crop -> VLM / open-vocab query,
vlm.py); per-bowl rim height from the props list (docs/PROPS.md).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..vision.detectors import Detector
from ..vision.geometry import bbox_mask, mask_points, pixel_ray, ray_plane_intersection
from ..vision.types import Detection, RGBDFrame


@dataclass
class BowlParams:
    labels: tuple[str, ...] = ("bowl", "cup")  # detector queries that count as a container
    min_score: float = 0.35
    table_z: float = 0.0  # m, table top in the base frame (AGENTS.md: z = 0 at the table)
    rim_height: float = 0.04  # m above the table where the box centre is assumed to lie. TODO: per prop
    depth_first: bool = True  # use depth when the camera has it
    dedupe_iou: float = 0.5  # "bowl" and "cup" boxes on the same object -> keep the best
    max_range: float = 1.5  # m


@dataclass
class Bowl:
    label: str
    position: np.ndarray  # (3,) rim centre, base frame
    radius: float  # m, from the box width
    score: float
    camera: str
    bbox: tuple[float, float, float, float]
    source: str  # "depth" | "table_plane"
    id: int = -1  # set by BowlTracker
    extras: dict = field(default_factory=dict)


def iou(a, b) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


class BowlFinder:
    def __init__(self, detector: Detector, camera: str = "webcam-2", params: BowlParams | None = None):
        self.detector, self.camera, self.p = detector, camera, params or BowlParams()

    def detections(self, frame: RGBDFrame) -> list[Detection]:
        dets = [d for q in self.p.labels for d in self.detector.detect(frame, q) if d.score >= self.p.min_score]
        dets.sort(key=lambda d: d.score, reverse=True)
        kept: list[Detection] = []
        for d in dets:
            if all(iou(d.bbox, k.bbox) < self.p.dedupe_iou for k in kept):
                kept.append(d)
        return kept

    def locate(self, frame: RGBDFrame, det: Detection, cam_pose) -> Bowl | None:
        k = frame.intrinsics
        x0, y0, x1, y1 = det.bbox
        u, v = (x0 + x1) / 2, (y0 + y1) / 2
        p_cam, source = None, "table_plane"
        if self.p.depth_first and frame.depth is not None and np.any(frame.depth > 0):
            depth = np.where(frame.depth < self.p.max_range, frame.depth, 0)
            pts = mask_points(depth, bbox_mask(depth.shape, det.bbox, 0.4), k)
            if len(pts) >= 12:
                p_cam, source = np.median(pts, axis=0), "depth"
        if p_cam is not None:
            p_base = cam_pose.transform_point(p_cam)
        else:
            ray = cam_pose.transform_vector(pixel_ray(u, v, k))
            plane_z = self.p.table_z + self.p.rim_height
            p_base = ray_plane_intersection(cam_pose.position, ray, [0, 0, plane_z], [0, 0, 1])
            if p_base is None:
                return None  # the ray never reaches the table (camera looking up)
        rng = float(np.linalg.norm(p_base - cam_pose.position))
        radius = 0.5 * (x1 - x0) * rng / k.fx
        return Bowl(det.label, p_base, float(radius), det.score, frame.camera, det.bbox, source)

    def find(self, robot) -> list[Bowl]:
        """Every bowl the camera sees now, best first."""
        frame = robot.get_frames([self.camera]).get(self.camera)
        if frame is None:
            return []
        cam = robot.camera_pose(self.camera)
        out = [b for d in self.detections(frame) if (b := self.locate(frame, d, cam)) is not None]
        for b in out:
            b.extras["t"] = frame.timestamp or robot.now()
        return out


@dataclass
class TrackerConfig:
    gate: float = 0.06  # m: farther than this from every track = a new bowl
    alpha: float = 0.5  # smoothing of the position (1 = no smoothing)
    max_missed: int = 10  # frames before a track is dropped


@dataclass
class _Track:
    bowl: Bowl
    missed: int = 0
    hits: int = 1


class BowlTracker:
    def __init__(self, config: TrackerConfig | None = None):
        self.c = config or TrackerConfig()
        self.tracks: dict[int, _Track] = {}
        self._next = 0

    def update(self, bowls: list[Bowl]) -> list[Bowl]:
        """Associate this frame's bowls with existing tracks (greedy nearest first)."""
        pairs = sorted(
            (float(np.linalg.norm(t.bowl.position - b.position)), tid, j)
            for tid, t in self.tracks.items() for j, b in enumerate(bowls)
        )
        used_t, used_b = set(), set()
        for d, tid, j in pairs:
            if d > self.c.gate or tid in used_t or j in used_b:
                continue
            t, b = self.tracks[tid], bowls[j]
            b.position = (1 - self.c.alpha) * t.bowl.position + self.c.alpha * b.position
            b.radius = (1 - self.c.alpha) * t.bowl.radius + self.c.alpha * b.radius
            b.id = tid
            t.bowl, t.missed, t.hits = b, 0, t.hits + 1
            used_t.add(tid)
            used_b.add(j)
        for tid in list(self.tracks):
            if tid not in used_t:
                self.tracks[tid].missed += 1
                if self.tracks[tid].missed > self.c.max_missed:
                    del self.tracks[tid]
        for j, b in enumerate(bowls):
            if j not in used_b:
                b.id = self._next
                self.tracks[self._next] = _Track(b)
                self._next += 1
        return self.bowls()

    def bowls(self, min_hits: int = 1) -> list[Bowl]:
        """Current bowls (seen in at least ``min_hits`` frames), left to right in y."""
        out = [t.bowl for t in self.tracks.values() if t.hits >= min_hits]
        return sorted(out, key=lambda b: -b.position[1])
