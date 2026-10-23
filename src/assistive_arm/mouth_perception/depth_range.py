"""Correct the fast monocular face range with metric depth (Depth Anything 3 on the Orin).

The 20 Hz face pipeline gets its range from the eye-corner span (``face_geometry``), which
assumes a population-average 90 mm span: a 10 % wrong span is a 10 % wrong distance, and
the error shows up as a height error too (the lips' ray is right, the point along it isn't).
Depth frames arrive at only a few Hz, so instead of replacing the fast path, each depth
frame measures this person's eye span: the face's depth (forehead, nose bridge, cheeks;
never the mouth) and the landmarks give it directly (``calibrate_eye_span``). A robust,
smoothed estimate then feeds back into ``FaceGeometryParams.eye_span_m``.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from ..vision.types import CameraIntrinsics
from . import face_geometry as fg


@dataclass
class DepthRangeConfig:
    min_span_m: float = 0.070  # plausible adult outer-eye-corner spans; outside = bad sample
    max_span_m: float = 0.115
    window: int = 15  # samples in the running median
    min_samples: int = 3  # before the estimate is used
    min_frontalness: float = 0.7  # side-on faces give poor spans


class DepthRangeCorrector:
    def __init__(self, cfg: DepthRangeConfig | None = None, prior_span_m: float = 0.090):
        self.cfg = cfg or DepthRangeConfig()
        self.prior = prior_span_m
        self.samples: deque = deque(maxlen=self.cfg.window)
        self.last: dict = {}

    def update(self, landmarks, k: CameraIntrinsics, depth: np.ndarray, params: fg.FaceGeometryParams) -> bool:
        """One depth frame with its face landmarks. Returns True if the sample was accepted."""
        px = fg.landmarks_to_pixels(landmarks, k)
        z_depth = fg.face_depth_from_image(px, depth, params)
        if z_depth is None:
            self.last = {"accepted": False, "why": "no depth at the face"}
            return False
        m = fg.measure_face(landmarks, k, None, fg.FaceGeometryParams(eye_span_m=self.prior))
        span = fg.calibrate_eye_span(landmarks, k, z_depth)
        ok = m.frontalness >= self.cfg.min_frontalness and self.cfg.min_span_m <= span <= self.cfg.max_span_m
        self.last = {"accepted": bool(ok), "depth_cm": round(z_depth * 100, 1),
                     "eye_span_estimate_cm": round(m.distance * 100, 1), "span_mm": round(span * 1000, 1),
                     "why": "" if ok else ("face turned" if m.frontalness < self.cfg.min_frontalness else "implausible span")}
        if ok:
            self.samples.append(span)
        return ok

    @property
    def ready(self) -> bool:
        return len(self.samples) >= self.cfg.min_samples

    @property
    def eye_span_m(self) -> float:
        """This person's eye span (m): the running median once there are enough samples."""
        return float(np.median(self.samples)) if self.ready else self.prior
