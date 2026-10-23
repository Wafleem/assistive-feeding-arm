"""Face landmarks -> mouth position, face normal and mouth opening (pure numpy).

Works with any landmark source in the MediaPipe Face Mesh topology (478 points,
``x, y`` normalised to the image, ``z`` relative depth in units of image width,
smaller = closer to the camera). Kept separate from the MediaPipe wrapper so it
is testable offline with synthetic faces.

Distance to the face (the webcam has no depth):
* with a depth image: the median depth over stable face points (forehead, nose
  bridge, cheeks). Never the mouth itself, which may be open: depth inside it is
  the mouth cavity, not the lips.
* monocular: the 3D distance between the outer eye corners is ~90 mm for adults
  and does not change when the head turns; ``z = f * eye_span_m / span_px``.
  Expect ~5-10% range error from the population spread; ``calibrate_eye_span``
  fixes it per person with one measured distance.
All outputs are in the camera frame (OpenCV axes: x right, y down, z forward), metres.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..vision.types import CameraIntrinsics

# MediaPipe Face Mesh indices
UPPER_INNER_LIP, LOWER_INNER_LIP = 13, 14
MOUTH_CORNER_L, MOUTH_CORNER_R = 78, 308  # inner corners
EYE_OUTER_R, EYE_OUTER_L = 33, 263
FACE_PLANE = (10, 151, 9, 168, 6, 234, 454, 93, 323, 132, 361, 152)  # forehead, nose bridge, cheeks, jaw, chin
DEPTH_POINTS = (10, 151, 9, 168, 6, 234, 454)  # stable, never inside the mouth


@dataclass
class FaceGeometryParams:
    eye_span_m: float = 0.090  # outer eye corner to outer eye corner; adult average (~85-100 mm)
    mar_closed: float = 0.05  # mouth aspect ratio with lips together. TODO: per-person calibration
    mar_open: float = 0.55  # mouth aspect ratio "open wide"
    depth_patch_px: int = 3  # half-size of the patch sampled around each depth point
    max_range: float = 1.5  # m


@dataclass
class FaceMeasurement:
    position: np.ndarray  # (3,) lips centre, camera frame, m
    normal: np.ndarray  # (3,) unit, out of the face toward the camera
    open_ratio: float  # 0..1 (normalised mouth aspect ratio)
    mar: float  # raw mouth aspect ratio
    distance: float  # m, camera to face
    distance_source: str  # "depth" | "eye_span"
    frontalness: float  # 1 = facing the camera, 0 = side-on


def landmarks_to_pixels(lm: np.ndarray, k: CameraIntrinsics) -> np.ndarray:
    """(N, 3) normalised landmarks -> (N, 3) pixels (u, v, z_px); z_px uses the width scale."""
    lm = np.asarray(lm, float)
    return np.column_stack([lm[:, 0] * k.width, lm[:, 1] * k.height, lm[:, 2] * k.width])


def mouth_aspect_ratio(px: np.ndarray) -> float:
    """Lip gap over mouth width (rotation and distance invariant)."""
    gap = np.linalg.norm(px[UPPER_INNER_LIP] - px[LOWER_INNER_LIP])
    width = np.linalg.norm(px[MOUTH_CORNER_L] - px[MOUTH_CORNER_R])
    return float(gap / width) if width > 1e-6 else 0.0


def normalise_mar(mar: float, p: FaceGeometryParams) -> float:
    return float(np.clip((mar - p.mar_closed) / (p.mar_open - p.mar_closed), 0.0, 1.0))


def face_depth_from_image(px: np.ndarray, depth: np.ndarray | None, p: FaceGeometryParams) -> float | None:
    """Median valid depth at the stable face points, or None (no depth camera / no returns)."""
    if depth is None or not np.any(depth > 0):
        return None
    h, w = depth.shape
    r = p.depth_patch_px
    vals = []
    for i in DEPTH_POINTS:
        u, v = int(round(px[i, 0])), int(round(px[i, 1]))
        if 0 <= u < w and 0 <= v < h:
            patch = depth[max(0, v - r) : v + r + 1, max(0, u - r) : u + r + 1]
            vals.extend(patch[(patch > 0) & (patch < p.max_range) & np.isfinite(patch)].ravel())
    return float(np.median(vals)) if len(vals) >= 5 else None


def face_depth_from_eye_span(px: np.ndarray, k: CameraIntrinsics, p: FaceGeometryParams) -> float:
    """Pinhole range from the 3D eye-corner span (rotation invariant: uses the landmark z too)."""
    span_px = float(np.linalg.norm(px[EYE_OUTER_L] - px[EYE_OUTER_R]))
    f = 0.5 * (k.fx + k.fy)
    return f * p.eye_span_m / max(span_px, 1e-6)


def calibrate_eye_span(lm: np.ndarray, k: CameraIntrinsics, measured_distance: float) -> float:
    """This person's eye span from one frame at a tape-measured distance (camera to eyes)."""
    px = landmarks_to_pixels(lm, k)
    span_px = float(np.linalg.norm(px[EYE_OUTER_L] - px[EYE_OUTER_R]))
    return measured_distance * span_px / (0.5 * (k.fx + k.fy))


def measure_face(lm: np.ndarray, k: CameraIntrinsics, depth: np.ndarray | None = None,
                 params: FaceGeometryParams | None = None) -> FaceMeasurement:
    """One face's landmarks (+ optional aligned depth) -> mouth pose in the camera frame."""
    p = params or FaceGeometryParams()
    px = landmarks_to_pixels(lm, k)
    z_face = face_depth_from_image(px, depth, p)
    source = "depth"
    if z_face is None:
        z_face, source = face_depth_from_eye_span(px, k, p), "eye_span"

    # Metric 3D landmarks: scale = metres per pixel at the face's distance.
    s = z_face / (0.5 * (k.fx + k.fy))
    ref_z = np.mean(px[list(DEPTH_POINTS), 2])  # landmark z is relative; anchor it at the depth points
    xyz = np.column_stack([(px[:, 0] - k.cx) * s, (px[:, 1] - k.cy) * s, z_face + (px[:, 2] - ref_z) * s])

    pts = xyz[list(FACE_PLANE)]
    c = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
    n = vt[-1]
    if np.dot(n, -c) < 0:  # out of the face = toward the camera
        n = -n
    lips = 0.5 * (xyz[UPPER_INNER_LIP] + xyz[LOWER_INNER_LIP])
    # An open mouth has no surface at the lips centre: put it on the lips' own line instead.
    lips[2] = 0.5 * (xyz[MOUTH_CORNER_L, 2] + xyz[MOUTH_CORNER_R, 2])
    mar = mouth_aspect_ratio(px)
    frontal = float(np.clip(np.dot(n, -c / np.linalg.norm(c)), 0.0, 1.0))
    return FaceMeasurement(lips, n, normalise_mar(mar, p), mar, float(z_face), source, frontal)
