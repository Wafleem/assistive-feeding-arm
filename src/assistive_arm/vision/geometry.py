"""Pixels + depth -> 3D points, surface normals (OpenCV camera convention)."""

from __future__ import annotations

import numpy as np

from .types import CameraIntrinsics


def deproject(u: float, v: float, z: float, k: CameraIntrinsics) -> np.ndarray:
    return np.array([(u - k.cx) * z / k.fx, (v - k.cy) * z / k.fy, z])


def project(p_cam, k: CameraIntrinsics) -> tuple[float, float]:
    x, y, z = p_cam
    return k.fx * x / z + k.cx, k.fy * y / z + k.cy


def mask_points(depth: np.ndarray, mask: np.ndarray, k: CameraIntrinsics, max_points: int = 5000) -> np.ndarray:
    """3D points (camera frame) for valid-depth pixels inside ``mask``."""
    vs, us = np.nonzero(mask & (depth > 0) & np.isfinite(depth))
    if len(us) > max_points:
        idx = np.linspace(0, len(us) - 1, max_points).astype(int)
        us, vs = us[idx], vs[idx]
    z = depth[vs, us]
    return np.column_stack([(us - k.cx) * z / k.fx, (vs - k.cy) * z / k.fy, z])


def fit_plane(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares plane: (centroid, unit normal). Normal sign is arbitrary."""
    if len(points) < 3:
        raise ValueError("need at least 3 points")
    c = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points - c, full_matrices=False)
    return c, vt[-1]


def bbox_mask(shape, bbox, scale: float = 1.0) -> np.ndarray:
    h, w = shape
    x0, y0, x1, y1 = bbox
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    hw, hh = (x1 - x0) / 2 * scale, (y1 - y0) / 2 * scale
    m = np.zeros((h, w), dtype=bool)
    m[max(0, int(cy - hh)) : min(h, int(np.ceil(cy + hh)) + 1), max(0, int(cx - hw)) : min(w, int(np.ceil(cx + hw)) + 1)] = True
    return m


def robust_center(points: np.ndarray) -> np.ndarray:
    """Median point, robust to a few background pixels leaking into the mask."""
    return np.median(points, axis=0)


def pixel_ray(u: float, v: float, k: CameraIntrinsics) -> np.ndarray:
    """Unit direction (camera frame) of the ray through pixel (u, v)."""
    d = np.array([(u - k.cx) / k.fx, (v - k.cy) / k.fy, 1.0])
    return d / np.linalg.norm(d)


def ray_plane_intersection(origin, direction, plane_point, plane_normal) -> np.ndarray | None:
    """Where a ray meets a plane; None if parallel or behind the origin.

    Monocular 3D for things resting on a known surface (bowls on the table):
    the pixel's ray from the camera, intersected with the table plane."""
    o, d = np.asarray(origin, float), np.asarray(direction, float)
    n, p0 = np.asarray(plane_normal, float), np.asarray(plane_point, float)
    denom = float(np.dot(n, d))
    if abs(denom) < 1e-9:
        return None
    s = float(np.dot(n, p0 - o)) / denom
    return o + s * d if s > 0 else None
