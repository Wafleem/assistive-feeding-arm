"""Depth map -> point cloud, lidar-style ranges and Viam wire formats. Pure numpy.

Conventions (as in src/assistive_arm/geometry.py):
- SI units inside (m, rad). Only the wire formats change units: Viam point clouds and
  raw depth maps are millimetres.
- Camera frame is OpenCV's: +z out of the lens, +x right, +y down.
- ``depth`` is z-depth in metres, 0 = invalid. A *range* is the distance along the
  pixel's ray (what a lidar measures): range = depth * |(x/z, y/z, 1)|.
"""

from __future__ import annotations

import struct
from dataclasses import asdict, dataclass

import numpy as np

# DA3METRIC-LARGE predicts depth normalised by focal length: metres = focal_px * raw / 300
# (Depth Anything 3 README, "Monocular Metric Depth").
DA3_METRIC_FOCAL_SCALE = 300.0
VIAM_DEPTH_MAGIC = b"DEPTHMAP"


@dataclass(frozen=True)
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @classmethod
    def from_hfov(cls, hfov_deg: float, width: int, height: int) -> "Intrinsics":
        """Pinhole intrinsics from a horizontal field of view (square pixels, centred principal point)."""
        f = (width / 2.0) / np.tan(np.deg2rad(hfov_deg) / 2.0)
        return cls(f, f, width / 2.0, height / 2.0, width, height)

    def scaled(self, width: int, height: int) -> "Intrinsics":
        """The same camera after resizing its image to ``width`` x ``height``."""
        sx, sy = width / self.width, height / self.height
        return Intrinsics(self.fx * sx, self.fy * sy, self.cx * sx, self.cy * sy, width, height)

    @property
    def focal(self) -> float:
        return (self.fx + self.fy) / 2.0


def da3_metric_to_metres(raw: np.ndarray, k_processed: Intrinsics) -> np.ndarray:
    """DA3METRIC output -> metres. ``k_processed`` is the camera at the resolution the network saw."""
    return raw * (k_processed.focal / DA3_METRIC_FOCAL_SCALE)


def resize_nearest(img: np.ndarray, height: int, width: int) -> np.ndarray:
    """Nearest-neighbour resize. Depth must not be interpolated: blending a foreground and a
    background pixel invents points in mid-air."""
    h, w = img.shape[:2]
    if (h, w) == (height, width):
        return img
    rows = np.minimum(((np.arange(height) + 0.5) * h / height).astype(int), h - 1)
    cols = np.minimum(((np.arange(width) + 0.5) * w / width).astype(int), w - 1)
    return img[rows[:, None], cols[None, :]]


def _rays(k: Intrinsics, shape: tuple[int, int], stride: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Per-pixel x/z and y/z on a ``stride`` grid."""
    vs, us = np.mgrid[0 : shape[0] : stride, 0 : shape[1] : stride]
    return (us - k.cx) / k.fx, (vs - k.cy) / k.fy


def ranges_from_depth(depth: np.ndarray, k: Intrinsics) -> np.ndarray:
    """Range along each pixel's ray, metres (0 where depth is invalid)."""
    xr, yr = _rays(k, depth.shape)
    d = np.where(np.isfinite(depth) & (depth > 0), depth, 0.0)
    return d * np.sqrt(1.0 + xr * xr + yr * yr)


def depth_to_points(
    depth: np.ndarray, k: Intrinsics, *, stride: int = 1, min_range: float = 0.0, max_range: float = np.inf
) -> np.ndarray:
    """(N, 3) float32 points in metres, camera frame, from valid pixels on a ``stride`` grid."""
    d = depth[::stride, ::stride]
    xr, yr = _rays(k, depth.shape, stride)
    rng = d * np.sqrt(1.0 + xr * xr + yr * yr)
    ok = np.isfinite(d) & (d > 0) & (rng >= min_range) & (rng <= max_range)
    return np.stack([xr[ok] * d[ok], yr[ok] * d[ok], d[ok]], axis=1).astype(np.float32)


def points_to_pcd(points_m: np.ndarray) -> bytes:
    """Binary PCD in millimetres, the unit Viam's frame system and 3D scene use."""
    pts = np.ascontiguousarray(np.asarray(points_m, dtype=np.float32).reshape(-1, 3) * 1000.0)
    n = len(pts)
    header = (
        "VERSION .7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\n"
        f"WIDTH {n}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {n}\nDATA binary\n"
    )
    return header.encode("ascii") + pts.tobytes()


def encode_viam_depth(depth_m: np.ndarray) -> bytes:
    """Viam raw depth (``image/vnd.viam.dep``): magic, width and height as big-endian
    uint64, then big-endian uint16 millimetres per pixel (0 = invalid)."""
    ok = np.isfinite(depth_m) & (depth_m > 0)
    mm = np.clip(np.where(ok, np.round(depth_m * 1000.0), 0), 0, 65535).astype(">u2")
    h, w = mm.shape
    return VIAM_DEPTH_MAGIC + struct.pack(">QQ", w, h) + mm.tobytes()


def colorize(depth_m: np.ndarray, min_range: float, max_range: float) -> np.ndarray:
    """(H, W, 3) uint8 for viewing: near = red, far = blue (turbo colormap), invalid = black."""
    ok = np.isfinite(depth_m) & (depth_m > 0)
    t = 1.0 - np.clip((np.where(ok, depth_m, max_range) - min_range) / (max_range - min_range), 0.0, 1.0)
    # Polynomial fit of Google's turbo colormap (t = 0 dark blue -> t = 1 dark red).
    r = 0.13572138 + t * (4.61539260 + t * (-42.66032258 + t * (132.13108234 + t * (-152.94239396 + t * 59.28637943))))
    g = 0.09140261 + t * (2.19418839 + t * (4.84296658 + t * (-14.18503333 + t * (4.27729857 + t * 2.82956604))))
    b = 0.10667330 + t * (12.64194608 + t * (-60.58204836 + t * (110.36276771 + t * (-89.90310912 + t * 27.34824973))))
    rgb = (np.clip(np.stack([r, g, b], axis=-1), 0.0, 1.0) * 255).astype(np.uint8)
    rgb[~ok] = 0
    return rgb


@dataclass(frozen=True)
class RangeParams:
    min_range: float = 0.1  # m: closer returns are dropped (like a lidar's blind zone)
    max_range: float = 10.0  # m: farther returns are dropped
    center_roi: float = 0.1  # fraction of width/height around the principal point for distance_m
    scan_beams: int = 64  # beams across the horizontal field of view
    scan_band: float = 0.2  # fraction of the image height, centred on cy, that the scan looks through
    # Robust "first return": a return needs this many pixels at or closer than it, so single
    # flying pixels on depth edges are ignored but small objects (a 5x5 px spoon) still count.
    # Not a percentile: a percentile of the whole view would hide anything smaller than it.
    min_support_px: int = 25


@dataclass(frozen=True)
class RangeReadings:
    """What a 2D lidar plus a rangefinder would report. Angles in degrees, distances in metres;
    None where there is no valid return. Bearing is + to the right of the optical axis
    (camera +x), elevation is + up."""

    distance_m: float | None  # median range in the centre ROI (the "rangefinder")
    nearest_m: float | None  # robust nearest return anywhere in view
    nearest_bearing_deg: float | None
    nearest_elevation_deg: float | None
    scan_ranges_m: list  # one per beam, left to right; None = no return
    scan_angle_min_deg: float  # bearing of the first beam's centre
    scan_angle_increment_deg: float
    valid_fraction: float  # share of pixels with a usable depth

    def to_dict(self) -> dict:
        return asdict(self)


def range_readings(depth: np.ndarray, k: Intrinsics, p: RangeParams) -> RangeReadings:
    h, w = depth.shape
    xr, yr = _rays(k, depth.shape)
    rng = ranges_from_depth(depth, k)
    ok = (rng >= p.min_range) & (rng <= p.max_range)

    def first_return(values: np.ndarray) -> float | None:
        """The ``min_support_px``-th smallest range; None without that much support."""
        n = p.min_support_px
        return round(float(np.partition(values, n - 1)[n - 1]), 4) if values.size >= n else None

    # Rangefinder: median over a small window around the optical axis.
    hw, hh = max(1, round(w * p.center_roi / 2)), max(1, round(h * p.center_roi / 2))
    u0, v0 = int(round(k.cx)), int(round(k.cy))
    roi = (slice(max(0, v0 - hh), min(h, v0 + hh + 1)), slice(max(0, u0 - hw), min(w, u0 + hw + 1)))
    centre = rng[roi][ok[roi]]
    distance = round(float(np.median(centre)), 4) if centre.size else None

    nearest = first_return(rng[ok])
    bearing = elevation = None
    if nearest is not None:
        idx = np.flatnonzero(ok)[np.argmin(np.abs(rng[ok] - nearest))]
        x, y = xr.flat[idx], yr.flat[idx]
        bearing = round(float(np.degrees(np.arctan(x))), 2)
        elevation = round(float(np.degrees(np.arctan2(-y, np.hypot(1.0, x)))), 2)

    # Scan: bin the pixels of a horizontal band by bearing; each beam reports its first return.
    band = np.abs(np.arange(h) - k.cy) <= max(0.5, p.scan_band * h / 2)
    bearings = np.arctan(xr[0])  # per column
    lo, hi = bearings[0], bearings[-1]
    inc = (hi - lo) / p.scan_beams
    beam_of_col = np.minimum(((bearings - lo) / inc).astype(int), p.scan_beams - 1)
    band_rng, band_ok = rng[band], ok[band]
    scan = [first_return(band_rng[:, beam_of_col == b][band_ok[:, beam_of_col == b]]) for b in range(p.scan_beams)]

    return RangeReadings(
        distance_m=distance,
        nearest_m=nearest,
        nearest_bearing_deg=bearing,
        nearest_elevation_deg=elevation,
        scan_ranges_m=scan,
        scan_angle_min_deg=round(float(np.degrees(lo + inc / 2)), 3),
        scan_angle_increment_deg=round(float(np.degrees(inc)), 3),
        valid_fraction=round(float(ok.mean()), 4),
    )
