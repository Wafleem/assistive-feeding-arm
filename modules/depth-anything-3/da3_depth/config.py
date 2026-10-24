"""The camera's Viam attributes, parsed into a frozen dataclass with documented defaults."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Mapping, Optional

from .geometry import RangeParams

MODELS = ("depth-pro", "da3-metric")  # keep in step with estimator.DEFAULT_WEIGHTS

DEVICES = ("auto", "cuda", "mps", "cpu")


@dataclass(frozen=True)
class CameraConfig:
    source: str  # the RGB camera to read (e.g. "webcam-2")
    # "depth-pro" (Apple Depth Pro, default): accurate close up, ~1.6 s per frame on the Orin.
    # "da3-metric" (Depth Anything 3 metric): ~0.15 s per frame, but ~2.5x too far at 0.25 m;
    # trust it from about 0.5 m out. Measurements in estimator.py.
    model: str = "depth-pro"
    # None = the model's default (estimator.DEFAULT_WEIGHTS): a Hugging Face repo id, or a local
    # path (Depth Pro: the .pt file; DA3: a directory, a metric model such as DA3METRIC-LARGE).
    weights: Optional[str] = None
    device: str = "auto"  # auto = CUDA with a CUDA build of torch (required then), else MPS, else CPU
    # Where the model runs. "cloud" (default): Depth Pro on a cloud GPU (cloud/depth_server.py)
    # at cloud_url, an SSH tunnel to the instance; "local": on this machine. With cloud_fallback
    # the local model takes over while the cloud is unreachable (rechecked every 30 s).
    compute: str = "cloud"
    cloud_url: str = "http://127.0.0.1:8800"
    cloud_timeout_s: float = 3.0
    cloud_fallback: bool = True
    process_res: int = 504  # DA3 only: input size (longest side, px). Depth Pro always uses 1536
    produce_hz: float = 2.0  # background inference rate (an upper bound: one frame at a time)
    stale_after_s: float = 5.0  # point clouds and ranges older than this are refused, not served
    source_timeout_s: float = 5.0  # per get_images call on the source camera
    # Intrinsics of the source frames. Order of preference: fx/fy here, then the source
    # camera's own properties, then hfov_deg, then (Depth Pro only) its own focal estimate.
    # Metric depth scales with the focal length, so a 5 % focal error is a 5 % distance
    # error: calibrate when it matters.
    fx: float | None = None
    fy: float | None = None
    cx: float | None = None
    cy: float | None = None
    hfov_deg: float | None = None
    stride: int = 2  # point-cloud pixel subsampling (2 = every other row and column)
    # Read-only preview page on 127.0.0.1:<preview_port> (0 = off). With vision_status_url set
    # (e.g. http://127.0.0.1:8765/status.json) it also shows how far away vision_preview's
    # mouths and spoons are.
    preview_port: int = 0
    vision_status_url: str = ""
    ranges: RangeParams = field(default_factory=RangeParams)

    @classmethod
    def from_attributes(cls, attrs: Mapping[str, Any]) -> "CameraConfig":
        """Validate Viam attributes. Unknown keys are errors, so typos don't pass silently."""
        range_keys = {"min_range_m": "min_range", "max_range_m": "max_range", "center_roi": "center_roi",
                      "scan_beams": "scan_beams", "scan_band": "scan_band", "min_support_px": "min_support_px"}
        own = {f.name for f in fields(cls)} - {"ranges"}
        unknown = set(attrs) - own - set(range_keys)
        if unknown:
            raise ValueError(f"unknown attributes {sorted(unknown)}; known: {sorted(own | set(range_keys))}")
        source = str(attrs.get("source") or "").strip()
        if not source:
            raise ValueError('"source" (the RGB camera name) is required')

        def num(key: str, kind=float):
            return None if attrs.get(key) is None else kind(attrs[key])

        kw = {k: num(k) for k in ("fx", "fy", "cx", "cy", "hfov_deg", "produce_hz", "stale_after_s", "source_timeout_s",
                                  "cloud_timeout_s")}
        kw.update(process_res=num("process_res", int), stride=num("stride", int), preview_port=num("preview_port", int))
        kw = {k: v for k, v in kw.items() if v is not None}
        if attrs.get("cloud_fallback") is not None:
            kw["cloud_fallback"] = bool(attrs["cloud_fallback"])
        for k in ("model", "weights", "device", "vision_status_url", "compute", "cloud_url"):
            if attrs.get(k) is not None:
                kw[k] = str(attrs[k]).strip()
        rp = {range_keys[k]: (int if k in ("scan_beams", "min_support_px") else float)(attrs[k])
              for k in range_keys if attrs.get(k) is not None}
        cfg = cls(source=source, ranges=RangeParams(**rp), **kw)
        cfg._check()
        return cfg

    def _check(self) -> None:
        r = self.ranges
        problems = [
            (self.model not in MODELS, f"model must be one of {MODELS}"),
            (self.compute not in ("cloud", "local"), "compute must be cloud or local"),
            (self.compute == "cloud" and self.model != "depth-pro", "the cloud server runs depth-pro only"),
            (self.compute == "cloud" and not self.cloud_url.startswith("http"), "cloud_url must be an http(s) URL"),
            (self.cloud_timeout_s <= 0, "cloud_timeout_s must be > 0"),
            (self.device not in DEVICES, f"device must be one of {DEVICES}"),
            ((self.fx is None) != (self.fy is None), "give fx and fy together"),
            ((self.cx is None) != (self.cy is None), "give cx and cy together"),
            (self.cx is not None and self.fx is None, "cx/cy need fx/fy"),
            (self.hfov_deg is not None and not 1 <= self.hfov_deg < 179, "hfov_deg must be in [1, 179)"),
            (not 56 <= self.process_res <= 2016, "process_res must be in [56, 2016]"),
            (self.produce_hz <= 0, "produce_hz must be > 0"),
            (self.stale_after_s <= 0 or self.source_timeout_s <= 0, "timeouts must be > 0"),
            (self.stride < 1, "stride must be >= 1"),
            (not 0 <= self.preview_port <= 65535, "preview_port must be in [0, 65535]"),
            (not 0 <= r.min_range < r.max_range, "need 0 <= min_range_m < max_range_m"),
            (not 0 < r.center_roi <= 1 or not 0 < r.scan_band <= 1, "center_roi and scan_band are fractions in (0, 1]"),
            (r.scan_beams < 1, "scan_beams must be >= 1"),
            (r.min_support_px < 1, "min_support_px must be >= 1"),
        ]
        bad = [msg for failed, msg in problems if failed]
        if bad:
            raise ValueError("; ".join(bad))
