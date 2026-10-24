"""Metric depth for one RGB frame: Depth Pro (default) or Depth Anything 3.

torch and the model packages are imported lazily. Measured on the Orin, 2026-10-24, on a
face ~0.23 m away (eye-corner span reference, same focal for all): Depth Pro 0.21 m,
MoGe-2 0.45 m, DA3METRIC 0.59 m. DA3 is ~6x faster (0.18 s vs 1.55 s per frame) but its
scale is only trustworthy at room scale (0.5 m+), like the benchmarks it was tested on.
"""

from __future__ import annotations

import importlib
import logging
import os
import sys
import types
import warnings
from typing import Optional, Protocol

import numpy as np

from .geometry import Intrinsics, da3_metric_to_metres, resize_nearest

LOGGER = logging.getLogger(__name__)
SKY_THRESHOLD = 0.3  # DA3's own threshold for its sky head (model/da3.py, compute_sky_mask)


DEFAULT_WEIGHTS = {
    "depth-pro": "apple/DepthPro",  # HF repo holding depth_pro.pt (1.9 GB), or a local .pt path
    "da3-metric": "depth-anything/DA3METRIC-LARGE",  # HF repo id or local directory
}
DEPTH_PRO_FILE = "depth_pro.pt"


class DepthEstimator(Protocol):
    device: str
    estimates_focal: bool  # True: predict() accepts k=None and sets focal_px itself
    focal_px: Optional[float]  # the focal length the last prediction used (px, at frame size)

    def load(self) -> None: ...

    def predict(self, rgb: np.ndarray, k: Optional[Intrinsics]) -> np.ndarray:
        """(H, W) metric z-depth in metres for an (H, W, 3) uint8 RGB frame, 0 = invalid."""
        ...


def make_estimator(cfg) -> "DepthEstimator":
    """The estimator a CameraConfig asks for."""
    weights = cfg.weights or DEFAULT_WEIGHTS[cfg.model]
    if cfg.model == "depth-pro":
        return DepthProEstimator(weights, cfg.device)
    return DA3Estimator(weights, cfg.device, cfg.process_res)


def _stub_export_modules() -> None:
    """depth_anything_3.api imports every export format (GLB, COLMAP, Gaussian video, pose
    alignment) at load time, which drags in pycolmap, trimesh, evo, moviepy, matplotlib and
    open3d. Inference never calls them, so if they're not installed, stand in empty modules."""
    for name in ("depth_anything_3.utils.export", "depth_anything_3.utils.pose_align"):
        try:
            importlib.import_module(name)
        except ImportError:
            stub = types.ModuleType(name)
            stub.__getattr__ = lambda attr, _n=name: _unavailable(_n, attr)
            sys.modules[name] = stub


def _unavailable(module: str, attr: str):
    if attr.startswith("__"):
        raise AttributeError(attr)

    def fail(*_a, **_k):
        raise RuntimeError(f"{module}.{attr} needs Depth Anything 3's full dependencies; inference doesn't")

    return fail


def _quiet_known_warnings() -> None:
    """viam-server logs a module's stderr at ERROR, so known-harmless warnings read as failures:
    torch on the Orin ("CC 8.7 not supported": its sm_80 kernels run fine on sm_87) and
    omegaconf's deprecation of the resolver API that DA3 calls at import."""
    warnings.filterwarnings("ignore", message=r"\s*Found GPU\d+ Orin which is of compute capability")
    warnings.filterwarnings("ignore", message=r"register_new_resolver\(\) is deprecated")


def pick_device(requested: str) -> str:
    """``auto``: CUDA with a CUDA build of torch, else MPS, else CPU.

    A CUDA build of torch is installed to use a GPU, so if that GPU is unusable ``auto``
    raises instead of falling back: on the Orin the CPU takes ~8 minutes per frame and
    starves teleop and viam-server (seen 2026-10-24 when the GPU failed to power on).
    Set ``device: cpu`` to run on the CPU on purpose."""
    import torch

    if requested != "auto":
        return requested
    if torch.version.cuda:
        try:
            x = torch.randn(1, 3, 16, 16, device="cuda")
            torch.nn.functional.conv2d(x, torch.randn(4, 3, 3, 3, device="cuda")).sum().item()
            return "cuda"
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"this torch is a CUDA build but the GPU is unusable ({e}); "
                               "set device: cpu to run on the CPU anyway") from e
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class DA3Estimator:
    """Wraps ``DepthAnything3.inference`` for metric models.

    PLAN
    1. Resize to ``process_res`` (DA3's upper_bound_resize: whole frame, no crop).
    2. DA3METRIC: metres = focal * raw / 300, with the focal at the resized resolution.
       DA3NESTED already outputs metres. Relative-depth models are refused.
    3. Drop sky pixels, then nearest-resize back to the source frame.
    """

    estimates_focal = False

    def __init__(self, weights: str, device: str = "auto", process_res: int = 504):
        self.weights = weights
        self.requested_device = device
        self.process_res = process_res
        self.device = "unloaded"
        self.kind = ""
        self.focal_px: Optional[float] = None
        self._model = None

    def load(self) -> None:
        os.environ.setdefault("DA3_LOG_LEVEL", "WARN")  # DA3 logs every step at INFO
        _quiet_known_warnings()
        _stub_export_modules()
        from depth_anything_3.api import DepthAnything3

        device = pick_device(self.requested_device)
        model = DepthAnything3.from_pretrained(self.weights).to(device).eval()
        name = getattr(model, "model_name", "")
        if name.startswith("da3metric"):
            self.kind = "metric"
        elif name.startswith("da3nested"):
            self.kind = "nested"
        else:
            raise ValueError(f"{self.weights} ({name}) predicts relative depth; use DA3METRIC-LARGE or DA3NESTED-*")
        self._model, self.device = model, device
        LOGGER.info("loaded %s (%s) on %s", self.weights, name, device)

    def predict(self, rgb: np.ndarray, k: Optional[Intrinsics]) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("load() first")
        if k is None:
            raise ValueError("Depth Anything 3 needs the focal length: set hfov_deg or fx/fy")
        self.focal_px = k.focal
        h, w = rgb.shape[:2]
        if (k.width, k.height) != (w, h):
            raise ValueError(f"intrinsics are for {k.width}x{k.height}, frame is {w}x{h}")
        pred = self._model.inference([rgb], process_res=self.process_res)
        depth = np.asarray(pred.depth[0], dtype=np.float32)
        ph, pw = depth.shape
        if self.kind == "metric":
            depth = da3_metric_to_metres(depth, k.scaled(pw, ph))
        sky = getattr(pred, "sky", None)
        if sky is not None:
            depth = np.where(np.asarray(sky[0]) > SKY_THRESHOLD, 0.0, depth).astype(np.float32)
        return resize_nearest(depth, h, w)


class DepthProEstimator:
    """Apple Depth Pro: metric depth from one image, with its own focal-length estimate.

    PLAN
    1. With intrinsics, pass fx (px at the frame's size) so the scale follows the
       calibration. Without them, Depth Pro estimates the focal itself (``focal_px``).
    2. It runs at a fixed 1536x1536 internally and returns depth at the frame's size.
    """

    estimates_focal = True

    def __init__(self, weights: str, device: str = "auto"):
        self.weights = weights
        self.requested_device = device
        self.device = "unloaded"
        self.focal_px: Optional[float] = None
        self._model = self._transform = self._torch = None

    def load(self) -> None:
        _quiet_known_warnings()
        import dataclasses

        import torch
        import depth_pro
        from depth_pro.depth_pro import DEFAULT_MONODEPTH_CONFIG_DICT

        path = self.weights
        if not os.path.isfile(path):
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(self.weights, DEPTH_PRO_FILE)
        device = pick_device(self.requested_device)
        precision = torch.half if device == "cuda" else torch.float32
        cfg = dataclasses.replace(DEFAULT_MONODEPTH_CONFIG_DICT, checkpoint_uri=path)
        model, transform = depth_pro.create_model_and_transforms(cfg, device=torch.device(device), precision=precision)
        self._model, self._transform, self._torch, self.device = model.eval(), transform, torch, device
        LOGGER.info("loaded Depth Pro (%s) on %s", path, device)

    def predict(self, rgb: np.ndarray, k: Optional[Intrinsics]) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("load() first")
        h, w = rgb.shape[:2]
        if k is not None and (k.width, k.height) != (w, h):
            raise ValueError(f"intrinsics are for {k.width}x{k.height}, frame is {w}x{h}")
        torch = self._torch
        f_px = None if k is None else torch.tensor(k.fx, device=self.device)  # Depth Pro pairs f with the width
        with torch.no_grad():
            out = self._model.infer(self._transform(rgb), f_px=f_px)
        self.focal_px = float(out["focallength_px"])
        depth = out["depth"].float().cpu().numpy()
        return np.where(np.isfinite(depth) & (depth > 0), depth, 0.0).astype(np.float32)
