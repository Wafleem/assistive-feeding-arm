"""Depth Anything 3 metric depth for one RGB frame. torch and depth_anything_3 are imported lazily."""

from __future__ import annotations

import importlib
import logging
import os
import sys
import types
import warnings
from typing import Protocol

import numpy as np

from .geometry import Intrinsics, da3_metric_to_metres, resize_nearest

LOGGER = logging.getLogger(__name__)
SKY_THRESHOLD = 0.3  # DA3's own threshold for its sky head (model/da3.py, compute_sky_mask)


class DepthEstimator(Protocol):
    device: str

    def load(self) -> None: ...

    def predict(self, rgb: np.ndarray, k: Intrinsics) -> np.ndarray:
        """(H, W) metric z-depth in metres for an (H, W, 3) uint8 RGB frame, 0 = invalid."""
        ...


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

    def __init__(self, weights: str, device: str = "auto", process_res: int = 504):
        self.weights = weights
        self.requested_device = device
        self.process_res = process_res
        self.device = "unloaded"
        self.kind = ""
        self._model = None

    def load(self) -> None:
        os.environ.setdefault("DA3_LOG_LEVEL", "WARN")  # DA3 logs every step at INFO
        # viam-server logs a module's stderr at ERROR, so known-harmless warnings read as failures:
        # torch on the Orin ("CC 8.7 not supported": its sm_80 kernels run fine on sm_87) and
        # omegaconf's deprecation of the resolver API that DA3 calls at import.
        warnings.filterwarnings("ignore", message=r"\s*Found GPU\d+ Orin which is of compute capability")
        warnings.filterwarnings("ignore", message=r"register_new_resolver\(\) is deprecated")
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

    def predict(self, rgb: np.ndarray, k: Intrinsics) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("load() first")
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
