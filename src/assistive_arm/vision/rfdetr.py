"""RGB-only spoon detection with a local RF-DETR Nano checkpoint.

PLAN: run the model on the Jetson and return image-space observations only.
Camera acquisition, previews, calibration and all robot actions belong outside
this backend. Importing this module never loads a model or contacts a service.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .types import Detection


@dataclass(frozen=True)
class RFDETRConfig:
    """Local checkpoint, inference device and minimum spoon confidence.

    ``weights_path`` must name an existing Nano checkpoint. Downloads are a
    separate deployment step. ``cuda`` selects the Jetson GPU; ``cpu`` is useful
    for diagnostics. The default threshold is 0.5, not a calibrated probability.
    """

    weights_path: str
    device: str = "cuda"
    threshold: float = 0.5

    def __post_init__(self) -> None:
        if not self.weights_path or not self.device.strip():
            raise ValueError("weights_path and device must be nonempty")
        if not np.isfinite(self.threshold) or not 0 <= self.threshold <= 1:
            raise ValueError("threshold must be finite and between 0 and 1")


class RFDETRSpoonDetector:
    """Return spoon boxes in source-image pixels, without inferred depth.

    RF-DETR takes RGB, unlike OpenCV's usual BGR arrays. Model loading is lazy
    and occurs once, on the first valid frame. One instance is intended for one
    inference worker.
    """

    def __init__(self, config: RFDETRConfig):
        self.config = config
        self._model = None

    def _load_model(self):
        if self._model is None:
            weights = Path(self.config.weights_path).expanduser().resolve()
            if not weights.is_file():
                raise FileNotFoundError(f"RF-DETR Nano checkpoint not found: {weights}")
            from rfdetr import RFDETRNano

            self._model = RFDETRNano(
                pretrain_weights=str(weights), device=self.config.device,
            )
        return self._model

    def detect_rgb(self, rgb: np.ndarray) -> list[Detection]:
        """Detect spoons in a nonempty H×W×3 uint8 RGB frame.

        Invalid model rows are discarded; a malformed output structure raises
        an error so the preview can show a failed detector rather than a false
        "no spoon" result. Boxes are clipped to the image extent.
        """
        if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8:
            raise ValueError("rgb must be a uint8 NumPy array")
        if rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) == 0:
            raise ValueError("rgb must have nonempty shape (height, width, 3)")

        model = self._load_model()
        prediction = model.predict(np.ascontiguousarray(rgb), threshold=self.config.threshold)
        boxes = np.asarray(prediction.xyxy, dtype=float)
        if boxes.ndim != 2 or boxes.shape[1] != 4:
            raise ValueError("RF-DETR returned malformed xyxy boxes")
        count = len(boxes)
        if count == 0:
            return []
        scores = np.asarray(prediction.confidence, dtype=float)
        class_ids = np.asarray(prediction.class_id)
        if scores.shape != (count,) or class_ids.shape != (count,):
            raise ValueError("RF-DETR returned mismatched confidence or class IDs")

        # New RF-DETR versions attach checkpoint-resolved names per detection.
        # Older versions expose an ID->name dict. Do not index the newer compact
        # class_names list: COCO's sparse IDs differ (spoon is not YOLO's ID 44).
        names = getattr(prediction, "data", {}).get("class_name")
        if names is not None:
            names = np.asarray(names, dtype=object)
            if names.shape != (count,):
                raise ValueError("RF-DETR returned mismatched class names")
        else:
            mapping = model.class_names
            if not isinstance(mapping, Mapping):
                raise ValueError("RF-DETR must provide per-detection names or an ID-to-name mapping")
            names = [mapping.get(class_id, "") for class_id in class_ids]

        height, width = rgb.shape[:2]
        out = []
        for box, score, class_id, name in zip(boxes, scores, class_ids, names):
            if not isinstance(class_id, (int, np.integer)) or isinstance(class_id, (bool, np.bool_)):
                continue
            if not isinstance(name, str) or name.strip().casefold() != "spoon":
                continue
            if not np.isfinite(score) or not self.config.threshold <= score <= 1:
                continue
            if not np.isfinite(box).all():
                continue
            clipped = np.clip(box, (0, 0, 0, 0), (width, height, width, height))
            x0, y0, x1, y1 = (float(value) for value in clipped)
            if x1 <= x0 or y1 <= y0:
                continue
            out.append(Detection(
                label="spoon", score=float(score), bbox=(x0, y0, x1, y1),
                extras={"model": "rf-detr-nano", "class_id": int(class_id)},
            ))
        return out
