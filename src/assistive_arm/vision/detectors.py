"""Open-vocabulary detectors behind one interface.

``SegmentationOracleDetector`` reads the simulator's ground-truth geom
segmentation, so the rest of the pipeline (depth -> 3D -> normal -> skill)
can be developed and regression-tested before any model is downloaded.
The model-backed detectors are thin, lazily-imported wrappers (Hugging Face
``transformers``: OWLv2, Grounding DINO; ``ultralytics``: YOLOv8 / YOLO-World);
they are not exercised by the offline tests.

Open-vocabulary detectors are good at "grape" or "cup" and weaker at
fine-grained or state questions ("the ripe strawberry", "is the spoon
loaded?"). Those go to a VLM on the crop (``food_perception.vlm``).
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from .geometry import bbox_mask
from .types import Detection, RGBDFrame


class Detector(Protocol):
    def detect(self, frame: RGBDFrame, query: str) -> list[Detection]: ...


class SegmentationOracleDetector:
    """Ground truth from sim: ``label -> [instance geom ids, ...]`` (built by ``SimRobot``).

    Returns one detection per visible instance, like a real detector would.
    Labels are matched case-insensitively, and a trailing plural "s" is
    forgiven ("grapes" finds "grape")."""

    def __init__(self, label_to_instances: dict[str, list[list[int]]]):
        self.instances = {k.lower(): v for k, v in label_to_instances.items()}

    def known_queries(self) -> list[str]:
        return sorted(self.instances)

    def _lookup(self, query: str):
        q = query.lower().strip()
        return self.instances.get(q) or self.instances.get(q.rstrip("s")) or self.instances.get(q[:-2] if q.endswith("es") else q)

    def detect(self, frame: RGBDFrame, query: str) -> list[Detection]:
        if frame.segmentation is None:
            return []
        out = []
        for k, geoms in enumerate(self._lookup(query) or []):
            mask = np.isin(frame.segmentation, geoms)
            if mask.sum() < 4:
                continue  # occluded or out of view
            vs, us = np.nonzero(mask)
            bbox = (float(us.min()), float(vs.min()), float(us.max() + 1), float(vs.max() + 1))
            out.append(Detection(query, 1.0, bbox, mask, extras={"instance": k}))
        return out


class NoisyDetector:
    """Wraps a detector and degrades it like a real model: box jitter, dropped
    masks, missed detections. Use in sim to test the pipeline's tolerance."""

    def __init__(self, inner: Detector, jitter_px: float = 3.0, drop_mask: bool = True, miss_rate: float = 0.1, seed: int = 0):
        self.inner, self.jitter, self.drop_mask, self.miss = inner, jitter_px, drop_mask, miss_rate
        self.rng = np.random.default_rng(seed)

    def detect(self, frame: RGBDFrame, query: str) -> list[Detection]:
        out = []
        for d in self.inner.detect(frame, query):
            if self.rng.random() < self.miss:
                continue
            b = np.asarray(d.bbox) + self.rng.normal(0, self.jitter, 4)
            out.append(Detection(d.label, d.score * 0.9, tuple(b), None if self.drop_mask else d.mask))
        return out


class OwlV2Detector:
    """OWLv2 open-vocabulary boxes (google/owlv2-base-patch16-ensemble). CPU-feasible, ~1 s/frame."""

    def __init__(self, model_id: str = "google/owlv2-base-patch16-ensemble", threshold: float = 0.2):
        from transformers import Owlv2ForObjectDetection, Owlv2Processor

        self.processor = Owlv2Processor.from_pretrained(model_id)
        self.model = Owlv2ForObjectDetection.from_pretrained(model_id).eval()
        self.threshold = threshold

    def detect(self, frame: RGBDFrame, query: str) -> list[Detection]:
        import torch

        inputs = self.processor(text=[[query]], images=frame.rgb, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        # OWLv2 pads to a square; post-process against the padded size.
        side = max(frame.rgb.shape[:2])
        res = self.processor.post_process_object_detection(
            outputs, threshold=self.threshold, target_sizes=torch.tensor([[side, side]])
        )[0]
        return [
            Detection(query, float(s), tuple(float(x) for x in b))
            for s, b in zip(res["scores"], res["boxes"])
        ]


class GroundingDinoDetector:
    """Grounding DINO via transformers (IDEA-Research/grounding-dino-tiny). GPU recommended.
    Check the post-processing argument names against your transformers version."""

    def __init__(self, model_id: str = "IDEA-Research/grounding-dino-tiny", box_threshold=0.3, text_threshold=0.25):
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).eval()
        self.box_threshold, self.text_threshold = box_threshold, text_threshold

    def detect(self, frame: RGBDFrame, query: str) -> list[Detection]:
        import torch

        text = query.lower().rstrip(".") + "."
        inputs = self.processor(images=frame.rgb, text=text, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        h, w = frame.rgb.shape[:2]
        res = self.processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, threshold=self.box_threshold,
            text_threshold=self.text_threshold, target_sizes=[(h, w)],
        )[0]
        return [Detection(query, float(s), tuple(float(x) for x in b)) for s, b in zip(res["scores"], res["boxes"])]


class UltralyticsDetector:
    """YOLOv8 / YOLO11 (closed set, e.g. COCO: "bowl", "cup", "spoon", "person") or
    YOLO-World (open vocabulary: any phrase) through ``ultralytics``. Real-time on the
    AGX Orin (export to TensorRT with ``model.export(format="engine")`` for ~3x more).

    * closed-set weights (``yolov8s.pt``): ``query`` must be one of ``model.names``;
      unknown queries return [] (check ``known_queries()``).
    * YOLO-World weights (``yolov8s-worldv2.pt``): the queries become the vocabulary.
    One inference per frame, cached, however many queries are asked of it.
    Optional dependency: ``pip install -e ".[yolo]"``; weights download on first use.
    """

    def __init__(self, weights: str = "yolov8s.pt", conf: float = 0.25, device: str | None = None, imgsz: int = 640):
        from ultralytics import YOLO  # lazy: heavy optional dependency

        self.model = YOLO(weights)
        self.open_vocab = "world" in weights.lower()
        self.conf, self.device, self.imgsz = conf, device, imgsz
        self._vocab: tuple[str, ...] = ()
        self._cache: tuple[int, tuple, list[Detection]] | None = None

    def known_queries(self) -> list[str]:
        return list(self._vocab) if self.open_vocab else sorted(self.model.names.values())

    def set_vocabulary(self, queries) -> None:
        """YOLO-World only: the phrases to look for (re-encodes the text prompts)."""
        self._vocab = tuple(dict.fromkeys(q.lower().strip() for q in queries))
        self.model.set_classes(list(self._vocab))
        self._cache = None

    def _run(self, frame: RGBDFrame) -> list[Detection]:
        key = (id(frame.rgb), frame.timestamp)
        if self._cache and self._cache[:2] == (key, self._vocab):
            return self._cache[2]
        res = self.model.predict(frame.rgb[..., ::-1], conf=self.conf, device=self.device, imgsz=self.imgsz,
                                 verbose=False)[0]  # ultralytics expects BGR numpy arrays
        names = res.names
        dets = [
            Detection(names[int(c)].lower(), float(s), tuple(float(x) for x in b))
            for b, s, c in zip(res.boxes.xyxy.tolist(), res.boxes.conf.tolist(), res.boxes.cls.tolist())
        ]
        self._cache = (key, self._vocab, dets)
        return dets

    def detect(self, frame: RGBDFrame, query: str) -> list[Detection]:
        q = query.lower().strip()
        if self.open_vocab and q not in self._vocab:
            self.set_vocabulary((*self._vocab, q))
        return [Detection(query, d.score, d.bbox) for d in self._run(frame) if d.label == q]


def mask_or_box(det: Detection, shape, scale: float = 0.6) -> np.ndarray:
    """The detection's mask, or the central part of its box when no mask (no SAM yet)."""
    return det.mask if det.mask is not None else bbox_mask(shape, det.bbox, scale)
