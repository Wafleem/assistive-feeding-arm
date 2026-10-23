"""RGB-only real-face lip landmarks for an observation preview.

PLAN: load a local MediaPipe Tasks model, process each RGB image independently,
and report image geometry. No depth, robot pose, readiness or motion is inferred.
This is separate from the calibrated 3D ``MouthDetector`` contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from pathlib import Path

import numpy as np


# Ordered contours from MediaPipe's FACEMESH_LIPS topology. Rendering closes each
# loop; the first point is not repeated in the result.
_OUTER_LIP = (61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
              291, 409, 270, 269, 267, 0, 37, 39, 40, 185)
_INNER_LIP = (78, 95, 88, 178, 87, 14, 317, 402, 318, 324,
              308, 415, 310, 311, 312, 13, 82, 81, 80, 191)
_MOUTH_CORNERS = (61, 291)
_INNER_MIDPOINTS = (13, 14)
Point2D = tuple[float, float]


@dataclass(frozen=True)
class MouthLandmarkerConfig:
    """Local model and detection thresholds; CPU delegate, no model download."""

    model_path: str
    num_faces: int = 1
    min_face_detection_confidence: float = 0.5
    min_face_presence_confidence: float = 0.5
    min_mouth_width_px: float = 1.0

    def __post_init__(self) -> None:
        if isinstance(self.num_faces, bool) or not isinstance(self.num_faces, int) or self.num_faces < 1:
            raise ValueError("num_faces must be a positive integer")
        for name in ("min_face_detection_confidence", "min_face_presence_confidence"):
            value = getattr(self, name)
            if not isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1")
        if not isfinite(self.min_mouth_width_px) or self.min_mouth_width_px <= 0:
            raise ValueError("min_mouth_width_px must be finite and positive")


@dataclass(frozen=True)
class MouthLandmarks2D:
    """Image measurements only, in pixels unless indicated otherwise.

    ``face_index`` is this frame's result order, not an identity or tracking ID.
    ``timestamp_s`` is the caller's capture timestamp (or None when unknown).
    ``aperture_ratio`` is raw inner-lip separation / outer-corner width. It is
    neither a calibrated 0..1 openness score nor a readiness/safety decision.
    Coordinates retain the model output and can lie outside the image border.
    """

    face_index: int
    timestamp_s: float | None
    outer_lip_px: tuple[Point2D, ...]
    inner_lip_px: tuple[Point2D, ...]
    bbox_px: tuple[float, float, float, float]
    center_px: Point2D
    aperture_ratio: float


class MediaPipeMouthLandmarker:
    """Lazy optional MediaPipe backend accepting uint8 HWC RGB arrays.

    IMAGE mode intentionally makes no temporal tracking assumptions. This class
    receives images; it does not acquire them or connect to robot hardware.
    """

    def __init__(self, config: MouthLandmarkerConfig):
        model_path = Path(config.model_path).expanduser()
        if not model_path.is_file():
            raise FileNotFoundError(f"Face Landmarker model is not a file: {model_path}")

        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        self.config = config
        self._mp = mp
        options = vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(
                model_asset_path=str(model_path), delegate=python.BaseOptions.Delegate.CPU,
            ),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=config.num_faces,
            min_face_detection_confidence=config.min_face_detection_confidence,
            min_face_presence_confidence=config.min_face_presence_confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._closed = False

    def detect_rgb(
        self, rgb: np.ndarray, *, timestamp_s: float | None = None,
    ) -> tuple[MouthLandmarks2D, ...]:
        """Return each usable face's lips; no face returns an empty tuple."""
        if self._closed:
            raise RuntimeError("Face Landmarker is closed")
        if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8:
            raise ValueError("rgb must be a uint8 numpy array")
        if rgb.ndim != 3 or rgb.shape[2] != 3 or not rgb.shape[0] or not rgb.shape[1]:
            raise ValueError("rgb must have nonempty HWC shape with 3 RGB channels")
        if timestamp_s is not None and not isfinite(timestamp_s):
            raise ValueError("timestamp_s must be finite or None")
        height, width = rgb.shape[:2]
        image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(rgb),
        )
        result = self._landmarker.detect(image)
        observations = []
        for face_index, face in enumerate(result.face_landmarks):
            observation = self._observation(face, face_index, width, height, timestamp_s)
            if observation is not None:
                observations.append(observation)
        return tuple(observations)

    def _observation(self, face, face_index, width, height, timestamp_s):
        indices = _OUTER_LIP + _INNER_LIP
        if len(face) <= max(indices):
            return None
        points = {i: np.array((face[i].x * width, face[i].y * height), dtype=float) for i in indices}
        all_points = np.stack(tuple(points.values()))
        if not np.isfinite(all_points).all():
            return None
        corner_a, corner_b = (points[i] for i in _MOUTH_CORNERS)
        mouth_width = float(np.linalg.norm(corner_b - corner_a))
        if mouth_width < self.config.min_mouth_width_px:
            return None
        upper, lower = (points[i] for i in _INNER_MIDPOINTS)
        center = (upper + lower) / 2
        lo, hi = all_points.min(axis=0), all_points.max(axis=0)
        return MouthLandmarks2D(
            face_index=face_index,
            timestamp_s=timestamp_s,
            outer_lip_px=tuple((float(points[i][0]), float(points[i][1])) for i in _OUTER_LIP),
            inner_lip_px=tuple((float(points[i][0]), float(points[i][1])) for i in _INNER_LIP),
            bbox_px=(float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1])),
            center_px=(float(center[0]), float(center[1])),
            aperture_ratio=float(np.linalg.norm(lower - upper)) / mouth_width,
        )

    def close(self) -> None:
        """Release the native task once; safe to call repeatedly."""
        if not self._closed:
            self._landmarker.close()
            self._closed = True

    def __enter__(self) -> MediaPipeMouthLandmarker:
        if self._closed:
            raise RuntimeError("Face Landmarker is closed")
        return self

    def __exit__(self, *_exc) -> None:
        self.close()
