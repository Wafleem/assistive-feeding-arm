"""Mouth detectors. One interface (``MouthDetector.observe(robot)``), several backends.

| Backend | Status | Use |
|---|---|---|
| ``MediaPipeMouthDetector`` | implemented | **the main path: a real person's face** (Face Landmarker, webcam or RGB-D) |
| ``SimOracleMouthDetector`` | TODO | sim ground truth (+ optional noise/dropouts/delay) |
| ``CameraSegMouthDetector`` | TODO | the vision path (mask + depth + face plane), testable on sim renders |
| ``FiducialMouthDetector`` | TODO, optional | ArUco tag on a printed face prop (rehearsal without a person) |

A printed mouth is always open: detectors for it report ``open_ratio = 1.0``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..vision.locate import Locator
from .face_geometry import FaceGeometryParams, FaceMeasurement, measure_face
from .types import MouthObservation


@dataclass
class OracleNoise:
    pos_sigma: float = 0.0  # m
    normal_sigma_deg: float = 0.0
    dropout: float = 0.0  # probability of no detection
    delay: float = 0.0  # s: report observations this old (camera + inference latency)


class SimOracleMouthDetector:
    """Ground truth from the MuJoCo scene: the "mouth" site (centre of the opening on the
    face plane, +z = face normal), open_ratio = 1.0 (the printed mouth is always open),
    with optional noise, dropouts and delay to rehearse real-detector behaviour.

    TODO(mouth_perception): implement with ``robot.site_pose("mouth")`` and ``robot.now()``.
    """

    def __init__(self, noise: OracleNoise | None = None, seed: int = 0):
        self.noise = noise or OracleNoise()
        self.rng = np.random.default_rng(seed)

    def observe(self, robot) -> MouthObservation | None:
        raise NotImplementedError("TODO(mouth_perception): SimOracleMouthDetector.observe")


class CameraSegMouthDetector:
    """The vision path: find "mouth" with any detector (sim: segmentation oracle) and use
    the "face" plane as the approach normal.

    Expected pitfall with the printed mouth: depth inside the opening is the catch cup
    behind it, several cm behind the face. Project the mouth centre onto the face plane
    before reporting it (verify in sim once the printed-mouth scene exists).

    TODO(mouth_perception): implement with ``Locator.locate(robot, "mouth", cameras,
    support_query="face")`` + the projection; confidence from the number of depth points;
    open_ratio from ``open_ratio_fn`` (1.0 for the printed mouth).
    """

    def __init__(self, locator: Locator, cameras=("scene", "wrist"), open_ratio_fn=None):
        self.locator, self.cameras = locator, list(cameras)
        self.open_ratio_fn = open_ratio_fn or (lambda robot: 1.0)

    def observe(self, robot) -> MouthObservation | None:
        raise NotImplementedError("TODO(mouth_perception): CameraSegMouthDetector.observe")


class MediaPipeMouthDetector:
    """Real faces: MediaPipe Face Landmarker (30+ fps on the AGX Orin, ~15 ms on a laptop CPU).

    observe(robot):
      1. RGB frame from ``camera`` (``robot.get_frames``); depth is used when the camera has it.
      2. Face Landmarker in VIDEO mode, one face (the closest/largest when several).
      3. ``face_geometry.measure_face``: lips centre, face normal (plane through forehead,
         nose bridge, cheeks, chin), mouth opening. Range from depth, else from the eye span.
      4. ``open_ratio``: the ``jawOpen`` blendshape when available (robust to lip shape),
         else the normalised mouth aspect ratio.
      5. Camera frame -> base frame with ``robot.camera_pose(camera)``.
      6. Confidence: 1.0 facing the camera, falling toward 0 as the head turns side-on
         (landmarks get unreliable there; the tracker ignores < 0.5).

    The model file (``face_landmarker.task``, ~4 MB) is not in git: download
    ``MODEL_URL`` into ``data/models/`` (``scripts/fetch_models.sh``).
    Pass ``landmarker`` to inject a fake in tests.
    TODO(mouth_perception): per-person calibration (eye span at a measured distance;
    "close" / "open wide" for the MAR range); run as a Viam vision service on the Orin.
    """

    MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
                 "face_landmarker/float16/1/face_landmarker.task")

    def __init__(self, camera: str = "webcam-2", model_path: str = "data/models/face_landmarker.task",
                 params: FaceGeometryParams | None = None, landmarker=None, use_blendshapes: bool = True,
                 min_frontalness: float = 0.3):
        self.camera, self.params, self.use_blendshapes = camera, params or FaceGeometryParams(), use_blendshapes
        self.min_frontalness = min_frontalness
        self.last: FaceMeasurement | None = None
        self.last_landmarks: np.ndarray | None = None  # (478, 3) normalised, for overlays
        self._last_ms = -1
        self._landmarker = landmarker or self._make_landmarker(model_path)

    def _make_landmarker(self, model_path: str):
        import mediapipe as mp  # optional dependency: pip install -e ".[face]"
        from mediapipe.tasks.python import BaseOptions, vision

        opts = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=2,
            output_face_blendshapes=self.use_blendshapes,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._mp = mp
        return vision.FaceLandmarker.create_from_options(opts)

    def landmarks(self, rgb: np.ndarray, t: float):
        """(landmarks (N, 3) normalised, blendshapes dict) of the largest face, or None."""
        ms = max(int(t * 1000), self._last_ms + 1)  # VIDEO mode needs increasing timestamps
        self._last_ms = ms
        if hasattr(self, "_mp"):
            image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
            res = self._landmarker.detect_for_video(image, ms)
        else:
            res = self._landmarker.detect_for_video(rgb, ms)
        faces = getattr(res, "face_landmarks", None) or []
        if not faces:
            return None
        arrs = [np.array([[l.x, l.y, l.z] for l in f]) if not isinstance(f, np.ndarray) else f for f in faces]
        i = int(np.argmax([np.ptp(a[:, 0]) * np.ptp(a[:, 1]) for a in arrs]))  # largest = closest
        shapes = {}
        bs = getattr(res, "face_blendshapes", None) or []
        if i < len(bs):
            shapes = {c.category_name: float(c.score) for c in bs[i]}
        return arrs[i], shapes

    def observe(self, robot) -> MouthObservation | None:
        frame = robot.get_frames([self.camera]).get(self.camera)
        if frame is None:
            return None
        t = frame.timestamp or robot.now()
        found = self.landmarks(frame.rgb, t)
        if found is None:
            self.last, self.last_landmarks = None, None
            return None
        lm, shapes = found
        self.last_landmarks = lm
        m = measure_face(lm, frame.intrinsics, frame.depth, self.params)
        self.last = m
        if m.frontalness < self.min_frontalness:
            return None
        open_ratio = shapes.get("jawOpen", m.open_ratio) if self.use_blendshapes else m.open_ratio
        cam = robot.camera_pose(self.camera)
        return MouthObservation(
            t=t,
            position=cam.transform_point(m.position),
            normal=cam.transform_vector(m.normal),
            open_ratio=float(open_ratio),
            confidence=float(np.clip((m.frontalness - self.min_frontalness) / (1 - self.min_frontalness), 0, 1)) ** 0.5,
            source="mediapipe",
            extras={"mar": m.mar, "distance": m.distance, "distance_source": m.distance_source,
                    "frontalness": m.frontalness, "camera": self.camera},
        )


class FiducialMouthDetector:
    """THE DEMO PATH. The printed face carries an ArUco tag at a known offset from the
    opening (both come from the same CAD file), so the mouth pose = tag pose x offset.

    TODO(mouth_perception):
      1. ``cv2.aruco.ArucoDetector`` (DICT_4X4_50), tag ~30-40 mm printed on paper and
         glued into a recess on the face plate; pose via ``solvePnP`` with the RealSense
         intrinsics, translation refined with depth on the tag's flat paper.
      2. mouth pose = tag pose x the CAD offset (store it in props/fake_mouth/).
      3. open_ratio = 1.0 (always open); confidence from the reprojection error.
      4. Fallback: the contrasting lip-colour ring + depth on the face plate (never
         inside the opening, where depth is the catch cup).
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("FiducialMouthDetector: see TODO in mouth_perception/detectors.py")

    def observe(self, robot) -> MouthObservation | None:  # pragma: no cover
        raise NotImplementedError
