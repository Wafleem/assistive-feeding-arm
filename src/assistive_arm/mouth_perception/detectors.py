"""Mouth detectors. One interface (``MouthDetector.observe(robot)``), several backends.

| Backend | Status | Use |
|---|---|---|
| ``SimOracleMouthDetector`` | TODO | sim ground truth (+ optional noise/dropouts/delay) |
| ``CameraSegMouthDetector`` | TODO | the real 3D path (mask + depth + face plane), testable on sim renders |
| ``MediaPipeMouthDetector`` | TODO | real faces: MediaPipe Face Landmarker on the Orin Nano |
| ``FiducialMouthDetector`` | TODO | the fake mouth: ArUco tag on the mannequin + measured offset; colour-blob fallback |
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..vision.locate import Locator
from .types import MouthObservation


@dataclass
class OracleNoise:
    pos_sigma: float = 0.0  # m
    normal_sigma_deg: float = 0.0
    dropout: float = 0.0  # probability of no detection
    delay: float = 0.0  # s: report observations this old (camera + inference latency)


class SimOracleMouthDetector:
    """Ground truth from the MuJoCo scene: the "mouth" site (position, +z = face normal)
    and the "jaw" joint (open_ratio = jaw / jaw_open_rad), with optional noise,
    dropouts and delay to rehearse real-detector behaviour.

    TODO(mouth_perception): implement with ``robot.site_pose("mouth")``,
    ``robot.scene_state()["jaw"]`` and ``robot.now()``.
    """

    def __init__(self, noise: OracleNoise | None = None, seed: int = 0, jaw_open_rad: float = 0.30):
        self.noise = noise or OracleNoise()
        self.rng = np.random.default_rng(seed)
        self.jaw_open_rad = jaw_open_rad

    def observe(self, robot) -> MouthObservation | None:
        raise NotImplementedError("TODO(mouth_perception): SimOracleMouthDetector.observe")


class CameraSegMouthDetector:
    """The camera path: find "mouth" with any vision detector (sim: segmentation oracle),
    use the "face" plane as the approach normal. open_ratio comes from ``open_ratio_fn``
    until a landmark model provides it.

    TODO(mouth_perception): implement with ``Locator.locate(robot, "mouth", cameras,
    support_query="face")``; confidence from the number of depth points.
    """

    def __init__(self, locator: Locator, cameras=("scene", "wrist"), open_ratio_fn=None):
        self.locator, self.cameras = locator, list(cameras)
        self.open_ratio_fn = open_ratio_fn or (lambda robot: 1.0)

    def observe(self, robot) -> MouthObservation | None:
        raise NotImplementedError("TODO(mouth_perception): CameraSegMouthDetector.observe")


class MediaPipeMouthDetector:
    """Real faces with MediaPipe Face Landmarker (runs at 30+ fps on the Orin Nano CPU/GPU).

    TODO(mouth_perception):
      1. ``mediapipe.tasks.vision.FaceLandmarker`` on the RGB frame (VIDEO mode, 1 face).
      2. Lips centre = mean of inner-lip landmarks (13 upper, 14 lower); corners 78 / 308.
      3. open_ratio = |13-14| / |78-308| (mouth aspect ratio), normalised per person
         with a 3-second calibration ("open wide", "close").
      4. 3D: deproject the lip pixels with aligned depth (median over a small patch);
         face normal = plane fit to forehead/cheek/chin landmarks (e.g. 10, 234, 454, 152),
         oriented toward the camera.
      5. Confidence from the landmarker's face presence score and depth validity.
      6. Publish as a Viam vision service on the Orin Nano (see docs/COMPUTE.md).
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("MediaPipeMouthDetector: see TODO in mouth_perception/detectors.py")

    def observe(self, robot) -> MouthObservation | None:  # pragma: no cover
        raise NotImplementedError


class FiducialMouthDetector:
    """The fake mouth: an ArUco tag on the mannequin's forehead + a measured tag->mouth offset.

    TODO(mouth_perception):
      1. ``cv2.aruco.ArucoDetector`` (DICT_4X4_50), tag ~30 mm, pose via ``solvePnP``
         with the RealSense intrinsics; refine translation with depth.
      2. mouth pose = tag pose x fixed offset (measure on the prop; store in props/).
      3. open_ratio from the servo jaw command (the prop reports it over serial) or a
         second tag on the jaw.
      4. Fallback: colour blob (red lips ring) + depth.
      This is the most robust option for the demo; MediaPipe is the "real face" story.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("FiducialMouthDetector: see TODO in mouth_perception/detectors.py")

    def observe(self, robot) -> MouthObservation | None:  # pragma: no cover
        raise NotImplementedError
