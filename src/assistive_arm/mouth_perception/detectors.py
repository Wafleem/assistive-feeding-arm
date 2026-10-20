"""Mouth detectors. One interface (``MouthDetector.observe(robot)``), several backends.

| Backend | Status | Use |
|---|---|---|
| ``FiducialMouthDetector`` | TODO | **the demo path**: ArUco tag on the printed face + the CAD tag->opening offset |
| ``CameraSegMouthDetector`` | TODO | the vision path (mask + depth + face plane), testable on sim renders |
| ``SimOracleMouthDetector`` | TODO | sim ground truth (+ optional noise/dropouts/delay) |
| ``MediaPipeMouthDetector`` | TODO, later | real faces: MediaPipe Face Landmarker (not needed for the printed mouth) |

The printed mouth is always open: detectors for it report ``open_ratio = 1.0``.
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
    """Real faces with MediaPipe Face Landmarker (30+ fps on the AGX Orin). LATER: not
    needed for the printed mouth, but it is the "works on a person" story.

    TODO(mouth_perception):
      1. ``mediapipe.tasks.vision.FaceLandmarker`` on the RGB frame (VIDEO mode, 1 face).
      2. Lips centre = mean of inner-lip landmarks (13 upper, 14 lower); corners 78 / 308.
      3. open_ratio = |13-14| / |78-308| (mouth aspect ratio), normalised per person
         with a 3-second calibration ("open wide", "close").
      4. 3D: deproject the lip pixels with aligned depth (median over a small patch);
         face normal = plane fit to forehead/cheek/chin landmarks (e.g. 10, 234, 454, 152),
         oriented toward the camera.
      5. Confidence from the landmarker's face presence score and depth validity.
      6. Publish as a Viam vision service on the AGX Orin (see docs/COMPUTE.md).
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("MediaPipeMouthDetector: see TODO in mouth_perception/detectors.py")

    def observe(self, robot) -> MouthObservation | None:  # pragma: no cover
        raise NotImplementedError


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
