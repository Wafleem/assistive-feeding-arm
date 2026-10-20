"""What mouth perception produces. All poses are in the arm base frame, metres, seconds."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


@dataclass
class MouthObservation:
    """One detector output for one camera frame (no filtering)."""

    t: float  # capture time of the frame (robot clock), NOT the time it was processed
    position: np.ndarray  # (3,) lips centre
    normal: np.ndarray  # (3,) unit, out of the face: the spoon/food approaches along -normal
    open_ratio: float  # 0 = closed .. 1 = wide open (mouth aspect ratio, normalised)
    confidence: float  # 0..1
    source: str = ""  # "sim_oracle", "camera_seg", "mediapipe", "fiducial", ...
    extras: dict = field(default_factory=dict)


@dataclass
class MouthState:
    """Filtered, predicted mouth state handed to feeding motion."""

    t: float  # time this state refers to (may be in the future: latency-compensated)
    position: np.ndarray
    velocity: np.ndarray
    normal: np.ndarray
    open_ratio: float
    is_open: bool
    confidence: float
    age: float  # seconds since the last accepted measurement (stale => don't move closer)

    @property
    def speed(self) -> float:
        return float(np.linalg.norm(self.velocity))


class MouthDetector(Protocol):
    """Anything that turns the current camera frames into a mouth observation."""

    def observe(self, robot) -> MouthObservation | None: ...
