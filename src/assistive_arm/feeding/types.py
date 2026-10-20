"""Feeding motion results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

DeliveryStatus = Literal["delivered", "retreated", "timeout", "declined", "refused", "failed"]


@dataclass
class DeliveryResult:
    status: DeliveryStatus
    reason: str = ""
    # Metrics the sim eval and the dashboard both log:
    distance_at_delivery: float | None = None  # bite point to lips, m
    lateral_error: float | None = None  # off the face normal, m
    max_speed_near_face: float = 0.0  # m/s within near_face_radius
    evidence: dict = field(default_factory=dict)
