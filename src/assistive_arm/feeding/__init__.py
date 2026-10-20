"""Section 4: feeding motion and face safety. See README.md in this folder."""

from .approach import deliver_bite, final_pose, retreat, staging_pose
from .config import FeedingConfig
from .safety import check_near_face, face_zone, speed_cap
from .types import DeliveryResult

__all__ = ["DeliveryResult", "FeedingConfig", "check_near_face", "deliver_bite", "face_zone", "final_pose",
           "retreat", "speed_cap", "staging_pose"]
