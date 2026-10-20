"""Feeding motion and face-safety parameters. Every number here is reviewed by the safety lead."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# Who/what is being fed. TODO(feeding): review these two modes with the safety lead.
#   "printed_mouth": the hackathon prop, a 3D-printed, always-open mouth with a removable
#                    catch cup behind the opening. The bite is placed insertion_depth INTO
#                    the opening and released; it drops into the cup, which is emptied after
#                    every use. Only ever valid for the printed prop.
#   "person":        a real person (future work). The bite stops `standoff` IN FRONT of the
#                    lips and the person closes the gap; nothing ever enters the mouth.
Target = Literal["printed_mouth", "person"]


@dataclass(frozen=True)
class FeedingConfig:
    target: Target = "person"  # SAFE DEFAULT; the demo config sets "printed_mouth" explicitly
    insertion_depth: float = 0.015  # m past the face plane into the opening (printed_mouth only)
    # Geometry
    standoff: float = 0.03  # m: person mode: the bite stops this far IN FRONT of the lips
    pre_approach: float = 0.15  # m: staging point out along the face normal
    head_radius: float = 0.12  # m: keep-out sphere around the head centre (mouth - 0.10 * normal)
    corridor_half_angle_deg: float = 20.0  # only this cone around the face normal may get closer than head_radius
    # Speeds
    staging_speed: float = 0.10  # m/s to the staging point
    final_speed: float = 0.02  # m/s along the normal in the final approach
    near_face_speed_cap: float = 0.03  # m/s anywhere within near_face_radius of the mouth
    near_face_radius: float = 0.15
    retreat_speed: float = 0.08  # m/s straight back out along the normal
    # Forces / timeouts
    max_force_near_face: float = 3.0  # N: any contact near the face => retreat
    wait_ready_timeout: float = 20.0  # s waiting for "open and still" (printed mouth: "still and tracked")
    bite_timeout: float = 10.0  # s holding the bite before retreating (person mode)
    servo_rate_hz: float = 20.0  # final-approach re-targeting rate
    confirm_each_bite: bool = True  # MANDATORY for the demo: ask before every approach
