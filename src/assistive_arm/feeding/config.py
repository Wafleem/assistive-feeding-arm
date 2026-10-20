"""Feeding motion and face-safety parameters. Every number here is reviewed by the safety lead."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FeedingConfig:
    # Geometry
    standoff: float = 0.03  # m: the bite stops this far IN FRONT of the lips; never inside the mouth
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
    wait_ready_timeout: float = 20.0  # s waiting for "open and still"
    bite_timeout: float = 10.0  # s holding the bite before retreating
    servo_rate_hz: float = 20.0  # final-approach re-targeting rate
    confirm_each_bite: bool = True  # MANDATORY for the demo: ask before every approach
