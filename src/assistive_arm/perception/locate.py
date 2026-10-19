"""Detection + depth -> a 3D target (position and surface normal) in the base frame.

Two cameras, two roles:

* the off-arm camera (``mount='base'``, on the wheelchair) gives the coarse
  location from a stable viewpoint the arm does not move;
* the wrist camera refines it from ~25 cm away just before contact, which
  also absorbs wheelchair sway and calibration error ("look before every move").
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .detectors import Detector, mask_or_box
from .geometry import bbox_mask, fit_plane, mask_points, robust_center
from .types import Target3D

if TYPE_CHECKING:  # robot_iface imports perception.types; avoid the cycle
    from ..robot_iface.base import RobotInterface


@dataclass
class LocateParams:
    normal_patch_scale: float = 2.5  # fit the surface normal on a patch this much larger than the box
    min_points: int = 12
    max_range: float = 1.5  # metres; ignore farther returns (D435 depth noise grows ~z^2)


class Locator:
    def __init__(self, detector: Detector, params: LocateParams | None = None):
        self.detector = detector
        self.params = params or LocateParams()

    def locate(self, robot: "RobotInterface", query: str, cameras: list[str] | None = None,
               support_query: str | None = None) -> Target3D | None:
        """Try cameras in order; return the first confident 3D fix.

        ``support_query`` names the surface the target sits on ("door",
        "elevator panel"). Its plane gives the approach normal, which a thin
        lever or a small knob cannot give reliably on its own."""
        names = cameras or list(robot.cameras())
        frames = robot.get_frames(names)
        for name in names:
            frame = frames.get(name)
            if frame is None:
                continue
            t = self._locate_in_frame(robot, frame, query, support_query)
            if t is not None:
                return t
        return None

    def _support_normal(self, frame, depth, support_query, center) -> np.ndarray | None:
        dets = self.detector.detect(frame, support_query)
        if not dets:
            return None
        pts = mask_points(depth, mask_or_box(max(dets, key=lambda d: d.score), depth.shape, scale=0.9), frame.intrinsics)
        pts = pts[np.linalg.norm(pts - center, axis=1) < 0.25]  # the part of the surface around the target
        if len(pts) < self.params.min_points:
            return None
        return fit_plane(pts)[1]

    def _locate_in_frame(self, robot, frame, query, support_query=None) -> Target3D | None:
        p = self.params
        dets = sorted(self.detector.detect(frame, query), key=lambda d: d.score, reverse=True)
        if not dets:
            return None
        det = dets[0]
        depth = np.where(frame.depth < p.max_range, frame.depth, 0)
        pts = mask_points(depth, mask_or_box(det, depth.shape), frame.intrinsics)
        if len(pts) < p.min_points:
            return None
        center = robust_center(pts)
        n = self._support_normal(frame, depth, support_query, center) if support_query else None
        source = "support" if n is not None else "patch"
        if n is None:
            patch = mask_points(depth, bbox_mask(depth.shape, det.bbox, p.normal_patch_scale), frame.intrinsics)
            # Keep the patch points near the target depth so a background wall does not tilt the fit.
            patch = patch[np.abs(patch[:, 2] - center[2]) < 0.05]
            if len(patch) < p.min_points:
                patch = pts
            _, n = fit_plane(patch)
        if np.dot(n, -center) < 0:  # make it face the camera
            n = -n
        cam = robot.camera_pose(frame.camera)
        return Target3D(
            label=query,
            position=cam.transform_point(center),
            normal=cam.transform_vector(n),
            score=det.score,
            camera=frame.camera,
            n_points=len(pts),
            extras={"normal_source": source},
        )
