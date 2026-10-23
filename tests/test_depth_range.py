"""Depth frames teach the fast face pipeline this person's eye span (offline, synthetic face)."""

import numpy as np

from assistive_arm.mouth_perception.depth_range import DepthRangeCorrector
from assistive_arm.mouth_perception.face_geometry import FaceGeometryParams, measure_face
from test_face_perception import K, synthetic_face


def scaled_face(span_m, dist):
    """A face whose eye span is span_m (the synthetic face has 90 mm): scale the head."""
    lm, lips, normal, cam = synthetic_face(dist=dist * 0.090 / span_m)  # same pixels as a smaller head closer
    return lm


def test_learns_a_narrow_eye_span_from_depth_and_fixes_the_range():
    true_span, true_dist = 0.080, 0.50
    lm = scaled_face(true_span, true_dist)
    depth = np.full((K.height, K.width), true_dist, np.float32)
    corr = DepthRangeCorrector()
    before = measure_face(lm, K).distance  # 90 mm prior: overestimates the range
    assert before > true_dist * 1.08
    for _ in range(5):
        assert corr.update(lm, K, depth, FaceGeometryParams())
    assert abs(corr.eye_span_m - true_span) < 0.002
    after = measure_face(lm, K, None, FaceGeometryParams(eye_span_m=corr.eye_span_m)).distance
    assert abs(after - true_dist) < 0.01


def test_implausible_or_missing_depth_is_ignored():
    lm = scaled_face(0.09, 0.5)
    corr = DepthRangeCorrector()
    assert not corr.update(lm, K, np.zeros((K.height, K.width), np.float32), FaceGeometryParams())
    assert not corr.update(lm, K, np.full((K.height, K.width), 2.0, np.float32), FaceGeometryParams())  # span ~36 cm
    assert not corr.ready and corr.eye_span_m == 0.090
