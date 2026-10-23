import numpy as np
import pytest

from da3_depth.geometry import (
    Intrinsics,
    RangeParams,
    colorize,
    da3_metric_to_metres,
    depth_to_points,
    encode_viam_depth,
    points_to_pcd,
    range_readings,
    resize_nearest,
)

K = Intrinsics.from_hfov(90.0, 64, 48)  # f = 32 px


def wall(z=2.0, k=K):
    return np.full((k.height, k.width), z, dtype=np.float32)


def test_hfov_and_scaling():
    assert K.fx == pytest.approx(32.0) and (K.cx, K.cy) == (32.0, 24.0)
    half = K.scaled(32, 24)
    assert (half.fx, half.cx, half.cy) == pytest.approx((16.0, 16.0, 12.0))


def test_da3_metric_scale_uses_focal_over_300():
    k = Intrinsics(600.0, 600.0, 320.0, 240.0, 640, 480)
    assert da3_metric_to_metres(np.array([1.0]), k)[0] == pytest.approx(2.0)


def test_resize_nearest_keeps_values():
    img = np.arange(6, dtype=np.float32).reshape(2, 3)
    out = resize_nearest(img, 4, 6)
    assert out.shape == (4, 6) and set(np.unique(out)) == set(img.ravel())
    assert resize_nearest(img, 2, 3) is img


def test_points_of_a_wall_lie_on_the_wall():
    pts = depth_to_points(wall(), K)
    assert pts.shape == (64 * 48, 3) and pts.dtype == np.float32
    assert np.allclose(pts[:, 2], 2.0)
    # The principal-point pixel deprojects onto the optical axis.
    centre = pts[(np.abs(pts[:, 0]) < 1e-6) & (np.abs(pts[:, 1]) < 1e-6)]
    assert len(centre) == 1


def test_points_respect_stride_invalid_and_range_limits():
    d = wall()
    d[0, 0] = 0.0
    d[0, 2] = np.nan
    assert len(depth_to_points(d, K, stride=2)) == 32 * 24 - 2
    # Corner rays are longer than 2.0 m (range = z * |ray|), so a 2.1 m limit drops them.
    assert 0 < len(depth_to_points(wall(), K, max_range=2.1)) < 64 * 48


def test_pcd_is_binary_millimetres():
    blob = points_to_pcd(np.array([[0.001, -0.002, 1.5]], dtype=np.float32))
    header, body = blob.split(b"DATA binary\n")
    assert b"POINTS 1" in header and b"FIELDS x y z" in header
    assert np.frombuffer(body, dtype=np.float32) == pytest.approx([1.0, -2.0, 1500.0])


def test_viam_raw_depth_round_trips_through_the_sdk():
    from viam.media.video import CameraMimeType, ViamImage

    d = np.array([[0.5, 0.0], [1.234, 70.0]], dtype=np.float32)  # 70 m saturates at 65535 mm
    arr = ViamImage(encode_viam_depth(d), CameraMimeType.VIAM_RAW_DEPTH).bytes_to_depth_array()
    assert arr == [[500, 0], [1234, 65535]]


def test_colorize_marks_invalid_black_and_near_warm():
    d = np.array([[0.2, 3.5, 0.0]], dtype=np.float32)
    rgb = colorize(d, 0.1, 5.0)
    assert rgb.shape == (1, 3, 3) and rgb.dtype == np.uint8
    assert tuple(rgb[0, 2]) == (0, 0, 0)
    assert rgb[0, 0, 0] > rgb[0, 0, 2] and rgb[0, 1, 2] > rgb[0, 1, 0]  # near is red, far is blue


def test_ranges_of_a_flat_wall():
    r = range_readings(wall(), K, RangeParams(scan_beams=9, scan_band=0.1, min_support_px=1))
    assert r.distance_m == pytest.approx(2.0, abs=0.01)
    assert r.nearest_m == pytest.approx(2.0, abs=1e-3)
    assert r.valid_fraction == 1.0
    scan = np.array(r.scan_ranges_m)
    assert len(scan) == 9 and np.argmin(scan) == 4 and scan[4] == pytest.approx(2.0, abs=0.01)
    # Each beam's first return is z / cos(bearing) somewhere inside the beam's wedge.
    centres = np.radians(r.scan_angle_min_deg + r.scan_angle_increment_deg * np.arange(9))
    half = np.radians(r.scan_angle_increment_deg) / 2
    inner, outer = np.maximum(np.abs(centres) - half, 0), np.abs(centres) + half
    assert np.all(scan >= 2.0 / np.cos(inner) - 0.02) and np.all(scan <= 2.0 / np.cos(outer) + 0.02)
    assert r.scan_angle_min_deg == pytest.approx(-r.scan_angle_min_deg - 8 * r.scan_angle_increment_deg, abs=1.0)


def test_nearest_object_bearing_and_empty_beams():
    d = wall(3.0)
    d[20:28, 48:56] = 0.8  # a box right of centre, at the image's mid height
    d[:, :12] = 0.0  # left edge (all of beam 0: beams are equal in angle, not columns): no depth
    r = range_readings(d, K, RangeParams(scan_beams=8, scan_band=0.2))  # box = 64 px, 2.4 % of the view
    assert 0.8 < r.nearest_m < 1.0  # a range, not z: the box is ~30 deg off-axis (0.8 / cos 30 = 0.92)
    assert r.nearest_bearing_deg > 20 and abs(r.nearest_elevation_deg) < 10
    assert r.scan_ranges_m[0] is None and min(v for v in r.scan_ranges_m if v) < 1.0
    assert r.distance_m == pytest.approx(3.0, abs=0.01)


def test_a_single_flying_pixel_is_not_a_return():
    d = wall(3.0)
    d[10, 10] = 0.3
    assert range_readings(d, K, RangeParams()).nearest_m > 2.9


def test_no_valid_depth_gives_none_not_zero():
    r = range_readings(np.zeros((48, 64), np.float32), K, RangeParams())
    assert r.distance_m is None and r.nearest_m is None and r.nearest_bearing_deg is None
    assert all(v is None for v in r.scan_ranges_m) and r.valid_fraction == 0.0
