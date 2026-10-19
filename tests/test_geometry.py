import numpy as np
import pytest

from assistive_arm.geometry import (
    Pose, arc_waypoints, axis_angle_to_mat, interpolate_poses, look_rotation, mat_to_quat, quat_to_mat,
    tool_pose_facing,
)
from assistive_arm.perception.geometry import deproject, fit_plane, mask_points, project
from assistive_arm.perception.types import CameraIntrinsics
from assistive_arm.robot_iface.viam_convert import mat_to_ov, ov_to_mat, pose_to_viam_dict, viam_dict_to_pose

rng = np.random.default_rng(0)


def random_rot():
    return axis_angle_to_mat(rng.normal(size=3), rng.uniform(-np.pi, np.pi))


def test_quat_roundtrip():
    for _ in range(50):
        r = random_rot()
        assert np.allclose(quat_to_mat(mat_to_quat(r)), r, atol=1e-9)


def test_pose_compose_inverse():
    a = Pose.from_matrix(rng.normal(size=3), random_rot())
    b = Pose.from_matrix(rng.normal(size=3), random_rot())
    ident = a @ a.inverse()
    assert np.allclose(ident.position, 0, atol=1e-9) and np.allclose(ident.rot, np.eye(3), atol=1e-9)
    p = rng.normal(size=3)
    assert np.allclose((a @ b).transform_point(p), a.transform_point(b.transform_point(p)))


def test_look_rotation_and_tool_pose():
    r = look_rotation([1, 0, 0], x_hint=[0, 0, 1])
    assert np.allclose(r[:, 2], [1, 0, 0]) and np.allclose(r[:, 0], [0, 0, 1])
    assert np.isclose(np.linalg.det(r), 1)
    pose = tool_pose_facing([0.5, 0, 0.3], [1, 0, 0], standoff=0.08)
    assert np.allclose(pose.position, [0.42, 0, 0.3]) and np.allclose(pose.z_axis, [1, 0, 0])


def test_interpolation_ends_on_target():
    a, b = Pose([0, 0, 0]), Pose.from_matrix([0.1, 0.2, 0.3], random_rot())
    path = interpolate_poses(a, b, 10)
    assert len(path) == 10 and path[-1].distance_to(b)[0] < 1e-9 and path[-1].distance_to(b)[1] < 1e-6


def test_arc_keeps_radius_and_rotates_grasp():
    start = Pose.from_matrix([0.5, 0.0, 0.3], look_rotation([1, 0, 0]))
    hinge = np.array([0.6, -0.7, 0.0])
    wps = arc_waypoints(start, hinge, [0, 0, 1], np.deg2rad(30), 15)
    r0 = np.linalg.norm((start.position - hinge)[:2])
    for w in wps:
        assert np.isclose(np.linalg.norm((w.position - hinge)[:2]), r0)
        assert np.isclose(w.position[2], 0.3)
    assert np.isclose(np.degrees(start.distance_to(wps[-1])[1]), 30, atol=1e-6)


@pytest.mark.parametrize("seed", range(5))
def test_viam_orientation_vector_roundtrip(seed):
    r = axis_angle_to_mat(np.random.default_rng(seed).normal(size=3), 1.0 + seed * 0.4)
    assert np.allclose(ov_to_mat(*mat_to_ov(r)), r, atol=1e-6)


def test_viam_orientation_vector_known_cases():
    assert np.allclose(mat_to_ov(np.eye(3)), (0, 0, 1, 0))
    # tool pointing straight down, no spin: Viam's common "o_z = -1" pose
    down = ov_to_mat(0, 0, -1, 0)
    assert np.allclose(down[:, 2], [0, 0, -1])
    assert np.allclose(ov_to_mat(*mat_to_ov(down)), down)
    # spin about z by 90 deg
    assert np.allclose(mat_to_ov(axis_angle_to_mat([0, 0, 1], np.pi / 2)), (0, 0, 1, 90))


def test_viam_pose_units():
    p = Pose.from_matrix([0.1, -0.2, 0.3], look_rotation([1, 0, 0], [0, 0, 1]))
    d = pose_to_viam_dict(p)
    assert np.allclose([d["x"], d["y"], d["z"]], [100, -200, 300])
    assert np.allclose([d["o_x"], d["o_y"], d["o_z"]], [1, 0, 0], atol=1e-9)
    back = viam_dict_to_pose(d)
    assert back.distance_to(p)[0] < 1e-9 and back.distance_to(p)[1] < 1e-6


def test_deproject_project_roundtrip():
    k = CameraIntrinsics.from_fovy(58, 640, 480)
    p = deproject(400, 100, 0.7, k)
    assert np.allclose(project(p, k), (400, 100))


def test_plane_fit_recovers_tilted_plane():
    k = CameraIntrinsics.from_fovy(58, 160, 120)
    n_true = np.array([0.2, -0.1, -1.0])
    n_true /= np.linalg.norm(n_true)
    # depth image of the plane n.x = d seen by the camera
    vs, us = np.mgrid[0:120, 0:160]
    rays = np.stack([(us - k.cx) / k.fx, (vs - k.cy) / k.fy, np.ones_like(us, float)], -1)
    depth = (-0.6 / (rays @ n_true)).astype(np.float32)
    pts = mask_points(depth, np.ones_like(depth, bool), k)
    _, n = fit_plane(pts)
    assert abs(abs(np.dot(n, n_true)) - 1) < 1e-6
