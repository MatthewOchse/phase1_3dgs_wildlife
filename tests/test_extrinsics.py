"""Tests for src/calibration/extrinsics.py."""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from calibration.extrinsics import (
    _pose_from_correspondences,
    _projection_matrix,
    load_rig_calibration,
    save_rig_calibration,
    triangulate_test_points,
)
from calibration.intrinsics import generate_charuco_board


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def board_and_dict():
    board, dictionary = generate_charuco_board(
        squares_x=5,
        squares_y=7,
        square_length_m=0.04,
        marker_length_m=0.03,
    )
    return board, dictionary


@pytest.fixture(scope="module")
def board_corners(board_and_dict):
    """3-D positions of all interior Charuco corners (in board/world frame)."""
    board, _ = board_and_dict
    # shape (N, 3) — Z == 0 for all corners since board is planar
    return board.getChessboardCorners().reshape(-1, 3)


@pytest.fixture(scope="module")
def synthetic_intrinsics():
    """Reasonable synthetic camera intrinsics for a 640×480 sensor."""
    K = np.array(
        [[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    dist = np.zeros(5, dtype=np.float64)
    return {"camera_matrix": K, "dist_coeffs": dist, "image_size": (640, 480)}


def _make_T_world_from_cam(tx: float, ty: float, tz_dist: float, rx: float = 0.0) -> np.ndarray:
    """Construct a camera pose looking at the Charuco board world frame.

    Board convention: corners lie at Z=0 in world; cameras must be at
    negative Z (i.e. the board's +Z axis points toward them).  This matches
    what solvePnP and ``board.generateImage`` expect: with R≈I the camera is
    in the -Z world direction and the board appears at positive Z in camera
    frame.

    Parameters
    ----------
    tx, ty:
        Lateral offset of the camera in world X/Y.
    tz_dist:
        Distance of the camera from the board along world -Z (must be > 0).
    rx:
        Additional X-axis tilt in radians (positive = tilting top away from board).
    """
    R_x = np.array(
        [
            [1, 0, 0],
            [0, math.cos(rx), -math.sin(rx)],
            [0, math.sin(rx), math.cos(rx)],
        ],
        dtype=np.float64,
    )
    # Camera world position: (tx, ty, -tz_dist) — behind the board plane
    p_cam_world = np.array([tx, ty, -tz_dist], dtype=np.float64)
    T_cam_from_world = np.eye(4, dtype=np.float64)
    T_cam_from_world[:3, :3] = R_x
    T_cam_from_world[:3, 3] = -R_x @ p_cam_world
    return np.linalg.inv(T_cam_from_world)


# ---------------------------------------------------------------------------
# Test 1: pose recovery from synthetic 3-D/2-D correspondences
# ---------------------------------------------------------------------------


def test_pose_recovery_four_virtual_cameras(board_corners, synthetic_intrinsics):
    """Place 4 cameras at known poses, project board corners, recover poses.

    Checks translation within 0.001 m and rotation within 0.1 degrees.
    No image detection is involved — we project synthetically, so there is
    zero noise and the test validates the PnP algebra.
    """
    K = np.array(synthetic_intrinsics["camera_matrix"], dtype=np.float64)
    dist = synthetic_intrinsics["dist_coeffs"]

    # 4 cameras around the board, all angled slightly downward
    known_poses = {
        "cam0": _make_T_world_from_cam(0.0, -0.5, 0.8, rx=math.radians(30)),
        "cam1": _make_T_world_from_cam(0.3, -0.5, 0.8, rx=math.radians(25)),
        "cam2": _make_T_world_from_cam(-0.2, -0.4, 0.7, rx=math.radians(35)),
        "cam3": _make_T_world_from_cam(0.1, -0.6, 0.9, rx=math.radians(20)),
    }

    obj_pts = board_corners.reshape(-1, 1, 3).astype(np.float64)

    for cam_id, T_world_from_cam in known_poses.items():
        T_cam_from_world = np.linalg.inv(T_world_from_cam)
        R_gt = T_cam_from_world[:3, :3]
        t_gt = T_cam_from_world[:3, 3:]

        rvec_gt, _ = cv2.Rodrigues(R_gt)
        img_pts, _ = cv2.projectPoints(obj_pts, rvec_gt, t_gt, K, dist)

        T_recovered = _pose_from_correspondences(obj_pts, img_pts, K, dist)
        assert T_recovered is not None, f"Pose recovery failed for {cam_id}"

        T_cam_rec = np.linalg.inv(T_recovered)
        R_rec = T_cam_rec[:3, :3]
        t_rec = T_cam_rec[:3, 3]

        # Translation error in metres
        t_err = np.linalg.norm(t_rec - t_gt.flatten())
        assert t_err < 1e-3, f"{cam_id}: translation error {t_err:.6f} m > 0.001 m"

        # Rotation error in degrees
        R_diff = R_rec @ R_gt.T
        angle_err = math.degrees(math.acos(min(1.0, (np.trace(R_diff) - 1) / 2)))
        assert angle_err < 0.1, f"{cam_id}: rotation error {angle_err:.4f}° > 0.1°"


# ---------------------------------------------------------------------------
# Test 2: round-trip save / load
# ---------------------------------------------------------------------------


def test_save_load_roundtrip(tmp_path, synthetic_intrinsics):
    """save_rig_calibration + load_rig_calibration preserves all values exactly."""
    T = np.array(
        [
            [1.0, 0.0, 0.0, 0.5],
            [0.0, 0.9848, -0.1736, -0.3],
            [0.0, 0.1736, 0.9848, 1.2],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    extrinsics = {"cam0": T, "cam1": T @ np.diag([1, 1, 1, 1])}
    intrinsics = {"cam0": synthetic_intrinsics, "cam1": synthetic_intrinsics}

    out = tmp_path / "rig.json"
    save_rig_calibration(extrinsics, intrinsics, out, board_config={"squares_x": 5})
    loaded = load_rig_calibration(out)

    for cam_id in ["cam0", "cam1"]:
        np.testing.assert_array_almost_equal(
            loaded[cam_id]["T_world_from_cam"],
            extrinsics[cam_id],
            decimal=10,
            err_msg=f"T_world_from_cam mismatch for {cam_id}",
        )
        np.testing.assert_array_almost_equal(
            loaded[cam_id]["camera_matrix"],
            synthetic_intrinsics["camera_matrix"],
            decimal=10,
        )
        np.testing.assert_array_almost_equal(
            loaded[cam_id]["dist_coeffs"],
            synthetic_intrinsics["dist_coeffs"],
            decimal=10,
        )


# ---------------------------------------------------------------------------
# Test 3: triangulation of a known 3-D point
# ---------------------------------------------------------------------------


def test_triangulation_known_point(synthetic_intrinsics):
    """Triangulate a known 3-D point from 3 cameras; recover within 0.1 mm."""
    K = np.array(synthetic_intrinsics["camera_matrix"], dtype=np.float64)
    dist = synthetic_intrinsics["dist_coeffs"]
    intr = synthetic_intrinsics

    # Ground-truth 3-D point (in world frame = board frame)
    P_gt = np.array([0.05, 0.06, 0.0], dtype=np.float64)

    cam_poses = {
        "cam0": _make_T_world_from_cam(0.0, -0.5, 0.8, rx=math.radians(30)),
        "cam1": _make_T_world_from_cam(0.3, -0.5, 0.8, rx=math.radians(25)),
        "cam2": _make_T_world_from_cam(-0.2, -0.4, 0.7, rx=math.radians(35)),
    }

    points_per_cam: dict[str, np.ndarray] = {}
    for cam_id, T_world_from_cam in cam_poses.items():
        T_cfw = np.linalg.inv(T_world_from_cam)
        R, t = T_cfw[:3, :3], T_cfw[:3, 3:]
        rvec, _ = cv2.Rodrigues(R)
        proj, _ = cv2.projectPoints(
            P_gt.reshape(1, 1, 3), rvec, t, K, dist
        )
        points_per_cam[cam_id] = proj.reshape(1, 2)

    pts3d = triangulate_test_points(
        points_per_cam,
        {c: intr for c in cam_poses},
        cam_poses,
    )

    assert pts3d.shape == (1, 3)
    err = np.linalg.norm(pts3d[0] - P_gt)
    assert err < 1e-4, f"Triangulation error {err:.6f} m > 0.1 mm"


# ---------------------------------------------------------------------------
# Test 4: triangulation requires at least two cameras
# ---------------------------------------------------------------------------


def test_triangulation_requires_two_cameras(synthetic_intrinsics):
    with pytest.raises(ValueError, match="at least two cameras"):
        triangulate_test_points(
            {"cam0": np.zeros((5, 2))},
            {"cam0": synthetic_intrinsics},
            {"cam0": np.eye(4)},
        )


# ---------------------------------------------------------------------------
# Test 5: projection matrix shape and sanity
# ---------------------------------------------------------------------------


def test_projection_matrix_shape(synthetic_intrinsics):
    T = _make_T_world_from_cam(0.0, -0.5, 0.8)
    P = _projection_matrix(synthetic_intrinsics, T)
    assert P.shape == (3, 4)
    # Last element of bottom-right of P relates to scale — just verify it's finite
    assert np.all(np.isfinite(P))
