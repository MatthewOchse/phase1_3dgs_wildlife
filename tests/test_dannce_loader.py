"""Tests for the DANNCE markerless_mouse_1 loader.

Requires the real dataset.  Run with:

    pytest -m dataset tests/test_dannce_loader.py -v

Skipped by default (no dataset marker in base pytest run).
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pytest

from io_utils.dannce_loader import (
    load_dannce_calibration,
    load_dannce_labels,
    load_dannce_session,
    load_dannce_sync,
)
from calibration.extrinsics import triangulate_test_points
import cv2

# ---------------------------------------------------------------------------
# Dataset fixture
# ---------------------------------------------------------------------------


def _dataset_root() -> Path:
    raw = os.environ.get("DANNCE_MM1_PATH", "~/datasets/dannce_mm1")
    return Path(raw).expanduser().resolve()


@pytest.fixture(scope="module")
def dataset_root():
    root = _dataset_root()
    if not (root / "label3d_dannce.mat").exists():
        pytest.skip(f"DANNCE dataset not found at {root}")
    return root


@pytest.fixture(scope="module")
def mat_path(dataset_root):
    return dataset_root / "label3d_dannce.mat"


@pytest.fixture(scope="module")
def calibration(mat_path, dataset_root):
    return load_dannce_calibration(mat_path, dataset_root)


@pytest.fixture(scope="module")
def session(dataset_root):
    return load_dannce_session(dataset_root)


# ---------------------------------------------------------------------------
# Calibration structure tests
# ---------------------------------------------------------------------------


@pytest.mark.dataset
def test_calibration_has_six_cameras(calibration):
    assert len(calibration["cameras"]) == 6


@pytest.mark.dataset
def test_calibration_camera_ids(calibration):
    ids = set(calibration["cameras"].keys())
    assert ids == {"Camera1", "Camera2", "Camera3", "Camera4", "Camera5", "Camera6"}


@pytest.mark.dataset
def test_camera_matrix_shape(calibration):
    for cam_id, cam in calibration["cameras"].items():
        K = cam["intrinsics"]["camera_matrix"]
        assert K.shape == (3, 3), f"{cam_id}: camera_matrix shape {K.shape}"


@pytest.mark.dataset
def test_camera_matrix_upper_triangular(calibration):
    """After transposing the stored matrix, K should be upper-triangular."""
    for cam_id, cam in calibration["cameras"].items():
        K = cam["intrinsics"]["camera_matrix"]
        assert abs(K[1, 0]) < 1e-6, f"{cam_id}: K[1,0]={K[1,0]} — not upper-triangular"
        assert abs(K[2, 0]) < 1e-6, f"{cam_id}: K[2,0]={K[2,0]} — not upper-triangular"
        assert abs(K[2, 1]) < 1e-6, f"{cam_id}: K[2,1]={K[2,1]} — not upper-triangular"
        assert K[2, 2] == pytest.approx(1.0), f"{cam_id}: K[2,2]={K[2,2]}"


@pytest.mark.dataset
def test_dist_coeffs_shape(calibration):
    for cam_id, cam in calibration["cameras"].items():
        d = cam["intrinsics"]["dist_coeffs"]
        assert d.ndim == 1, f"{cam_id}: dist_coeffs has ndim {d.ndim}"
        assert len(d) == 5, f"{cam_id}: dist_coeffs has {len(d)} elements"


@pytest.mark.dataset
def test_world_from_camera_is_valid_transform(calibration):
    for cam_id, cam in calibration["cameras"].items():
        T = cam["extrinsics"]["world_from_camera"]
        assert T.shape == (4, 4), f"{cam_id}: T shape {T.shape}"

        # Bottom row must be [0, 0, 0, 1]
        np.testing.assert_array_almost_equal(
            T[3, :], [0, 0, 0, 1], decimal=10,
            err_msg=f"{cam_id}: bottom row of T_world_from_cam is not [0,0,0,1]"
        )

        # Upper-left 3×3 must be a proper rotation (det ≈ +1)
        R = T[:3, :3]
        det = np.linalg.det(R)
        assert abs(det - 1.0) < 1e-5, f"{cam_id}: rotation det={det:.6f}"


@pytest.mark.dataset
def test_focal_lengths_plausible(calibration):
    """Focal lengths should be in a physically reasonable range (200–5000 px)."""
    for cam_id, cam in calibration["cameras"].items():
        K = cam["intrinsics"]["camera_matrix"]
        fx, fy = K[0, 0], K[1, 1]
        assert 200 < fx < 5000, f"{cam_id}: fx={fx} out of plausible range"
        assert 200 < fy < 5000, f"{cam_id}: fy={fy} out of plausible range"


# ---------------------------------------------------------------------------
# Triangulation round-trip
# ---------------------------------------------------------------------------


@pytest.mark.dataset
def test_triangulation_round_trip(calibration):
    """Project a known 3D point into all 6 cameras and triangulate back."""
    # Use a point in the middle of the arena (approximate world centre)
    P_gt = np.array([50.0, 50.0, 100.0], dtype=np.float64)

    cam_ids = sorted(calibration["cameras"].keys())
    points_per_cam: dict[str, np.ndarray] = {}
    intrinsics_per_cam: dict[str, dict] = {}
    extrinsics_per_cam: dict[str, np.ndarray] = {}

    for cam_id in cam_ids:
        cam = calibration["cameras"][cam_id]
        K = cam["intrinsics"]["camera_matrix"]
        dist = cam["intrinsics"]["dist_coeffs"]
        T_world_from_cam = cam["extrinsics"]["world_from_camera"]
        T_cam_from_world = np.linalg.inv(T_world_from_cam)
        R = T_cam_from_world[:3, :3]
        t = T_cam_from_world[:3, 3:]
        rvec, _ = cv2.Rodrigues(R)
        proj, _ = cv2.projectPoints(P_gt.reshape(1, 1, 3), rvec, t, K, dist)
        points_per_cam[cam_id] = proj.reshape(1, 2)
        intrinsics_per_cam[cam_id] = {"camera_matrix": K, "dist_coeffs": dist}
        extrinsics_per_cam[cam_id] = T_world_from_cam

    pts3d = triangulate_test_points(points_per_cam, intrinsics_per_cam, extrinsics_per_cam)
    err = float(np.linalg.norm(pts3d[0] - P_gt))
    assert err < 1.0, f"Triangulation error {err:.4f} mm > 1 mm"


# ---------------------------------------------------------------------------
# Session loading tests
# ---------------------------------------------------------------------------


@pytest.mark.dataset
def test_session_has_six_cameras(session):
    assert len(session.cameras) == 6


@pytest.mark.dataset
def test_session_fps_plausible(session):
    assert 10.0 <= session.fps <= 120.0, f"fps={session.fps} out of plausible range"


@pytest.mark.dataset
def test_session_frame_count(session):
    assert session.n_timesteps >= 100, f"Only {session.n_timesteps} timesteps"


@pytest.mark.dataset
def test_session_all_cameras_same_frame_count(session):
    counts = {cam_id: len(frames) for cam_id, frames in session.frames_per_camera.items()}
    assert len(set(counts.values())) == 1, f"Frame counts differ: {counts}"


@pytest.mark.dataset
def test_session_frame0_all_cameras_consistent_shape(session):
    """Frame 0 from all cameras should have the same image dimensions."""
    ts = session.get_frame_at_index(0)
    shapes = {}
    for cam_id, frame in ts.frames.items():
        img = frame.image
        assert img is not None
        shapes[cam_id] = img.shape

    height_set = set(s[0] for s in shapes.values())
    width_set = set(s[1] for s in shapes.values())
    assert len(height_set) == 1, f"Inconsistent heights: {shapes}"
    assert len(width_set) == 1, f"Inconsistent widths: {shapes}"


@pytest.mark.dataset
def test_session_summary_string(session):
    s = session.summary()
    assert "markerless_mouse_1" in s
    assert "Camera" in s


# ---------------------------------------------------------------------------
# Sync
# ---------------------------------------------------------------------------


@pytest.mark.dataset
def test_sync_hardware_type(mat_path):
    sync = load_dannce_sync(mat_path)
    assert sync["type"] == "hardware"
    assert sync["n_frames"] > 0


@pytest.mark.dataset
def test_sync_frame_indices_sequential(mat_path):
    sync = load_dannce_sync(mat_path)
    for cam_id, indices in sync["per_camera_frame_indices"].items():
        np.testing.assert_array_equal(
            indices,
            np.arange(len(indices), dtype=np.int32),
            err_msg=f"{cam_id}: frame indices are not sequential",
        )


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------


@pytest.mark.dataset
def test_labels_present(mat_path):
    labels = load_dannce_labels(mat_path)
    assert len(labels) > 0


@pytest.mark.dataset
def test_label_shapes(mat_path):
    labels = load_dannce_labels(mat_path)
    for cam_id, ld in labels.items():
        n = len(ld["frame_indices"])
        assert ld["data_2d"].shape == (n, 44), \
            f"{cam_id}: data_2d shape {ld['data_2d'].shape}"
        assert ld["data_3d"].shape == (n, 66), \
            f"{cam_id}: data_3d shape {ld['data_3d'].shape}"
