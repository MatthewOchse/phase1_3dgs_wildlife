"""Tests for src/calibration/intrinsics.py."""

import json
import warnings
from pathlib import Path

import cv2
import numpy as np
import pytest

from calibration.intrinsics import (
    detect_charuco_corners,
    generate_charuco_board,
    load_intrinsics,
    save_intrinsics,
    validate_intrinsics,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_board(squares_x: int = 5, squares_y: int = 7):
    return generate_charuco_board(
        squares_x=squares_x,
        squares_y=squares_y,
        square_length_m=0.04,
        marker_length_m=0.03,
    )


# ---------------------------------------------------------------------------
# Detection test
# ---------------------------------------------------------------------------


def test_detect_corners_on_synthetic_image():
    """Corners should be detected on a cleanly rendered board image."""
    board, dictionary = _make_board()
    board_img = board.generateImage((600, 800))

    corners, ids = detect_charuco_corners(board_img, board, dictionary)

    assert corners is not None, "Expected corners to be detected on synthetic image"
    assert ids is not None, "Expected IDs to be returned alongside corners"
    assert len(corners) == len(ids)
    # A 5×7 board has (5-1)×(7-1) = 24 interior corners
    assert len(corners) == 24, f"Expected 24 corners, got {len(corners)}"


def test_detect_corners_returns_none_on_blank_image():
    """A blank image should produce no detections."""
    board, dictionary = _make_board()
    blank = np.ones((600, 800), dtype=np.uint8) * 255

    corners, ids = detect_charuco_corners(blank, board, dictionary)

    assert corners is None
    assert ids is None


# ---------------------------------------------------------------------------
# Save / load round-trip
# ---------------------------------------------------------------------------


def test_save_load_roundtrip_preserves_camera_matrix(tmp_path):
    """Camera matrix must survive a JSON save/load cycle exactly."""
    camera_matrix = np.array(
        [[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    dist_coeffs = np.array([0.1, -0.2, 0.0, 0.0, 0.05], dtype=np.float64)
    intrinsics = {
        "camera_matrix": camera_matrix,
        "dist_coeffs": dist_coeffs,
        "reprojection_error_per_image": [0.5, 0.6],
        "mean_reprojection_error": 0.55,
        "image_size": (640, 480),
        "n_images_used": 2,
    }

    out = tmp_path / "intrinsics.json"
    save_intrinsics(intrinsics, out)
    loaded = load_intrinsics(out)

    np.testing.assert_array_equal(
        loaded["camera_matrix"],
        camera_matrix,
        err_msg="Camera matrix changed after save/load",
    )
    np.testing.assert_array_equal(loaded["dist_coeffs"], dist_coeffs)
    assert loaded["image_size"] == [640, 480]
    assert loaded["n_images_used"] == 2


def test_save_produces_valid_json(tmp_path):
    """The output file must be parseable JSON with the expected keys."""
    camera_matrix = np.eye(3, dtype=np.float64)
    intrinsics = {
        "camera_matrix": camera_matrix,
        "dist_coeffs": np.zeros(5, dtype=np.float64),
        "reprojection_error_per_image": [],
        "mean_reprojection_error": 0.0,
        "image_size": (640, 480),
        "n_images_used": 0,
    }
    out = tmp_path / "intrinsics.json"
    save_intrinsics(intrinsics, out)

    with out.open() as fh:
        parsed = json.load(fh)

    assert "camera_matrix" in parsed
    assert "dist_coeffs" in parsed


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_validate_intrinsics_warns_on_high_error():
    """validate_intrinsics should emit a UserWarning when error exceeds threshold."""
    bad_intrinsics = {
        "camera_matrix": np.eye(3),
        "dist_coeffs": np.zeros(5),
        "mean_reprojection_error": 2.5,
        "image_size": (640, 480),
        "n_images_used": 10,
    }

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = validate_intrinsics(bad_intrinsics, max_acceptable_error=1.0)

    assert result is False
    assert any(issubclass(w.category, UserWarning) for w in caught), (
        "Expected a UserWarning for high reprojection error"
    )


def test_validate_intrinsics_passes_on_low_error():
    """validate_intrinsics should return True and emit no warning for good calibration."""
    good_intrinsics = {
        "camera_matrix": np.eye(3),
        "dist_coeffs": np.zeros(5),
        "mean_reprojection_error": 0.3,
        "image_size": (640, 480),
        "n_images_used": 20,
    }

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = validate_intrinsics(good_intrinsics, max_acceptable_error=1.0)

    assert result is True
    assert not any(issubclass(w.category, UserWarning) for w in caught)
