"""Camera intrinsic calibration using a Charuco board (OpenCV 4.x new API)."""

from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Board generation
# ---------------------------------------------------------------------------


def generate_charuco_board(
    squares_x: int,
    squares_y: int,
    square_length_m: float,
    marker_length_m: float,
    dictionary_name: str = "DICT_5X5_100",
    save_path: Optional[Path] = None,
    image_size: tuple[int, int] = (800, 600),
) -> tuple[cv2.aruco.CharucoBoard, cv2.aruco.Dictionary]:
    """Create a Charuco board and optionally save a printable PNG.

    Parameters
    ----------
    squares_x:
        Number of chessboard squares along the x-axis.
    squares_y:
        Number of chessboard squares along the y-axis.
    square_length_m:
        Physical side length of each chessboard square in metres.
    marker_length_m:
        Physical side length of each ArUco marker in metres.
        Must be smaller than ``square_length_m``.
    dictionary_name:
        Name of the ArUco dictionary constant, e.g. ``'DICT_5X5_100'``.
    save_path:
        If provided, write the board image as a PNG at this path.
    image_size:
        ``(width, height)`` in pixels of the generated board image.

    Returns
    -------
    board:
        The ``cv2.aruco.CharucoBoard`` instance.
    dictionary:
        The corresponding ArUco dictionary.
    """
    dict_id = getattr(cv2.aruco, dictionary_name)
    dictionary = cv2.aruco.getPredefinedDictionary(dict_id)
    board = cv2.aruco.CharucoBoard(
        (squares_x, squares_y), square_length_m, marker_length_m, dictionary
    )

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        board_img = board.generateImage(image_size)
        cv2.imwrite(str(save_path), board_img)
        logger.info("Saved Charuco board image to %s", save_path)

    return board, dictionary


# ---------------------------------------------------------------------------
# Corner detection
# ---------------------------------------------------------------------------


def detect_charuco_corners(
    image: np.ndarray,
    board: cv2.aruco.CharucoBoard,
    dictionary: cv2.aruco.Dictionary,
    min_corners: int = 4,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Detect ArUco markers and interpolate Charuco corners in an image.

    Uses the ``CharucoDetector`` introduced in OpenCV 4.6+. Partial board
    visibility is handled gracefully — as long as enough corners are found
    the result is usable for calibration.

    Parameters
    ----------
    image:
        Grayscale or BGR image.
    board:
        The Charuco board definition.
    dictionary:
        The ArUco dictionary used by the board.
    min_corners:
        Minimum number of corners required to consider detection successful.

    Returns
    -------
    charuco_corners:
        Array of shape ``(N, 1, 2)`` with sub-pixel 2-D corner positions,
        or ``None`` if detection failed.
    charuco_ids:
        Array of shape ``(N, 1)`` with corner IDs,
        or ``None`` if detection failed.
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    detector = cv2.aruco.CharucoDetector(board)
    charuco_corners, charuco_ids, _marker_corners, _marker_ids = detector.detectBoard(
        gray
    )

    if charuco_corners is None or len(charuco_corners) < min_corners:
        return None, None

    return charuco_corners, charuco_ids


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def calibrate_camera_from_images(
    image_paths: list[Path],
    board: cv2.aruco.CharucoBoard,
    dictionary: cv2.aruco.Dictionary,
    image_size: tuple[int, int],
    min_corners: int = 4,
) -> dict:
    """Estimate camera intrinsics from a set of Charuco calibration images.

    Iterates over the supplied images, detects Charuco corners in each,
    accumulates the correspondences, then calls ``cv2.calibrateCamera``.
    Per-image reprojection error is computed after calibration.

    Parameters
    ----------
    image_paths:
        Paths to calibration images (any OpenCV-readable format).
    board:
        Charuco board definition matching the physical target.
    dictionary:
        ArUco dictionary used by the board.
    image_size:
        ``(width, height)`` of the images — all images must share the same
        resolution.
    min_corners:
        Minimum detected corners to accept an image for calibration.

    Returns
    -------
    dict with keys:
        ``camera_matrix`` (3×3 ndarray),
        ``dist_coeffs`` (1-D ndarray),
        ``reprojection_error_per_image`` (list of floats, one per used image),
        ``mean_reprojection_error`` (float),
        ``image_size`` (tuple),
        ``n_images_used`` (int).

    Raises
    ------
    RuntimeError
        If fewer than two images provide usable corner detections.
    """
    all_obj_pts: list[np.ndarray] = []
    all_img_pts: list[np.ndarray] = []
    used_paths: list[Path] = []

    for path in image_paths:
        img = cv2.imread(str(path))
        if img is None:
            logger.warning("Could not read image: %s — skipping", path)
            continue

        corners, ids = detect_charuco_corners(img, board, dictionary, min_corners)
        if corners is None:
            logger.debug("No corners detected in %s — skipping", path)
            continue

        obj_pts, img_pts = board.matchImagePoints(corners, ids)
        if obj_pts is None or len(obj_pts) < min_corners:
            continue

        all_obj_pts.append(obj_pts)
        all_img_pts.append(img_pts)
        used_paths.append(path)

    if len(all_obj_pts) < 2:
        raise RuntimeError(
            f"Only {len(all_obj_pts)} image(s) yielded detections; "
            "need at least 2 for calibration."
        )

    logger.info("Calibrating from %d images …", len(all_obj_pts))
    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        all_obj_pts, all_img_pts, image_size, None, None
    )
    logger.info("Overall RMS reprojection error: %.4f px", rms)

    per_image_errors: list[float] = []
    for obj_pts, img_pts, rvec, tvec in zip(all_obj_pts, all_img_pts, rvecs, tvecs):
        projected, _ = cv2.projectPoints(obj_pts, rvec, tvec, camera_matrix, dist_coeffs)
        err = float(np.sqrt(np.mean((img_pts - projected) ** 2)))
        per_image_errors.append(err)

    return {
        "camera_matrix": camera_matrix,
        "dist_coeffs": dist_coeffs.flatten(),
        "reprojection_error_per_image": per_image_errors,
        "mean_reprojection_error": float(np.mean(per_image_errors)),
        "image_size": image_size,
        "n_images_used": len(all_obj_pts),
    }


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------


def save_intrinsics(intrinsics_dict: dict, output_path: Path) -> None:
    """Serialise intrinsics to a JSON file.

    Numpy arrays are converted to nested Python lists so they round-trip
    through JSON without loss of precision at float64.

    Parameters
    ----------
    intrinsics_dict:
        Dict as returned by :func:`calibrate_camera_from_images`.
    output_path:
        Destination ``.json`` file path. Parent directories are created.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    serialisable = {}
    for key, value in intrinsics_dict.items():
        if isinstance(value, np.ndarray):
            serialisable[key] = value.tolist()
        else:
            serialisable[key] = value

    with output_path.open("w") as fh:
        json.dump(serialisable, fh, indent=2)

    logger.info("Intrinsics saved to %s", output_path)


def load_intrinsics(path: Path) -> dict:
    """Load intrinsics from a JSON file produced by :func:`save_intrinsics`.

    Lists are converted back to numpy arrays for ``camera_matrix`` and
    ``dist_coeffs``; all other fields are returned as-is.

    Parameters
    ----------
    path:
        Path to the ``.json`` file.

    Returns
    -------
    dict
        Same structure as the dict returned by
        :func:`calibrate_camera_from_images`.
    """
    with Path(path).open() as fh:
        data = json.load(fh)

    data["camera_matrix"] = np.array(data["camera_matrix"], dtype=np.float64)
    data["dist_coeffs"] = np.array(data["dist_coeffs"], dtype=np.float64)
    return data


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_intrinsics(
    intrinsics_dict: dict,
    max_acceptable_error: float = 1.0,
) -> bool:
    """Check whether the calibration quality meets the acceptance threshold.

    Parameters
    ----------
    intrinsics_dict:
        Dict as returned by :func:`calibrate_camera_from_images` or
        :func:`load_intrinsics`.
    max_acceptable_error:
        Maximum mean reprojection error in pixels considered acceptable.
        Defaults to 1.0 px — values above this indicate poor calibration.

    Returns
    -------
    bool
        ``True`` if the mean reprojection error is within threshold.
    """
    error = intrinsics_dict.get("mean_reprojection_error", float("inf"))
    if error > max_acceptable_error:
        warnings.warn(
            f"Mean reprojection error {error:.4f} px exceeds the acceptable "
            f"threshold of {max_acceptable_error:.1f} px. "
            "Consider recapturing calibration images with better coverage.",
            UserWarning,
            stacklevel=2,
        )
        return False
    logger.info("Intrinsics validation passed (error=%.4f px)", error)
    return True
