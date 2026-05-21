"""Multi-camera extrinsic calibration using a shared Charuco board world frame.

Coordinate convention
---------------------
* **World frame** = the Charuco board frame.  The board lies in the Z=0 plane;
  X points right along the board's first column of squares, Y points down along
  the first row, Z points out of the board surface toward the cameras.
* **Camera frame** follows the standard OpenCV convention: X right, Y down,
  +Z pointing into the scene (away from the camera).
* ``T_world_from_cam`` is a 4×4 homogeneous matrix that transforms a point
  expressed in camera coordinates into world (board) coordinates:
      p_world = T_world_from_cam @ p_cam
  Its inverse ``T_cam_from_world`` is what OpenCV's solvePnP returns
  (it maps world points to camera points for projection).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_SOFTWARE_VERSION = "0.1.0"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _projection_matrix(intrinsics: dict, T_world_from_cam: np.ndarray) -> np.ndarray:
    """Build a 3×4 projection matrix P = K @ [R | t] (world → pixel).

    Parameters
    ----------
    intrinsics:
        Dict with ``camera_matrix`` (3×3) and ``dist_coeffs``.
    T_world_from_cam:
        4×4 homogeneous pose matrix.

    Returns
    -------
    np.ndarray
        3×4 projection matrix.
    """
    K = np.array(intrinsics["camera_matrix"], dtype=np.float64)
    T_cam_from_world = np.linalg.inv(np.array(T_world_from_cam, dtype=np.float64))
    Rt = T_cam_from_world[:3, :]  # 3×4
    return K @ Rt


def _pose_from_correspondences(
    obj_pts: np.ndarray,
    img_pts: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> Optional[np.ndarray]:
    """Estimate camera pose from 3D–2D correspondences via PnP.

    Parameters
    ----------
    obj_pts:
        Array of shape ``(N, 1, 3)`` or ``(N, 3)`` — 3-D world points.
    img_pts:
        Array of shape ``(N, 1, 2)`` or ``(N, 2)`` — corresponding pixels.
    camera_matrix:
        3×3 intrinsic matrix K.
    dist_coeffs:
        Distortion coefficient vector.

    Returns
    -------
    np.ndarray or None
        4×4 ``T_world_from_cam`` if solvePnP succeeds, else ``None``.
    """
    ret, rvec, tvec = cv2.solvePnP(
        obj_pts, img_pts, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not ret:
        return None

    R, _ = cv2.Rodrigues(rvec)
    T_cam_from_world = np.eye(4, dtype=np.float64)
    T_cam_from_world[:3, :3] = R
    T_cam_from_world[:3, 3] = tvec.flatten()
    return np.linalg.inv(T_cam_from_world)


# ---------------------------------------------------------------------------
# Pose estimation
# ---------------------------------------------------------------------------


def estimate_camera_pose_from_board(
    image: np.ndarray,
    intrinsics: dict,
    board: cv2.aruco.CharucoBoard,
    dictionary: cv2.aruco.Dictionary,
    min_corners: int = 4,
) -> Optional[np.ndarray]:
    """Detect the Charuco board in an image and return the camera's 6-DoF pose.

    The board defines the world frame (see module docstring for conventions).
    Detection uses OpenCV 4.x's ``CharucoDetector``; partial board visibility
    is handled as long as ``min_corners`` interior corners are found.

    Parameters
    ----------
    image:
        Grayscale or BGR image.
    intrinsics:
        Dict with ``camera_matrix`` (3×3 ndarray) and ``dist_coeffs``.
    board:
        Charuco board definition.
    dictionary:
        ArUco dictionary used by the board.
    min_corners:
        Minimum detected corners to attempt pose estimation.

    Returns
    -------
    np.ndarray or None
        4×4 ``T_world_from_cam`` homogeneous matrix, or ``None`` if the board
        could not be detected or pose estimation failed.
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    detector = cv2.aruco.CharucoDetector(board)
    charuco_corners, charuco_ids, _, _ = detector.detectBoard(gray)

    if charuco_corners is None or len(charuco_corners) < min_corners:
        logger.debug("Pose estimation: insufficient corners detected (%s)", charuco_corners)
        return None

    obj_pts, img_pts = board.matchImagePoints(charuco_corners, charuco_ids)
    if obj_pts is None or len(obj_pts) < min_corners:
        return None

    K = np.array(intrinsics["camera_matrix"], dtype=np.float64)
    dist = np.array(intrinsics["dist_coeffs"], dtype=np.float64)
    return _pose_from_correspondences(obj_pts, img_pts, K, dist)


# ---------------------------------------------------------------------------
# Rig extrinsic calibration
# ---------------------------------------------------------------------------


def calibrate_rig_extrinsics(
    image_paths_per_camera: dict[str, Path],
    intrinsics_per_camera: dict[str, dict],
    board: cv2.aruco.CharucoBoard,
    dictionary: cv2.aruco.Dictionary,
) -> dict[str, np.ndarray]:
    """Estimate world-from-camera poses for an N-camera rig in one shared capture.

    Each camera captures the same Charuco board simultaneously.  The board
    defines the common world frame, so all poses are expressed in the same
    coordinate system without any additional hand-eye calibration step.

    Parameters
    ----------
    image_paths_per_camera:
        Mapping from camera ID to the path of its single calibration image.
    intrinsics_per_camera:
        Mapping from camera ID to its intrinsics dict (``camera_matrix``,
        ``dist_coeffs``).
    board:
        Charuco board definition — must match the physical target.
    dictionary:
        ArUco dictionary used by the board.

    Returns
    -------
    dict[str, np.ndarray]
        Mapping from camera ID to its 4×4 ``T_world_from_cam`` matrix.

    Raises
    ------
    RuntimeError
        If no camera yields a valid pose estimate.
    """
    extrinsics: dict[str, np.ndarray] = {}

    for cam_id, img_path in image_paths_per_camera.items():
        intr = intrinsics_per_camera[cam_id]
        img = cv2.imread(str(img_path))
        if img is None:
            logger.warning("Could not read image for camera %s: %s", cam_id, img_path)
            continue

        pose = estimate_camera_pose_from_board(img, intr, board, dictionary)
        if pose is None:
            logger.warning("Pose estimation failed for camera %s", cam_id)
            continue

        extrinsics[cam_id] = pose
        logger.info("Camera %s: pose estimated successfully", cam_id)

    if not extrinsics:
        raise RuntimeError("No camera yielded a valid pose estimate.")

    return extrinsics


# ---------------------------------------------------------------------------
# Multi-view triangulation (DLT)
# ---------------------------------------------------------------------------


def triangulate_test_points(
    points_per_camera: dict[str, np.ndarray],
    intrinsics_per_camera: dict[str, dict],
    extrinsics_per_camera: dict[str, np.ndarray],
) -> np.ndarray:
    """Triangulate 2-D observations from multiple cameras to 3-D via DLT.

    Implements the linear multi-view Direct Linear Transform (DLT).  For each
    3-D point, the constraint that projected pixel = observed pixel is
    linearised to ``A @ X = 0`` (2 rows per camera), then solved by SVD.
    Using all cameras simultaneously is more accurate than averaging pairwise
    ``cv2.triangulatePoints`` results because it minimises a single global
    algebraic error rather than combining separately noisy pair solutions.

    Points are undistorted before building the linear system so that the
    projection matrices can be the simple pinhole form K @ [R|t].

    Parameters
    ----------
    points_per_camera:
        Mapping from camera ID to ``(N, 2)`` array of observed pixel
        coordinates.  All cameras must observe the same N points (in the same
        order).
    intrinsics_per_camera:
        Mapping from camera ID to intrinsics dict (``camera_matrix``,
        ``dist_coeffs``).
    extrinsics_per_camera:
        Mapping from camera ID to 4×4 ``T_world_from_cam``.

    Returns
    -------
    np.ndarray
        Array of shape ``(N, 3)`` containing triangulated 3-D world points.

    Raises
    ------
    ValueError
        If fewer than two cameras are provided.
    """
    cam_ids = list(points_per_camera.keys())
    if len(cam_ids) < 2:
        raise ValueError("Need at least two cameras for triangulation.")

    # Pre-compute projection matrices and undistort all points
    proj_matrices: list[np.ndarray] = []
    undistorted: list[np.ndarray] = []  # each (N, 1, 2)

    for cam_id in cam_ids:
        intr = intrinsics_per_camera[cam_id]
        K = np.array(intr["camera_matrix"], dtype=np.float64)
        dist = np.array(intr["dist_coeffs"], dtype=np.float64)
        pts = np.array(points_per_camera[cam_id], dtype=np.float64).reshape(-1, 1, 2)

        ud = cv2.undistortPoints(pts, K, dist, P=K)  # keep in pixel space
        undistorted.append(ud.reshape(-1, 2))

        P = _projection_matrix(intr, extrinsics_per_camera[cam_id])
        proj_matrices.append(P)

    n_points = undistorted[0].shape[0]
    result = np.zeros((n_points, 3), dtype=np.float64)

    for i in range(n_points):
        rows = []
        for j, (P, ud) in enumerate(zip(proj_matrices, undistorted)):
            u, v = float(ud[i, 0]), float(ud[i, 1])
            # Two linearly independent equations from x × (PX) = 0
            rows.append(u * P[2, :] - P[0, :])
            rows.append(v * P[2, :] - P[1, :])

        A = np.stack(rows)  # (2*N_cams, 4)
        _, _, Vt = np.linalg.svd(A)
        X_h = Vt[-1]  # homogeneous solution
        result[i] = X_h[:3] / X_h[3]

    return result


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_extrinsics(
    extrinsics_dict: dict[str, np.ndarray],
    intrinsics_dict: dict[str, dict],
    test_image_per_camera: dict[str, np.ndarray],
    board: cv2.aruco.CharucoBoard,
    dictionary: cv2.aruco.Dictionary,
) -> dict:
    """Validate rig extrinsics by triangulating board corners and reprojecting.

    For each camera's test image:
      1. Detect Charuco corners and retrieve their 3-D board positions.
      2. Identify corners visible in at least two cameras.
      3. Triangulate those corners using :func:`triangulate_test_points`.
      4. Project the triangulated 3-D points back into each camera.
      5. Compute RMS reprojection error against the original detections.

    Parameters
    ----------
    extrinsics_dict:
        Mapping from camera ID to 4×4 ``T_world_from_cam``.
    intrinsics_dict:
        Mapping from camera ID to intrinsics dict.
    test_image_per_camera:
        Mapping from camera ID to a test image (BGR or grayscale ndarray).
    board:
        Charuco board definition.
    dictionary:
        ArUco dictionary.

    Returns
    -------
    dict with keys:
        ``mean_error_per_camera`` (dict[str, float]) and
        ``overall_mean_error`` (float).
    """
    detector = cv2.aruco.CharucoDetector(board)

    # ---- Detect corners and 3-D positions in each camera ----
    detections: dict[str, dict] = {}
    for cam_id, img in test_image_per_camera.items():
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        corners, ids, _, _ = detector.detectBoard(gray)
        if corners is None or len(corners) < 4:
            logger.warning("Validation: insufficient corners in camera %s", cam_id)
            continue
        obj_pts, img_pts = board.matchImagePoints(corners, ids)
        flat_ids = ids.flatten().tolist()
        detections[cam_id] = {
            "ids": flat_ids,
            "img_pts": img_pts.reshape(-1, 2),
            "obj_pts": obj_pts.reshape(-1, 3),
        }

    if len(detections) < 2:
        raise RuntimeError("Validation requires at least two cameras with detections.")

    # ---- Find corner IDs visible in all detected cameras ----
    common_ids = set(detections[next(iter(detections))]["ids"])
    for cam_id in detections:
        common_ids &= set(detections[cam_id]["ids"])

    if not common_ids:
        raise RuntimeError("No corners visible in all cameras simultaneously.")

    common_ids_list = sorted(common_ids)

    # ---- Build per-camera 2-D observations for common corners ----
    points_per_cam: dict[str, np.ndarray] = {}
    obj_pts_known: Optional[np.ndarray] = None

    for cam_id, det in detections.items():
        idx = [det["ids"].index(cid) for cid in common_ids_list]
        points_per_cam[cam_id] = det["img_pts"][idx]
        if obj_pts_known is None:
            obj_pts_known = det["obj_pts"][idx]

    # ---- Triangulate ----
    pts3d = triangulate_test_points(
        points_per_cam,
        {c: intrinsics_dict[c] for c in points_per_cam},
        {c: extrinsics_dict[c] for c in points_per_cam},
    )

    # ---- Reproject into each camera and compute error ----
    mean_error_per_camera: dict[str, float] = {}

    for cam_id in points_per_cam:
        intr = intrinsics_dict[cam_id]
        K = np.array(intr["camera_matrix"], dtype=np.float64)
        dist = np.array(intr["dist_coeffs"], dtype=np.float64)
        T_cam_from_world = np.linalg.inv(extrinsics_dict[cam_id])
        R = T_cam_from_world[:3, :3]
        t = T_cam_from_world[:3, 3]
        rvec, _ = cv2.Rodrigues(R)

        projected, _ = cv2.projectPoints(pts3d, rvec, t, K, dist)
        observed = points_per_cam[cam_id]
        err = float(np.sqrt(np.mean((observed - projected.reshape(-1, 2)) ** 2)))
        mean_error_per_camera[cam_id] = err
        logger.info("Camera %s validation reprojection error: %.4f px", cam_id, err)

    overall = float(np.mean(list(mean_error_per_camera.values())))
    logger.info("Overall multi-view reprojection error: %.4f px", overall)

    return {
        "mean_error_per_camera": mean_error_per_camera,
        "overall_mean_error": overall,
    }


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------


def _to_serialisable(obj):
    """Recursively convert numpy arrays to lists for JSON."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _to_serialisable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serialisable(v) for v in obj]
    return obj


def save_rig_calibration(
    extrinsics_dict: dict[str, np.ndarray],
    intrinsics_dict: dict[str, dict],
    output_path: Path,
    board_config: Optional[dict] = None,
) -> None:
    """Save a complete rig calibration (all cameras) to a single JSON file.

    Parameters
    ----------
    extrinsics_dict:
        Mapping from camera ID to 4×4 ``T_world_from_cam``.
    intrinsics_dict:
        Mapping from camera ID to intrinsics dict.
    output_path:
        Destination JSON path.  Parent directories are created.
    board_config:
        Optional dict of board parameters to embed as metadata.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cameras: dict = {}
    for cam_id in set(list(extrinsics_dict.keys()) + list(intrinsics_dict.keys())):
        entry: dict = {}
        if cam_id in intrinsics_dict:
            intr = intrinsics_dict[cam_id]
            entry["camera_matrix"] = _to_serialisable(intr["camera_matrix"])
            entry["dist_coeffs"] = _to_serialisable(intr.get("dist_coeffs", []))
            entry["image_size"] = intr.get("image_size", None)
            entry["mean_reprojection_error"] = intr.get("mean_reprojection_error", None)
        if cam_id in extrinsics_dict:
            entry["T_world_from_cam"] = _to_serialisable(extrinsics_dict[cam_id])
        cameras[cam_id] = entry

    payload = {
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "software_version": _SOFTWARE_VERSION,
            "board_config": board_config or {},
            "coordinate_convention": (
                "World frame = Charuco board frame (Z=0 plane). "
                "T_world_from_cam maps camera coords to world coords. "
                "OpenCV camera convention: +Z into scene."
            ),
        },
        "cameras": cameras,
    }

    with output_path.open("w") as fh:
        json.dump(payload, fh, indent=2)

    logger.info("Rig calibration saved to %s (%d cameras)", output_path, len(cameras))


def load_rig_calibration(path: Path) -> dict[str, dict]:
    """Load a rig calibration JSON produced by :func:`save_rig_calibration`.

    Parameters
    ----------
    path:
        Path to the JSON file.

    Returns
    -------
    dict[str, dict]
        Mapping from camera ID to a dict containing:
        ``camera_matrix`` (3×3 ndarray), ``dist_coeffs`` (1-D ndarray),
        ``T_world_from_cam`` (4×4 ndarray), ``image_size``, and
        ``mean_reprojection_error`` if present.
    """
    with Path(path).open() as fh:
        payload = json.load(fh)

    cameras_raw = payload["cameras"]
    result: dict[str, dict] = {}

    for cam_id, entry in cameras_raw.items():
        cam: dict = {}
        if "camera_matrix" in entry:
            cam["camera_matrix"] = np.array(entry["camera_matrix"], dtype=np.float64)
        if "dist_coeffs" in entry:
            cam["dist_coeffs"] = np.array(entry["dist_coeffs"], dtype=np.float64)
        if "T_world_from_cam" in entry:
            cam["T_world_from_cam"] = np.array(entry["T_world_from_cam"], dtype=np.float64)
        for scalar_key in ("image_size", "mean_reprojection_error"):
            if scalar_key in entry:
                cam[scalar_key] = entry[scalar_key]
        result[cam_id] = cam

    return result
