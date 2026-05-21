#!/usr/bin/env python3
"""Load the DANNCE markerless_mouse_1 session and print a summary.

Usage
-----
    python scripts/load_dannce_mm1.py
    python scripts/load_dannce_mm1.py --validate-triangulation
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from io_utils.dannce_loader import (
    extract_dannce_demo_frames,
    load_dannce_labels,
    load_dannce_session,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load and inspect the DANNCE markerless_mouse_1 dataset."
    )
    parser.add_argument(
        "--validate-triangulation",
        action="store_true",
        help=(
            "Project labelled 3D keypoints into each camera, compute "
            "reprojection error against the 2D labels, and report per-camera "
            "mean and max error."
        ),
    )
    return parser.parse_args()


def _dataset_root() -> Path:
    raw = os.environ.get("DANNCE_MM1_PATH", "~/datasets/dannce_mm1")
    return Path(raw).expanduser().resolve()


def print_session_summary(capture) -> None:
    print()
    print(capture.summary())
    print()


def print_intrinsics(capture) -> None:
    print("=== Camera Intrinsics ===")
    for cam_id in sorted(capture.cameras):
        cam = capture.calibration["cameras"][cam_id]
        K = cam["intrinsics"]["camera_matrix"]
        size = cam["intrinsics"]["image_size"]
        dist = cam["intrinsics"]["dist_coeffs"]
        print(
            f"  {cam_id}: fx={K[0,0]:.1f}  fy={K[1,1]:.1f}  "
            f"cx={K[0,2]:.1f}  cy={K[1,2]:.1f}  "
            f"size={size[0]}×{size[1]}  "
            f"dist=[{', '.join(f'{d:.4f}' for d in dist)}]"
        )
    print()


def print_camera_positions(capture) -> None:
    print("=== Camera World Positions (mm) ===")
    for cam_id in sorted(capture.cameras):
        cam = capture.calibration["cameras"][cam_id]
        T_wfc = cam["extrinsics"]["world_from_camera"]
        # Camera centre in world = T_wfc[:3, 3]
        C = T_wfc[:3, 3]
        print(f"  {cam_id}: [{C[0]:8.2f}, {C[1]:8.2f}, {C[2]:8.2f}]")
    print()


def validate_triangulation(capture, dataset_root: Path) -> None:
    """Project labelled 3D points into each camera; report reprojection error."""
    import cv2
    from calibration.extrinsics import triangulate_test_points

    mat_path = dataset_root / "label3d_dannce.mat"
    labels = load_dannce_labels(mat_path)

    if not labels:
        print("No labels available — skipping triangulation validation.")
        return

    cal = capture.calibration["cameras"]
    cam_ids = sorted(cal.keys())

    print("=== Reprojection Error (labelled frames) ===")
    all_errors: dict[str, list[float]] = {c: [] for c in cam_ids}
    skipped = 0

    # Use Camera1 as reference for labeled frame list
    ref_cam = cam_ids[0]
    ld_ref = labels[ref_cam]
    n_joints = ld_ref["data_3d"].shape[1] // 3

    for frame_i in range(len(ld_ref["frame_indices"])):
        for j in range(n_joints):
            X = ld_ref["data_3d"][frame_i, j * 3:j * 3 + 3]
            if np.any(np.isnan(X)) or np.all(X == 0):
                skipped += 1
                continue

            for cam_id in cam_ids:
                cam = cal[cam_id]
                K = cam["intrinsics"]["camera_matrix"]
                dist = cam["intrinsics"]["dist_coeffs"]
                T_wfc = cam["extrinsics"]["world_from_camera"]
                T_cfw = np.linalg.inv(T_wfc)
                R = T_cfw[:3, :3]
                t = T_cfw[:3, 3:]
                rvec, _ = cv2.Rodrigues(R)

                gt_2d = labels[cam_id]["data_2d"][frame_i, j * 2:j * 2 + 2]
                if np.any(np.isnan(gt_2d)):
                    continue

                proj, _ = cv2.projectPoints(X.reshape(1, 1, 3), rvec, t, K, dist)
                err = float(np.linalg.norm(proj[0, 0] - gt_2d))
                if np.isfinite(err):
                    all_errors[cam_id].append(err)

    for cam_id in cam_ids:
        errs = all_errors[cam_id]
        if errs:
            print(
                f"  {cam_id}: mean={np.mean(errs):.2f} px  "
                f"max={np.max(errs):.2f} px  "
                f"(n={len(errs)})"
            )
        else:
            print(f"  {cam_id}: no valid labeled points")

    overall = [e for errs in all_errors.values() for e in errs]
    if overall:
        print(
            f"\n  Overall: mean={np.mean(overall):.2f} px  "
            f"max={np.max(overall):.2f} px\n"
        )
        if np.mean(overall) > 5.0:
            print(
                "  WARNING: mean reprojection error > 5 px — "
                "check coordinate convention in the loader."
            )
        else:
            print("  All cameras sub-5 px — loader convention is correct.")
    print()


def main() -> int:
    args = parse_args()
    root = _dataset_root()

    if not (root / "label3d_dannce.mat").exists():
        logger.error(
            "Dataset not found at %s\n"
            "Run: python scripts/setup_dannce_mm1.py",
            root,
        )
        return 1

    logger.info("Loading session from %s …", root)
    capture = load_dannce_session(root)

    print_session_summary(capture)
    print_intrinsics(capture)
    print_camera_positions(capture)

    if args.validate_triangulation:
        validate_triangulation(capture, root)

    # Extract 5 demo frames
    demo_dir = REPO_ROOT / "output" / "figures" / "dannce_demo_frames"
    logger.info("Extracting 5 demo frames → %s", demo_dir)
    extract_dannce_demo_frames(capture, demo_dir, n_frames=5)
    logger.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
