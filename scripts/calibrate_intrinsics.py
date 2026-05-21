#!/usr/bin/env python3
"""CLI entry point for camera intrinsic calibration.

Usage
-----
    python scripts/calibrate_intrinsics.py \\
        --camera-id cam0 \\
        --images-dir data/calibration/cam0 \\
        --output output/calibration \\
        --board-config configs/board.yaml

Board config YAML schema
------------------------
::

    squares_x: 5
    squares_y: 7
    square_length_m: 0.04
    marker_length_m: 0.03
    dictionary: DICT_5X5_100   # optional, default DICT_5X5_100
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2
import yaml

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from calibration.intrinsics import (
    calibrate_camera_from_images,
    generate_charuco_board,
    save_intrinsics,
    validate_intrinsics,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate camera intrinsics from Charuco calibration images."
    )
    parser.add_argument(
        "--camera-id",
        required=True,
        help="Identifier for the camera (used in the output filename).",
    )
    parser.add_argument(
        "--images-dir",
        required=True,
        type=Path,
        help="Directory containing calibration images.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output directory; intrinsics saved to <output>/intrinsics_<camera-id>.json.",
    )
    parser.add_argument(
        "--board-config",
        required=True,
        type=Path,
        help="Path to YAML file with board dimensions.",
    )
    return parser.parse_args()


def load_board_config(path: Path) -> dict:
    with path.open() as fh:
        cfg = yaml.safe_load(fh)
    required = {"squares_x", "squares_y", "square_length_m", "marker_length_m"}
    missing = required - cfg.keys()
    if missing:
        raise ValueError(f"Board config missing keys: {missing}")
    return cfg


def main() -> int:
    args = parse_args()

    # ---- Load board config ----
    cfg = load_board_config(args.board_config)
    dictionary_name = cfg.get("dictionary", "DICT_5X5_100")
    board, dictionary = generate_charuco_board(
        squares_x=cfg["squares_x"],
        squares_y=cfg["squares_y"],
        square_length_m=cfg["square_length_m"],
        marker_length_m=cfg["marker_length_m"],
        dictionary_name=dictionary_name,
    )
    logger.info(
        "Board: %dx%d squares, square=%.3fm, marker=%.3fm, dict=%s",
        cfg["squares_x"],
        cfg["squares_y"],
        cfg["square_length_m"],
        cfg["marker_length_m"],
        dictionary_name,
    )

    # ---- Collect images ----
    images_dir = args.images_dir
    if not images_dir.is_dir():
        logger.error("images-dir does not exist: %s", images_dir)
        return 1

    image_paths = sorted(
        p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        logger.error("No images found in %s", images_dir)
        return 1
    logger.info("Found %d image(s) in %s", len(image_paths), images_dir)

    # ---- Determine image size from first readable image ----
    first = cv2.imread(str(image_paths[0]))
    if first is None:
        logger.error("Cannot read first image: %s", image_paths[0])
        return 1
    h, w = first.shape[:2]
    image_size = (w, h)

    # ---- Calibrate ----
    try:
        intrinsics = calibrate_camera_from_images(
            image_paths=image_paths,
            board=board,
            dictionary=dictionary,
            image_size=image_size,
        )
    except RuntimeError as exc:
        logger.error("Calibration failed: %s", exc)
        return 1

    logger.info(
        "Used %d/%d images. Mean reprojection error: %.4f px",
        intrinsics["n_images_used"],
        len(image_paths),
        intrinsics["mean_reprojection_error"],
    )

    # ---- Validate ----
    validate_intrinsics(intrinsics)

    # ---- Save ----
    output_path = args.output / f"intrinsics_{args.camera_id}.json"
    save_intrinsics(intrinsics, output_path)
    logger.info("Intrinsics written to %s", output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
