#!/usr/bin/env python3
"""CLI entry point for multi-camera rig extrinsic calibration.

Each camera must have one image of the shared Charuco board taken
simultaneously (or with the board stationary).  Intrinsics JSONs for every
camera must already exist (produced by calibrate_intrinsics.py).

Usage
-----
    python scripts/calibrate_rig.py \\
        --captures-dir data/calibration/rig_capture \\
        --intrinsics-dir output/calibration \\
        --output output/calibration/rig.json \\
        --board-config configs/board.yaml

Directory layout expected under ``--captures-dir``
----------------------------------------------------
One image file per camera, named <camera_id>.<ext>, e.g.::

    rig_capture/
        cam0.jpg
        cam1.jpg
        cam2.jpg

Intrinsics files under ``--intrinsics-dir``::

    output/calibration/
        intrinsics_cam0.json
        intrinsics_cam1.json
        intrinsics_cam2.json
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from calibration.extrinsics import calibrate_rig_extrinsics, save_rig_calibration
from calibration.intrinsics import generate_charuco_board, load_intrinsics

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate extrinsic poses for a multi-camera rig via a Charuco board."
    )
    parser.add_argument(
        "--captures-dir",
        required=True,
        type=Path,
        help="Directory with one board image per camera (named <camera_id>.<ext>).",
    )
    parser.add_argument(
        "--intrinsics-dir",
        required=True,
        type=Path,
        help="Directory containing intrinsics_<camera_id>.json files.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output path for the combined rig calibration JSON.",
    )
    parser.add_argument(
        "--board-config",
        required=True,
        type=Path,
        help="YAML file with board dimensions (squares_x, squares_y, square_length_m, "
        "marker_length_m, dictionary).",
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

    # ---- Board ----
    cfg = load_board_config(args.board_config)
    board, dictionary = generate_charuco_board(
        squares_x=cfg["squares_x"],
        squares_y=cfg["squares_y"],
        square_length_m=cfg["square_length_m"],
        marker_length_m=cfg["marker_length_m"],
        dictionary_name=cfg.get("dictionary", "DICT_5X5_100"),
    )

    # ---- Discover cameras from captures dir ----
    captures_dir = args.captures_dir
    if not captures_dir.is_dir():
        logger.error("captures-dir does not exist: %s", captures_dir)
        return 1

    image_paths: dict[str, Path] = {}
    for p in sorted(captures_dir.iterdir()):
        if p.suffix.lower() in IMAGE_EXTENSIONS:
            cam_id = p.stem
            image_paths[cam_id] = p

    if not image_paths:
        logger.error("No images found in %s", captures_dir)
        return 1

    logger.info("Found %d camera images: %s", len(image_paths), list(image_paths.keys()))

    # ---- Load intrinsics ----
    intrinsics_dir = args.intrinsics_dir
    intrinsics: dict = {}
    for cam_id in list(image_paths.keys()):
        intr_path = intrinsics_dir / f"intrinsics_{cam_id}.json"
        if not intr_path.exists():
            logger.warning("No intrinsics found for camera %s at %s — skipping", cam_id, intr_path)
            image_paths.pop(cam_id)
            continue
        intrinsics[cam_id] = load_intrinsics(intr_path)
        logger.info("Loaded intrinsics for %s", cam_id)

    if len(intrinsics) < 2:
        logger.error("Need intrinsics for at least 2 cameras; got %d", len(intrinsics))
        return 1

    # ---- Extrinsic calibration ----
    try:
        extrinsics = calibrate_rig_extrinsics(
            image_paths_per_camera=image_paths,
            intrinsics_per_camera=intrinsics,
            board=board,
            dictionary=dictionary,
        )
    except RuntimeError as exc:
        logger.error("Extrinsic calibration failed: %s", exc)
        return 1

    logger.info(
        "Poses estimated for cameras: %s", list(extrinsics.keys())
    )

    # ---- Save ----
    save_rig_calibration(
        extrinsics_dict=extrinsics,
        intrinsics_dict=intrinsics,
        output_path=args.output,
        board_config=cfg,
    )
    logger.info("Rig calibration written to %s", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
