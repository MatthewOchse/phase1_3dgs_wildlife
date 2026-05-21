# Progress Log

## Week 1 — Project Scaffolding

- Initialised repository structure with `src/` layout covering calibration, sync, segmentation, features, geometry, and io modules.
- Created `pyproject.toml` with full dependency list and dev extras.
- Set up conda environment `wildsplat` for the project.
- Placeholder documentation written for algorithms and architecture decisions.

## Step 1 — Camera Intrinsic Calibration (complete)

- Implemented `src/calibration/intrinsics.py`: Charuco board generation, corner detection via `CharucoDetector` (OpenCV 4.x API), multi-image calibration, JSON save/load, and reprojection-error validation.
- Added `scripts/calibrate_intrinsics.py` CLI accepting camera ID, images directory, output path, and YAML board config.
- Tests in `tests/test_intrinsics.py` cover synthetic detection, save/load round-trip, and warning behaviour on high error.

## Step 2 — Multi-camera Extrinsic Calibration (complete)

- Implemented `src/calibration/extrinsics.py`: board-based 6-DoF pose estimation via `matchImagePoints` + `solvePnP` (OpenCV 4.13 API), rig calibration from one synchronised capture, multi-view DLT triangulation, reprojection-error validation, and unified rig JSON save/load with metadata.
- Added `scripts/calibrate_rig.py` CLI accepting a captures directory, intrinsics directory, board YAML config, and output path.
- Tests in `tests/test_extrinsics.py`: synthetic 4-camera pose recovery (< 0.001 m, < 0.1°), save/load round-trip, triangulation of a known 3-D point (< 0.1 mm), and error-handling checks.
