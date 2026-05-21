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
