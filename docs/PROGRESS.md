# Progress Log

## Week 1 — Project Scaffolding

- Initialised repository structure with `src/` layout covering calibration, sync, segmentation, features, geometry, and io_utils modules.
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

## Step 3 — Data Loading and Synchronisation Layer (complete)

### 3a — DANNCE dataset download script
- Created `scripts/setup_dannce_mm1.py`: idempotent download, extraction, and verification of the DANNCE markerless_mouse_1 dataset. Supports `--force` and `--verify-only` flags.
- Actual download: **5.4 GB** zip (not ~1 GB as the README implies). Each camera has **6 MP4 segments** named by global start frame index (0.mp4, 3000.mp4, …, 15000.mp4), not a single file.
- MAT file keys: `params`, `sync`, `labelData`, `camnames`, `com`. Labels are present (81 labeled frames).

### 3b — Dataset inspection
- Created `scripts/inspect_dannce_mm1.py` and `docs/DATASET_DANNCE_MM1.md`.
- Key quirks discovered:
  - **K and R are stored transposed** in the MAT file. Must apply `.T` to both before using as OpenCV matrices.
  - **100 fps**, not 50 fps. DANNCE estimates pose at 50 Hz by subsampling; raw video is 100 Hz.
  - **Hardware sync** — `data_frame` is `[0, 1, …, 17999]` identically across all cameras. No correction needed.
  - **Distortion:** `RDistort = [k1, k2, k3]`, `TDistort = [p1, p2]`; OpenCV order `[k1, k2, p1, p2, k3]`.
  - **No audio track** — cross-correlation sync not applicable to this dataset.

### 3c — Canonical MultiViewCapture and sync layer
- Created `src/io_utils/dataset.py`: `CaptureFrame` (lazy image loading, video-backed or image-backed), `Timestep`, and `MultiViewCapture` with `from_video_files`, `from_image_directories`, `get_timestep`, `get_frame_at_index`, `iter_timesteps`, `n_timesteps`, `summary`.
- Created `src/sync/alignment.py`: audio cross-correlation offsets via ffmpeg + scipy, offset application, clap detection, clap-based validation.
- Note: `src/io` was renamed to `src/io_utils` because `io` is a **frozen Python built-in** and cannot be shadowed on `sys.path`.
- 29 passing unit tests in `tests/test_dataset.py`, all without the real dataset.

### 3d — DANNCE adapter
- Created `src/io_utils/dannce_loader.py` with full adapter: calibration loading (with transposed K/R correction and distortion reordering), sync extraction, label loading, session loading with segmented-video frame resolution, and `extract_dannce_demo_frames` utility.
- Created `scripts/load_dannce_mm1.py` CLI with `--validate-triangulation` flag.
- **Reprojection error on labeled frames:**

  | Camera | Mean | Max |
  |--------|------|-----|
  | Camera1 | 0.83 px | 1.61 px |
  | Camera2 | 0.21 px | 0.47 px |
  | Camera3 | 0.14 px | 0.28 px |
  | Camera4 | 0.76 px | 1.58 px |
  | Camera5 | 0.14 px | 0.39 px |
  | Camera6 | 0.35 px | 1.34 px |
  | **Overall** | **0.40 px** | **1.61 px** |

  All sub-pixel. Loader coordinate convention confirmed correct.

- 19/19 dataset tests passing (`pytest -m dataset`).

### Open questions
- Camera1 and Camera4 have higher reprojection error (~0.8 px) than Camera3 and Camera5 (~0.14 px). This may reflect calibration quality differences in the original DANNCE data, not a loader bug.
- The `src/io_utils` module name (renamed from `src/io` to avoid collision with Python's frozen built-in) is now reflected in the README directory structure.
