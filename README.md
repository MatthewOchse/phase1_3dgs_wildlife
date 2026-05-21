# Phase 1: Multi-view Wildlife Reconstruction Pipeline

This repository implements the first phase of a master's thesis project at the University of Pretoria on 3D Gaussian Splatting for wildlife biometric re-identification. The pipeline ingests synchronised multi-view video captured from a fixed camera rig, performs camera calibration, image segmentation, and feature-based reconstruction to produce per-animal 3D representations suitable for training a Gaussian Splatting model used in downstream re-identification tasks.

## Setup

Activate the existing conda environment and install the package in editable mode:

```bash
conda activate wildsplat
pip install -e ".[dev]"
```

## Quick Start

```bash
# Step 1 & 2: calibration (requires physical Charuco board and capture rig)
python scripts/calibrate_intrinsics.py --camera-id cam0 --images-dir data/calibration/cam0 \
    --output output/calibration --board-config configs/board.yaml
python scripts/calibrate_rig.py --captures-dir data/calibration/rig_capture \
    --intrinsics-dir output/calibration --output output/calibration/rig.json \
    --board-config configs/board.yaml

# Step 3: download and load the DANNCE markerless_mouse_1 dataset
python scripts/setup_dannce_mm1.py
python scripts/load_dannce_mm1.py --validate-triangulation
```

## Directory Structure

```
phase1_3dgs_wildlife/
├── data/
│   ├── samples/          # Versioned sample clips for testing
│   ├── captures/         # Raw multi-view captures (gitignored)
│   ├── calibration/      # Calibration boards and results (gitignored)
│   └── public/           # Downloaded public datasets (gitignored)
├── docs/
│   ├── PROGRESS.md             # Weekly progress log
│   ├── ALGORITHMS.md           # Algorithm explanations per step
│   ├── DECISIONS.md            # Architecture Decision Records
│   ├── SETUP.md                # Environment setup notes
│   └── DATASET_DANNCE_MM1.md  # DANNCE dataset format documentation
├── notebooks/            # Exploratory Jupyter notebooks
├── output/               # Pipeline outputs (gitignored)
├── scripts/              # CLI entry points and utilities
├── src/
│   ├── calibration/      # Intrinsic and extrinsic calibration
│   ├── features/         # Feature detection and matching
│   ├── geometry/         # Triangulation and visual hull
│   ├── io_utils/         # Data loading — MultiViewCapture, DANNCE adapter
│   ├── segmentation/     # SAM2, classical, and LiDAR segmentation
│   └── sync/             # Multi-view frame synchronisation
└── tests/                # Pytest test suite (pytest -m dataset for real-data tests)
```
