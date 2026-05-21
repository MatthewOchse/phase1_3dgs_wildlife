# Phase 1: Multi-view Wildlife Reconstruction Pipeline

This repository implements the first phase of a master's thesis project at the University of Pretoria on 3D Gaussian Splatting for wildlife biometric re-identification. The pipeline ingests synchronised multi-view video captured from a fixed camera rig, performs camera calibration, image segmentation, and feature-based reconstruction to produce per-animal 3D representations suitable for training a Gaussian Splatting model used in downstream re-identification tasks.

## Setup

Activate the existing conda environment and install the package in editable mode:

```bash
conda activate wildsplat
pip install -e ".[dev]"
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
│   ├── PROGRESS.md       # Weekly progress log
│   ├── ALGORITHMS.md     # Algorithm explanations per step
│   ├── DECISIONS.md      # Architecture Decision Records
│   └── SETUP.md          # Environment setup notes
├── notebooks/            # Exploratory Jupyter notebooks
├── output/               # Pipeline outputs (gitignored)
├── scripts/              # One-off utility scripts
├── src/
│   ├── calibration/      # Intrinsic and extrinsic calibration
│   ├── features/         # Feature detection and matching
│   ├── geometry/         # Triangulation and visual hull
│   ├── io/               # Data loading and writing utilities
│   ├── segmentation/     # SAM2, classical, and LiDAR segmentation
│   └── sync/             # Multi-view frame synchronisation
└── tests/                # Pytest test suite
```
