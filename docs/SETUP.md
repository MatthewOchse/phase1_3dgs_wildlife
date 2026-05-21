# Environment Setup

## Requirements

- Python 3.11 or later
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) (recommended)
- CUDA-capable GPU recommended for segmentation and Gaussian Splatting steps

## Activating the Environment

The project uses the existing `wildsplat` conda environment:

```bash
conda activate wildsplat
```

## Installing the Package

From the repository root, install in editable mode so that changes to `src/` are picked up immediately:

```bash
pip install -e ".[dev]"
```

## SAM2

SAM2 (Segment Anything Model 2) will be installed separately in a later step due to its custom build requirements. Instructions will be added to this document once the segmentation step is reached.

## Verifying the Installation

Run the smoke tests to confirm all submodules are importable:

```bash
pytest tests/test_smoke.py -v
```

## DANNCE markerless_mouse_1 Dataset

The pipeline uses the DANNCE markerless_mouse_1 dataset (6-camera synchronised
mouse recording from the Dunn lab at Duke University) as its initial test dataset.

### Where to store it

Keep the dataset **outside** the repository to avoid accidentally committing
large binary files and to keep `git status` clean. The recommended location is
`~/datasets/dannce_mm1`. The project reads the path from the environment
variable `DANNCE_MM1_PATH`; if unset it defaults to `~/datasets/dannce_mm1`.

Set it permanently in your shell profile (e.g. `~/.bashrc` or `~/.zshrc`):

```bash
export DANNCE_MM1_PATH=~/datasets/dannce_mm1
```

The dataset directory is already listed in `.gitignore` if placed inside the
repo. Do not override this.

### Downloading the dataset

Run the setup script once from the repo root with the `wildsplat` environment
active:

```bash
conda activate wildsplat
python scripts/setup_dannce_mm1.py
```

**Expected download size:** ~1 GB for the video zip, ~100 KB for the
calibration `.mat` file. On a typical university connection (50 Mbps) this
takes roughly 3–5 minutes.

**Expected layout after setup:**

```
~/datasets/dannce_mm1/
    label3d_dannce.mat
    videos/
        Camera1/0.mp4
        Camera2/0.mp4
        Camera3/0.mp4
        Camera4/0.mp4
        Camera5/0.mp4
        Camera6/0.mp4
```

### Useful flags

```bash
# Check an existing layout without downloading anything
python scripts/setup_dannce_mm1.py --verify-only

# Wipe and re-download from scratch
python scripts/setup_dannce_mm1.py --force
```

### Troubleshooting

- If the download stalls or the size check fails, the tinyurl redirect may
  have changed. Check the DANNCE repository README for the current link.
- `label3d_dannce.mat` is a MATLAB v5 file; if `scipy.io.loadmat` raises an
  error, ensure `scipy >= 1.7` is installed in the environment.
