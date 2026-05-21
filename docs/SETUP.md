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
