"""End-to-end smoke test for the DANNCE loader.

Requires the real dataset.  Run with:

    pytest -m dataset tests/test_dannce_smoke.py -v
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from io_utils.dannce_loader import load_dannce_session


def _dataset_root() -> Path:
    raw = os.environ.get("DANNCE_MM1_PATH", "~/datasets/dannce_mm1")
    return Path(raw).expanduser().resolve()


@pytest.fixture(scope="module")
def dataset_root():
    root = _dataset_root()
    if not (root / "label3d_dannce.mat").exists():
        pytest.skip(f"DANNCE dataset not found at {root}")
    return root


@pytest.mark.dataset
def test_end_to_end_smoke(dataset_root, tmp_path):
    """Load the session, read frame at t=1.0 s, save all 6 to disk."""
    capture = load_dannce_session(dataset_root)

    # Basic sanity
    assert len(capture.cameras) == 6
    assert capture.fps > 0
    assert capture.n_timesteps > 100

    # Get timestep at 1 second
    ts = capture.get_timestep(1.0)
    assert len(ts.frames) == 6

    # Load and save each camera's frame
    for cam_id, frame in ts.frames.items():
        img = frame.image
        assert img is not None, f"image is None for {cam_id}"
        assert img.size > 0, f"empty image for {cam_id}"

        out_path = tmp_path / f"{cam_id}_t1s.jpg"
        import cv2
        cv2.imwrite(str(out_path), img)
        assert out_path.exists(), f"saved file missing for {cam_id}"
        assert out_path.stat().st_size > 1000, f"file too small for {cam_id}"
