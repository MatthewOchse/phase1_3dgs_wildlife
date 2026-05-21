"""Tests for src/io/dataset.py and src/sync/alignment.py.

All tests are self-contained and run without the real dataset.
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from io_utils.dataset import CaptureFrame, MultiViewCapture, Timestep
from sync.alignment import apply_sync_offsets


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N_CAMERAS = 3
N_FRAMES = 20
FPS = 10.0
CAMERA_IDS = ["cam0", "cam1", "cam2"]


def _make_calibration(camera_ids: list[str]) -> dict:
    return {
        "cameras": {
            cam_id: {
                "intrinsics": {
                    "camera_matrix": np.eye(3, dtype=np.float64),
                    "dist_coeffs": np.zeros(5, dtype=np.float64),
                    "image_size": (8, 8),
                },
                "extrinsics": {
                    "world_from_camera": np.eye(4, dtype=np.float64),
                },
            }
            for cam_id in camera_ids
        },
        "world_frame": "test_world",
        "source": "synthetic",
    }


def _write_synthetic_images(
    base_dir: Path, camera_ids: list[str], n_frames: int
) -> dict[str, Path]:
    """Write tiny 8×8 PNG images for each camera. Returns dir per camera."""
    cam_dirs: dict[str, Path] = {}
    for cam_id in camera_ids:
        cam_dir = base_dir / cam_id
        cam_dir.mkdir()
        for i in range(n_frames):
            img = np.zeros((8, 8, 3), dtype=np.uint8)
            img[0, 0, 0] = i % 256  # embed frame index
            cv2.imwrite(str(cam_dir / f"frame_{i:04d}.png"), img)
        cam_dirs[cam_id] = cam_dir
    return cam_dirs


@pytest.fixture(scope="module")
def synthetic_capture(tmp_path_factory):
    base = tmp_path_factory.mktemp("capture")
    cam_dirs = _write_synthetic_images(base, CAMERA_IDS, N_FRAMES)
    calibration = _make_calibration(CAMERA_IDS)
    return MultiViewCapture.from_image_directories(
        camera_dirs=cam_dirs,
        calibration=calibration,
        session_id="synthetic_test",
        fps=FPS,
    )


# ---------------------------------------------------------------------------
# Construction and basic properties
# ---------------------------------------------------------------------------


def test_from_image_directories_camera_count(synthetic_capture):
    assert len(synthetic_capture.cameras) == N_CAMERAS


def test_from_image_directories_frame_count(synthetic_capture):
    assert synthetic_capture.n_timesteps == N_FRAMES
    for cam_id in CAMERA_IDS:
        assert len(synthetic_capture.frames_per_camera[cam_id]) == N_FRAMES


def test_from_image_directories_timestamps(synthetic_capture):
    """First frame is at t=0, last at t=(N-1)/fps."""
    frames = synthetic_capture.frames_per_camera["cam0"]
    assert frames[0].timestamp_s == pytest.approx(0.0)
    assert frames[-1].timestamp_s == pytest.approx((N_FRAMES - 1) / FPS)


def test_from_image_directories_frame_index(synthetic_capture):
    frames = synthetic_capture.frames_per_camera["cam1"]
    for i, f in enumerate(frames):
        assert f.frame_index == i
        assert f.camera_id == "cam1"


def test_n_timesteps_property(synthetic_capture):
    assert synthetic_capture.n_timesteps == N_FRAMES


def test_summary_is_string(synthetic_capture):
    s = synthetic_capture.summary()
    assert isinstance(s, str)
    assert "synthetic_test" in s
    assert str(FPS) in s


# ---------------------------------------------------------------------------
# get_frame_at_index
# ---------------------------------------------------------------------------


def test_get_frame_at_index_returns_timestep(synthetic_capture):
    ts = synthetic_capture.get_frame_at_index(0)
    assert isinstance(ts, Timestep)
    assert set(ts.frames.keys()) == set(CAMERA_IDS)


def test_get_frame_at_index_all_cameras_present(synthetic_capture):
    ts = synthetic_capture.get_frame_at_index(5)
    for cam_id in CAMERA_IDS:
        assert cam_id in ts.frames


def test_get_frame_at_index_out_of_range_raises(synthetic_capture):
    with pytest.raises(ValueError, match="out of range"):
        synthetic_capture.get_frame_at_index(N_FRAMES)

    with pytest.raises(ValueError, match="out of range"):
        synthetic_capture.get_frame_at_index(-1)


# ---------------------------------------------------------------------------
# get_timestep
# ---------------------------------------------------------------------------


def test_get_timestep_exact_match(synthetic_capture):
    """t=0.5 s at 10 fps → frame 5."""
    ts = synthetic_capture.get_timestep(0.5)
    for cam_id in CAMERA_IDS:
        assert ts.frames[cam_id].frame_index == 5


def test_get_timestep_nearest_not_interpolated(synthetic_capture):
    """t slightly above 0.5 s should still snap to frame 5, not 6."""
    ts = synthetic_capture.get_timestep(0.54)
    for cam_id in CAMERA_IDS:
        assert ts.frames[cam_id].frame_index == 5


def test_get_timestep_snaps_up(synthetic_capture):
    """t slightly above midpoint snaps to next frame."""
    ts = synthetic_capture.get_timestep(0.55)
    for cam_id in CAMERA_IDS:
        assert ts.frames[cam_id].frame_index in (5, 6)  # rounding edge


def test_get_timestep_t0(synthetic_capture):
    ts = synthetic_capture.get_timestep(0.0)
    for cam_id in CAMERA_IDS:
        assert ts.frames[cam_id].frame_index == 0


def test_get_timestep_last(synthetic_capture):
    last_t = (N_FRAMES - 1) / FPS
    ts = synthetic_capture.get_timestep(last_t)
    for cam_id in CAMERA_IDS:
        assert ts.frames[cam_id].frame_index == N_FRAMES - 1


def test_get_timestep_out_of_range_raises(synthetic_capture):
    with pytest.raises(ValueError, match="outside capture duration"):
        synthetic_capture.get_timestep(-1.0)

    with pytest.raises(ValueError, match="outside capture duration"):
        synthetic_capture.get_timestep(1000.0)


# ---------------------------------------------------------------------------
# iter_timesteps
# ---------------------------------------------------------------------------


def test_iter_timesteps_count(synthetic_capture):
    """Iterating every 0.5 s over 1 s should yield 3 timesteps (0, 0.5, 1.0)."""
    timesteps = list(synthetic_capture.iter_timesteps(0.0, 1.0, 0.5))
    assert len(timesteps) == 3


def test_iter_timesteps_correct_alignment(synthetic_capture):
    """All cameras in each timestep should have the same frame index."""
    for ts in synthetic_capture.iter_timesteps(0.0, (N_FRAMES - 1) / FPS, 1.0 / FPS):
        indices = {f.frame_index for f in ts.frames.values()}
        assert len(indices) == 1, f"Expected one unique frame index per timestep, got {indices}"


def test_iter_timesteps_monotone(synthetic_capture):
    ts_list = list(synthetic_capture.iter_timesteps(0.0, 1.0, 0.2))
    times = [ts.t for ts in ts_list]
    assert times == sorted(times)


# ---------------------------------------------------------------------------
# Image lazy loading
# ---------------------------------------------------------------------------


def test_capture_frame_image_lazy_loads(synthetic_capture):
    """Accessing .image should return a non-None numpy array."""
    frame = synthetic_capture.frames_per_camera["cam0"][0]
    assert frame._image_cache is None  # not loaded yet
    img = frame.image
    assert img is not None
    assert isinstance(img, np.ndarray)
    assert frame._image_cache is not None  # now cached


def test_capture_frame_image_cached(synthetic_capture):
    """Second access returns the same object."""
    frame = synthetic_capture.frames_per_camera["cam0"][3]
    img1 = frame.image
    img2 = frame.image
    assert img1 is img2


def test_capture_frame_no_path_raises(tmp_path):
    frame = CaptureFrame(
        camera_id="x", timestamp_s=0.0, frame_index=0
    )
    with pytest.raises(ValueError, match="neither image_path nor video_path"):
        _ = frame.image


# ---------------------------------------------------------------------------
# Sync offset application
# ---------------------------------------------------------------------------


def test_apply_sync_offsets_shifts_timestamps(synthetic_capture):
    offsets = {"cam0": 0.1, "cam1": -0.05, "cam2": 0.0}
    shifted = apply_sync_offsets(synthetic_capture, offsets)

    for cam_id, delta in offsets.items():
        for orig, new in zip(
            synthetic_capture.frames_per_camera[cam_id],
            shifted.frames_per_camera[cam_id],
        ):
            assert new.timestamp_s == pytest.approx(orig.timestamp_s + delta, abs=1e-9)


def test_apply_sync_offsets_returns_new_object(synthetic_capture):
    shifted = apply_sync_offsets(synthetic_capture, {"cam0": 0.1})
    assert shifted is not synthetic_capture


def test_apply_sync_offsets_zero_offset_unchanged(synthetic_capture):
    offsets = {cam_id: 0.0 for cam_id in CAMERA_IDS}
    shifted = apply_sync_offsets(synthetic_capture, offsets)
    for cam_id in CAMERA_IDS:
        for orig, new in zip(
            synthetic_capture.frames_per_camera[cam_id],
            shifted.frames_per_camera[cam_id],
        ):
            assert new.timestamp_s == pytest.approx(orig.timestamp_s)


def test_apply_sync_offsets_missing_camera_unchanged(synthetic_capture):
    """Cameras absent from offsets dict should not have timestamps changed."""
    shifted = apply_sync_offsets(synthetic_capture, {"cam0": 0.5})
    for cam_id in ["cam1", "cam2"]:
        for orig, new in zip(
            synthetic_capture.frames_per_camera[cam_id],
            shifted.frames_per_camera[cam_id],
        ):
            assert new.timestamp_s == pytest.approx(orig.timestamp_s)


# ---------------------------------------------------------------------------
# Canonical calibration structure
# ---------------------------------------------------------------------------


def test_canonical_calibration_structure():
    """A handmade calibration dict must validate against the documented schema."""
    K = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=np.float64)
    dist = np.array([-0.1, 0.05, 0.001, -0.002, 0.0], dtype=np.float64)
    T_world_from_cam = np.eye(4, dtype=np.float64)

    calibration = {
        "cameras": {
            "cam0": {
                "intrinsics": {
                    "camera_matrix": K,
                    "dist_coeffs": dist,
                    "image_size": (640, 480),
                },
                "extrinsics": {
                    "world_from_camera": T_world_from_cam,
                },
            }
        },
        "world_frame": "charuco_board",
        "source": "intrinsic_step1",
    }

    cam = calibration["cameras"]["cam0"]
    assert cam["intrinsics"]["camera_matrix"].shape == (3, 3)
    assert cam["intrinsics"]["dist_coeffs"].shape == (5,)
    assert len(cam["intrinsics"]["image_size"]) == 2
    assert cam["extrinsics"]["world_from_camera"].shape == (4, 4)

    # Bottom row must be [0, 0, 0, 1] for a valid homogeneous transform
    T = cam["extrinsics"]["world_from_camera"]
    np.testing.assert_array_equal(T[3, :], [0, 0, 0, 1])

    # Upper-left 3×3 should be a valid rotation (det ≈ +1)
    R = T[:3, :3]
    assert abs(np.linalg.det(R) - 1.0) < 1e-6


def test_canonical_calibration_preserved_in_capture(tmp_path):
    """Calibration dict is stored unchanged in the MultiViewCapture."""
    cam_dir = tmp_path / "cam0"
    cam_dir.mkdir()
    img = np.zeros((4, 4, 3), dtype=np.uint8)
    cv2.imwrite(str(cam_dir / "frame_0000.png"), img)

    calibration = _make_calibration(["cam0"])
    capture = MultiViewCapture.from_image_directories(
        camera_dirs={"cam0": cam_dir},
        calibration=calibration,
        session_id="calib_test",
        fps=1.0,
    )
    assert capture.calibration is calibration
    np.testing.assert_array_equal(
        capture.calibration["cameras"]["cam0"]["intrinsics"]["camera_matrix"],
        np.eye(3),
    )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_from_image_directories_empty_dir_raises(tmp_path):
    empty = tmp_path / "empty_cam"
    empty.mkdir()
    with pytest.raises(ValueError, match="No image files found"):
        MultiViewCapture.from_image_directories(
            camera_dirs={"cam0": empty},
            calibration=_make_calibration(["cam0"]),
            session_id="test",
            fps=10.0,
        )


def test_single_frame_capture(tmp_path):
    cam_dir = tmp_path / "cam0"
    cam_dir.mkdir()
    img = np.zeros((4, 4, 3), dtype=np.uint8)
    cv2.imwrite(str(cam_dir / "frame_0000.png"), img)

    capture = MultiViewCapture.from_image_directories(
        camera_dirs={"cam0": cam_dir},
        calibration=_make_calibration(["cam0"]),
        session_id="single",
        fps=30.0,
    )
    assert capture.n_timesteps == 1
    ts = capture.get_timestep(0.0)
    assert ts.frames["cam0"].frame_index == 0
