"""DANNCE dataset adapter.

Translates the DANNCE markerless_mouse_1 dataset's native formats into the
pipeline's canonical ``MultiViewCapture`` representation.

Key convention differences handled here
----------------------------------------
* **K and R are transposed** in the MAT file.  ``K_opencv = K_stored.T`` and
  ``R_opencv = r_stored.T``.
* **Distortion order:** DANNCE stores ``RDistort = [k1, k2, k3]`` and
  ``TDistort = [p1, p2]``; OpenCV order is ``[k1, k2, p1, p2, k3]``.
* **World-to-camera** convention in the MAT file; the canonical format uses
  **camera-to-world** (``T_world_from_cam``).
* **Segmented video files:** each camera has up to 6 MP4 segments named by
  the global start frame index (e.g. ``0.mp4``, ``3000.mp4``).  The loader
  resolves any global frame index to ``(segment_path, local_offset)``.
* **1-indexed MATLAB frame numbers** in ``data_sampleID``; ``data_frame`` is
  already 0-indexed and is used directly.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import scipy.io

from io_utils.dataset import CaptureFrame, MultiViewCapture

logger = logging.getLogger(__name__)

_DEFAULT_DATASET_ROOT = Path("~/datasets/dannce_mm1").expanduser()
_N_CAMERAS = 6
_FRAMES_PER_SEGMENT = 3000   # all segments in markerless_mouse_1 have 3000 frames
_VIDEO_GLOB = "*.mp4"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_dataset_root() -> Path:
    raw = os.environ.get("DANNCE_MM1_PATH", str(_DEFAULT_DATASET_ROOT))
    return Path(raw).expanduser().resolve()


def _load_mat(mat_path: Path) -> dict:
    return scipy.io.loadmat(
        str(mat_path), struct_as_record=False, squeeze_me=True
    )


def _build_dist_coeffs(rd: np.ndarray, td: np.ndarray) -> np.ndarray:
    """Convert DANNCE distortion arrays to the OpenCV [k1,k2,p1,p2,k3] order."""
    k1, k2, k3 = float(rd[0]), float(rd[1]), float(rd[2])
    p1, p2 = float(td[0]), float(td[1])
    return np.array([k1, k2, p1, p2, k3], dtype=np.float64)


def _get_image_size(dataset_root: Path, cam_id: str) -> tuple[int, int]:
    """Probe the first video segment of a camera for (width, height)."""
    cam_dir = dataset_root / "videos" / cam_id
    segs = sorted(cam_dir.glob(_VIDEO_GLOB), key=lambda p: int(p.stem))
    if not segs:
        raise FileNotFoundError(f"No video files found for {cam_id} in {cam_dir}")
    cap = cv2.VideoCapture(str(segs[0]))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return w, h


def _probe_fps(dataset_root: Path) -> float:
    """Read FPS from Camera1's first segment."""
    seg = next(
        iter(sorted((dataset_root / "videos" / "Camera1").glob(_VIDEO_GLOB),
                    key=lambda p: int(p.stem)))
    )
    cap = cv2.VideoCapture(str(seg))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return float(fps)


def _resolve_segment(
    dataset_root: Path,
    cam_id: str,
    global_frame: int,
) -> tuple[Path, int]:
    """Map a global frame index to ``(segment_path, local_offset)``."""
    cam_dir = dataset_root / "videos" / cam_id
    segs = sorted(cam_dir.glob(_VIDEO_GLOB), key=lambda p: int(p.stem))
    # Each segment's stem is its global start frame index
    seg_starts = [int(s.stem) for s in segs]

    # Find the last segment whose start ≤ global_frame
    chosen_seg = segs[0]
    chosen_start = seg_starts[0]
    for seg, start in zip(segs, seg_starts):
        if start <= global_frame:
            chosen_seg = seg
            chosen_start = start
        else:
            break

    local_offset = global_frame - chosen_start
    return chosen_seg, local_offset


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_dannce_calibration(
    mat_path: Path,
    dataset_root: Optional[Path] = None,
) -> dict:
    """Load DANNCE calibration and return the canonical calibration dict.

    Applies the following transformations to match the canonical format:

    * Transposes ``K`` and ``r`` from their DANNCE storage convention.
    * Reorders distortion coefficients to OpenCV ``[k1, k2, p1, p2, k3]``.
    * Inverts the world-to-camera transform to produce ``T_world_from_cam``.
    * Probes each camera's video to determine image size.

    Parameters
    ----------
    mat_path:
        Path to ``label3d_dannce.mat``.
    dataset_root:
        Dataset root for probing video dimensions.  Defaults to the value of
        ``DANNCE_MM1_PATH`` or ``~/datasets/dannce_mm1``.

    Returns
    -------
    dict
        Canonical calibration dict (see ``src/io_utils/dataset.py`` docstring).
    """
    if dataset_root is None:
        dataset_root = _get_dataset_root()

    mat = _load_mat(mat_path)
    params = mat["params"]
    camnames = mat.get("camnames", [f"Camera{i+1}" for i in range(len(params))])
    if isinstance(camnames, str):
        camnames = [camnames]

    cameras: dict = {}
    for i, p in enumerate(params):
        cam_id = str(camnames[i])

        # Transpose from DANNCE convention
        K = p.K.T.astype(np.float64)       # (3, 3) upper-triangular
        R = p.r.T.astype(np.float64)       # (3, 3) proper rotation
        t = p.t.astype(np.float64)         # (3,)

        dist = _build_dist_coeffs(p.RDistort, p.TDistort)

        # Image size from video
        try:
            w, h = _get_image_size(dataset_root, cam_id)
        except FileNotFoundError:
            logger.warning("Cannot probe image size for %s — using (0, 0)", cam_id)
            w, h = 0, 0

        # World-to-camera → camera-to-world
        T_cam_from_world = np.eye(4, dtype=np.float64)
        T_cam_from_world[:3, :3] = R
        T_cam_from_world[:3, 3] = t
        T_world_from_cam = np.linalg.inv(T_cam_from_world)

        cameras[cam_id] = {
            "intrinsics": {
                "camera_matrix": K,
                "dist_coeffs": dist,
                "image_size": (w, h),
            },
            "extrinsics": {
                "world_from_camera": T_world_from_cam,
            },
        }
        logger.debug(
            "%s: fx=%.1f fy=%.1f cx=%.1f cy=%.1f  size=%dx%d",
            cam_id, K[0, 0], K[1, 1], K[0, 2], K[1, 2], w, h,
        )

    return {
        "cameras": cameras,
        "world_frame": "DANNCE L-frame world coordinates (mm)",
        "source": "DANNCE_mm1",
    }


def load_dannce_sync(mat_path: Path) -> dict:
    """Extract the sync structure from the MAT file.

    For the markerless_mouse_1 dataset, cameras are hardware-synchronised so
    ``data_frame`` is simply ``[0, 1, 2, …, 17999]`` for every camera.  This
    function returns a simple summary dict confirming hardware sync and
    recording the total frame count.

    Parameters
    ----------
    mat_path:
        Path to ``label3d_dannce.mat``.

    Returns
    -------
    dict with keys:
        ``type`` (``"hardware"``),
        ``n_frames`` (total frames per camera),
        ``fps_estimated`` (None — determined from video),
        ``per_camera_frame_indices`` (dict[str, np.ndarray]).
    """
    mat = _load_mat(mat_path)
    sync = mat["sync"]
    camnames = mat.get("camnames", [f"Camera{i+1}" for i in range(len(sync))])
    if isinstance(camnames, str):
        camnames = [camnames]

    per_camera: dict[str, np.ndarray] = {}
    for i, s in enumerate(sync):
        cam_id = str(camnames[i])
        per_camera[cam_id] = s.data_frame.astype(np.int32)

    n_frames = int(sync[0].data_frame.shape[0])
    return {
        "type": "hardware",
        "n_frames": n_frames,
        "fps_estimated": None,
        "per_camera_frame_indices": per_camera,
    }


def load_dannce_labels(mat_path: Path) -> dict:
    """Extract manual 3D pose labels from the MAT file.

    Returns
    -------
    dict[str, dict] or empty dict
        Keys are camera IDs.  Each value has:
        ``frame_indices`` (np.ndarray), ``data_2d`` (N, 44), ``data_3d`` (N, 66).
        Returns ``{}`` if ``labelData`` is absent or empty.
    """
    mat = _load_mat(mat_path)
    if "labelData" not in mat:
        logger.info("No labelData key in MAT file")
        return {}

    label_data = mat["labelData"]
    if label_data is None or (hasattr(label_data, "__len__") and len(label_data) == 0):
        return {}

    camnames = mat.get("camnames", [f"Camera{i+1}" for i in range(len(label_data))])
    if isinstance(camnames, str):
        camnames = [camnames]

    result: dict = {}
    for i, ld in enumerate(label_data):
        cam_id = str(camnames[i])
        result[cam_id] = {
            "frame_indices": ld.data_frame.astype(np.int32),
            "data_2d": ld.data_2d.astype(np.float64),
            "data_3d": ld.data_3d.astype(np.float64),
        }

    n_frames = len(label_data[0].data_frame)
    n_joints = label_data[0].data_3d.shape[1] // 3
    logger.info("Labels: %d frames, %d joints per frame, %d cameras", n_frames, n_joints, len(result))
    return result


def list_dannce_sessions(dataset_root: Path) -> list[str]:
    """Return available session IDs for the given DANNCE dataset root.

    Currently always returns ``["markerless_mouse_1"]`` for this dataset.
    Designed so additional session support can be added later.
    """
    return ["markerless_mouse_1"]


def load_dannce_session(
    dataset_root: Optional[Path] = None,
    session_id: str = "markerless_mouse_1",
) -> MultiViewCapture:
    """Load the DANNCE markerless_mouse_1 session as a ``MultiViewCapture``.

    Resolves each camera's segmented videos, builds ``CaptureFrame`` objects
    with the correct ``(video_path, video_frame_offset)`` for every frame, and
    attaches calibration, sync metadata, and labels.

    Parameters
    ----------
    dataset_root:
        Path to the dataset root.  Defaults to ``DANNCE_MM1_PATH`` env var or
        ``~/datasets/dannce_mm1``.
    session_id:
        Session identifier (currently only ``"markerless_mouse_1"`` is valid).

    Returns
    -------
    MultiViewCapture
        Fully populated capture with 6 cameras and ~18 000 frames each.
    """
    if dataset_root is None:
        dataset_root = _get_dataset_root()
    dataset_root = Path(dataset_root)

    mat_path = dataset_root / "label3d_dannce.mat"
    if not mat_path.exists():
        raise FileNotFoundError(f"label3d_dannce.mat not found at {mat_path}")

    logger.info("Loading DANNCE session '%s' from %s", session_id, dataset_root)

    # -- Calibration --
    calibration = load_dannce_calibration(mat_path, dataset_root)

    # -- Sync --
    sync_info = load_dannce_sync(mat_path)
    n_frames = sync_info["n_frames"]

    # -- FPS from video --
    fps = _probe_fps(dataset_root)
    logger.info("FPS from video: %.1f  |  total frames per camera: %d", fps, n_frames)

    # -- Labels --
    labels = load_dannce_labels(mat_path)

    # -- Camera list from calibration --
    cameras = sorted(calibration["cameras"].keys())

    # -- Build CaptureFrame lists for each camera --
    frames_per_camera: dict[str, list[CaptureFrame]] = {}

    for cam_id in cameras:
        cam_dir = dataset_root / "videos" / cam_id
        if not cam_dir.is_dir():
            logger.warning("Video directory not found for %s: %s", cam_id, cam_dir)
            continue

        # Sort segments by numeric stem
        segs = sorted(cam_dir.glob(_VIDEO_GLOB), key=lambda p: int(p.stem))
        seg_starts = [int(s.stem) for s in segs]
        # Pre-build segment index: global_frame → (seg_path, local_offset)
        # For efficiency, compute directly from starts list
        frame_list: list[CaptureFrame] = []

        seg_idx = 0
        for global_frame in range(n_frames):
            # Advance segment pointer
            while (seg_idx + 1 < len(seg_starts)
                   and seg_starts[seg_idx + 1] <= global_frame):
                seg_idx += 1

            local_offset = global_frame - seg_starts[seg_idx]
            frame_list.append(
                CaptureFrame(
                    camera_id=cam_id,
                    timestamp_s=global_frame / fps,
                    frame_index=global_frame,
                    video_path=segs[seg_idx],
                    video_frame_offset=local_offset,
                )
            )

        frames_per_camera[cam_id] = frame_list
        logger.info("Camera %s: %d frames across %d segment(s)", cam_id, n_frames, len(segs))

    return MultiViewCapture(
        cameras=cameras,
        frames_per_camera=frames_per_camera,
        fps=fps,
        calibration=calibration,
        session_id=session_id,
        metadata={"sync": sync_info, "labels": labels},
    )


# ---------------------------------------------------------------------------
# Utility: extract demo frames
# ---------------------------------------------------------------------------


def extract_dannce_demo_frames(
    capture: MultiViewCapture,
    output_dir: Path,
    n_frames: int = 10,
) -> None:
    """Save N evenly-spaced 6-camera grid images to *output_dir*.

    Useful for a quick visual confirmation that loading works end-to-end.
    Each grid is saved as ``frame_XXXXXXX.jpg`` where the number is the
    global frame index.

    Parameters
    ----------
    capture:
        Loaded ``MultiViewCapture``.
    output_dir:
        Directory to write grid images.  Created if it does not exist.
    n_frames:
        Number of evenly-spaced timesteps to extract.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    duration = (capture.n_timesteps - 1) / capture.fps
    times = np.linspace(0.0, duration, n_frames)

    for t in times:
        ts = capture.get_timestep(float(t))
        global_idx = next(iter(ts.frames.values())).frame_index

        n_cams = len(ts.frames)
        ncols = 3
        nrows = (n_cams + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(18, nrows * 6))
        axes = np.array(axes).reshape(-1)
        fig.suptitle(f"Frame {global_idx}  (t={t:.2f} s)", fontsize=13)

        for ax, (cam_id, frame) in zip(axes, sorted(ts.frames.items())):
            img_bgr = frame.image
            img_rgb = img_bgr[:, :, ::-1]
            ax.imshow(img_rgb)
            ax.set_title(cam_id, fontsize=10)
            ax.axis("off")
        for ax in axes[n_cams:]:
            ax.axis("off")

        plt.tight_layout()
        out_path = output_dir / f"frame_{global_idx:07d}.jpg"
        plt.savefig(str(out_path), dpi=72, bbox_inches="tight")
        plt.close()
        logger.info("Saved demo frame %d → %s", global_idx, out_path.name)
