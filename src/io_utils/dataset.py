"""Canonical multi-view capture data structures.

All downstream pipeline code works against these types.  Dataset-specific
loaders (DANNCE, AcinoSet, custom Kruger rigs) produce a ``MultiViewCapture``
and the rest of the pipeline never sees raw MAT files or vendor path layouts.

Canonical calibration dict
--------------------------
::

    {
        "cameras": {
            camera_id (str): {
                "intrinsics": {
                    "camera_matrix": np.ndarray  shape (3, 3),
                    "dist_coeffs":   np.ndarray  shape (5,)  [k1, k2, p1, p2, k3],
                    "image_size":    tuple        (width, height)
                },
                "extrinsics": {
                    "world_from_camera": np.ndarray  shape (4, 4)
                        # Homogeneous transform: p_world = T @ p_cam
                        # This is the INVERSE of OpenCV's [R|t] convention.
                }
            }
        },
        "world_frame": str,   # human description of the world origin
        "source":      str,   # dataset / calibration procedure name
    }

Note: extrinsics are stored as **camera-to-world** (``T_world_from_cam``).
Dataset adapters are responsible for inverting the dataset's native convention
(e.g. DANNCE's world-to-camera) before storing here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CaptureFrame
# ---------------------------------------------------------------------------


@dataclass
class CaptureFrame:
    """A single image frame from one camera at one point in time.

    Frames can be backed by an image file on disk or by a position within a
    video file.  The raw pixel data is **lazy-loaded** on first access of the
    ``image`` property and cached for the lifetime of the object.

    Parameters
    ----------
    camera_id:
        Identifier matching a key in the parent ``MultiViewCapture.cameras``.
    timestamp_s:
        Absolute timestamp in seconds from the start of the capture session.
    frame_index:
        Global 0-based frame index within this camera's sequence.
    image_path:
        Path to an image file (JPEG, PNG, …).  Set for image-directory-backed
        captures.
    video_path:
        Path to the video file containing this frame.  Set for video-backed
        captures.
    video_frame_offset:
        0-based index of this frame *within* ``video_path``.  When seeking a
        segmented video, the adapter stores the local offset into the segment
        here, not the global frame index.
    """

    camera_id: str
    timestamp_s: float
    frame_index: int
    image_path: Optional[Path] = None
    video_path: Optional[Path] = None
    video_frame_offset: Optional[int] = None

    def __post_init__(self):
        self._image_cache: Optional[np.ndarray] = None

    @property
    def image(self) -> np.ndarray:
        """Load and return the frame as a BGR uint8 ndarray (cached)."""
        if self._image_cache is None:
            self._image_cache = self._load()
        return self._image_cache

    def _load(self) -> np.ndarray:
        if self.image_path is not None:
            img = cv2.imread(str(self.image_path))
            if img is None:
                raise IOError(f"Could not read image: {self.image_path}")
            return img

        if self.video_path is not None and self.video_frame_offset is not None:
            cap = cv2.VideoCapture(str(self.video_path))
            cap.set(cv2.CAP_PROP_POS_FRAMES, self.video_frame_offset)
            ret, frame = cap.read()
            cap.release()
            if not ret:
                raise IOError(
                    f"Could not read frame {self.video_frame_offset} "
                    f"from {self.video_path}"
                )
            return frame

        raise ValueError(
            f"CaptureFrame for camera '{self.camera_id}' frame {self.frame_index} "
            "has neither image_path nor video_path set."
        )


# ---------------------------------------------------------------------------
# Timestep
# ---------------------------------------------------------------------------


@dataclass
class Timestep:
    """A synchronised set of frames from all cameras at one logical time.

    Parameters
    ----------
    t:
        The logical timestamp in seconds (may differ slightly from individual
        frame timestamps after sync-offset adjustment).
    frames:
        Mapping from camera ID to the selected ``CaptureFrame``.
    """

    t: float
    frames: dict[str, CaptureFrame]


# ---------------------------------------------------------------------------
# MultiViewCapture
# ---------------------------------------------------------------------------

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


class MultiViewCapture:
    """A synchronised multi-camera capture session.

    This is the central data structure used throughout the pipeline.
    Construct it via the class methods :meth:`from_video_files` or
    :meth:`from_image_directories`; access frames via :meth:`get_timestep`
    or :meth:`iter_timesteps`.

    Parameters
    ----------
    cameras:
        Ordered list of camera IDs.
    frames_per_camera:
        Mapping from camera ID to the ordered list of ``CaptureFrame`` objects.
        All cameras must have the same number of frames.
    fps:
        Frames per second of the capture.
    calibration:
        Canonical calibration dict (see module docstring).
    session_id:
        Human-readable identifier for the recording session.
    metadata:
        Free-form dict for dataset-specific extras (e.g. sync info, labels).
    """

    def __init__(
        self,
        cameras: list[str],
        frames_per_camera: dict[str, list[CaptureFrame]],
        fps: float,
        calibration: dict,
        session_id: str,
        metadata: Optional[dict] = None,
    ) -> None:
        self.cameras = cameras
        self.frames_per_camera = frames_per_camera
        self.fps = fps
        self.calibration = calibration
        self.session_id = session_id
        self.metadata = metadata or {}

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_video_files(
        cls,
        camera_video_paths: dict[str, Path],
        calibration: dict,
        session_id: str,
        fps: float,
    ) -> "MultiViewCapture":
        """Build a capture from one video file per camera.

        Probes each video to determine its frame count.  Creates a
        ``CaptureFrame`` per frame storing the video path and local offset
        so that pixel data is loaded lazily.

        Parameters
        ----------
        camera_video_paths:
            Mapping from camera ID to the path of its video file.
        calibration:
            Canonical calibration dict.
        session_id:
            Session identifier string.
        fps:
            Recording frame rate (used to compute timestamps).
        """
        cameras = sorted(camera_video_paths)
        frames_per_camera: dict[str, list[CaptureFrame]] = {}

        for cam_id in cameras:
            vid_path = Path(camera_video_paths[cam_id])
            cap = cv2.VideoCapture(str(vid_path))
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            reported_fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()

            if reported_fps > 0 and abs(reported_fps - fps) > 1.0:
                logger.warning(
                    "Camera %s: requested fps=%.1f but video reports %.1f",
                    cam_id, fps, reported_fps,
                )

            frames_per_camera[cam_id] = [
                CaptureFrame(
                    camera_id=cam_id,
                    timestamp_s=i / fps,
                    frame_index=i,
                    video_path=vid_path,
                    video_frame_offset=i,
                )
                for i in range(n_frames)
            ]
            logger.debug("Camera %s: %d frames from %s", cam_id, n_frames, vid_path.name)

        return cls(
            cameras=cameras,
            frames_per_camera=frames_per_camera,
            fps=fps,
            calibration=calibration,
            session_id=session_id,
        )

    @classmethod
    def from_image_directories(
        cls,
        camera_dirs: dict[str, Path],
        calibration: dict,
        session_id: str,
        fps: float,
    ) -> "MultiViewCapture":
        """Build a capture from one directory of images per camera.

        Images within each directory are sorted lexicographically (so
        zero-padded names like ``frame_0000.png`` give the correct order).

        Parameters
        ----------
        camera_dirs:
            Mapping from camera ID to the directory containing its image files.
        calibration:
            Canonical calibration dict.
        session_id:
            Session identifier string.
        fps:
            Logical frame rate used to assign timestamps.
        """
        cameras = sorted(camera_dirs)
        frames_per_camera: dict[str, list[CaptureFrame]] = {}

        for cam_id in cameras:
            cam_dir = Path(camera_dirs[cam_id])
            image_paths = sorted(
                p for p in cam_dir.iterdir()
                if p.suffix.lower() in _IMAGE_EXTENSIONS
            )
            if not image_paths:
                raise ValueError(
                    f"No image files found in directory for camera '{cam_id}': {cam_dir}"
                )
            frames_per_camera[cam_id] = [
                CaptureFrame(
                    camera_id=cam_id,
                    timestamp_s=i / fps,
                    frame_index=i,
                    image_path=img_path,
                )
                for i, img_path in enumerate(image_paths)
            ]
            logger.debug(
                "Camera %s: %d frames from %s", cam_id, len(image_paths), cam_dir
            )

        return cls(
            cameras=cameras,
            frames_per_camera=frames_per_camera,
            fps=fps,
            calibration=calibration,
            session_id=session_id,
        )

    # ------------------------------------------------------------------
    # Frame access
    # ------------------------------------------------------------------

    @property
    def n_timesteps(self) -> int:
        """Number of synchronised timesteps (= frames in the shortest camera)."""
        if not self.frames_per_camera:
            return 0
        return min(len(frames) for frames in self.frames_per_camera.values())

    def get_frame_at_index(self, frame_index: int) -> Timestep:
        """Return the synchronised frame set at a given integer index.

        Parameters
        ----------
        frame_index:
            0-based index.  Must be in ``[0, n_timesteps)``.

        Raises
        ------
        ValueError
            If ``frame_index`` is out of range.
        """
        if frame_index < 0 or frame_index >= self.n_timesteps:
            raise ValueError(
                f"frame_index {frame_index} is out of range "
                f"[0, {self.n_timesteps}) for session '{self.session_id}'."
            )
        frames = {
            cam_id: self.frames_per_camera[cam_id][frame_index]
            for cam_id in self.cameras
            if cam_id in self.frames_per_camera
        }
        t = sum(f.timestamp_s for f in frames.values()) / len(frames)
        return Timestep(t=t, frames=frames)

    def get_timestep(self, t: float) -> Timestep:
        """Return the nearest synchronised frame set to timestamp ``t``.

        Finds the frame index closest to ``t`` seconds and returns the
        corresponding frames from all cameras.  Does **not** interpolate.

        Parameters
        ----------
        t:
            Query timestamp in seconds.

        Raises
        ------
        ValueError
            If ``t`` is outside the capture duration by more than half a frame.
        """
        duration = (self.n_timesteps - 1) / self.fps
        half_frame = 0.5 / self.fps

        if t < -half_frame or t > duration + half_frame:
            raise ValueError(
                f"Timestamp t={t:.4f} s is outside capture duration "
                f"[0, {duration:.4f}] s for session '{self.session_id}'."
            )

        frame_index = int(round(t * self.fps))
        frame_index = max(0, min(frame_index, self.n_timesteps - 1))
        return self.get_frame_at_index(frame_index)

    def iter_timesteps(
        self, start: float, end: float, step: float
    ) -> Iterator[Timestep]:
        """Yield synchronised frame sets at regular intervals.

        Parameters
        ----------
        start:
            Start timestamp in seconds (inclusive).
        end:
            End timestamp in seconds (inclusive).
        step:
            Time step between successive timesteps in seconds.

        Yields
        ------
        Timestep
            One per step from ``start`` to ``end``.
        """
        t = start
        while t <= end + 1e-9:
            yield self.get_timestep(t)
            t += step

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Return a human-readable description of the capture."""
        duration = (self.n_timesteps - 1) / self.fps if self.n_timesteps > 1 else 0
        lines = [
            f"MultiViewCapture — session: {self.session_id}",
            f"  cameras    : {', '.join(self.cameras)}  ({len(self.cameras)} total)",
            f"  fps        : {self.fps:.1f}",
            f"  timesteps  : {self.n_timesteps}",
            f"  duration   : {duration:.2f} s",
        ]
        if self.calibration and "cameras" in self.calibration:
            lines.append(f"  calibration: {self.calibration.get('source', 'unknown')} "
                         f"/ world frame: {self.calibration.get('world_frame', 'unknown')}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"MultiViewCapture(session={self.session_id!r}, "
            f"cameras={len(self.cameras)}, "
            f"n_timesteps={self.n_timesteps}, "
            f"fps={self.fps})"
        )
