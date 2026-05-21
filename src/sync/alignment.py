"""Multi-view temporal synchronisation utilities.

Two synchronisation strategies are supported:

Hardware sync (e.g. DANNCE)
    Cameras are triggered by a shared electronic pulse.  Frame indices are
    identical across cameras by construction; no residual correction is needed.
    ``audio_cross_correlate_offsets`` returns zero offsets for these captures
    (or warns if no audio track is present).

Software sync (e.g. our own Kruger field captures)
    Each camera records independently.  If all cameras capture audio, the
    shared clap at the start of the take can be detected and used to align
    the streams.  ``audio_cross_correlate_offsets`` extracts audio via
    ``ffmpeg``, cross-correlates against the first camera's track, and
    returns per-camera time offsets in seconds.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import scipy.io.wavfile
import scipy.signal

from io_utils.dataset import MultiViewCapture, CaptureFrame, Timestep

logger = logging.getLogger(__name__)

_FFMPEG_AVAILABLE: Optional[bool] = None


def _check_ffmpeg() -> bool:
    global _FFMPEG_AVAILABLE
    if _FFMPEG_AVAILABLE is None:
        _FFMPEG_AVAILABLE = shutil.which("ffmpeg") is not None
    return _FFMPEG_AVAILABLE


# ---------------------------------------------------------------------------
# Audio extraction
# ---------------------------------------------------------------------------


def _extract_audio_to_wav(video_path: Path, out_wav: Path, sample_rate: int = 44100) -> bool:
    """Extract the audio track of *video_path* to a mono WAV file.

    Returns ``True`` on success, ``False`` if the video has no audio or
    extraction fails.
    """
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn",                         # no video
        "-acodec", "pcm_s16le",        # 16-bit PCM
        "-ac", "1",                    # mono
        "-ar", str(sample_rate),       # resample to target rate
        str(out_wav),
    ]
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=120
    )
    if result.returncode != 0:
        if "no audio" in result.stderr.lower() or "audio stream" in result.stderr.lower():
            logger.debug("No audio track in %s", video_path.name)
        else:
            logger.debug(
                "ffmpeg failed for %s: %s", video_path.name, result.stderr[-300:]
            )
        return False
    if not out_wav.exists() or out_wav.stat().st_size == 0:
        return False
    return True


# ---------------------------------------------------------------------------
# Cross-correlation sync
# ---------------------------------------------------------------------------


def audio_cross_correlate_offsets(
    video_paths: dict[str, Path],
    sample_rate: int = 44100,
) -> dict[str, float]:
    """Estimate per-camera time offsets via audio cross-correlation.

    Extracts the audio track of each video using ``ffmpeg``, then
    cross-correlates each camera's audio against the first camera's audio
    using ``scipy.signal.correlate``.  The lag at the correlation peak gives
    the time offset to apply so that all cameras share the same time origin.

    This approach requires all cameras to have captured a shared acoustic
    event (e.g. a clap at the start of the take).  For hardware-synchronised
    captures with no audio (e.g. DANNCE), zero offsets are returned.

    Parameters
    ----------
    video_paths:
        Mapping from camera ID to video file path.
    sample_rate:
        Target audio sample rate for extraction and correlation (Hz).

    Returns
    -------
    dict[str, float]
        Per-camera time offsets in seconds.  A positive offset means the
        camera's frames are *late* relative to the reference — call
        :func:`apply_sync_offsets` with negative values to advance timestamps.
        The reference camera always has offset 0.0.
    """
    if not _check_ffmpeg():
        logger.warning(
            "ffmpeg not found — cannot extract audio for sync. "
            "Returning zero offsets."
        )
        return {cam_id: 0.0 for cam_id in video_paths}

    cam_ids = list(video_paths)
    reference_id = cam_ids[0]
    offsets: dict[str, float] = {reference_id: 0.0}

    with tempfile.TemporaryDirectory(prefix="dannce_sync_") as tmp_str:
        tmp = Path(tmp_str)

        # Extract audio for all cameras
        audio_tracks: dict[str, np.ndarray] = {}
        for cam_id, vid_path in video_paths.items():
            wav_path = tmp / f"{cam_id}.wav"
            success = _extract_audio_to_wav(vid_path, wav_path, sample_rate)
            if not success:
                logger.warning(
                    "Camera %s: no audio track — using zero offset", cam_id
                )
                continue
            _, audio = scipy.io.wavfile.read(str(wav_path))
            audio = audio.astype(np.float32)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)  # already forced mono via ffmpeg
            audio_tracks[cam_id] = audio

        if reference_id not in audio_tracks:
            logger.warning(
                "Reference camera %s has no audio — returning zero offsets",
                reference_id,
            )
            return {cam_id: 0.0 for cam_id in video_paths}

        ref_audio = audio_tracks[reference_id]

        for cam_id in cam_ids[1:]:
            if cam_id not in audio_tracks:
                offsets[cam_id] = 0.0
                continue

            cam_audio = audio_tracks[cam_id]

            # Cross-correlate (full mode): peak index gives lag
            correlation = scipy.signal.correlate(ref_audio, cam_audio, mode="full")
            lags = scipy.signal.correlation_lags(
                len(ref_audio), len(cam_audio), mode="full"
            )
            lag_samples = int(lags[np.argmax(np.abs(correlation))])
            offset_s = lag_samples / sample_rate
            offsets[cam_id] = float(offset_s)
            logger.info(
                "Camera %s vs %s: lag=%d samples (%.4f s)",
                cam_id, reference_id, lag_samples, offset_s,
            )

    return offsets


# ---------------------------------------------------------------------------
# Offset application
# ---------------------------------------------------------------------------


def apply_sync_offsets(
    capture: MultiViewCapture,
    offsets: dict[str, float],
) -> MultiViewCapture:
    """Return a new capture with per-camera timestamp offsets applied.

    Timestamps for each camera are shifted by ``offsets[camera_id]`` seconds.
    Cameras not present in *offsets* are left unchanged.  The frames are
    shallow-copied so the underlying image data is shared.

    Parameters
    ----------
    capture:
        Source capture.
    offsets:
        Mapping from camera ID to time offset in seconds to *add* to
        each frame's ``timestamp_s``.

    Returns
    -------
    MultiViewCapture
        New capture with adjusted timestamps.
    """
    new_frames: dict[str, list[CaptureFrame]] = {}

    for cam_id, frame_list in capture.frames_per_camera.items():
        delta = offsets.get(cam_id, 0.0)
        adjusted = []
        for f in frame_list:
            new_f = CaptureFrame(
                camera_id=f.camera_id,
                timestamp_s=f.timestamp_s + delta,
                frame_index=f.frame_index,
                image_path=f.image_path,
                video_path=f.video_path,
                video_frame_offset=f.video_frame_offset,
            )
            adjusted.append(new_f)
        new_frames[cam_id] = adjusted

    return MultiViewCapture(
        cameras=capture.cameras,
        frames_per_camera=new_frames,
        fps=capture.fps,
        calibration=capture.calibration,
        session_id=capture.session_id,
        metadata=dict(capture.metadata),
    )


# ---------------------------------------------------------------------------
# Clap detection
# ---------------------------------------------------------------------------


def detect_clap_in_audio(
    audio: np.ndarray,
    sample_rate: int,
    window_ms: int = 10,
) -> Optional[float]:
    """Find the loudest transient in an audio signal and return its timestamp.

    Uses a simple energy-envelope approach: computes the RMS energy in
    overlapping windows and returns the time of the window with peak energy.

    Parameters
    ----------
    audio:
        1-D float audio array.
    sample_rate:
        Sample rate in Hz.
    window_ms:
        RMS window length in milliseconds.

    Returns
    -------
    float or None
        Timestamp of the loudest transient in seconds, or ``None`` if the
        audio is silent (all zeros).
    """
    if np.all(audio == 0) or len(audio) == 0:
        return None

    window_samples = max(1, int(sample_rate * window_ms / 1000))
    n_windows = len(audio) // window_samples
    if n_windows == 0:
        return None

    energy = np.array([
        float(np.sqrt(np.mean(audio[i * window_samples:(i + 1) * window_samples] ** 2)))
        for i in range(n_windows)
    ])
    peak_window = int(np.argmax(energy))
    peak_time_s = (peak_window * window_samples + window_samples // 2) / sample_rate
    return float(peak_time_s)


# ---------------------------------------------------------------------------
# Validation via clap
# ---------------------------------------------------------------------------


def validate_sync_with_clap(
    capture: MultiViewCapture,
    clap_timestamp_hint: float,
    search_window_s: float = 2.0,
    sample_rate: int = 44100,
) -> dict[str, float]:
    """Validate synchronisation by detecting a clap across all cameras.

    For each camera's primary video, extracts audio in a window around
    ``clap_timestamp_hint`` and locates the loudest transient.  Reports the
    residual timing difference relative to the median clap time.

    Returns zero residuals for cameras with no audio track (with a warning),
    so the function is safe to call on DANNCE-style captures.

    Parameters
    ----------
    capture:
        The loaded capture (provides video paths).
    clap_timestamp_hint:
        Approximate time of the clap in seconds (to narrow the search window).
    search_window_s:
        Audio window to extract around the hint (seconds, centred).
    sample_rate:
        Audio extraction sample rate.

    Returns
    -------
    dict[str, float]
        Residual offset per camera in **milliseconds** relative to the median
        detected clap time.  Smaller is better; > 33 ms is a concern at 30 fps.
    """
    if not _check_ffmpeg():
        logger.warning("ffmpeg not available — cannot validate sync with clap")
        return {cam_id: 0.0 for cam_id in capture.cameras}

    # Collect primary video path for each camera
    video_paths: dict[str, Optional[Path]] = {}
    for cam_id in capture.cameras:
        frames = capture.frames_per_camera.get(cam_id, [])
        vid_path = next(
            (f.video_path for f in frames if f.video_path is not None), None
        )
        video_paths[cam_id] = vid_path

    clap_times: dict[str, Optional[float]] = {}

    with tempfile.TemporaryDirectory(prefix="clap_sync_") as tmp_str:
        tmp = Path(tmp_str)
        half = search_window_s / 2

        for cam_id, vid_path in video_paths.items():
            if vid_path is None:
                logger.warning("Camera %s: no video path — skipping clap detection", cam_id)
                clap_times[cam_id] = None
                continue

            wav_path = tmp / f"{cam_id}.wav"
            # Extract only the relevant window using ffmpeg -ss / -t
            start_s = max(0.0, clap_timestamp_hint - half)
            cmd = [
                "ffmpeg", "-y",
                "-ss", str(start_s),
                "-i", str(vid_path),
                "-t", str(search_window_s),
                "-vn", "-acodec", "pcm_s16le", "-ac", "1", "-ar", str(sample_rate),
                str(wav_path),
            ]
            result = subprocess.run(cmd, capture_output=True, timeout=60)
            if result.returncode != 0 or not wav_path.exists():
                logger.warning("Camera %s: audio extraction failed — skipping", cam_id)
                clap_times[cam_id] = None
                continue

            _, audio = scipy.io.wavfile.read(str(wav_path))
            audio = audio.astype(np.float32)
            local_t = detect_clap_in_audio(audio, sample_rate)
            if local_t is None:
                clap_times[cam_id] = None
            else:
                clap_times[cam_id] = start_s + local_t

    valid = [t for t in clap_times.values() if t is not None]
    if not valid:
        logger.warning("No audio tracks found — returning zero residuals")
        return {cam_id: 0.0 for cam_id in capture.cameras}

    median_t = float(np.median(valid))
    residuals: dict[str, float] = {}
    for cam_id, t in clap_times.items():
        if t is None:
            residuals[cam_id] = 0.0
            logger.warning("Camera %s: no clap detected — residual set to 0", cam_id)
        else:
            residuals[cam_id] = (t - median_t) * 1000.0  # ms
            logger.info(
                "Camera %s: clap at %.4f s, residual %.2f ms", cam_id, t, residuals[cam_id]
            )
    return residuals
