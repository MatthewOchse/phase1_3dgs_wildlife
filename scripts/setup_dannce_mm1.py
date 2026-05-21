#!/usr/bin/env python3
"""Download and organise the DANNCE markerless_mouse_1 dataset.

Target layout after setup
--------------------------
<dataset_root>/
    label3d_dannce.mat
    videos/
        Camera1/0.mp4
        Camera2/0.mp4
        ...
        Camera6/0.mp4

Usage
-----
    python scripts/setup_dannce_mm1.py               # download if needed
    python scripts/setup_dannce_mm1.py --verify-only # check existing layout
    python scripts/setup_dannce_mm1.py --force       # wipe and re-download
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import requests
import scipy.io
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VIDEOS_URL = "https://tinyurl.com/DANNCEmm1vids"
CALIBRATION_URL = (
    "https://github.com/spoonsso/dannce/raw/master/demo/"
    "markerless_mouse_1/label3d_dannce.mat"
)
N_CAMERAS = 6
MIN_DOWNLOAD_BYTES = 100 * 1024 * 1024  # 100 MB sanity floor
MAT_MAGIC = b"MATLAB"
REQUIRED_MAT_KEYS = {"params", "sync"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_dataset_root() -> Path:
    raw = os.environ.get("DANNCE_MM1_PATH", "~/datasets/dannce_mm1")
    return Path(raw).expanduser().resolve()


def download_file(url: str, dest: Path, description: str = "Downloading") -> None:
    """Stream-download *url* to *dest* with a tqdm progress bar."""
    logger.info("GET %s → %s", url, dest)
    response = requests.get(url, stream=True, allow_redirects=True, timeout=60)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0)) or None
    dest.parent.mkdir(parents=True, exist_ok=True)

    with (
        open(dest, "wb") as fh,
        tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=description,
            ncols=80,
        ) as bar,
    ):
        for chunk in response.iter_content(chunk_size=1024 * 256):
            if chunk:
                fh.write(chunk)
                bar.update(len(chunk))

    logger.info("Saved %s (%.1f MB)", dest.name, dest.stat().st_size / 1e6)


def verify_mat(mat_path: Path) -> None:
    """Check that *mat_path* is a loadable MATLAB file with required keys."""
    with mat_path.open("rb") as fh:
        magic = fh.read(6)
    if magic != MAT_MAGIC:
        raise RuntimeError(
            f"{mat_path.name} does not look like a MATLAB file "
            f"(first bytes: {magic!r})"
        )
    data = scipy.io.loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)
    missing = REQUIRED_MAT_KEYS - {k for k in data if not k.startswith("__")}
    if missing:
        raise RuntimeError(
            f"{mat_path.name} is missing required keys: {missing}"
        )
    logger.info("label3d_dannce.mat verified (keys: %s)", sorted(
        k for k in data if not k.startswith("__")
    ))


def camera_dirs(videos_dir: Path) -> list[Path]:
    return sorted(
        d for d in videos_dir.iterdir()
        if d.is_dir() and d.name.startswith("Camera")
    )


def has_video_files(cam_dir: Path) -> bool:
    return any(cam_dir.glob("*.mp4")) or any(cam_dir.glob("*.avi"))


def print_tree(root: Path, prefix: str = "", max_depth: int = 3, depth: int = 0) -> None:
    if depth >= max_depth:
        return
    entries = sorted(root.iterdir()) if root.is_dir() else []
    for i, entry in enumerate(entries):
        connector = "└── " if i == len(entries) - 1 else "├── "
        size_str = ""
        if entry.is_file():
            size_mb = entry.stat().st_size / 1e6
            size_str = f"  ({size_mb:.1f} MB)" if size_mb >= 0.1 else ""
        print(f"{prefix}{connector}{entry.name}{size_str}")
        if entry.is_dir():
            extension = "    " if i == len(entries) - 1 else "│   "
            print_tree(entry, prefix + extension, max_depth, depth + 1)


# ---------------------------------------------------------------------------
# Check whether dataset is already set up
# ---------------------------------------------------------------------------


def is_already_setup(root: Path) -> bool:
    mat = root / "label3d_dannce.mat"
    videos = root / "videos"
    if not mat.exists():
        return False
    if not videos.is_dir():
        return False
    dirs = camera_dirs(videos)
    if len(dirs) < N_CAMERAS:
        return False
    return all(has_video_files(d) for d in dirs)


# ---------------------------------------------------------------------------
# Download video zip
# ---------------------------------------------------------------------------


def download_and_extract_videos(root: Path) -> None:
    videos_dir = root / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="dannce_mm1_") as tmp_str:
        tmp = Path(tmp_str)
        zip_path = tmp / "dannce_mm1_videos.zip"

        # -- Download --
        download_file(VIDEOS_URL, zip_path, description="Videos zip")

        size_bytes = zip_path.stat().st_size
        logger.info("Downloaded %.1f MB", size_bytes / 1e6)
        if size_bytes < MIN_DOWNLOAD_BYTES:
            raise RuntimeError(
                f"Downloaded file is only {size_bytes / 1e6:.1f} MB — "
                "expected at least 100 MB. The URL may have changed or "
                "the download was truncated."
            )

        # -- Extract --
        extract_dir = tmp / "extracted"
        extract_dir.mkdir()
        logger.info("Extracting zip …")
        with zipfile.ZipFile(zip_path, "r") as zf:
            members = zf.namelist()
            logger.info("Zip contains %d entries", len(members))
            zf.extractall(extract_dir)

        # -- Discover camera folders --
        # The zip may place folders at the top level or inside a subdirectory.
        cam_candidates = []
        for candidate in sorted(extract_dir.rglob("Camera*")):
            if candidate.is_dir() and has_video_files(candidate):
                cam_candidates.append(candidate)

        if not cam_candidates:
            # List what we got to help diagnose
            found = [str(p.relative_to(extract_dir)) for p in extract_dir.rglob("*")]
            raise RuntimeError(
                "Could not find Camera* directories with video files in the zip.\n"
                f"Extracted contents (first 30): {found[:30]}"
            )

        logger.info("Found %d camera directories in zip", len(cam_candidates))

        # -- Move camera folders into videos/ --
        for cam_dir in cam_candidates:
            dest = videos_dir / cam_dir.name
            if dest.exists():
                shutil.rmtree(dest)
            shutil.move(str(cam_dir), str(dest))
            logger.info("Moved %s → %s", cam_dir.name, dest)

    logger.info("Temporary files cleaned up")


# ---------------------------------------------------------------------------
# Download calibration file
# ---------------------------------------------------------------------------


def download_calibration(root: Path) -> None:
    mat_path = root / "label3d_dannce.mat"
    download_file(CALIBRATION_URL, mat_path, description="label3d_dannce.mat")
    verify_mat(mat_path)


# ---------------------------------------------------------------------------
# Verify final layout
# ---------------------------------------------------------------------------


def verify_layout(root: Path) -> bool:
    ok = True

    mat_path = root / "label3d_dannce.mat"
    if not mat_path.exists():
        logger.error("MISSING: label3d_dannce.mat")
        ok = False
    else:
        try:
            verify_mat(mat_path)
            logger.info("OK: label3d_dannce.mat")
        except Exception as exc:
            logger.error("INVALID label3d_dannce.mat: %s", exc)
            ok = False

    videos_dir = root / "videos"
    if not videos_dir.is_dir():
        logger.error("MISSING: videos/ directory")
        return False

    dirs = camera_dirs(videos_dir)
    if len(dirs) < N_CAMERAS:
        logger.error(
            "Expected %d camera dirs, found %d: %s",
            N_CAMERAS,
            len(dirs),
            [d.name for d in dirs],
        )
        ok = False
    else:
        logger.info("OK: %d camera directories", len(dirs))

    for cam_dir in dirs:
        vids = list(cam_dir.glob("*.mp4")) + list(cam_dir.glob("*.avi"))
        if not vids:
            logger.error("NO video files in %s", cam_dir.name)
            ok = False
        else:
            total_mb = sum(v.stat().st_size for v in vids) / 1e6
            logger.info(
                "OK: %s — %d video file(s), %.1f MB total",
                cam_dir.name,
                len(vids),
                total_mb,
            )

    return ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and verify the DANNCE markerless_mouse_1 dataset."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Wipe the dataset directory and re-download from scratch.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Skip downloads; only verify the existing layout.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = get_dataset_root()
    logger.info("Dataset root: %s", root)

    # -- Force wipe --
    if args.force:
        if root.exists():
            logger.warning("--force: removing %s", root)
            shutil.rmtree(root)
        else:
            logger.info("--force: directory does not exist, nothing to remove")

    root.mkdir(parents=True, exist_ok=True)

    # -- Verify-only mode --
    if args.verify_only:
        logger.info("--verify-only: skipping downloads")
        ok = verify_layout(root)
        if ok:
            logger.info("Verification PASSED")
            print("\nDataset layout:")
            print(str(root))
            print_tree(root)
        else:
            logger.error("Verification FAILED")
        return 0 if ok else 1

    # -- Already set up? --
    if is_already_setup(root):
        logger.info("Dataset already set up — skipping download")
        logger.info("Run with --force to re-download, or --verify-only to re-verify")
        ok = verify_layout(root)
        print("\nDataset layout:")
        print(str(root))
        print_tree(root)
        return 0 if ok else 1

    # -- Download videos if missing --
    videos_dir = root / "videos"
    if not videos_dir.is_dir() or not any(camera_dirs(videos_dir)):
        logger.info("Downloading video files (~1 GB, may take several minutes) …")
        download_and_extract_videos(root)
    else:
        logger.info("Videos directory exists; checking camera completeness …")
        dirs = camera_dirs(videos_dir)
        if len(dirs) < N_CAMERAS or not all(has_video_files(d) for d in dirs):
            logger.warning("Camera directories incomplete — re-downloading videos")
            download_and_extract_videos(root)

    # -- Download calibration if missing --
    mat_path = root / "label3d_dannce.mat"
    if not mat_path.exists():
        logger.info("Downloading calibration file …")
        download_calibration(root)
    else:
        logger.info("label3d_dannce.mat already present — verifying …")
        verify_mat(mat_path)

    # -- Final verification --
    logger.info("Running final verification …")
    ok = verify_layout(root)

    print("\nDataset layout:")
    print(str(root))
    print_tree(root)

    if ok:
        logger.info("Setup complete. Dataset is ready.")
    else:
        logger.error("Setup finished with errors. Check the log above.")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
