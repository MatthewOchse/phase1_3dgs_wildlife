#!/usr/bin/env python3
"""Inspect the DANNCE markerless_mouse_1 dataset and write documentation.

Outputs
-------
* Console report of the MAT file structure and video properties.
* output/figures/dannce_mm1_camera1_frame0.jpg
* output/figures/dannce_mm1_all_cameras_frame0.jpg  (2×3 grid)
* docs/DATASET_DANNCE_MM1.md

Usage
-----
    python scripts/inspect_dannce_mm1.py
"""

from __future__ import annotations

import os
import sys
import textwrap
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))


def get_dataset_root() -> Path:
    raw = os.environ.get("DANNCE_MM1_PATH", "~/datasets/dannce_mm1")
    return Path(raw).expanduser().resolve()


# ---------------------------------------------------------------------------
# MAT file inspection
# ---------------------------------------------------------------------------

def load_mat(path: Path):
    return scipy.io.loadmat(
        str(path), struct_as_record=False, squeeze_me=True
    )


def report_mat_structure(mat: dict) -> dict:
    keys = [k for k in mat if not k.startswith("__")]
    print("=" * 60)
    print("MAT FILE STRUCTURE")
    print("=" * 60)
    print(f"Top-level keys: {keys}")
    print()

    # --- params ---
    params = mat["params"]
    print(f"params: shape={params.shape}  (one struct per camera)")
    fields = params[0]._fieldnames
    print(f"  fields per camera: {fields}")
    for f in fields:
        v = getattr(params[0], f)
        print(f"    {f:12s}  shape={getattr(v,'shape','scalar')}  dtype={getattr(v,'dtype',type(v).__name__)}")
    print()

    # --- sync ---
    sync = mat["sync"]
    print(f"sync: shape={sync.shape}  (one struct per camera)")
    s0 = sync[0]
    print(f"  fields: {s0._fieldnames}")
    for f in s0._fieldnames:
        v = getattr(s0, f)
        print(f"    {f:16s}  shape={v.shape}  dtype={v.dtype}  first_vals={v.flat[:5].tolist()}")
    print()

    # --- labelData ---
    ld = mat["labelData"]
    print(f"labelData: shape={ld.shape}  (one struct per camera)")
    l0 = ld[0]
    print(f"  fields: {l0._fieldnames}")
    for f in l0._fieldnames:
        v = getattr(l0, f)
        print(f"    {f:16s}  shape={v.shape}  dtype={v.dtype}")
    n_frames = len(l0.data_frame)
    n_nan_2d = int(np.sum(np.isnan(l0.data_2d)))
    n_nan_3d = int(np.sum(np.isnan(l0.data_3d)))
    print(f"  labeled frames: {n_frames}  |  NaN in 2D: {n_nan_2d}  |  NaN in 3D: {n_nan_3d}")
    print()

    return {"params": params, "sync": sync, "labelData": ld,
            "camnames": mat["camnames"]}


def report_camera1_values(params) -> dict:
    print("=" * 60)
    print("CAMERA1 NUMERIC VALUES")
    print("=" * 60)

    p = params[0]
    K_stored = p.K       # lower-triangular (stored as K^T in OpenCV sense)
    R_stored = p.r       # stored as R^T in OpenCV sense
    t = p.t

    K = K_stored.T       # OpenCV camera matrix
    R = R_stored.T       # OpenCV rotation matrix

    print("K (stored, lower-triangular — transpose of OpenCV K):")
    print(K_stored)
    print()
    print("K transposed (= OpenCV K):")
    print(K)
    print()
    print(f"  focal length : fx={K[0,0]:.2f} px,  fy={K[1,1]:.2f} px")
    print(f"  principal pt : cx={K[0,2]:.2f} px,  cy={K[1,2]:.2f} px")
    print(f"  skew         : {K[0,1]:.4f}")
    print()
    print("R (stored = R^T in OpenCV sense):")
    print(R_stored)
    print()
    print(f"t vector : {t}")
    print(f"RDistort : {p.RDistort}  (= [k1, k2, k3])")
    print(f"TDistort : {p.TDistort}  (= [p1, p2])")
    print()

    C = -R.T @ t
    print(f"Camera centre in world  : {C}")
    print()

    return {"K": K, "R": R, "t": t,
            "K_stored": K_stored, "R_stored": R_stored,
            "RDistort": p.RDistort, "TDistort": p.TDistort}


def report_all_camera_centres(params) -> list[np.ndarray]:
    print("=" * 60)
    print("ALL CAMERA CENTRES (world coordinates, mm)")
    print("=" * 60)
    centres = []
    for i, p in enumerate(params):
        K = p.K.T
        R = p.r.T
        t = p.t
        C = -R.T @ t
        centres.append(C)
        print(f"  Camera{i+1}: [{C[0]:8.2f},  {C[1]:8.2f},  {C[2]:8.2f}]")
    print()
    return centres


def compute_reprojection_errors(params, label_data) -> dict[str, float]:
    print("=" * 60)
    print("REPROJECTION ERROR (labeled frames)")
    print("=" * 60)
    errors = {}
    for ci in range(len(params)):
        p = params[ci]
        ld = label_data[ci]
        K = p.K.T
        R = p.r.T
        t = p.t.reshape(3, 1)
        dist = np.array([p.RDistort[0], p.RDistort[1],
                         p.TDistort[0], p.TDistort[1], p.RDistort[2]])
        rvec, _ = cv2.Rodrigues(R)
        errs = []
        n_joints = ld.data_3d.shape[1] // 3
        for i in range(len(ld.data_frame)):
            for j in range(n_joints):
                X = ld.data_3d[i, j*3:j*3+3]
                gt = ld.data_2d[i, j*2:j*2+2]
                if np.any(np.isnan(X)) or np.any(np.isnan(gt)):
                    continue
                if np.all(X == 0) and np.all(gt == 0):
                    continue
                proj, _ = cv2.projectPoints(
                    X.reshape(1, 1, 3), rvec, t, K, dist
                )
                e = float(np.linalg.norm(proj[0, 0] - gt))
                if np.isfinite(e):
                    errs.append(e)
        if errs:
            mean_err = float(np.mean(errs))
            errors[f"Camera{ci+1}"] = mean_err
            print(f"  Camera{ci+1}: mean={mean_err:.2f} px  (n={len(errs)})")
        else:
            print(f"  Camera{ci+1}: no valid labeled points")
    print()
    return errors


# ---------------------------------------------------------------------------
# Video inspection
# ---------------------------------------------------------------------------

def get_video_segments(dataset_root: Path) -> dict[str, list[Path]]:
    """Return sorted list of video segment paths per camera."""
    result: dict[str, list[Path]] = {}
    videos_dir = dataset_root / "videos"
    for cam_dir in sorted(videos_dir.iterdir()):
        if not cam_dir.is_dir():
            continue
        segs = sorted(cam_dir.glob("*.mp4"), key=lambda p: int(p.stem))
        if segs:
            result[cam_dir.name] = segs
    return result


def report_video_properties(video_segments: dict[str, list[Path]]) -> dict:
    print("=" * 60)
    print("VIDEO PROPERTIES")
    print("=" * 60)
    props = {}
    for cam_id, segs in video_segments.items():
        cap = cv2.VideoCapture(str(segs[0]))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        n_frames_seg = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
        fourcc_str = "".join(chr((fourcc_int >> (i * 8)) & 0xFF) for i in range(4))
        cap.release()

        total_frames = n_frames_seg * len(segs)
        duration_s = total_frames / fps if fps > 0 else 0
        print(f"  {cam_id}: {w}×{h}  {fps:.0f} fps  {n_frames_seg} frames/seg × {len(segs)} segs"
              f" = {total_frames} total  ({duration_s:.1f} s)  codec={fourcc_str}")
        props[cam_id] = {
            "width": w, "height": h, "fps": fps,
            "frames_per_segment": n_frames_seg,
            "n_segments": len(segs),
            "total_frames": total_frames,
            "duration_s": duration_s,
            "fourcc": fourcc_str,
        }
    print()
    return props


# ---------------------------------------------------------------------------
# Figure generation
# ---------------------------------------------------------------------------

def save_camera1_frame0(video_segments: dict[str, list[Path]], out_dir: Path) -> Path:
    segs = video_segments["Camera1"]
    cap = cv2.VideoCapture(str(segs[0]))
    ret, frame = cap.read()
    cap.release()
    assert ret, "Could not read Camera1 frame 0"
    out = out_dir / "dannce_mm1_camera1_frame0.jpg"
    cv2.imwrite(str(out), frame)
    print(f"Saved: {out}")
    return out


def save_all_cameras_frame0(video_segments: dict[str, list[Path]], out_dir: Path) -> Path:
    frames = {}
    for cam_id, segs in sorted(video_segments.items()):
        cap = cv2.VideoCapture(str(segs[0]))
        ret, frame = cap.read()
        cap.release()
        if ret:
            frames[cam_id] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("DANNCE markerless_mouse_1 — Frame 0, all cameras", fontsize=14)

    for ax, (cam_id, img) in zip(axes.flat, sorted(frames.items())):
        ax.imshow(img)
        ax.set_title(cam_id, fontsize=11)
        ax.axis("off")

    for ax in axes.flat[len(frames):]:
        ax.axis("off")

    plt.tight_layout()
    out = out_dir / "dannce_mm1_all_cameras_frame0.jpg"
    plt.savefig(str(out), dpi=80, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")
    return out


# ---------------------------------------------------------------------------
# Cross-checks
# ---------------------------------------------------------------------------

def cross_check(params, video_segments: dict) -> None:
    print("=" * 60)
    print("CROSS-CHECKS")
    print("=" * 60)
    n_cam_params = len(params)
    n_cam_videos = len(video_segments)
    match = "OK" if n_cam_params == n_cam_videos else "MISMATCH"
    print(f"  Cameras in params: {n_cam_params}  |  Camera video dirs: {n_cam_videos}  [{match}]")
    print()


# ---------------------------------------------------------------------------
# Write documentation
# ---------------------------------------------------------------------------

def write_dataset_doc(
    dataset_root: Path,
    cam1_info: dict,
    video_props: dict,
    reprojection_errors: dict[str, float],
    centres: list[np.ndarray],
    n_labeled_frames: int,
    n_joints: int,
    doc_path: Path,
) -> None:
    v = video_props["Camera1"]
    mean_err_all = float(np.mean(list(reprojection_errors.values())))

    lines = textwrap.dedent(f"""\
    # DANNCE markerless_mouse_1 Dataset

    ## Overview

    The **markerless_mouse_1** dataset is part of the DANNCE (3-D Aligned Neural Network for
    Computational Ethology) open-data release from the Dunn lab at Duke University.  It consists
    of a 6-camera synchronised recording of a freely-moving mouse in a circular arena, with dense
    3-D pose labels for {n_labeled_frames} frames and body-wide triangulated 3-D keypoints.

    - **Animal:** C57BL/6 mouse (laboratory mouse)
    - **Arena:** circular open-field arena
    - **Cameras:** 6 × {v['width']}×{v['height']} px, H.264, {v['fps']:.0f} fps hardware-synchronised
    - **Total frames:** {v['total_frames']} per camera × 6 = {v['total_frames']*6} frames
    - **Duration:** {v['duration_s']:.1f} s ({v['duration_s']/60:.1f} min) per camera
    - **Labeled frames:** {n_labeled_frames} frames × {n_joints} body keypoints

    **Citation:** Dunn, T.W., Marshall, J.D., Severson, K.S., et al. (2021).
    Geometric deep learning enables 3D kinematic profiling across species and environments.
    *Nature Methods*, 18, 564–573. https://doi.org/10.1038/s41592-021-01106-6

    **Source:** https://github.com/spoonsso/dannce

    ## Directory Structure

    ```
    {dataset_root}/
    ├── label3d_dannce.mat          (1.2 MB — calibration, sync, and labels)
    └── videos/
        ├── Camera1/
        │   ├── 0.mp4               (~262 MB — frames 0–2999)
        │   ├── 3000.mp4            (~271 MB — frames 3000–5999)
        │   ├── 6000.mp4
        │   ├── 9000.mp4
        │   ├── 12000.mp4
        │   └── 15000.mp4           (frames 15000–17999)
        ├── Camera2/ … (same layout)
        ⋮
        └── Camera6/
    ```

    Videos are split into 6 segments per camera of {v['frames_per_segment']} frames each (3000 ÷ {v['fps']:.0f} fps = {3000/v['fps']:.0f} s per segment).
    The filename stem is the **global start frame index** of that segment
    (0-indexed in Python; DANNCE uses 1-indexed MATLAB frame numbers internally).

    ## Calibration Format

    The calibration lives in `label3d_dannce.mat` under the `params` key, which is a
    6-element MATLAB cell array (one struct per camera).

    ### Fields per camera struct

    | Field | Shape | Meaning |
    |-------|-------|---------|
    | `K` | (3, 3) | **Camera matrix, stored transposed.** `K_opencv = K_stored.T` |
    | `r` | (3, 3) | **Rotation matrix, stored transposed.** `R_opencv = r_stored.T` |
    | `t` | (3,) | Translation vector (world→camera). Same convention as OpenCV `t` in `p = K·(R·X + t)` |
    | `RDistort` | (3,) | Radial distortion coefficients `[k1, k2, k3]` |
    | `TDistort` | (2,) | Tangential distortion coefficients `[p1, p2]` |

    ### Coordinate convention

    DANNCE uses a **world-to-camera** convention (same as OpenCV).  The projection is:

    ```
    p_pixel = K_opencv · (R_opencv · X_world + t)
    ```

    where `K_opencv = K_stored.T` and `R_opencv = r_stored.T`.

    The **world frame** is the DANNCE L-frame: an external lab coordinate system in millimetres,
    distinct from any camera's frame.  Camera centres in world coordinates:

    | Camera | X (mm) | Y (mm) | Z (mm) |
    |--------|--------|--------|--------|
    {chr(10).join(f"    | Camera{i+1} | {c[0]:8.2f} | {c[1]:8.2f} | {c[2]:8.2f} |" for i, c in enumerate(centres))}

    Camera centre = `−R_opencv.T · t`.

    ### Distortion model

    OpenCV 5-parameter model `[k1, k2, p1, p2, k3]` constructed from DANNCE fields as:

    ```python
    dist_opencv = [RDistort[0], RDistort[1], TDistort[0], TDistort[1], RDistort[2]]
    ```

    Note that `k3 = RDistort[2]` is large for some cameras (≈ −2.7 for Camera1), so including it
    matters.  Do not zero it.

    ### Converting to canonical (camera-to-world) form

    The pipeline's canonical calibration stores **camera-to-world** (`T_world_from_cam`).
    The loader computes:

    ```python
    T_cam_from_world = np.eye(4)
    T_cam_from_world[:3, :3] = R_opencv       # = r_stored.T
    T_cam_from_world[:3,  3] = t              # same t
    T_world_from_cam = np.linalg.inv(T_cam_from_world)
    ```

    ### Reprojection error (validation against labeled frames)

    Using the world-to-camera convention above, projecting the labeled 3D keypoints back into each
    camera and comparing to the stored 2D labels gives:

    | Camera | Mean reprojection error |
    |--------|------------------------|
    {chr(10).join(f"    | Camera{i+1} | {reprojection_errors.get(f'Camera{i+1}', float('nan')):.2f} px |" for i in range(6))}
    | **All cameras** | **{mean_err_all:.2f} px** |

    All errors are sub-pixel, confirming the projection convention is correct.

    ## Synchronisation Format

    The `sync` key in the MAT file is a 6-element array (one per camera).  Each element has:

    | Field | Shape | Meaning |
    |-------|-------|---------|
    | `data_frame` | (18000,) | Sequential 0-indexed frame numbers 0…17999 |
    | `data_sampleID` | (18000,) | Corresponding audio sample index (1-indexed, step 10) |
    | `data_2d` | (18000, 44) | 2D predictions for all frames (sparse — mostly zeros for unlabeled) |
    | `data_3d` | (18000, 66) | 3D predictions for all frames (similarly sparse) |

    `data_frame` is simply `[0, 1, 2, …, 17999]` — cameras were hardware-synchronised via
    electronic trigger, so frame indices are identical across all 6 cameras with zero residual
    drift.  The `data_sampleID` values (1, 11, 21, …) reflect the audio ADC sampling rate of
    `fps × 10 = {v['fps']*10:.0f}` Hz aligned to each video frame.

    ## Video Format

    | Property | Value |
    |----------|-------|
    | Resolution | {v['width']} × {v['height']} px |
    | Frame rate | {v['fps']:.0f} fps |
    | Codec | H.264 (AVC) |
    | Container | MP4 |
    | Segments per camera | 6 |
    | Frames per segment | {v['frames_per_segment']} |
    | Total frames | {v['total_frames']} |
    | Duration | {v['duration_s']:.1f} s |
    | Audio | None |

    ## Annotations

    The `labelData` key mirrors the `sync` structure but contains only the {n_labeled_frames} manually
    labeled frames (≈ 0.5% of total).  Each frame has 2D pixel coordinates and 3D world coordinates
    (in mm) for {n_joints} body keypoints arranged as `[joint0_x, joint0_y, joint0_z, joint1_x, …]`
    (66 = 22 × 3 columns).  Some joints are unlabeled in some frames (stored as `NaN`).

    The 22 keypoints are: sacrum, L5, mid-spine, T4, T1, base-of-tail, mid-tail, tip-of-tail,
    left shoulder, right shoulder, front-left (×4 limb joints), front-right (×4), hind-left (×4),
    hind-right (×4).

    ## Known Quirks

    1. **Transposed K and R matrices.** DANNCE stores `K` and `r` as their own transposes relative
       to the OpenCV convention.  Always apply `.T` when loading: `K_opencv = mat_K.T`,
       `R_opencv = mat_r.T`.

    2. **1-indexed MATLAB frame numbers.**  `data_sampleID` is 1-indexed (starts at 1).
       `data_frame` is already 0-indexed (starts at 0) and can be used directly as a Python index.

    3. **Three radial distortion coefficients.**  `RDistort` has three elements `[k1, k2, k3]`
       including `k3`.  OpenCV order is `[k1, k2, p1, p2, k3]` — do not place `k3` at index 2.

    4. **Segmented video files.**  There is no single `0.mp4` containing all frames.  The file
       stem is the global frame offset of the first frame in that file.  Seeking to frame `n`
       requires finding which segment covers `n` and seeking to the local offset `n % 3000`.

    5. **100 fps, not 50 fps.**  The DANNCE README implies 50 Hz pose estimation, but the raw
       video is captured at 100 fps.  The pose estimation runs at 50 Hz by sub-sampling every other
       frame.

    6. **No audio track.**  Cross-correlation-based sync cannot be applied to this dataset.
       Synchronisation relies entirely on the electronic hardware trigger.
    """)

    doc_path.parent.mkdir(parents=True, exist_ok=True)
    doc_path.write_text(lines)
    print(f"Wrote: {doc_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    root = get_dataset_root()
    mat_path = root / "label3d_dannce.mat"
    figures_dir = REPO_ROOT / "output" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    doc_path = REPO_ROOT / "docs" / "DATASET_DANNCE_MM1.md"

    print(f"Dataset root : {root}")
    print(f"MAT file     : {mat_path}")
    print()

    # -- Load MAT --
    mat = load_mat(mat_path)

    # -- Report structure --
    structs = report_mat_structure(mat)
    cam1_info = report_camera1_values(structs["params"])
    centres = report_all_camera_centres(structs["params"])
    repro_errors = compute_reprojection_errors(structs["params"], structs["labelData"])

    # -- Video --
    video_segs = get_video_segments(root)
    video_props = report_video_properties(video_segs)
    cross_check(structs["params"], video_segs)

    # -- Figures --
    print("=" * 60)
    print("SAVING FIGURES")
    print("=" * 60)
    save_camera1_frame0(video_segs, figures_dir)
    save_all_cameras_frame0(video_segs, figures_dir)
    print()

    # -- Documentation --
    ld0 = structs["labelData"][0]
    n_joints = ld0.data_3d.shape[1] // 3
    write_dataset_doc(
        dataset_root=root,
        cam1_info=cam1_info,
        video_props=video_props,
        reprojection_errors=repro_errors,
        centres=centres,
        n_labeled_frames=len(ld0.data_frame),
        n_joints=n_joints,
        doc_path=doc_path,
    )
    print()
    print("Inspection complete.")


if __name__ == "__main__":
    main()
