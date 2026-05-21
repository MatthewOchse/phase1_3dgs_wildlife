# DANNCE markerless_mouse_1 Dataset

## Overview

The **markerless_mouse_1** dataset is part of the DANNCE (3-D Aligned Neural Network for
Computational Ethology) open-data release from the Dunn lab at Duke University.  It consists
of a 6-camera synchronised recording of a freely-moving mouse in a circular arena, with dense
3-D pose labels for 81 frames and body-wide triangulated 3-D keypoints.

- **Animal:** C57BL/6 mouse (laboratory mouse)
- **Arena:** circular open-field arena
- **Cameras:** 6 × 1152×1024 px, H.264, 100 fps hardware-synchronised
- **Total frames:** 18000 per camera × 6 = 108000 frames
- **Duration:** 180.0 s (3.0 min) per camera
- **Labeled frames:** 81 frames × 22 body keypoints

**Citation:** Dunn, T.W., Marshall, J.D., Severson, K.S., et al. (2021).
Geometric deep learning enables 3D kinematic profiling across species and environments.
*Nature Methods*, 18, 564–573. https://doi.org/10.1038/s41592-021-01106-6

**Source:** https://github.com/spoonsso/dannce

## Directory Structure

```
/home/matthew/datasets/dannce_mm1/
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

Videos are split into 6 segments per camera of 3000 frames each (3000 ÷ 100 fps = 30 s per segment).
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
    | Camera1 |  -199.48 |  -129.34 |    63.44 |
| Camera2 |   321.65 |   216.84 |   145.93 |
| Camera3 |    24.05 |   355.81 |    71.19 |
| Camera4 |   327.64 |   -65.30 |    63.27 |
| Camera5 |  -230.39 |   167.36 |   142.22 |
| Camera6 |   114.57 |  -236.28 |   156.54 |

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
    | Camera1 | 0.83 px |
| Camera2 | 0.21 px |
| Camera3 | 0.14 px |
| Camera4 | 0.76 px |
| Camera5 | 0.14 px |
| Camera6 | 0.35 px |
| **All cameras** | **0.40 px** |

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
`fps × 10 = 1000` Hz aligned to each video frame.

## Video Format

| Property | Value |
|----------|-------|
| Resolution | 1152 × 1024 px |
| Frame rate | 100 fps |
| Codec | H.264 (AVC) |
| Container | MP4 |
| Segments per camera | 6 |
| Frames per segment | 3000 |
| Total frames | 18000 |
| Duration | 180.0 s |
| Audio | None |

## Annotations

The `labelData` key mirrors the `sync` structure but contains only the 81 manually
labeled frames (≈ 0.5% of total).  Each frame has 2D pixel coordinates and 3D world coordinates
(in mm) for 22 body keypoints arranged as `[joint0_x, joint0_y, joint0_z, joint1_x, …]`
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
