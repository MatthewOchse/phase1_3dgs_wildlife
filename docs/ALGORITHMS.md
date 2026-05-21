# Algorithm Explanations

## Table of Contents

1. [Camera Calibration (Intrinsics and Extrinsics)](#camera-calibration)
2. [Multi-view Synchronisation](#multi-view-synchronisation)
3. [Image Segmentation (SAM2, Classical, LiDAR)](#image-segmentation)
4. [Feature Detection and Matching](#feature-detection-and-matching)
5. [Triangulation](#triangulation)
6. [Visual Hull Construction](#visual-hull-construction)

---

## Camera Calibration

### What intrinsic calibration does

A camera does not record the world metrically — it projects 3-D points onto a 2-D sensor through a lens that introduces its own distortion. Intrinsic calibration recovers the parameters that describe this projection so we can invert it: given a pixel coordinate, we can compute the corresponding ray in 3-D space. Without accurate intrinsics every downstream step — triangulation, stereo rectification, pose estimation — accumulates systematic error that no amount of algorithmic cleverness can correct.

The intrinsic model has two components. The **camera matrix** (also called the projection matrix or K) encodes focal length in pixels (`fx`, `fy`) and the principal point (`cx`, `cy`), which is the pixel coordinate of the optical axis. These four numbers define the pinhole projection `[u, v, 1]^T = K [X/Z, Y/Z, 1]^T`. The **distortion coefficients** model lens aberrations — primarily radial distortion (barrel or pincushion) described by coefficients `k1, k2, k3`, and tangential distortion from lens misalignment described by `p1, p2`. OpenCV's `calibrateCamera` estimates all of these simultaneously.

### Why Charuco over a plain checkerboard

A standard checkerboard requires every corner to be visible — if any part of the board is occluded or moves out of frame, that image is discarded. A **Charuco board** embeds unique ArUco markers inside each chessboard square. Because each marker has a known ID, we can identify and localise any subset of the interior chessboard corners even when the board is partially outside the field of view or occluded. This is important in a wildlife capture setup where the rig may not perfectly frame a large calibration target, and where we want to use oblique viewing angles (high tilt gives better coverage of distortion at the sensor periphery).

The workflow is: detect ArUco markers → use their known geometry to localise the board → interpolate the sub-pixel chessboard corner positions between adjacent markers → accumulate `(3-D object point, 2-D image point)` correspondences across many frames → run non-linear optimisation to minimise reprojection error.

### What an acceptable reprojection error looks like

**Reprojection error** measures how far the calibrated model's predicted pixel positions deviate from the actually detected corner positions. It is reported in pixels as an RMS value. As a rule of thumb:

| Error | Assessment |
|-------|------------|
| < 0.5 px | Excellent — suitable for metric reconstruction |
| 0.5 – 1.0 px | Good — acceptable for most 3-D tasks |
| 1.0 – 2.0 px | Marginal — recapture more images with better coverage |
| > 2.0 px | Poor — likely motion blur, wrong board size, or too few images |

For our pipeline, which feeds directly into 3DGS training, we target **under 1.0 px**. Errors consistently above this threshold usually stem from too few calibration frames, insufficient angular variation, or incorrect physical board measurements in the config.

## Multi-view Synchronisation

To be populated in [Step 3].

## Image Segmentation

To be populated in [Step 4].

## Feature Detection and Matching

To be populated in [Step 5].

## Triangulation

To be populated in [Step 6].

## Visual Hull Construction

To be populated in [Step 7].
