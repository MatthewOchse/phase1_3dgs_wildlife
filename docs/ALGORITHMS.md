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

### Extrinsic calibration and the rig world frame

**Extrinsic calibration** determines where each camera sits in 3-D space relative to every other camera.  A 6-DoF (six degrees of freedom) pose has three translational components (where the camera's origin is) and three rotational components (which way the camera is pointing).  Together they form a 4×4 homogeneous transformation matrix.

Rather than calibrating camera pairs independently and chaining the results, we use a single Charuco board to define a common **world frame** for the whole rig.  Each camera captures the same board simultaneously; a PnP (Perspective-n-Point) solve against the board's known 3-D corner positions yields that camera's pose relative to the board in one shot.  Because all cameras share the same reference object, their poses are automatically expressed in the same coordinate system — no multi-camera bundle adjustment is required.

**Coordinate convention used throughout this pipeline:**

* The **world frame** is the Charuco board frame.  The board lies in the Z = 0 plane; X points right along the first column of squares, Y points down along the first row, Z points outward toward the cameras.
* Each **camera frame** follows the standard OpenCV convention: X right, Y down, +Z pointing into the scene (away from the camera).
* We store `T_world_from_cam` — a 4×4 matrix that converts a point in camera coordinates to world coordinates: `p_world = T_world_from_cam @ p_cam`.  Its inverse is what solvePnP returns internally.

**Validation via triangulation reprojection:**  After estimating all poses we verify them by triangulating a set of known board corners from all cameras simultaneously (using multi-view DLT, see below) and projecting the triangulated 3-D points back into each camera's image plane.  The resulting reprojection error should be sub-pixel; consistently higher errors indicate a mis-estimated pose or stale intrinsics.

### DLT triangulation

The **Direct Linear Transform (DLT)** converts multi-view 2-D observations into a 3-D world point algebraically.  For a single 3-D point **X** observed at pixel (u, v) by camera *i* with 3×4 projection matrix **P**ᵢ, the cross-product constraint `(u, v, 1) × (Pᵢ X) = 0` yields two linearly independent equations.  Writing **P**ᵢ row-wise as [p₁; p₂; p₃], the two equations are:

```
u · p₃ᵀ X  −  p₁ᵀ X = 0
v · p₃ᵀ X  −  p₂ᵀ X = 0
```

Stacking these for all N cameras gives a (2N × 4) homogeneous linear system **A X = 0**.  The least-squares solution in the absence of noise is the right singular vector of **A** corresponding to its smallest singular value — obtained directly from an SVD of **A**.  The result is a homogeneous 4-vector; dividing by its fourth component yields the Euclidean 3-D point.

The multi-view DLT is preferred over averaging pairwise `cv2.triangulatePoints` results because it minimises a single global algebraic error.  Pairwise averaging is biased when cameras have unequal noise levels or when the baseline between a particular pair is short.  Our implementation first undistorts the 2-D observations into the normalised image plane so that the projection matrices remain the simple pinhole form K [R | t] without an extra distortion correction term in the linear system.

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
