# Architecture Decision Records

## ADR-001: Fixed Multi-camera Rig with Known Calibration over SfM

We use a fixed multi-camera rig with pre-computed intrinsic and extrinsic calibration rather than Structure-from-Motion (SfM) because the capture scenario is controlled and repeatable: cameras are mounted at known positions around a defined capture zone, and the rig geometry does not change between sessions. SfM is designed for uncontrolled, unordered image collections where camera poses are unknown, which adds significant computational overhead and introduces pose estimation errors that compound downstream. By calibrating the rig once with a physical target (e.g. a charuco board) and storing the result, we obtain exact camera matrices that remain valid across all capture sessions, enabling deterministic and fast reconstruction without the non-convex optimisation step that SfM requires.

## ADR-002: Canonical Internal Calibration Format with Dataset-Specific Adapters

All downstream pipeline code works against a single canonical calibration dict format (OpenCV-style 3×3 camera matrix, 5-element ``[k1, k2, p1, p2, k3]`` distortion vector, and camera-to-world ``T_world_from_cam`` 4×4 extrinsics). Dataset-specific quirks are isolated in adapter modules (e.g. ``src/io_utils/dannce_loader.py``) that translate from the raw dataset format into this canonical form before handing data to the pipeline.

This design means adding a new dataset (AcinoSet cheetah captures, our own Kruger field rig) requires writing one new adapter function rather than touching any reconstruction or segmentation code. It also makes unit testing straightforward: synthetic calibration dicts can be constructed directly in tests without reference to any real dataset file.

The DANNCE dataset illustrates why this isolation matters: DANNCE stores its camera matrix ``K`` and rotation matrix ``R`` **transposed** relative to OpenCV convention, uses a ``[k1, k2, k3]`` + ``[p1, p2]`` split for distortion rather than OpenCV's interleaved ``[k1, k2, p1, p2, k3]`` order, and represents extrinsics as world-to-camera rather than the canonical camera-to-world. The adapter handles all three differences in one place (``load_dannce_calibration``), and the rest of the pipeline never sees them.
