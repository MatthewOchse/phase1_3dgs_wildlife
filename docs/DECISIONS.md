# Architecture Decision Records

## ADR-001: Fixed Multi-camera Rig with Known Calibration over SfM

We use a fixed multi-camera rig with pre-computed intrinsic and extrinsic calibration rather than Structure-from-Motion (SfM) because the capture scenario is controlled and repeatable: cameras are mounted at known positions around a defined capture zone, and the rig geometry does not change between sessions. SfM is designed for uncontrolled, unordered image collections where camera poses are unknown, which adds significant computational overhead and introduces pose estimation errors that compound downstream. By calibrating the rig once with a physical target (e.g. a charuco board) and storing the result, we obtain exact camera matrices that remain valid across all capture sessions, enabling deterministic and fast reconstruction without the non-convex optimisation step that SfM requires.
