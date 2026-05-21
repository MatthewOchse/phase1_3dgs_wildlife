import importlib


SUBMODULES = [
    "calibration",
    "sync",
    "segmentation",
    "io_utils",
    "geometry",
    "features",
]


def test_src_importable():
    import src  # noqa: F401


def test_submodules_importable():
    for name in SUBMODULES:
        mod = importlib.import_module(name)
        assert mod is not None, f"Failed to import {name}"
