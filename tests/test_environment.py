"""Environment guard tests: the native stack must actually import and be in range.

These are deliberately not "trivial import tests". conda-forge publishes `opencv 5.0.0`
builds with NO Python bindings (no `py3XX` build string), so a dependency bump can produce a
perfectly green `pixi install` followed by `ModuleNotFoundError: cv2` at runtime. A solve
succeeding is not evidence that the bindings exist -- importing is. Same reasoning applies to
onnxruntime, whose macOS wheel tags moved out from under the default `__osx` floor at 1.24.1.
"""

import numpy as np


def test_opencv_bindings_are_importable_and_below_5():
    """cv2 must import and stay on the 4.x line that actually ships Python bindings."""
    import cv2

    major = int(cv2.__version__.split(".")[0])
    assert major == 4, f"expected an opencv 4.x build with Python bindings, got {cv2.__version__}"


def test_opencv_has_the_functions_the_methods_rely_on():
    """Guard the specific cv2 entry points Method 1/2 need, including contrib (SIFT)."""
    import cv2

    assert hasattr(cv2, "matchTemplate")
    assert hasattr(cv2, "SIFT_create")
    assert hasattr(cv2, "watershed")


def test_onnxruntime_is_importable_and_pinned():
    """onnxruntime must import and match the pin that is proven to resolve on osx-arm64."""
    import onnxruntime

    assert onnxruntime.__version__ == "1.23.2"
    assert "CPUExecutionProvider" in onnxruntime.get_available_providers()


def test_numpy_is_2x():
    """numpy 2.x is required: the conda-forge cv2/onnxruntime builds target the numpy 2 ABI."""
    assert int(np.__version__.split(".")[0]) == 2


def test_numpy_and_opencv_interoperate():
    """The classic numpy-2 ABI break shows up here, not at import: exercise a real call."""
    import cv2

    image = np.zeros((16, 16, 3), dtype=np.uint8)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    assert gray.shape == (16, 16)
    assert gray.dtype == np.uint8
