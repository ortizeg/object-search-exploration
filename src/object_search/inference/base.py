"""The inference abstractions, ported and generalized from the sibling project.

The sibling ``basketball-2d-to-3d`` had one return type -- ``list[Detection]``. This project's
inferencers return dense token grids (DINOv2), variable keypoint sets (SuperPoint) and mask
prototypes (FastSAM), so :class:`BaseInferencer` is generic over its output type and the
output-decoding step is a pluggable :class:`PostProcessor` strategy. Keeping decoding out of
the session-management code is the half of the ported design that lets one ``ONNXInferencer``
serve three very different models.

**This is a port, not a dependency.** Nothing here imports ``basketball_2d_to_3d``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt


class PostProcessor[OutT_co](Protocol):
    """Decode a model's raw ONNX outputs into the typed result a method wants.

    ``OutT_co`` appears only in return position, so PEP 695 infers it covariant -- a
    ``PostProcessor[Match]`` is usable where a ``PostProcessor[Detection]`` is expected when
    ``Match`` is a subtype, which is the correct direction for a producer.

    Args:
        outputs: The list of arrays ``onnxruntime`` returned from ``session.run``.
        orig_w: Width of the original (pre-preprocess) image, for coordinate mapping.
        orig_h: Height of the original image.

    Returns:
        The method-specific decoded output.
    """

    def __call__(
        self,
        outputs: list[npt.NDArray[Any]],
        orig_w: int,
        orig_h: int,
    ) -> OutT_co: ...


class BaseInferencer[OutT](ABC):
    """A model wrapper that turns one BGR image into one typed output.

    Generic over ``OutT`` because this project's models do not share a return type. The
    concrete :class:`~object_search.inference.onnx_inferencer.ONNXInferencer` supplies the
    ONNX-specific machinery; a hypothetical non-ONNX inferencer could subclass this too,
    though the project constraint is ONNX Runtime for every learned model.
    """

    @abstractmethod
    def predict(self, image: npt.NDArray[np.uint8]) -> OutT:
        """Run the full pipeline -- preprocess, session, post-process -- on one BGR image."""
        raise NotImplementedError
