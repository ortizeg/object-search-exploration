"""ONNX inference: one wrapper that validates its input contract at load (INFRA-09).

Import the pieces from here::

    from object_search.inference import ONNXInferencer, ONNXInputSpec, ONNXContractError
"""

from object_search.inference.base import BaseInferencer, PostProcessor
from object_search.inference.dinov2 import DINOv2Inferencer
from object_search.inference.onnx_inferencer import (
    ONNXContractError,
    ONNXInferencer,
    ONNXInputSpec,
    PreprocessInfo,
)
from object_search.inference.superpoint import SuperPointInferencer, SuperPointResult

__all__ = [
    "BaseInferencer",
    "DINOv2Inferencer",
    "ONNXContractError",
    "ONNXInferencer",
    "ONNXInputSpec",
    "PostProcessor",
    "PreprocessInfo",
    "SuperPointInferencer",
    "SuperPointResult",
]
