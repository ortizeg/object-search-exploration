"""ONNX inference: one wrapper that validates its input contract at load (INFRA-09).

Import the pieces from here::

    from object_search.inference import ONNXInferencer, ONNXInputSpec, ONNXContractError
"""

from object_search.inference.base import BaseInferencer, PostProcessor
from object_search.inference.onnx_inferencer import (
    ONNXContractError,
    ONNXInferencer,
    ONNXInputSpec,
    PreprocessInfo,
)

__all__ = [
    "BaseInferencer",
    "ONNXContractError",
    "ONNXInferencer",
    "ONNXInputSpec",
    "PostProcessor",
    "PreprocessInfo",
]
