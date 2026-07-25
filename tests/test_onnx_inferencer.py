"""Tests for ONNXInferencer's init-time input-contract validation (INFRA-09).

Every test builds a *real* tiny ONNX model on disk and loads it under onnxruntime, so the
validation being exercised is the same code path a production model hits -- not a mock.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
import onnx
import pytest
from onnx import TensorProto, helper

from object_search.inference import ONNXContractError, ONNXInferencer, ONNXInputSpec

# Pin the CPU provider: the dev machine also exposes CoreML, and a fixed provider keeps the
# tiny-model runs identical run to run.
_CPU = ["CPUExecutionProvider"]


def _make_tiny_model(
    path: Path,
    *,
    shape: list[int | str],
    dtype: int = TensorProto.FLOAT,
    input_name: str = "pixel_values",
) -> None:
    """Write a minimal single-node Identity ONNX model with the requested input signature."""
    inp = helper.make_tensor_value_info(input_name, dtype, shape)
    out = helper.make_tensor_value_info("out", dtype, shape)
    node = helper.make_node("Identity", inputs=[input_name], outputs=["out"])
    graph = helper.make_graph([node], "tiny", [inp], [out])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    # onnxruntime 1.23.2 accepts IR version 10; pin it so a newer onnx default cannot break load.
    model.ir_version = 10
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


def _shape_postprocessor(
    outputs: list[npt.NDArray[np.generic]],
    orig_w: int,
    orig_h: int,
) -> tuple[int, ...]:
    """A trivial post-processor: return the first output's shape, to prove predict() ran."""
    return tuple(int(dim) for dim in outputs[0].shape)


def _spec(shape: tuple[int | str, ...], dtype: str = "tensor(float)") -> ONNXInputSpec:
    return ONNXInputSpec(
        shape=shape,
        dtype=dtype,
        mean=(0.0, 0.0, 0.0),
        std=(1.0, 1.0, 1.0),
        resize="stretch",
    )


def test_missing_file_raises_filenotfound(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        ONNXInferencer(
            tmp_path / "does_not_exist.onnx",
            _shape_postprocessor,
            _spec((1, 3, 8, 8)),
            providers=_CPU,
        )


def test_correct_spec_constructs_and_predict_returns_postprocessor_output(tmp_path: Path) -> None:
    model_path = tmp_path / "tiny.onnx"
    _make_tiny_model(model_path, shape=[1, 3, 8, 8])

    inferencer: ONNXInferencer[tuple[int, ...]] = ONNXInferencer(
        model_path,
        _shape_postprocessor,
        _spec((1, 3, 8, 8)),
        providers=_CPU,
    )
    assert inferencer.output_names == ("out",)

    scene = np.zeros((8, 8, 3), dtype=np.uint8)
    result = inferencer.predict(scene)
    assert result == (1, 3, 8, 8)


def test_dynamic_dims_are_accepted(tmp_path: Path) -> None:
    """Symbolic H/W on both sides must be skipped, not rejected (the real-model case)."""
    model_path = tmp_path / "dyn.onnx"
    _make_tiny_model(model_path, shape=[1, 3, "height", "width"])

    inferencer: ONNXInferencer[tuple[int, ...]] = ONNXInferencer(
        model_path,
        _shape_postprocessor,
        _spec((1, 3, "height", "width")),
        providers=_CPU,
    )
    scene = np.zeros((28, 28, 3), dtype=np.uint8)
    # A dynamic model runs at whatever size preprocess produced (original, for stretch).
    assert inferencer.predict(scene) == (1, 3, 28, 28)


def test_wrong_shape_raises_contract_error_at_construction(tmp_path: Path) -> None:
    model_path = tmp_path / "tiny.onnx"
    _make_tiny_model(model_path, shape=[1, 3, 8, 8])

    with pytest.raises(ONNXContractError) as excinfo:
        ONNXInferencer(
            model_path,
            _shape_postprocessor,
            _spec((1, 3, 16, 16)),  # model is 8x8; spec claims 16x16
            providers=_CPU,
        )
    message = str(excinfo.value)
    # Names both the actual and the expected values (both mismatching dims reported).
    assert "8" in message
    assert "16" in message
    assert "dim[2]" in message
    assert "dim[3]" in message


def test_wrong_dtype_raises_contract_error_at_construction(tmp_path: Path) -> None:
    model_path = tmp_path / "tiny.onnx"
    _make_tiny_model(model_path, shape=[1, 3, 8, 8], dtype=TensorProto.FLOAT)

    with pytest.raises(ONNXContractError) as excinfo:
        ONNXInferencer(
            model_path,
            _shape_postprocessor,
            _spec((1, 3, 8, 8), dtype="tensor(int64)"),  # model is float
            providers=_CPU,
        )
    message = str(excinfo.value)
    assert "tensor(float)" in message
    assert "tensor(int64)" in message


def test_rank_mismatch_raises(tmp_path: Path) -> None:
    model_path = tmp_path / "tiny.onnx"
    _make_tiny_model(model_path, shape=[1, 3, 8, 8])

    with pytest.raises(ONNXContractError, match="rank mismatch"):
        ONNXInferencer(
            model_path,
            _shape_postprocessor,
            _spec((1, 3, 8)),  # rank 3 vs model rank 4
            providers=_CPU,
        )


def test_named_input_not_found_raises(tmp_path: Path) -> None:
    model_path = tmp_path / "tiny.onnx"
    _make_tiny_model(model_path, shape=[1, 3, 8, 8], input_name="pixel_values")

    spec = ONNXInputSpec(
        input_name="not_there",
        shape=(1, 3, 8, 8),
        mean=(0.0, 0.0, 0.0),
        std=(1.0, 1.0, 1.0),
    )
    with pytest.raises(ONNXContractError, match="not found"):
        ONNXInferencer(model_path, _shape_postprocessor, spec, providers=_CPU)


def test_snap_to_multiple_requires_size_multiple() -> None:
    with pytest.raises(ValueError, match="size_multiple"):
        ONNXInputSpec(
            shape=(1, 3, "h", "w"),
            mean=(0.0, 0.0, 0.0),
            std=(1.0, 1.0, 1.0),
            resize="snap-to-multiple",
        )
