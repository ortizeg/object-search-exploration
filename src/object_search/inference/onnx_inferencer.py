"""``ONNXInferencer`` -- an ONNX Runtime wrapper that validates its INPUT contract at load.

Ported and adapted from ``basketball-2d-to-3d``; this file does **not** import that project.

What INFRA-09 buys, and what it deliberately does not
-----------------------------------------------------
A wrong model fails at *construction*, before a single image is passed: the declared input
dtype and the declared input shape are checked against a typed :class:`ONNXInputSpec`, and a
mismatch raises :class:`ONNXContractError`. That turns "silently wrong ONNX inference" -- the
single most common failure mode this layer exists to prevent -- into a loud load-time error.

**Only the INPUT is validated. Output shapes are never asserted.** All three models this
project uses declare *symbolic* output dimensions: DINOv2's token count is
``floor(H/14)*floor(W/14)+1``, SuperPoint's keypoint count is genuinely data-dependent, and
FastSAM's anchor count depends on resolution. A validator demanding static output extents
would reject every working model (verified in ``.planning/research/MODELS.md``). Output
*names* are recorded and their presence is checked; their lengths are not.

Pre/post-processing contract (project constraint: state it in the docstring, exactly)
-------------------------------------------------------------------------------------
:class:`ONNXInputSpec` *is* the preprocessing contract as a typed object, so an inferencer
cannot be constructed without stating every step. :meth:`ONNXInferencer.preprocess` executes
it faithfully: BGR->RGB per ``color_order``; resize per ``resize`` policy
(``stretch`` / ``letterbox`` / ``snap-to-multiple``, the last honouring ``size_multiple``);
multiply by ``scale``; subtract ``mean`` and divide by ``std``; transpose to ``layout``; add
the batch dim; cast float32. It returns a :class:`PreprocessInfo` carrying the exact scale
and pad factors, because a caller cannot map a model-space box back to image pixels without
them. ``snap-to-multiple`` snaps each side to the nearest multiple of ``size_multiple``
rather than padding, because DINOv2 silently drops trailing pixels on a non-multiple side --
an up-to-13-pixel misalignment that would show up as a systematic box offset, not an error.

Reproducibility -- the real threats, and a warning about a false one
--------------------------------------------------------------------
``intra_op_num_threads`` / ``inter_op_num_threads`` are exposed as **latency / CPU-contention
controls only**. They are NOT a determinism measure: research measured ORT thread count,
OpenCV thread count, BLAS thread count and argmax tie order as all producing bit-identical
output, and ``use_deterministic_compute`` is a no-op on the CPU execution provider. What
actually threatens reproducibility is set/dict iteration order, NMS tie-breaking (sort by
``(-score, y, x)``, never score alone), config-hash key ordering, and library-version drift.
The last of those is guarded by :attr:`ONNXInferencer.model_sha256` (EVAL-09 provenance).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import numpy.typing as npt
import onnxruntime as ort
from loguru import logger
from pydantic import BaseModel, ConfigDict, model_validator

from object_search import provenance
from object_search.inference.base import BaseInferencer, PostProcessor

# ONNX tensor-type string -> numpy dtype name, for readable error messages.
ONNX_TYPE_TO_NUMPY: dict[str, str] = {
    "tensor(float)": "float32",
    "tensor(float16)": "float16",
    "tensor(double)": "float64",
    "tensor(int32)": "int32",
    "tensor(int64)": "int64",
    "tensor(uint8)": "uint8",
    "tensor(int8)": "int8",
    "tensor(bool)": "bool",
}


class ONNXContractError(ValueError):
    """The model on disk violates its declared :class:`ONNXInputSpec`.

    A named subclass of ``ValueError`` so a caller can tell a contract violation apart from
    any other bad argument.
    """


class ONNXInputSpec(BaseModel):
    """The full preprocessing + input contract for one ONNX model, as a frozen object.

    Making this a typed model rather than prose is what makes the project's "pre-processing
    must be explicitly documented" constraint *structural*: you cannot build an inferencer
    without stating every field below.

    Attributes:
        input_name: The graph input to bind. ``None`` = the model's first input.
        dtype: Expected ONNX tensor type string, e.g. ``"tensor(float)"``.
        shape: Expected input shape. ``int`` dims are validated; ``str`` dims are dynamic
            and skipped.
        layout: ``"NCHW"`` or ``"NHWC"``.
        color_order: Channel order the model wants. Scenes arrive BGR (OpenCV).
        scale: Multiply pixels by this first (``1/255`` maps ``0..255`` to ``0..1``).
        mean: Per-channel mean subtracted after scaling.
        std: Per-channel std divided after mean subtraction.
        resize: ``"stretch"`` ignores aspect ratio; ``"letterbox"`` preserves it with
            padding; ``"snap-to-multiple"`` resizes each side to the nearest multiple of
            ``size_multiple`` (required by patch models such as DINOv2).
        size_multiple: The multiple for ``"snap-to-multiple"`` (DINOv2 needs 14).
        interpolation: ``"bilinear"`` or ``"bicubic"`` (DINOv2 wants bicubic).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    input_name: str | None = None
    dtype: str = "tensor(float)"
    shape: tuple[int | str, ...]
    layout: Literal["NCHW", "NHWC"] = "NCHW"
    color_order: Literal["RGB", "BGR"] = "RGB"
    scale: float = 1.0 / 255.0
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    resize: Literal["stretch", "letterbox", "snap-to-multiple"] = "stretch"
    size_multiple: int | None = None
    interpolation: Literal["bilinear", "bicubic"] = "bilinear"

    @model_validator(mode="after")
    def _snap_needs_a_multiple(self) -> ONNXInputSpec:
        if self.resize == "snap-to-multiple" and (
            self.size_multiple is None or self.size_multiple < 1
        ):
            raise ValueError(
                "resize='snap-to-multiple' requires size_multiple >= 1 "
                "(e.g. 14 for DINOv2); got size_multiple="
                f"{self.size_multiple!r}"
            )
        return self


class PreprocessInfo:
    """The preprocessed tensor plus the exact factors needed to invert the resize.

    A method maps a model-space coordinate back to the original image with
    ``orig = (coord - pad) / scale`` per axis, so these factors travel with the tensor rather
    than being recomputed (and mis-recomputed) at the call site.
    """

    __slots__ = ("input_h", "input_w", "pad_x", "pad_y", "scale_x", "scale_y", "tensor")

    def __init__(
        self,
        tensor: npt.NDArray[np.float32],
        scale_x: float,
        scale_y: float,
        pad_x: int,
        pad_y: int,
        input_w: int,
        input_h: int,
    ) -> None:
        self.tensor = tensor
        self.scale_x = scale_x
        self.scale_y = scale_y
        self.pad_x = pad_x
        self.pad_y = pad_y
        self.input_w = input_w
        self.input_h = input_h


# A neutral grey letterbox fill, matching the long-standing YOLO/Ultralytics convention.
_LETTERBOX_FILL = 114


class ONNXInferencer[OutT](BaseInferencer[OutT]):
    """Load an ONNX model, validate its input contract at construction, and run it.

    See the module docstring for the full pre/post-processing and reproducibility contract.

    Args:
        model_path: Path to the ``.onnx`` file. Missing -> ``FileNotFoundError``.
        post_processor: Strategy that decodes raw outputs into ``OutT``.
        input_spec: The :class:`ONNXInputSpec` the model must satisfy.
        providers: ONNX Runtime execution providers. ``None`` = the runtime default.
        intra_op_num_threads: Latency / CPU-contention control only (see module docstring).
        inter_op_num_threads: Latency / CPU-contention control only.
    """

    def __init__(
        self,
        model_path: Path | str,
        post_processor: PostProcessor[OutT],
        input_spec: ONNXInputSpec,
        providers: list[str] | None = None,
        *,
        intra_op_num_threads: int | None = None,
        inter_op_num_threads: int | None = None,
    ) -> None:
        self._model_path = Path(model_path)
        if not self._model_path.is_file():
            raise FileNotFoundError(f"ONNX model file not found: {self._model_path}")

        self._post_processor = post_processor
        self.input_spec = input_spec
        self._model_sha256: str | None = None

        session_options = ort.SessionOptions()
        if intra_op_num_threads is not None:
            session_options.intra_op_num_threads = intra_op_num_threads
        if inter_op_num_threads is not None:
            session_options.inter_op_num_threads = inter_op_num_threads

        logger.info(f"Loading ONNX model from {self._model_path}")
        self._session = ort.InferenceSession(
            str(self._model_path),
            sess_options=session_options,
            providers=providers or ort.get_available_providers(),
        )

        model_input = self._resolve_input()
        self._input_name: str = model_input.name
        actual_dtype: str = model_input.type
        actual_shape: list[int | str | None] = model_input.shape

        self._validate_dtype(actual_dtype, input_spec.dtype)
        self._validate_shape(actual_shape, list(input_spec.shape))

        self.output_names: tuple[str, ...] = tuple(o.name for o in self._session.get_outputs())
        if not self.output_names:
            raise ONNXContractError(f"model {self._model_path} declares no outputs")

        logger.info(
            f"ONNX session ready (input={self._input_name}, dtype={actual_dtype}, "
            f"shape={actual_shape}, outputs={list(self.output_names)}, "
            f"providers={self._session.get_providers()})"
        )

    # -- construction-time validation --------------------------------------------------

    def _resolve_input(self) -> ort.NodeArg:
        """The named input from the spec, or the first input when the spec leaves it None."""
        inputs = self._session.get_inputs()
        wanted = self.input_spec.input_name
        if wanted is None:
            return inputs[0]
        for candidate in inputs:
            if candidate.name == wanted:
                return candidate
        available = [candidate.name for candidate in inputs]
        raise ONNXContractError(
            f"input {wanted!r} not found in model {self._model_path}; available inputs: {available}"
        )

    @staticmethod
    def _validate_dtype(actual: str, expected: str) -> None:
        if actual != expected:
            raise ONNXContractError(
                f"ONNX input dtype mismatch: model declares {actual!r} "
                f"(numpy {ONNX_TYPE_TO_NUMPY.get(actual, '?')}), but the spec expects "
                f"{expected!r} (numpy {ONNX_TYPE_TO_NUMPY.get(expected, '?')})"
            )

    @staticmethod
    def _validate_shape(
        actual: list[int | str | None],
        expected: list[int | str],
    ) -> None:
        """Rank must match; every static dim must match. Dynamic dims are wildcards.

        Reports *every* mismatching index, not just the first, so a model that disagrees in
        two places is fixed in one pass.
        """
        if len(actual) != len(expected):
            raise ONNXContractError(
                f"ONNX input rank mismatch: model has {len(actual)} dims {actual}, "
                f"spec expects {len(expected)} dims {expected}"
            )
        mismatches: list[str] = []
        for i, (actual_dim, expected_dim) in enumerate(zip(actual, expected, strict=True)):
            # A dimension is dynamic (and skipped) if either side is symbolic or unknown.
            if isinstance(expected_dim, str) or isinstance(actual_dim, str) or actual_dim is None:
                continue
            if actual_dim != expected_dim:
                mismatches.append(f"dim[{i}]: model={actual_dim}, spec={expected_dim}")
        if mismatches:
            raise ONNXContractError(
                f"ONNX input shape mismatch: {'; '.join(mismatches)}. "
                f"Model shape={actual}, spec shape={expected}"
            )

    # -- provenance --------------------------------------------------------------------

    @property
    def model_sha256(self) -> str:
        """SHA-256 of the model file, computed once and cached (EVAL-09 provenance)."""
        if self._model_sha256 is None:
            self._model_sha256 = provenance.file_sha256(self._model_path)
        return self._model_sha256

    # -- pre-processing ----------------------------------------------------------------

    def _resize_plan(self, orig_w: int, orig_h: int) -> tuple[int, int, float, float, int, int]:
        """Return ``(target_w, target_h, scale_x, scale_y, pad_x, pad_y)`` for the policy."""
        spec = self.input_spec
        h_idx, w_idx = (2, 3) if spec.layout == "NCHW" else (1, 2)
        dim_h = spec.shape[h_idx]
        dim_w = spec.shape[w_idx]

        if spec.resize == "snap-to-multiple":
            multiple = spec.size_multiple or 1
            target_w = max(multiple, round(orig_w / multiple) * multiple)
            target_h = max(multiple, round(orig_h / multiple) * multiple)
            return target_w, target_h, target_w / orig_w, target_h / orig_h, 0, 0

        target_w = dim_w if isinstance(dim_w, int) else orig_w
        target_h = dim_h if isinstance(dim_h, int) else orig_h

        if spec.resize == "letterbox":
            scale = min(target_w / orig_w, target_h / orig_h)
            new_w = max(1, round(orig_w * scale))
            new_h = max(1, round(orig_h * scale))
            return (
                target_w,
                target_h,
                scale,
                scale,
                (target_w - new_w) // 2,
                (target_h - new_h) // 2,
            )

        # stretch
        return target_w, target_h, target_w / orig_w, target_h / orig_h, 0, 0

    def preprocess(self, image: npt.NDArray[np.uint8]) -> PreprocessInfo:
        """Execute :class:`ONNXInputSpec` on one BGR image; return tensor + resize factors.

        This is the general RGB, 3-channel path. A grayscale-luma backend such as SuperPoint
        overrides this in its own method module (that preprocessing is single-file and
        method-specific by design).
        """
        spec = self.input_spec
        orig_h, orig_w = int(image.shape[0]), int(image.shape[1])

        if image.ndim == 3 and image.shape[2] == 3 and spec.color_order == "RGB":
            converted = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            converted = image

        interp = cv2.INTER_CUBIC if spec.interpolation == "bicubic" else cv2.INTER_LINEAR
        target_w, target_h, scale_x, scale_y, pad_x, pad_y = self._resize_plan(orig_w, orig_h)

        if spec.resize == "letterbox":
            new_w = max(1, round(orig_w * scale_x))
            new_h = max(1, round(orig_h * scale_y))
            fitted = cv2.resize(converted, (new_w, new_h), interpolation=interp)
            canvas = np.full(
                (target_h, target_w, converted.shape[2]), _LETTERBOX_FILL, dtype=np.uint8
            )
            canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = fitted
            resized = canvas
        else:
            resized = cv2.resize(converted, (target_w, target_h), interpolation=interp)

        # Scale, then (x - mean) / std, all in float32. Steps are explicit so numpy's operator
        # typing stays float32 end to end rather than widening to a mixed-dtype result.
        mean = np.asarray(spec.mean, dtype=np.float32)
        std = np.asarray(spec.std, dtype=np.float32)
        scaled: npt.NDArray[np.float32] = np.asarray(resized, dtype=np.float32) * np.float32(
            spec.scale
        )
        normalized: npt.NDArray[np.float32] = (scaled - mean) / std

        if spec.layout == "NCHW":
            normalized = np.transpose(normalized, (2, 0, 1))
        batched = np.ascontiguousarray(np.expand_dims(normalized, axis=0), dtype=np.float32)

        return PreprocessInfo(
            tensor=batched,
            scale_x=scale_x,
            scale_y=scale_y,
            pad_x=pad_x,
            pad_y=pad_y,
            input_w=target_w,
            input_h=target_h,
        )

    # -- inference ---------------------------------------------------------------------

    def predict(self, image: npt.NDArray[np.uint8]) -> OutT:
        """Preprocess -> ``session.run`` -> post-process, for one BGR image."""
        orig_h, orig_w = int(image.shape[0]), int(image.shape[1])
        info = self.preprocess(image)
        outputs = self._session.run(None, {self._input_name: info.tensor})
        return self._post_processor(outputs, orig_w, orig_h)
