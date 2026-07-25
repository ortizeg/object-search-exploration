"""``DINOv2Inferencer`` -- dense patch tokens from ``onnx-community/dinov2-small-ONNX``.

Method 3 (``dino-dense``, Phase 6) and Method 5 (``propose-retrieve``, Phase 7) both read a
dense grid of patch tokens out of DINOv2-small. This one inferencer is the single download and
single preprocessing contract they share, so Phase 7 reuses it rather than duplicating it.

The verified contract (exact numbers, per project constraint -- runtime-verified in
``.planning/research/MODELS.md`` and gated by the ``Adopt`` verdict in
``docs/library-reviews/dinov2.md``)
------------------------------------------------------------------------------------------------
Input
  * name ``pixel_values``; dtype ``float32`` (``tensor(float)``); layout **NCHW**; shape
    ``[batch, 3, height, width]`` with **all four dims dynamic**.
  * colour order **RGB** (scenes arrive BGR from OpenCV and are converted).
  * scale ``1/255``; mean ``[0.485, 0.456, 0.406]``; std ``[0.229, 0.224, 0.225]``.
  * resize **snap-to-multiple(14)** with **bicubic** interpolation and **NO centre-crop**.
    HF's default preprocessor shortest-edge-resizes then centre-crops to 224x224, which
    silently discards the image border -- Methods 3 and 5 need the whole scene.

Output
  * name ``last_hidden_state`` (the only output; there is no ``pooler_output``).
  * shape ``[batch, floor(H/14)*floor(W/14) + 1 (+ n_register), 384]``; index 0 is the CLS
    token; ``D = 384`` for dinov2-small.

Three silent-bug guards this inferencer exists to enforce
--------------------------------------------------------
1. **Snap each side to a multiple of 14 and return the exact scale factors.** DINOv2 does *not*
   validate the multiple-of-14 requirement: the stride-14 patch conv silently floor-divides, so
   a 225 px side yields 16 patches (224 px of content) and the trailing pixel row/column vanish.
   The result is a **systematic spatial offset** in the similarity map, not an error. Snapping
   removes it; :class:`PreprocessInfo` carries the ``scale_x``/``scale_y`` needed to map a token
   coordinate back to an image pixel.

2. **Strip the CLS token and any register tokens with ``[1 + n_register:]``, ``n_register``
   derived from the token count -- never hardcoded to 1.** dinov2-small has ``n_register == 0``,
   but HF itself shipped a bug here, and a with-registers variant would silently shift the whole
   feature map by a few patches. :meth:`_derive_layout` computes
   ``n_register = tokens - 1 - floor(H/14)*floor(W/14)`` and raises :class:`ONNXContractError` if
   it goes negative; :meth:`_probe_layout` runs the model at **two swapped aspect ratios** at
   load and raises if the derived register count disagrees. That turns a silent spatial shift
   into a load-time error -- the same INFRA-09 discipline the base class applies to the input.

3. **Reshape the patch tokens height-first, ``(gh, gw, D)``.** A transposed similarity map is a
   plausible-looking bug -- it renders as a perfectly reasonable heatmap and is wrong. Row-major
   height-first flattening is the standard ViT ordering and is proven, not assumed, by the
   non-square off-centre fixture in ``tests/test_dinov2.py``.

Positional-embedding interpolation is fixed at export time (HF's ``align_corners=False`` differs
from FB's reference ``offset=0.1`` + antialiasing). It is not re-derived or "corrected" here; the
export in use is the fp32 ``onnx-community`` graph at the pinned revision recorded in
:data:`object_search.inference.models.MODEL_REGISTRY`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt

from object_search.inference.onnx_inferencer import (
    ONNXContractError,
    ONNXInferencer,
    ONNXInputSpec,
)

# The DINOv2 patch stride, embedding width, and the verified preprocessing constants. Module
# level so importing the module (and the model-free tests) covers the spec construction.
DINOV2_PATCH: int = 14
DINOV2_EMBED_DIM: int = 384
DINOV2_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
DINOV2_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

# The full input contract as a typed object -- snap-to-multiple(14), bicubic, RGB, no crop.
DINOV2_INPUT_SPEC: ONNXInputSpec = ONNXInputSpec(
    input_name="pixel_values",
    dtype="tensor(float)",
    shape=(1, 3, "height", "width"),
    layout="NCHW",
    color_order="RGB",
    scale=1.0 / 255.0,
    mean=DINOV2_MEAN,
    std=DINOV2_STD,
    resize="snap-to-multiple",
    size_multiple=DINOV2_PATCH,
    interpolation="bicubic",
)

# Two deliberately non-square, swapped probe sizes (multiples of 14) used at load to pin
# n_register. Swapping the aspect ratio makes a transpose or an off-by-one in the register
# arithmetic disagree between the two runs.
_PROBE_SIZES: tuple[tuple[int, int], ...] = ((28, 42), (42, 28))  # (height, width)


def _raw_last_hidden_state(
    outputs: list[npt.NDArray[np.generic]],
    orig_w: int,
    orig_h: int,
) -> npt.NDArray[np.float32]:
    """Post-processor for the base ``predict`` path: the raw ``last_hidden_state`` array.

    ``predict`` therefore returns the raw ``[1, tokens, D]`` tensor (CLS still attached). The
    spatial API callers actually want is :meth:`DINOv2Inferencer.dense_tokens`, which strips CLS
    and registers and reshapes to a grid.
    """
    return np.asarray(outputs[0], dtype=np.float32)


class DINOv2Inferencer(ONNXInferencer[npt.NDArray[np.float32]]):
    """Load DINOv2-small, validate its input *and* its token layout at construction, run it.

    The base :class:`ONNXInferencer` validates the input contract at load. This subclass adds a
    load-time validation of the **output token layout** (:meth:`_probe_layout`): it runs the
    model at two swapped non-square aspect ratios, derives ``n_register`` from each, and raises
    :class:`ONNXContractError` if they disagree or if the embedding width is not
    :data:`DINOV2_EMBED_DIM`.

    Args:
        model_path: Path to ``dinov2_small.onnx`` (fetched by ``pixi run fetch-models``).
        providers: ONNX Runtime execution providers; ``None`` = the runtime default.
        intra_op_num_threads: Latency / CPU-contention control only (see base module docstring).
        inter_op_num_threads: Latency / CPU-contention control only.
    """

    def __init__(
        self,
        model_path: Path | str,
        providers: list[str] | None = None,
        *,
        intra_op_num_threads: int | None = None,
        inter_op_num_threads: int | None = None,
    ) -> None:
        super().__init__(
            model_path,
            _raw_last_hidden_state,
            DINOV2_INPUT_SPEC,
            providers=providers,
            intra_op_num_threads=intra_op_num_threads,
            inter_op_num_threads=inter_op_num_threads,
        )
        # Pin the register count and embedding width at load, from real forward passes.
        self.n_register, self.embed_dim = self._probe_layout()

    # -- load-time token-layout validation ---------------------------------------------

    @staticmethod
    def _derive_layout(tokens: int, input_h: int, input_w: int) -> tuple[int, int, int]:
        """Return ``(gh, gw, n_register)`` for a token count at a snapped input size.

        ``gh = input_h // 14`` and ``gw = input_w // 14`` are fixed by the (already snapped)
        input, so ``n_register = tokens - 1 - gh*gw`` is determined. A negative register count
        means the token count is smaller than a single CLS plus the patch grid -- the model does
        not match its declared contract, so this raises :class:`ONNXContractError` rather than
        letting a downstream reshape fail cryptically or, worse, silently mis-slice.

        This is deliberately model-free (pure arithmetic on integers) so CI gates it without the
        gitignored weight present.
        """
        grid = (input_h // DINOV2_PATCH) * (input_w // DINOV2_PATCH)
        n_register = tokens - 1 - grid
        if n_register < 0:
            raise ONNXContractError(
                f"DINOv2 token-count mismatch: model returned {tokens} tokens at "
                f"{input_h}x{input_w} (grid {input_h // DINOV2_PATCH}x{input_w // DINOV2_PATCH} "
                f"= {grid} patches), which is fewer than 1 CLS + {grid} patch tokens. Derived "
                f"n_register={n_register} is negative -- the model violates its DINOv2 contract."
            )
        return input_h // DINOV2_PATCH, input_w // DINOV2_PATCH, n_register

    def _probe_layout(self) -> tuple[int, int]:
        """Run two swapped non-square probes; return ``(n_register, embed_dim)``, validated.

        Running both a tall and a wide probe means an off-by-one in the register arithmetic, or a
        transposed reading of the token sequence, disagrees between the two -- so the disagreement
        is caught at load rather than surfacing as a silent spatial shift on real images.
        """
        derived: list[int] = []
        embed_dims: list[int] = []
        for probe_h, probe_w in _PROBE_SIZES:
            blank = np.zeros((probe_h, probe_w, 3), dtype=np.uint8)
            info = self.preprocess(blank)
            outputs = self._session.run(None, {self._input_name: info.tensor})
            last_hidden = np.asarray(outputs[0])
            tokens = int(last_hidden.shape[1])
            embed_dims.append(int(last_hidden.shape[2]))
            _, _, n_register = self._derive_layout(tokens, info.input_h, info.input_w)
            derived.append(n_register)

        if len(set(derived)) != 1:
            raise ONNXContractError(
                f"DINOv2 register-count probe disagreed across aspect ratios: derived "
                f"{derived} at sizes {_PROBE_SIZES}. A stable model yields one register count; a "
                f"disagreement means the token sequence is being read wrong (e.g. transposed)."
            )
        if len(set(embed_dims)) != 1 or embed_dims[0] != DINOV2_EMBED_DIM:
            raise ONNXContractError(
                f"DINOv2 embedding-width mismatch: probes returned {embed_dims}, expected "
                f"{DINOV2_EMBED_DIM} for dinov2-small. Wrong model file loaded?"
            )
        return derived[0], embed_dims[0]

    # -- the spatial API ---------------------------------------------------------------

    def dense_tokens(
        self,
        image: npt.NDArray[np.uint8],
    ) -> tuple[npt.NDArray[np.float32], float, float]:
        """Return ``(grid, scale_x, scale_y)`` for one BGR scene.

        ``grid`` has shape ``(H//14, W//14, 384)`` where ``H``/``W`` are the *snapped* input
        sizes, ordered **height-first** (row-major, the standard ViT flattening). ``scale_x`` and
        ``scale_y`` are ``snapped / original`` per axis: a token ``(gy, gx)`` maps to an original
        pixel via ``orig_x = (gx * 14 + 7) / scale_x`` (patch centre), and likewise for ``y``.
        The CLS token and any register tokens are stripped with the derived ``[1 + n_register:]``
        slice before the reshape.
        """
        info = self.preprocess(image)
        outputs = self._session.run(None, {self._input_name: info.tensor})
        last_hidden = np.asarray(outputs[0], dtype=np.float32)  # (1, tokens, D)
        tokens = int(last_hidden.shape[1])
        embed_dim = int(last_hidden.shape[2])

        gh, gw, n_register = self._derive_layout(tokens, info.input_h, info.input_w)
        # A per-input consistency check against the load-time constants: any drift is a bug.
        if n_register != self.n_register or embed_dim != self.embed_dim:
            raise ONNXContractError(
                f"DINOv2 layout drift at run time: derived n_register={n_register}, "
                f"embed_dim={embed_dim}; load-time constants were n_register={self.n_register}, "
                f"embed_dim={self.embed_dim}."
            )

        # Strip CLS + registers, then reshape the gh*gw patch tokens height-first.
        patch = last_hidden[0, 1 + n_register :, :]  # (gh*gw, D)
        grid = patch.reshape(gh, gw, embed_dim)
        return grid, info.scale_x, info.scale_y
