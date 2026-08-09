"""``OWLv2Inferencer`` -- per-patch class embeddings + boxes from OWLv2 (ONNX, image-guided).

Method 4 (``owlv2-oneshot``) uses OWLv2 in its **image-conditioned (one-shot) detection** mode:
the exemplar crop is encoded as a *query image*, one query embedding is selected from it, and
every patch of the scene is scored by cosine similarity to that query embedding. Both encodes go
through the **same single-input vision graph** -- there is no second "query" input to the ONNX
model; the method simply runs :meth:`embed_image` twice (once on the exemplar crop, once on the
scene). That is why this inferencer is a plain single-input :class:`ONNXInferencer` even though the
detection mode is "two image" -- the two-image logic lives, readably, in the method module.

The runtime package imports **nothing** from ``torch`` or ``transformers``; it loads the exported
``.onnx`` under ONNX Runtime. The exporter (``transformers`` + ``torch``, both Apache-2.0) lives
only in the ``export`` pixi env (see ``docs/library-reviews/owlv2.md`` and
``scripts/export_owlv2.py``).

The contract (exact numbers -- to be RUNTIME-VERIFIED at export, then pinned here)
----------------------------------------------------------------------------------
Unlike the other three learned models, OWLv2's numbers below are the **documented HuggingFace
processor / architecture constants**, not yet runtime-verified in ``.planning/research/MODELS.md``
(no weights are fetched in the authoring environment). ``scripts/export_owlv2.py`` asserts the
graph I/O contract at export, and the ``sha256`` in the registry is pinned from the first verified
fetch -- the same discipline the other models already carry. Any number that turns out wrong at
export is fixed here and in :data:`OWLV2_INPUT_SPEC`, not worked around downstream.

Input
  * name ``pixel_values``; dtype ``float32`` (``tensor(float)``); layout **NCHW**; shape
    ``[batch, 3, 960, 960]`` -- the spatial dims are **static 960** (OWLv2's learned position
    embeddings fix the input resolution; a different size would silently mis-index them).
  * colour order **RGB** (scenes arrive BGR from OpenCV and are converted).
  * **preprocessing is OWLv2's own policy, NOT the base ``stretch``/``letterbox`` path**, so
    :meth:`preprocess` is overridden here (as the base docstring anticipates for non-standard
    backends). In order: (1) rescale ``*1/255``; (2) **pad bottom-right to a square** of side
    ``max(H, W)`` with the grey value :data:`OWLV2_PAD_VALUE` (``0.5`` in the rescaled ``[0, 1]``
    space) -- OWLv2 pads *bottom-right*, not centred like a letterbox, so the content origin stays
    top-left and normalized boxes need no pad offset; (3) resize the square to ``960x960``
    **bilinear**; (4) normalize with the CLIP mean/std :data:`OWLV2_MEAN` / :data:`OWLV2_STD`.

Output (defined by the export wrapper in ``scripts/export_owlv2.py`` -- names are ours)
  * ``class_embeds`` f32 ``[batch, num_patches, 512]`` -- the projected per-patch class-head
    embeddings (``512`` = OWLv2 projection dim), **not yet L2-normalized**; the method normalizes.
  * ``pred_boxes`` f32 ``[batch, num_patches, 4]`` -- per-patch boxes as ``(cx, cy, w, h)``
    **normalized to ``[0, 1]``** over the padded square. At ``960`` with patch ``16``,
    ``num_patches == 60 * 60 == 3600``.
  * ``logit_shift`` / ``logit_scale`` f32 ``[batch, num_patches, 1]`` -- OWLv2's own learned,
    **query-independent** per-patch score-calibration terms (the ``Owlv2ClassPredictionHead``
    ``logit_shift``/``logit_scale`` Linear(1) layers applied to the pre-projection 768-dim vision
    features -- the same tensor HF's own class head feeds them when a query IS present, so this is
    not a repurposing). The method applies ``(cosine + logit_shift) * logit_scale`` to recalibrate
    raw cosine before thresholding -- HF's own formula, computed from the SCENE alone, never the
    query crop.

Only the **input** is validated at load (INFRA-09): ``num_patches`` is a fixed property of the
960/16 grid, but the four outputs' patch dim exports as symbolic, so no output-shape assertion is
made here. Mapping a normalized box back to scene pixels (multiply by ``max(H, W)``, clip) and
selecting the query embedding are pure arithmetic and live in ``search/owlv2_oneshot.py`` so CI
gates them with synthetic tensors -- no gitignored weight required.

Licence: OWLv2 is **Apache-2.0** (Google), the same permissive tier as DINOv2 -- no AGPL/§13 or
non-commercial constraint. The weights are gitignored (INFRA-11) and arrive only via
``pixi run -e export export-owlv2``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt

from object_search.inference.onnx_inferencer import (
    ONNXInferencer,
    ONNXInputSpec,
    PreprocessInfo,
)

# The verified OWLv2 operating point and constants (see the module docstring / library review).
OWLV2_IMAGE_SIZE: int = 960  # static square input; OWLv2 position embeddings fix this
OWLV2_PATCH: int = 16  # patch stride -> 60x60 = 3600 patches at 960
OWLV2_EMBED_DIM: int = 512  # class-head projection dim for owlv2-base-patch16
OWLV2_PAD_VALUE: float = 0.5  # grey pad in the rescaled [0, 1] space (OWLv2 pads bottom-right)
# CLIP normalization constants, which OWLv2 inherits verbatim.
OWLV2_MEAN: tuple[float, float, float] = (0.48145466, 0.4578275, 0.40821073)
OWLV2_STD: tuple[float, float, float] = (0.26862954, 0.26130258, 0.27577711)

# The full input contract as a typed object. ``resize="stretch"`` and the mean/std below are
# recorded for the load-time INPUT validation and for provenance; the ACTUAL preprocessing is the
# OWLv2 pad-bottom-right-then-resize policy implemented in ``preprocess`` below, NOT the base
# stretch path (the base ``preprocess`` is overridden). Stated so the spec cannot silently drift
# from the override.
OWLV2_INPUT_SPEC: ONNXInputSpec = ONNXInputSpec(
    input_name="pixel_values",
    dtype="tensor(float)",
    shape=(1, 3, OWLV2_IMAGE_SIZE, OWLV2_IMAGE_SIZE),
    layout="NCHW",
    color_order="RGB",
    scale=1.0 / 255.0,
    mean=OWLV2_MEAN,
    std=OWLV2_STD,
    resize="stretch",  # nominal; preprocess() overrides with pad-bottom-right-to-square + resize
    interpolation="bilinear",
)


@dataclass(frozen=True, eq=False)
class Owlv2Embeddings:
    """One image's per-patch OWLv2 outputs: class embeddings, boxes, and score calibration.

    ``eq=False`` because the fields are NumPy arrays (element-wise ``==`` has no single truth
    value); these are never compared for equality.

    Attributes:
        class_embeds: ``(num_patches, 512)`` projected class-head embeddings, **not** yet
            L2-normalized -- the method normalizes once so the choice is visible in one file.
        boxes_cxcywh: ``(num_patches, 4)`` predicted boxes as ``(cx, cy, w, h)`` normalized to
            ``[0, 1]`` over the padded square. Map to scene pixels by multiplying by
            ``max(orig_h, orig_w)`` (the padded-square side), then clipping.
        logit_shift: ``(num_patches,)`` OWLv2's learned, query-independent additive calibration
            term per patch (already through the export wrapper's Linear(1); see the module
            docstring). Computed from this image's own vision features only.
        logit_scale: ``(num_patches,)`` OWLv2's learned, query-independent multiplicative
            calibration term per patch (already through Linear(1) + ELU + 1 in the export wrapper,
            so it is always ``> 0``). Computed from this image's own vision features only.
    """

    class_embeds: npt.NDArray[np.float32]
    boxes_cxcywh: npt.NDArray[np.float32]
    logit_shift: npt.NDArray[np.float32]
    logit_scale: npt.NDArray[np.float32]


def owlv2_preprocess_tensor(image: npt.NDArray[np.uint8]) -> tuple[npt.NDArray[np.float32], int]:
    """Execute OWLv2's exact preprocessing on one BGR image; return ``(tensor, square_side)``.

    Pure arithmetic (no ONNX session, no weight), so CI gates it. Steps, in OWLv2's order:
    BGR->RGB; rescale ``*1/255``; **pad bottom-right** to a square of side ``max(H, W)`` with the
    grey :data:`OWLV2_PAD_VALUE`; resize the square to ``960x960`` bilinear; normalize with the
    CLIP mean/std. Returns the ``[1, 3, 960, 960]`` float32 tensor and ``square_side == max(H, W)``
    (the factor that maps a normalized box back to scene pixels).
    """
    orig_h, orig_w = int(image.shape[0]), int(image.shape[1])
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.ndim == 3 else image

    # 1. rescale to [0, 1].
    scaled = np.asarray(rgb, dtype=np.float32) * np.float32(1.0 / 255.0)

    # 2. pad bottom-right to a square with grey OWLV2_PAD_VALUE (content origin stays top-left).
    side = max(orig_h, orig_w)
    square = np.full((side, side, 3), OWLV2_PAD_VALUE, dtype=np.float32)
    square[:orig_h, :orig_w] = scaled

    # 3. resize the square to the fixed 960x960 input, bilinear.
    resized = cv2.resize(
        square, (OWLV2_IMAGE_SIZE, OWLV2_IMAGE_SIZE), interpolation=cv2.INTER_LINEAR
    ).astype(np.float32)

    # 4. normalize with the CLIP mean/std.
    mean = np.asarray(OWLV2_MEAN, dtype=np.float32)
    std = np.asarray(OWLV2_STD, dtype=np.float32)
    normalized = (resized - mean) / std

    # NCHW + batch dim, contiguous float32.
    batched = np.ascontiguousarray(
        np.expand_dims(np.transpose(normalized, (2, 0, 1)), axis=0), dtype=np.float32
    )
    return batched, side


class OWLv2Inferencer(ONNXInferencer[Owlv2Embeddings]):
    """Load OWLv2 (ONNX), validate its input contract at load, encode one image to patch outputs.

    The base :class:`ONNXInferencer` validates the input dtype/shape at construction (INFRA-09).
    This subclass overrides :meth:`preprocess` with OWLv2's pad-bottom-right-then-resize policy and
    returns per-patch class embeddings, normalized boxes, and score-calibration terms. No output
    shape is asserted: the four outputs' patch dim exports as symbolic.

    Args:
        model_path: Path to ``owlv2_base_patch16.onnx`` (produced by
            ``pixi run -e export export-owlv2``; gitignored, Apache-2.0). Missing ->
            ``FileNotFoundError``.
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
            self._pack_embeddings,
            OWLV2_INPUT_SPEC,
            providers=providers,
            intra_op_num_threads=intra_op_num_threads,
            inter_op_num_threads=inter_op_num_threads,
        )

    def preprocess(self, image: npt.NDArray[np.uint8]) -> PreprocessInfo:
        """OWLv2's own preprocessing (overrides the base stretch/letterbox path).

        Delegates to :func:`owlv2_preprocess_tensor`. The returned ``scale_x``/``scale_y`` are the
        uniform ``960 / max(H, W)`` factor and the pad offsets are ``0`` (OWLv2 pads bottom-right,
        so a token's normalized coordinate already maps to the top-left content); the method maps
        boxes from the normalized ``pred_boxes`` directly, so these factors are informational.
        """
        tensor, side = owlv2_preprocess_tensor(image)
        scale = OWLV2_IMAGE_SIZE / side
        return PreprocessInfo(
            tensor=tensor,
            scale_x=scale,
            scale_y=scale,
            pad_x=0,
            pad_y=0,
            input_w=OWLV2_IMAGE_SIZE,
            input_h=OWLV2_IMAGE_SIZE,
        )

    def _named_outputs(
        self, outputs: list[npt.NDArray[np.generic]]
    ) -> tuple[
        npt.NDArray[np.float32],
        npt.NDArray[np.float32],
        npt.NDArray[np.float32],
        npt.NDArray[np.float32],
    ]:
        """Map ``outputs`` to ``(class_embeds, pred_boxes, logit_shift, logit_scale)``.

        Falls back to the documented export order if a re-export renamed the outputs.
        """
        named = dict(zip(self.output_names, outputs, strict=True))
        try:
            class_embeds = named["class_embeds"]
            pred_boxes = named["pred_boxes"]
            logit_shift = named["logit_shift"]
            logit_scale = named["logit_scale"]
        except KeyError:  # a re-export renamed the outputs -- fall back to the documented order
            class_embeds, pred_boxes = outputs[0], outputs[1]
            logit_shift, logit_scale = outputs[2], outputs[3]
        return (
            np.asarray(class_embeds, dtype=np.float32),
            np.asarray(pred_boxes, dtype=np.float32),
            np.asarray(logit_shift, dtype=np.float32),
            np.asarray(logit_scale, dtype=np.float32),
        )

    def _pack_embeddings(
        self,
        outputs: list[npt.NDArray[np.generic]],
        orig_w: int,
        orig_h: int,
    ) -> Owlv2Embeddings:
        """Post-processor for the base ``predict`` path: drop the batch dim and package outputs.

        ``orig_w``/``orig_h`` are unused because ``pred_boxes`` are normalized (resolution-free);
        the method maps them to scene pixels with the image it already holds. ``logit_shift``/
        ``logit_scale`` arrive as ``(num_patches, 1)``; squeezed to ``(num_patches,)`` here so the
        method scores against a plain 1-D array, matching ``class_embeds``' patch-major layout.
        """
        class_embeds, pred_boxes, logit_shift, logit_scale = self._named_outputs(outputs)
        return Owlv2Embeddings(
            class_embeds=np.ascontiguousarray(class_embeds[0], dtype=np.float32),
            boxes_cxcywh=np.ascontiguousarray(pred_boxes[0], dtype=np.float32),
            logit_shift=np.ascontiguousarray(logit_shift[0, :, 0], dtype=np.float32),
            logit_scale=np.ascontiguousarray(logit_scale[0, :, 0], dtype=np.float32),
        )

    def embed_image(self, image: npt.NDArray[np.uint8]) -> Owlv2Embeddings:
        """Encode one BGR image to per-patch ``(class_embeds, boxes_cxcywh, shift, scale)``.

        The single-image unit used twice by ``owlv2-oneshot`` (query crop, then scene). It knows
        nothing about exemplars or scoring -- the two-image image-guided logic lives in the method.
        """
        orig_h, orig_w = int(image.shape[0]), int(image.shape[1])
        info = self.preprocess(image)
        outputs = self._session.run(None, {self._input_name: info.tensor})
        return self._pack_embeddings([np.asarray(o) for o in outputs], orig_w, orig_h)
