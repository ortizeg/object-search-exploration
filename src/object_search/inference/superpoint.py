"""``SuperPointInferencer`` -- variable-count keypoints + descriptors from SuperPoint (ONNX).

Method 2 (``sparse-geo``, Phase 5) runs SuperPoint as its **learned** keypoint backend, behind
the same backend boundary as the classical SIFT/AKAZE/ORB detectors. The only two things the
rest of ``sparse_geo.py`` needs to know about it are that its keypoints carry **no geometric
frame** (no scale, no orientation) and that its descriptors are **already L2-normalized** -- both
of which flow directly from the verified contract below.

The verified contract (exact numbers, per project constraint -- runtime-verified in
``.planning/research/MODELS.md`` and gated by the ``Trial`` verdict in
``docs/library-reviews/superpoint.md``)
------------------------------------------------------------------------------------------------
Input
  * name ``image``; dtype ``float32`` (``tensor(float)``); layout **NCHW**; shape
    ``[1, 1, height, width]`` -- **batch fixed at 1**, a **single grayscale channel**, H/W
    dynamic.
  * colour: **BT.601 luma** ``gray = 0.299*R + 0.587*G + 0.114*B``, which
    ``cv2.cvtColor(bgr, COLOR_BGR2GRAY)`` reproduces (METHOD-11: the equivalence is written down,
    because the two paths differ in rounding and can shift a borderline keypoint).
  * range **[0, 1]** -- scale by ``1/255``. **NO mean subtraction, NO std division.** SuperPoint
    wants raw luma; applying ImageNet normalization here would silently wreck detection.
  * stride 8. Non-multiple sides are silently floored (trailing rows/columns dropped -- a
    coordinate-range truncation, not an error), so this inferencer **pads bottom/right with zeros
    to the next multiple of 8**. Padding on the far edges preserves the top-left origin, so
    keypoint coordinates need no remapping.

Output (all three read from the graph and confirmed at runtime; they share one symbolic ``N``)
  * ``keypoints`` **int64** ``[1, N, 2]`` -- ``(x, y)`` in **input-image pixels**, integer. No
    sub-pixel refinement, and crucially **no scale or orientation**: a SuperPoint keypoint carries
    no frame, which is exactly what makes single-correspondence 4-DoF voting invalid for this
    backend (``single-4dof`` raises; ``translation-2dof`` is the SuperPoint default).
  * ``scores`` **float32** ``[1, N]`` -- detector confidence, floor ``0.0005`` (baked in).
  * ``descriptors`` **float32** ``[1, N, 256]`` -- **already L2-normalized** (measured
    ``||d|| = 1.0000``). **Do not re-normalize.** kNN is therefore a plain matmul: cosine
    ``= D_crop @ D_scene.T``, squared-L2 ``= 2 - 2*cos``.

Effective border is **8 px**, not the configured ``remove_borders=4`` (the border mask is applied
on the 8x-upsampled score grid). Method 2 never gets a correspondence within 8 px of the scene
edge, which matters when an instance is clipped by the frame.

Licence (carried from the library review): the **weights are MagicLeap non-commercial
research-only** and the DERIVATIVES clause covers this ONNX file -- never redistribute it. The
file is gitignored (INFRA-11) and arrives only via ``pixi run fetch-models``.

Why this inferencer overrides ``preprocess``
--------------------------------------------
The base :class:`ONNXInferencer.preprocess` is the general RGB, 3-channel, mean/std path. This
model is single-channel grayscale with no mean/std, so it overrides ``preprocess`` with the
SuperPoint-specific luma-and-pad steps above -- exactly the override the base docstring
anticipates. Only the **input** is validated at load (INFRA-09): ``N`` is genuinely
data-dependent, so no output-shape assertion is possible or wanted.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
import numpy.typing as npt

from object_search.inference.onnx_inferencer import (
    ONNXInferencer,
    ONNXInputSpec,
    PreprocessInfo,
)

# The verified SuperPoint constants (all baked into the v1.0.0 graph; documented, not settable).
SUPERPOINT_DESCRIPTOR_DIM: int = 256
SUPERPOINT_STRIDE: int = 8  # snap sides to a multiple of this (pad, do not resize)
SUPERPOINT_BORDER_PX: int = 8  # effective border exclusion, NOT the configured remove_borders=4
SUPERPOINT_DETECTION_THRESHOLD: float = 0.0005  # baked-in score floor
SUPERPOINT_NMS_RADIUS: int = 4  # baked-in NMS radius

# The full input contract as a typed object -- grayscale [1,1,H,W], scale 1/255, NO mean/std.
# ``mean``/``std`` are required by :class:`ONNXInputSpec` but are **inert here**: ``preprocess``
# is overridden and never consults them. They are set to the identity (0 / 1) so that even if a
# future caller routed through the base path, it would apply no normalization -- matching the
# "no mean, no std" contract rather than silently introducing one.
SUPERPOINT_INPUT_SPEC: ONNXInputSpec = ONNXInputSpec(
    input_name="image",
    dtype="tensor(float)",
    shape=(1, 1, "height", "width"),
    layout="NCHW",
    color_order="BGR",  # inert: preprocess is overridden and computes BT.601 luma itself
    scale=1.0 / 255.0,
    mean=(0.0, 0.0, 0.0),  # inert -- see note above
    std=(1.0, 1.0, 1.0),  # inert -- see note above
    resize="stretch",  # inert: preprocess pads to a multiple of 8 rather than resizing
    interpolation="bilinear",
)


class SuperPointResult(NamedTuple):
    """One SuperPoint forward pass, decoded into aligned arrays (all share the same ``N``).

    ``keypoints`` are integer ``(x, y)`` in input-image pixels; ``descriptors`` are **already
    L2-normalized**. There is deliberately no ``scale`` or ``angle`` field -- SuperPoint keypoints
    are frameless, and representing that absence structurally is what lets Method 2's voting layer
    reject ``single-4dof`` for this backend.
    """

    keypoints: npt.NDArray[np.int64]  # (N, 2) -- x, y in input pixels
    scores: npt.NDArray[np.float32]  # (N,)
    descriptors: npt.NDArray[np.float32]  # (N, 256) -- already L2-normalized


# -- pure pre/post-processing (model-free, so CI gates them without the gitignored weight) ------
#
# The SuperPoint preprocessing (BT.601 luma + pad-to-stride) and output decoding are the two most
# fragile pieces of this inferencer and the exact contract the project constraints require pinned
# with exact numbers. They are pulled out as free functions -- mirroring ``owlv2_preprocess_tensor``
# -- so they are testable in CI with no weight; the instance methods below simply delegate.


def superpoint_preprocess(
    image: npt.NDArray[np.uint8], *, scale: float = SUPERPOINT_INPUT_SPEC.scale
) -> PreprocessInfo:
    """BGR-or-grayscale -> BT.601 luma, scaled to [0, 1], padded to a multiple of 8.

    Accepts either a BGR ``(H, W, 3)`` scene/crop or an already-grayscale ``(H, W)`` array
    (Method 2 grayscales the scene once and hands the single-channel crop and scene here, so
    both shapes occur). The luma weighting is BT.601 either way -- ``cv2.COLOR_BGR2GRAY`` for a
    colour input, or the caller's already-BT.601 gray untouched.

    Padding is on the **bottom/right only**, so the top-left origin is preserved and keypoint
    coordinates come back directly in original-image pixels with **no** remapping -- hence the
    returned ``scale_x``/``scale_y`` are ``1.0`` and ``pad_x``/``pad_y`` are ``0``. ``scale`` is
    the input contract's ``1/255`` (passed in so the value has a single source of truth).
    """
    luma = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    orig_h, orig_w = int(luma.shape[0]), int(luma.shape[1])

    # Snap each side up to a multiple of the stride (pad, never resize): non-multiple sides are
    # silently floored by the graph, dropping trailing rows/columns -- a coordinate truncation.
    pad_h = (-orig_h) % SUPERPOINT_STRIDE
    pad_w = (-orig_w) % SUPERPOINT_STRIDE
    if pad_h or pad_w:
        luma = cv2.copyMakeBorder(luma, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)

    scaled = np.asarray(luma, dtype=np.float32) * np.float32(scale)
    tensor = np.ascontiguousarray(scaled[np.newaxis, np.newaxis, :, :], dtype=np.float32)
    return PreprocessInfo(
        tensor=tensor,
        scale_x=1.0,
        scale_y=1.0,
        pad_x=0,
        pad_y=0,
        input_w=orig_w + pad_w,
        input_h=orig_h + pad_h,
    )


def superpoint_decode(
    outputs: list[npt.NDArray[np.generic]],
    output_names: tuple[str, ...],
) -> SuperPointResult:
    """Decode the three outputs into a :class:`SuperPointResult`, mapped by output name.

    The v1.0.0 graph names its outputs ``keypoints`` / ``scores`` / ``descriptors``; they are
    looked up by name (falling back to positional order if a re-export ever renames them) so a
    graph-order change cannot silently swap them. The batch dimension (fixed at 1) is dropped.
    """
    named = dict(zip(output_names, outputs, strict=True))
    try:
        raw_kpts = named["keypoints"]
        raw_scores = named["scores"]
        raw_desc = named["descriptors"]
    except KeyError:  # a re-export renamed the outputs -- fall back to the documented order
        raw_kpts, raw_scores, raw_desc = outputs[0], outputs[1], outputs[2]

    keypoints = np.asarray(raw_kpts, dtype=np.int64).reshape(-1, 2)
    scores = np.asarray(raw_scores, dtype=np.float32).reshape(-1)
    descriptors = np.asarray(raw_desc, dtype=np.float32).reshape(-1, SUPERPOINT_DESCRIPTOR_DIM)
    return SuperPointResult(keypoints, scores, descriptors)


class SuperPointInferencer(ONNXInferencer[SuperPointResult]):
    """Load SuperPoint (ONNX), validate its grayscale input contract at load, run it.

    The base :class:`ONNXInferencer` validates the input dtype/shape at construction (INFRA-09).
    This subclass overrides :meth:`preprocess` with SuperPoint's single-channel luma-and-pad
    contract and decodes the three named outputs into a :class:`SuperPointResult`. No output shape
    is asserted, because ``N`` (the keypoint count) is genuinely data-dependent -- which is exactly
    the property Method 2's low-keypoint guard (METHOD-04c) relies on.

    Args:
        model_path: Path to ``superpoint.onnx`` (fetched by ``pixi run fetch-models``; gitignored,
            MagicLeap non-commercial weights). Missing -> ``FileNotFoundError``.
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
            self._decode,
            SUPERPOINT_INPUT_SPEC,
            providers=providers,
            intra_op_num_threads=intra_op_num_threads,
            inter_op_num_threads=inter_op_num_threads,
        )

    # -- pre-processing (the SuperPoint-specific override) -----------------------------

    def preprocess(self, image: npt.NDArray[np.uint8]) -> PreprocessInfo:
        """SuperPoint's grayscale luma-and-pad contract; delegates to :func:`superpoint_preprocess`.

        Kept as an override (the base is the general RGB mean/std path) so the model's single-
        channel, no-normalization contract is honoured; the pure logic lives in the free function
        so it is testable without the gitignored weight.
        """
        return superpoint_preprocess(image, scale=self.input_spec.scale)

    # -- output decoding ---------------------------------------------------------------

    def _decode(
        self,
        outputs: list[npt.NDArray[np.generic]],
        orig_w: int,
        orig_h: int,
    ) -> SuperPointResult:
        """Decode the three outputs; delegates to :func:`superpoint_decode`.

        ``orig_w`` / ``orig_h`` are unused: preprocess pads on the far edges only, so keypoint
        coordinates are already in original pixels. The name-mapped decode logic lives in the free
        function so it is testable without the gitignored weight.
        """
        return superpoint_decode(outputs, self.output_names)

    # -- the detection API -------------------------------------------------------------

    def detect(self, image: npt.NDArray[np.uint8]) -> SuperPointResult:
        """Run SuperPoint on one BGR-or-grayscale image; return frameless keypoints + descriptors.

        A thin, named alias for :meth:`predict` -- ``detect`` reads better at the Method 2 call
        site, where the intent is "detect keypoints", not "predict a class".
        """
        return self.predict(image)
