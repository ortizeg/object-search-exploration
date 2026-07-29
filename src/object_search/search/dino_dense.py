"""Method 3 -- DINOv2 dense-token best-part matching (the general-purpose default).

What it does
------------
Embed the exemplar crop and the whole scene into DINOv2 dense patch tokens. Score every scene
token by the **mean of its top-``match_tokens`` cosine similarities to the crop's own tokens**
(``max-token`` scoring, the default) -- "how well does the single best-matching part of the
exemplar explain this location". Threshold the resulting high-contrast similarity map at a
calibrated cut, run connected components **at that threshold**, keep each component whose area is
consistent with the exemplar, and turn it into a box. Where NCC (Method 1) correlates raw
intensities and so misses instances under pose or lighting change, DINOv2 features are
appearance-robust, so this is the method that finds "the same object, differently posed".

A ``prototype`` scoring mode (mean-pool the crop tokens into ONE vector, then dot) is retained as
a readable baseline, but it produces a **low-contrast map on richly-textured objects** where every
instance fuses into a single image-spanning blob -- the reason ``max-token`` is the default.

This file is meant to be read top to bottom by an ML practitioner. Readability outranks DRY
(project convention): the numbered steps ``# 1.`` .. ``# 9.`` in :func:`search` match the
headings in ``docs/methods/dino-dense.md`` one-for-one, and a step may inline a few lines
rather than reach for a shared helper if that reads better standalone.

Pre-processing (exact)
----------------------
The ONNX numbers -- ``pixel_values`` f32 NCHW, RGB, scale ``1/255``, mean
``[0.485,0.456,0.406]``, std ``[0.229,0.224,0.225]``, **bicubic** resize, **snap each side to a
multiple of 14**, and **NO centre-crop** -- live in :class:`DINOv2Inferencer`'s docstring and
are not duplicated here; this module reuses that ONE inferencer rather than re-deriving the
contract. Two method-level preprocessing choices *are* stated here because they are this
method's, not the backbone's:

- **The scene input resolution is capped** at ``scene_max_side`` (default 1568). DINOv2's
  stride-14 patches are coarse, so a higher input resolution buys a finer similarity map -- but
  an uncapped 6000 px scene is 180k+ tokens and OOMs. When a scene's long side exceeds the cap
  it is downscaled *and the cap is logged*, never silently truncated. Scenes at or below the cap
  run at native resolution.
- **The exemplar crop is taken from the original BGR scene** and embedded on its own. Under the
  default ``max-token`` scoring its patch tokens are kept as a **bank** (one vector per part);
  under ``prototype`` scoring they are mean-pooled into one vector. All tokens are L2-normalized.
- **Fixed-size letterbox (opt-in, ``fixed_input_side``).** ``None`` (default) is the native/cap
  path above -- byte-identical on the chipset/synthetic sets. When set to a multiple of 14, the
  (capped) scene is letterboxed into ONE fixed ``fixed_input_side`` x ``fixed_input_side`` square
  by a **single uniform aspect-preserving scale** ``side / max(h, w)`` with the content placed
  top-left and the bottom/right strip filled with a **constant pad**. Because every plan then hits
  the same input resolution, onnxruntime allocates its CUDA arena once instead of per-scene -- the
  GPU-OOM fix for varied-resolution plans. Two consequences are handled explicitly: **padding
  tokens are masked** (a token whose patch centre lands in the padded region is dropped from the
  threshold calibration, and the padded pixels of the similarity map are set below any threshold,
  so connected components fire only in real content); and **boxes map back through the single
  letterbox scale** (top-left placement => zero pad offset => one division by the combined
  cap x letterbox scale, no offset term). The side is a multiple of 14, so
  :class:`DINOv2Inferencer`'s snap-to-multiple(14) resize is a no-op on the letterboxed input and
  its scale factors come back as 1.0. Determinism is preserved (fixed interpolation + constant pad).

Post-processing (exact)
-----------------------
- **All tokens are L2-normalized before any dot product**, so every score is a genuine cosine in
  ``[-1, 1]`` rather than an unnormalized dot dominated by token *magnitude* (DINOv2's high-norm
  background artifact tokens in particular). Order matters: normalize, then dot.
- **``max-token`` scoring (default):** each scene token's score is the **mean of its top-k cosines
  to the crop token bank**, where ``k = match_tokens``. This best-matching-part score is high on
  true instances and low on background -- a far sharper contrast than the mean-pooled ``prototype``
  dot, which averages the crop's diverse parts into a mushy vector that matches everything weakly.
- **The similarity MAP is bilinearly upsampled, not the tokens.** Upsampling the ``(gh, gw)``
  cosine map to pixel resolution -- using the scale factors :meth:`DINOv2Inferencer.dense_tokens`
  returns so token centres land on their true pixels -- is cheap and correct; upsampling the
  384-d tokens first would be 384x the work for the same map.
- **Threshold via ``common.calibration`` OR the local ``contrast`` strategy (default).** Absolute
  cosine thresholds do not transfer across images, so the cut is calibrated per image from the
  score distribution. ``contrast`` blends a background anchor (``mean + std``) with a foreground
  anchor (a fraction of the high percentile); on the high-contrast ``max-token`` map this tracks
  the per-image optimum, where the old ``gmm`` posterior cut sat in the background shoulder and
  under-thresholded (every instance fused into one image-spanning box).
- **MATCH components are grown at the accept threshold ITSELF, never at a sub-threshold floor.** A
  component's box is therefore the extent of the above-threshold region only; a low-contrast
  shoulder below the threshold cannot bridge distinct instances into one blob. ``label 0`` (the
  background) is skipped explicitly -- emitting it would be one image-sized false positive.
- **Component area is bounded to the exemplar's size.** A component below ``min_area_frac`` x the
  exemplar area is a fragment; one above ``max_area_frac`` x it is a merged/background blob. Both
  are dropped so neither speckle nor a swallowing blob is emitted. The bounds scale with the
  exemplar, so they hold across the 300x canvas range without per-image tuning.
- **Sub-threshold candidates are retained** (EVAL-08): components found at a floor
  ``candidate_margin`` below the accept threshold are logged (candidate LOG only, never emitted as
  matches) so an offline threshold sweep can rebuild a PR curve. There is **no single-best
  short-circuit** (METHOD-12) -- every above-threshold component survives as a match.

Known failure modes
--------------------
- **Stride-14 coarseness.** Even with high-res inference and upsampling, boxes are quantised to
  ~14 px patches; a small object a few pixels across is a single fuzzy token. This is the
  headline limitation and the reason for the sliding-window / FeatUp backlog items.
- **Very large scenes hit the resolution cap.** Above ``scene_max_side`` the scene is downscaled
  (and it is logged), so effective localisation on a 6000 px image is coarser than the cap
  suggests.
- **Small objects below the token grid.** On the fixed-scale chipset (chips ~36-46 px) a whole
  instance spans ~2-3 stride-14 tokens, so its box is imprecise or merges with a neighbour and
  often misses the IoU-0.5 bar. This is the same coarseness limit as above and is why NCC, not
  this method, owns the flat-chip regime; ``max-token`` scoring does not change the grid pitch.
- **Weights absent.** DINOv2 weights are gitignored and fetched by ``pixi run fetch-models``.
  With no weight present the method returns ``outcome=error`` with a ``model_unavailable`` note
  rather than raising, so the sample renderer and the API degrade honestly.

ROBUSTNESS BACKLOG
------------------
Deferred deliberately (mirrored in ``docs/methods/dino-dense.md`` and
``docs/ROBUSTNESS-BACKLOG.md``); none is built in this phase:

- **Sliding-window backbone inference** for very large scenes, so localisation no longer
  degrades at the resolution cap.
- **Adaptive input resolution** -- size the scene so the exemplar spans >= N stride-14 tokens
  (clamped to a hard max) instead of a fixed ``scene_max_side``. Measured ~6x chipset recall
  (0.077 -> 0.554 on a small-chip subset); deferred because it costs latency, does not fix the
  flat-chip precision, and chipset is NCC's regime (see docs/reports/dino-dense-improvement.md).
- **Learned feature upsampling (FeatUp)** to recover sub-patch localisation from the stride-14
  grid without a full high-res forward pass.
- **SAM-based box refinement** -- snap each coarse component box to the nearest segment mask.
- **Spatially-structured (not order-free) part matching.** ``max-token`` scoring already does
  many-to-many token similarity (DONE -- it replaced the mean-pooled prototype and lifted textured
  F1 from ~0.03 to ~0.70), but it pools the top-k cosines with no geometric constraint on WHERE the
  matching parts sit. Adding a spatial-consistency term (parts must be arranged like the exemplar)
  would cut clutter false positives further.
- **DINOv3 backbone swap** once a clean ONNX export exists.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Literal

import cv2
import numpy as np
import numpy.typing as npt
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, field_validator

from object_search.inference import DINOv2Inferencer, models, resolve_providers
from object_search.inference.dinov2 import DINOV2_PATCH
from object_search.schemas import (
    BBox,
    Candidate,
    Diagnostics,
    ExemplarBox,
    HeatmapPayload,
    LatencyBreakdown,
    Match,
    MethodError,
    SearchOutcome,
    SearchResult,
)
from object_search.search.common import calibration, viz
from object_search.search.registry import register_method

# -- Method-level constants (properties of the METHOD, not of a query, so not config fields) --

_METHOD_VERSION = "1.0.0"
_MODEL_KEY = "dinov2-small"  # the MODEL_REGISTRY key this method reuses (shared with Method 5)
# Pin the CPU provider so a run is bit-identical machine to machine (dev boxes also expose
# CoreML, whose kernels differ). Reproducibility is a hard project constraint.
_PROVIDERS = resolve_providers()
# The exemplar's own prototype is L2-normalized, so its cosine similarity with itself is exactly
# 1.0. That is the self-match score the "self-similarity" calibrator anchors on.
_SELF_MATCH_SCORE = 1.0
# A component whose box overlaps the exemplar by at least this is the exemplar's own region,
# labelled is_exemplar rather than dropped or counted as a fresh discovery (METHOD-04c).
_EXEMPLAR_IOU = 0.5
_EPS = 1e-12  # guards a zero-norm division; a genuinely zero token is background, not a match
# --- "contrast" calibration constants (properties of the max-token map, not per-query knobs) ---
# The accept threshold is a 50/50 blend of two anchors on the similarity-map distribution:
#   * a BACKGROUND anchor  = mean + _CONTRAST_STD_K * std   (where the background bulk ends), and
#   * a FOREGROUND anchor   = _CONTRAST_PEAK_FRAC * p{_CONTRAST_PEAK_PCTL}  (a fraction of the
#     near-peak, i.e. relative to the strongest matches rather than an absolute cosine).
# Blending the two tracks the per-image optimum far better than a single Gaussian-mixture cut,
# which sits in the background shoulder on this high-contrast map (see docs/methods/dino-dense.md).
# Chosen on the EVAL-20 textured set: pooled F1 ~0.70 vs ~0.44 for gmm, on a broad plateau (not a
# knife-edge fit). These are NOT fit to the ground-truth boxes -- only to the score distribution.
_CONTRAST_STD_K = 1.0
_CONTRAST_PEAK_PCTL = 99.5
_CONTRAST_PEAK_FRAC = 0.85
_CONTRAST_BLEND = 0.5  # weight on the background anchor; (1 - this) on the foreground anchor


class DinoDenseConfig(BaseModel):
    """Frozen config for :func:`search`; its JSON Schema drives the UI form (one source).

    Every field carries a ``description`` because that text becomes the form's help string --
    this is the single place it is written, so it must be written here.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    scene_max_side: int = Field(
        default=1568,
        ge=DINOV2_PATCH,
        description=(
            "Cap on the scene's longest side (pixels) before DINOv2 inference. Higher = a finer "
            "similarity map but more tokens and memory; a scene above this is downscaled and the "
            "cap is logged. 1568 = 112 patches, a good high-res/'safe memory' balance."
        ),
    )
    calibration: Literal["contrast", "fixed", "self-similarity", "ratio", "gmm"] = Field(
        default="contrast",
        description=(
            "How the accept threshold is chosen when `threshold` is None. `contrast` (default) "
            "blends a background anchor (mean+std) with a foreground anchor (a fraction of the "
            "high percentile) -- tuned for the high-contrast max-token map, where it tracks the "
            "per-image optimum. `gmm` fits two modes but its cut sits in the background shoulder "
            "here (it was the old default and shipped a single full-frame box). Absolute cosine "
            "thresholds do not transfer across images, which is why calibration is the default."
        ),
    )
    scoring: Literal["max-token", "prototype"] = Field(
        default="max-token",
        description=(
            "How a scene token is scored against the exemplar crop. `max-token` scores each scene "
            "token by the mean of its top-`match_tokens` cosine similarities to the crop's own "
            "tokens (best-matching-part) -- this keeps the crop's spatial detail and yields a "
            "high-contrast map where instances separate cleanly. `prototype` mean-pools the crop "
            "tokens into ONE vector first; simpler but low-contrast on richly-textured objects "
            "(all instances fuse into one blob), so it is retained only as a baseline."
        ),
    )
    match_tokens: int = Field(
        default=3,
        ge=1,
        description=(
            "For `max-token` scoring: how many of the closest crop tokens are averaged per scene "
            "token. 1 = pure nearest-token max (sharpest, noisiest); a few smooths single-token "
            "flukes without washing the contrast back out."
        ),
    )
    threshold: float | None = Field(
        default=None,
        description=(
            "Fixed accept threshold on the raw cosine similarity. None => use the calibrator."
        ),
    )
    candidate_margin: float = Field(
        default=0.1,
        ge=0.0,
        description=(
            "How far below the accept threshold a component is still logged as a sub-threshold "
            "candidate for offline PR sweeps (EVAL-08). Larger keeps more near-misses."
        ),
    )
    min_component_area: int = Field(
        default=4,
        ge=1,
        description=(
            "Absolute floor on connected-component area in pixels; smaller blobs are dropped as "
            "noise. The effective floor is the LARGER of this and `min_area_frac` x exemplar area."
        ),
    )
    min_area_frac: float = Field(
        default=0.12,
        ge=0.0,
        description=(
            "Size-relative floor: a component smaller than this fraction of the exemplar's area "
            "is a fragment, not an instance, and is dropped. Ties the noise floor to the object "
            "so it holds across the 300x canvas range without hand-tuning per image."
        ),
    )
    max_area_frac: float = Field(
        default=8.0,
        gt=0.0,
        description=(
            "Size-relative ceiling: a component larger than this multiple of the exemplar's area "
            "is a merged/background blob (an instance scaled up 1.6x with a rotated bounding box "
            "tops out near 5x), not a single instance, and is dropped so it cannot swallow "
            "several instances into one image-spanning false positive."
        ),
    )
    max_candidates: int = Field(
        default=50,
        ge=1,
        description=(
            "How many top components (with raw scores) to keep for the EVAL-08 candidate log."
        ),
    )
    seed: int = Field(
        default=0,
        ge=0,
        description="random_state for the gmm calibrator (its only genuinely stochastic step).",
    )
    retain_frac: float = Field(
        default=0.7,
        gt=0.0,
        le=1.0,
        description=(
            "self-similarity accepts scores above self_score * retain_frac (self_score=1.0)."
        ),
    )
    fixed_input_side: int | None = Field(
        default=None,
        ge=DINOV2_PATCH,
        description=(
            "Opt-in GPU-OOM fix for varied-resolution scenes (e.g. floor plans). When None "
            "(default) the scene runs at its native size capped at `scene_max_side` -- the "
            "committed path, byte-identical on the chipset/synthetic sets. When set to a multiple "
            "of 14, EVERY scene is letterboxed (one uniform aspect-preserving scale + a constant "
            "bottom-right pad) into a single fixed `fixed_input_side` x `fixed_input_side` input, "
            "so onnxruntime sees ONE input resolution across all plans and the CUDA memory arena "
            "is allocated once. Padding tokens are masked out before thresholding, and boxes map "
            "back through the single letterbox scale. Must be a multiple of 14 (the DINOv2 patch "
            "stride); a non-multiple is rejected at construction."
        ),
    )

    @field_validator("fixed_input_side")
    @classmethod
    def _fixed_input_side_is_multiple_of_patch(cls, value: int | None) -> int | None:
        """Reject a ``fixed_input_side`` that is not a multiple of the DINOv2 patch stride (14).

        A non-multiple would be silently floor-divided by the stride-14 patch conv (a systematic
        spatial offset, not an error -- the exact silent bug :class:`DINOv2Inferencer` guards), so
        it is caught at construction. ``None`` (the native/cap path) is always valid.
        """
        if value is not None and value % DINOV2_PATCH != 0:
            raise ValueError(
                f"fixed_input_side={value} must be a multiple of {DINOV2_PATCH} "
                f"(the DINOv2 patch stride); got remainder {value % DINOV2_PATCH}"
            )
        return value


@dataclass(frozen=True)
class _Component:
    """One connected component carried through post-processing as a box, score and area."""

    box: BBox
    score: float
    area: int


# -- the ONE reused inferencer (module-level lazy singleton) --------------------------------
# The SearchFn contract is (image, exemplar, config) -> SearchResult and shares nothing else
# with the app, so a method cannot read app.state. Instead the DINOv2 inferencer is built once,
# lazily, from the gitignored weight on disk and cached here, so every query -- API, CLI, or
# sample renderer -- reuses the SAME loaded model and pays the session-init cost once.
_inferencer: DINOv2Inferencer | None = None
_inferencer_loaded = False


def _get_inferencer() -> DINOv2Inferencer | None:
    """Return the shared :class:`DINOv2Inferencer`, or ``None`` when the weight is absent.

    Loads once and caches (including the absent result, so a missing weight is not re-probed on
    every call). Absence is a legitimate state -- weights are gitignored -- so it returns
    ``None`` and lets :func:`search` degrade to an honest ``outcome=error`` rather than raising.
    """
    global _inferencer, _inferencer_loaded
    if _inferencer_loaded:
        return _inferencer
    _inferencer_loaded = True
    path = models.models_dir() / models.MODEL_REGISTRY[_MODEL_KEY].dest
    if not path.is_file():
        logger.info(
            "dino-dense: {!r} weight absent at {}; returning model_unavailable "
            "(run `pixi run fetch-models --only {}`)",
            _MODEL_KEY,
            path,
            _MODEL_KEY,
        )
        _inferencer = None
        return None
    _inferencer = DINOv2Inferencer(path, providers=_PROVIDERS)
    return _inferencer


def reset_inferencer_cache() -> None:
    """Drop the cached inferencer so the next call re-probes disk. For test isolation only."""
    global _inferencer, _inferencer_loaded
    _inferencer = None
    _inferencer_loaded = False


# -- pure-numpy helpers (model-free, so CI gates them without the gitignored weight) --------


def _l2_normalize(vectors: npt.NDArray[np.floating], axis: int) -> npt.NDArray[np.float32]:
    """L2-normalize along ``axis``; a zero vector stays zero (guarded, never NaN)."""
    arr = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=axis, keepdims=True)
    normalized: npt.NDArray[np.float32] = (arr / np.maximum(norms, _EPS)).astype(np.float32)
    return normalized


def _prototype_from_grid(crop_grid: npt.NDArray[np.floating]) -> npt.NDArray[np.float32]:
    """Mean-pool a ``(gh, gw, D)`` crop token grid into one **L2-normalized** prototype.

    The pool comes first, the normalization second: normalizing per-token before pooling would
    weight every token equally regardless of magnitude, but the intended prototype is the mean
    embedding, normalized once so its self-cosine is 1.0.
    """
    tokens = np.asarray(crop_grid, dtype=np.float32).reshape(-1, crop_grid.shape[-1])
    pooled = tokens.mean(axis=0)
    return _l2_normalize(pooled, axis=0)


def _similarity_map(
    grid: npt.NDArray[np.floating],
    prototype: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    """Cosine similarity of every ``(gh, gw)`` scene token with the prototype.

    Both sides are L2-normalized before the dot product, so this is genuine cosine similarity in
    ``[-1, 1]`` rather than an unnormalized dot dominated by token magnitude. The prototype is
    already normalized by :func:`_prototype_from_grid`; the grid is normalized here.
    """
    gh, gw, dim = grid.shape
    flat_norm = _l2_normalize(np.asarray(grid, dtype=np.float32).reshape(-1, dim), axis=1)
    return (flat_norm @ prototype).reshape(gh, gw).astype(np.float32)


def _contrast_threshold(sim_token: npt.NDArray[np.floating]) -> tuple[float, str]:
    """Blend a background anchor (``mean + k*std``) with a foreground anchor
    (``frac * high-percentile``) into one accept threshold, returning ``(threshold, reason)``.

    The max-token map is high-contrast and heavy-tailed: a background bulk plus a thin tail of
    true-instance tokens. ``mean + k*std`` marks where the bulk ends; ``frac * p99.5`` marks a
    fraction of the near-peak. Their blend lands between the background and the instances across a
    wide range of images -- where a two-Gaussian posterior cut does not, because the tiny
    foreground weight drags its boundary down into the background shoulder. Operates on the
    TOKEN-resolution map so it is invariant to the upsample factor.
    """
    arr = np.asarray(sim_token, dtype=np.float64).reshape(-1)
    background = float(arr.mean() + _CONTRAST_STD_K * arr.std())
    foreground = _CONTRAST_PEAK_FRAC * float(np.percentile(arr, _CONTRAST_PEAK_PCTL))
    threshold = _CONTRAST_BLEND * background + (1.0 - _CONTRAST_BLEND) * foreground
    reason = (
        f"blend of background mean+{_CONTRAST_STD_K:g}std={background:.4f} and foreground "
        f"{_CONTRAST_PEAK_FRAC:g}*p{_CONTRAST_PEAK_PCTL:g}={foreground:.4f} "
        f"(w={_CONTRAST_BLEND:g}) -> cut at {threshold:.4f}"
    )
    return threshold, reason


def _crop_token_bank(crop_grid: npt.NDArray[np.floating]) -> npt.NDArray[np.float32]:
    """Flatten a ``(gh, gw, D)`` crop token grid into an ``(M, D)`` bank of **L2-normalized**
    tokens -- the crop's parts kept separate, as opposed to :func:`_prototype_from_grid` which
    collapses them to one vector. ``max-token`` scoring correlates each scene token against this
    whole bank, so a scene location scores high only where it resembles some ACTUAL part of the
    exemplar rather than the washed-out average of all its parts.
    """
    tokens = np.asarray(crop_grid, dtype=np.float32).reshape(-1, crop_grid.shape[-1])
    return _l2_normalize(tokens, axis=1)


def _maxtoken_similarity_map(
    grid: npt.NDArray[np.floating],
    bank: npt.NDArray[np.float32],
    match_tokens: int,
) -> npt.NDArray[np.float32]:
    """Score every scene token by the **mean of its top-``match_tokens`` cosine similarities** to
    the crop token bank (best-matching-part matching).

    Both sides are L2-normalized, so ``scene_norm @ bank.T`` is a ``(N_scene, M)`` matrix of genuine
    cosines. Taking, per scene token, the mean of its ``k`` largest entries answers "how well does
    the single best-matching crop part explain this location", which is high on true instances and
    low on background -- a far sharper contrast than one mean-pooled prototype produces. ``k`` is
    clamped to the bank size so a tiny crop still works.
    """
    gh, gw, dim = grid.shape
    scene = _l2_normalize(np.asarray(grid, dtype=np.float32).reshape(-1, dim), axis=1)
    cosines = scene @ bank.T  # (N_scene, M): every scene token vs every crop token
    k = min(match_tokens, cosines.shape[1])
    # np.partition puts the k largest in the last k columns (unordered) -- all the mean needs.
    topk = np.partition(cosines, -k, axis=1)[:, -k:]
    return topk.mean(axis=1).reshape(gh, gw).astype(np.float32)


def _upsample_similarity(
    sim_map: npt.NDArray[np.floating],
    scale_x: float,
    scale_y: float,
) -> npt.NDArray[np.float32]:
    """Bilinearly upsample the ``(gh, gw)`` similarity MAP to the model-input pixel resolution.

    The token grid covers a snapped input of ``gh*14 x gw*14`` pixels; the inferencer's
    ``scale_x``/``scale_y`` (``snapped / input``) invert that snap, so the true input size is
    ``gw*14 / scale_x`` by ``gh*14 / scale_y``. ``cv2.resize`` maps source cell centre
    ``(gx+0.5)`` to destination fraction ``(gx+0.5)/gw``, which is exactly each patch centre, so
    a token peak lands on its pixel. The MAP is upsampled, never the 384-d tokens.
    """
    gh, gw = sim_map.shape
    target_w = max(1, round(gw * DINOV2_PATCH / scale_x))
    target_h = max(1, round(gh * DINOV2_PATCH / scale_y))
    upsampled = cv2.resize(
        np.asarray(sim_map, dtype=np.float32),
        (target_w, target_h),
        interpolation=cv2.INTER_LINEAR,
    )
    return np.asarray(upsampled, dtype=np.float32)


# -- fixed-size letterbox (opt-in GPU-OOM fix; model-free, so CI gates it without the weight) ----
# A similarity strictly BELOW any cosine/threshold, written into the padded region so connected
# components can never fire there. It must be below the candidate floor's own minimum (which is
# clamped to -1.0), so -2.0 -- outside the [-1, 1] cosine range and below every possible cut,
# including the candidate_margin floor -- guarantees padded pixels are never foreground.
_PAD_SENTINEL: float = -2.0
# The constant pad value for the letterboxed canvas (a flat, deterministic border).
_LETTERBOX_PAD_VALUE: int = 0


def _fixed_letterbox(
    image: npt.NDArray[np.uint8], side: int
) -> tuple[npt.NDArray[np.uint8], float, int, int]:
    """Letterbox a BGR image into a ``side`` x ``side`` square via ONE uniform scale + a pad.

    The single aspect-preserving scale is ``side / max(h, w)``, so the content fits exactly inside
    the square with no distortion; the resized content is placed at the TOP-LEFT and the remaining
    bottom/right strip is filled with a constant value (:data:`_LETTERBOX_PAD_VALUE`). A top-left
    placement means the pad offset is zero, so a box in the square maps back to the original by a
    single division by the scale -- no offset term. Deterministic (fixed interpolation + constant
    pad). Returns ``(canvas, scale, content_w, content_h)`` where ``content_w/h`` are the resized
    content extent in square pixels (everything at or beyond them is padding).
    """
    h, w = int(image.shape[0]), int(image.shape[1])
    scale = side / max(h, w)
    content_w = min(side, max(1, round(w * scale)))
    content_h = min(side, max(1, round(h * scale)))
    resized = cv2.resize(image, (content_w, content_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((side, side, 3), _LETTERBOX_PAD_VALUE, dtype=np.uint8)
    canvas[:content_h, :content_w] = resized
    return np.ascontiguousarray(canvas), scale, content_w, content_h


def _content_token_mask(gh: int, gw: int, content_w: int, content_h: int) -> npt.NDArray[np.bool_]:
    """Boolean ``(gh, gw)`` grid: ``True`` where a token's patch CENTRE lands inside the content.

    A token ``(gy, gx)`` on a multiple-of-14 (unsnapped) input has its patch centre at pixel
    ``(gx*14 + 7, gy*14 + 7)``; it is a content token when that centre falls within
    ``[0, content_w) x [0, content_h)`` and a padding token otherwise. Used to exclude padding
    tokens from the threshold calibration so the padded region cannot drag the cut around.
    """
    centre_x = np.arange(gw) * DINOV2_PATCH + DINOV2_PATCH // 2
    centre_y = np.arange(gh) * DINOV2_PATCH + DINOV2_PATCH // 2
    valid_x = centre_x < content_w
    valid_y = centre_y < content_h
    return np.outer(valid_y, valid_x)


def _extract_components(
    sim_full: npt.NDArray[np.float32],
    floor: float,
    min_area: int,
    max_area: float,
    cap_scale: float,
    orig_w: int,
    orig_h: int,
) -> list[_Component]:
    """Threshold at ``floor``, run connected components, and return boxes in ORIGINAL pixels.

    ``cv2.connectedComponentsWithStats`` numbers the background 0 and each foreground blob 1..N.
    **Label 0 is skipped explicitly** -- it is the whole non-matching background and emitting it
    would be one image-sized false positive. Each remaining component's stats bbox is scaled from
    the (capped) inference resolution back to original scene pixels by dividing out ``cap_scale``.
    A component outside ``[min_area, max_area]`` (both in capped-inference pixels) is dropped as a
    fragment or a merged/background blob; ``max_area <= 0`` disables the ceiling.
    """
    mask = (sim_full >= floor).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    components: list[_Component] = []
    for label in range(1, count):  # 1.. : SKIP label 0, the background component
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area or (max_area > 0 and area > max_area):
            continue
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        score = float(sim_full[labels == label].max())
        try:
            box = BBox(
                x=max(0, round(x / cap_scale)),
                y=max(0, round(y / cap_scale)),
                w=max(1, round(w / cap_scale)),
                h=max(1, round(h / cap_scale)),
            ).clipped_to(orig_w, orig_h)
        except ValueError:
            continue  # a rounded box that fell entirely off-image; drop it
        components.append(_Component(box=box, score=score, area=area))
    return components


def _empty_or_error(
    outcome: SearchOutcome,
    note: str,
    *,
    threshold: float | None,
    latency: LatencyBreakdown,
    metrics: dict[str, float],
    heatmap: HeatmapPayload | None = None,
    error: MethodError | None = None,
    candidates: tuple[Candidate, ...] = (),
) -> SearchResult:
    """Build an honest zero-match result (``EMPTY`` or ``ERROR``) carrying a note saying why."""
    return SearchResult(
        method="dino-dense",
        method_version=_METHOD_VERSION,
        outcome=outcome,
        matches=(),
        latency=latency,
        threshold_applied=threshold,
        candidates=candidates,
        diagnostics=Diagnostics(notes=(note,), metrics=metrics, similarity_heatmap=heatmap),
        error=error,
    )


@register_method(
    name="dino-dense",
    description="DINOv2 dense-token prototype cosine similarity with high-res upsampling.",
    version=_METHOD_VERSION,
    config_model=DinoDenseConfig,
)
def search(
    image: npt.NDArray[np.uint8],
    exemplar: ExemplarBox,
    config: BaseModel,
) -> SearchResult:
    """Find every instance of ``exemplar`` in ``image`` by DINOv2 dense-token similarity."""
    # The registry types config as BaseModel (method-agnostic); the registered config_model
    # guarantees the concrete type. Narrow once here and fail loudly if the contract is violated.
    if not isinstance(config, DinoDenseConfig):
        raise TypeError(
            f"dino-dense.search requires a DinoDenseConfig, got {type(config).__name__}"
        )

    orig_h, orig_w = int(image.shape[0]), int(image.shape[1])
    t_start = perf_counter()

    # 1. Get the ONE shared DINOv2 inferencer. Absent weight => an honest model_unavailable error
    #    (outcome=error), never a raise -- so the sample renderer and API degrade gracefully.
    inferencer = _get_inferencer()
    if inferencer is None:
        latency = LatencyBreakdown(preprocess_ms=0.0, inference_ms=0.0, postprocess_ms=0.0)
        return _empty_or_error(
            SearchOutcome.ERROR,
            f"DINOv2 weight {_MODEL_KEY!r} is absent; run `pixi run fetch-models --only "
            f"{_MODEL_KEY}`. No search was run.",
            threshold=None,
            latency=latency,
            metrics={},
            error=MethodError(kind="model_unavailable", message=f"{_MODEL_KEY} weight not on disk"),
        )

    # 2. Embed the exemplar crop into DINOv2 tokens. `max-token` keeps every crop token (a bank of
    #    the object's parts); `prototype` mean-pools them into one vector. The bank is the default
    #    because a single prototype washes out contrast on textured objects (see the `scoring` doc).
    ex = exemplar.box
    crop = np.ascontiguousarray(image[ex.y : ex.y2, ex.x : ex.x2], dtype=np.uint8)
    crop_grid, _, _ = inferencer.dense_tokens(crop)
    prototype = _prototype_from_grid(crop_grid)
    token_bank = _crop_token_bank(crop_grid)

    # 3. Run the SCENE at high resolution, capped at scene_max_side. A scene above the cap is
    #    downscaled and the cap is LOGGED -- never silently truncated. dense_tokens then snaps
    #    each side to a multiple of 14 and returns the scale factors that invert that snap.
    long_side = max(orig_h, orig_w)
    cap_engaged = long_side > config.scene_max_side
    if cap_engaged:
        cap_scale = config.scene_max_side / long_side
        capped = np.ascontiguousarray(
            cv2.resize(
                image,
                (max(1, round(orig_w * cap_scale)), max(1, round(orig_h * cap_scale))),
                interpolation=cv2.INTER_AREA,
            ),
            dtype=np.uint8,
        )
        logger.info(
            "dino-dense: scene {}x{} exceeds scene_max_side={}; downscaling by {:.4f} to {}x{}",
            orig_w,
            orig_h,
            config.scene_max_side,
            cap_scale,
            capped.shape[1],
            capped.shape[0],
        )
    else:
        cap_scale = 1.0
        capped = image

    # 3b. OPT-IN fixed-size letterbox (GPU-OOM fix). When `fixed_input_side` is set, letterbox the
    #     (capped) scene into ONE fixed square so onnxruntime sees a single input resolution across
    #     all plans (the CUDA arena is allocated once). `effective_scale` is the combined
    #     original -> model-input scale used to map boxes back; `content_extent` marks where the
    #     real content ends and the constant pad begins (None => native/cap path, no pad). The None
    #     path leaves `model_input`/`effective_scale` exactly as the cap branch produced -- byte-
    #     identical on chipset/synthetic.
    if config.fixed_input_side is not None:
        model_input, letterbox_scale, content_w, content_h = _fixed_letterbox(
            capped, config.fixed_input_side
        )
        effective_scale = cap_scale * letterbox_scale
        content_extent: tuple[int, int] | None = (content_w, content_h)
        logger.info(
            "dino-dense: letterboxed scene into a fixed {side}x{side} input "
            "(content {cw}x{ch}, scale x{ls:.4f}); one onnxruntime input shape for every plan",
            side=config.fixed_input_side,
            cw=content_w,
            ch=content_h,
            ls=letterbox_scale,
        )
    else:
        model_input = capped
        effective_scale = cap_scale
        content_extent = None

    t_infer = perf_counter()
    grid, scale_x, scale_y = inferencer.dense_tokens(model_input)
    inference_ms = (perf_counter() - t_infer) * 1000.0
    gh, gw = grid.shape[0], grid.shape[1]

    # 4. Score every scene token -> a (gh, gw) cosine-similarity map. `max-token` uses the mean of
    #    each token's top-k cosines to the crop bank (high contrast); `prototype` dots against the
    #    single mean-pooled vector (the low-contrast baseline).
    if config.scoring == "max-token":
        sim_token = _maxtoken_similarity_map(grid, token_bank, config.match_tokens)
    else:
        sim_token = _similarity_map(grid, prototype)

    # 5. Bilinearly upsample the MAP (not the tokens) to pixel resolution using the scale factors.
    sim_full = _upsample_similarity(sim_token, scale_x, scale_y)

    # 5b. Fixed-letterbox PADDING MASK. Tokens whose patch centre lands in the padded region are
    #     excluded from calibration (so the pad cannot drag the cut around), and the padded pixels
    #     of the upsampled map are set BELOW any threshold so connected components can only fire in
    #     the real content. On the native/cap path (content_extent is None) nothing is masked --
    #     `calib_values` is the full token map, byte-identical to before.
    if content_extent is not None:
        content_w, content_h = content_extent
        token_valid = _content_token_mask(gh, gw, content_w, content_h)
        calib_values = sim_token[token_valid]
        sim_full[content_h:, :] = _PAD_SENTINEL
        sim_full[:, content_w:] = _PAD_SENTINEL
    else:
        calib_values = sim_token.reshape(-1)

    # 6. Calibrate the accept threshold from the token-resolution similarity distribution. A pinned
    #    config.threshold wins; otherwise "contrast" (the default, tuned for this map) is computed
    #    locally, and the classical strategies delegate to the shared calibrator. self_score is 1.0
    #    (the L2-normalized prototype's self-cosine). Each path yields a (strategy, reason) for the
    #    diagnostics note. `calib_values` is the content-only token distribution under the fixed
    #    letterbox, else the full token map.
    if config.threshold is not None:
        threshold = config.threshold
        calib_strategy, calib_reason = "fixed", f"caller-pinned threshold {threshold:.4f}"
    elif config.calibration == "contrast":
        threshold, calib_reason = _contrast_threshold(calib_values)
        calib_strategy = "contrast"
    else:
        calib = calibration.calibrate(
            calib_values.reshape(-1),
            strategy=config.calibration,
            fixed_threshold=None,
            self_score=_SELF_MATCH_SCORE,
            retain_frac=config.retain_frac,
            seed=config.seed,
        )
        threshold, calib_strategy, calib_reason = calib.threshold, calib.strategy, calib.reason

    # 7. MATCH components are grown at the accept threshold ITSELF -- NOT at a floor below it. The
    #    box of a match is therefore the extent of the above-threshold region only; a low-contrast
    #    shoulder just below the threshold can no longer bridge distinct instances into one
    #    image-spanning blob (the bug that made this method return a single full-frame box).
    #    Area bounds are tied to the exemplar so fragments and merged blobs drop out at any canvas
    #    scale (both bounds are in model-input pixels, hence the effective_scale^2 factor).
    #    `effective_scale` is the combined original -> model-input scale: the cap scale on the
    #    native path, and cap x letterbox on the fixed-input path, so boxes map back through the
    #    one scale (bottom-right pad => zero offset). METHOD-12: EVERY component above threshold
    #    survives -- no single-best short-circuit, so components emit as many instances as present.
    exemplar_area_capped = ex.w * ex.h * effective_scale * effective_scale
    min_area = max(config.min_component_area, round(config.min_area_frac * exemplar_area_capped))
    max_area = config.max_area_frac * exemplar_area_capped
    match_components = _extract_components(
        sim_full, threshold, min_area, max_area, effective_scale, orig_w, orig_h
    )
    accepted = sorted(match_components, key=lambda c: (-c.score, c.box.y, c.box.x))
    matches = _build_matches(accepted, ex)

    # 8. CANDIDATE components (EVAL-08) are grown at a floor candidate_margin BELOW the threshold so
    #    an offline PR sweep can rebuild the curve from sub-threshold near-misses. These feed the
    #    candidate LOG ONLY and are never emitted as matches, so their coarser (possibly merged)
    #    boxes cannot pollute the returned detections.
    candidate_floor = max(-1.0, threshold - config.candidate_margin)
    cand_components = _extract_components(
        sim_full, candidate_floor, min_area, max_area, effective_scale, orig_w, orig_h
    )
    ordered = sorted(cand_components, key=lambda c: (-c.score, c.box.y, c.box.x))
    candidates = tuple(
        Candidate(box=c.box, score=c.score) for c in ordered[: config.max_candidates]
    )

    # 9. Assemble diagnostics (the similarity heatmap is the debug overlay) and the result.
    postprocess_ms = max(0.0, (perf_counter() - t_start) * 1000.0 - inference_ms)
    latency = LatencyBreakdown(
        preprocess_ms=0.0, inference_ms=inference_ms, postprocess_ms=postprocess_ms
    )
    heatmap = viz.heatmap_png_b64(sim_full)
    metrics: dict[str, float] = {
        "threshold": threshold,
        "self_score": _SELF_MATCH_SCORE,
        "grid_h": float(gh),
        "grid_w": float(gw),
        "cap_scale": cap_scale,
        "cap_engaged": 1.0 if cap_engaged else 0.0,
        "n_components": float(len(match_components)),
        "n_candidates": float(len(candidates)),
        "n_matches": float(len(matches)),
        "sim_max": float(sim_token.max()),
        "sim_mean": float(sim_token.mean()),
    }
    notes = (
        f"calibration[{calib_strategy}]: {calib_reason}",
        (
            f"kept {len(matches)} match(es) from {len(match_components)} component(s) "
            f"on a {gh}x{gw} token grid; threshold {threshold:.4f} on cosine similarity"
            + (
                f"; scene capped to {config.scene_max_side}px (x{cap_scale:.4f})"
                if cap_engaged
                else ""
            )
        ),
    )

    if not matches:
        return _empty_or_error(
            SearchOutcome.EMPTY,
            (
                f"no DINOv2 component cleared the calibrated threshold {threshold:.4f} "
                f"(best {float(sim_token.max()):.4f}); {calib_strategy}: {calib_reason}"
            ),
            threshold=threshold,
            latency=latency,
            metrics=metrics,
            heatmap=heatmap,
            candidates=candidates,
        )

    return SearchResult(
        method="dino-dense",
        method_version=_METHOD_VERSION,
        outcome=SearchOutcome.OK,
        matches=matches,
        latency=latency,
        threshold_applied=threshold,
        candidates=candidates,
        diagnostics=Diagnostics(notes=notes, metrics=metrics, similarity_heatmap=heatmap),
    )


def _build_matches(accepted: list[_Component], exemplar: BBox) -> tuple[Match, ...]:
    """Turn accepted components into Matches, labelling the exemplar's own region is_exemplar.

    The exemplar is part of the scene it is searched in, so one accepted component overlaps it;
    that component is labelled rather than dropped (which understates recall) or counted silently
    as a discovery (which overstates the method), per METHOD-04c. Matches are returned in the
    canonical ``(-score, y, x)`` order.
    """
    exemplar_idx: int | None = None
    best_iou = _EXEMPLAR_IOU
    for i, component in enumerate(accepted):
        overlap = component.box.iou(exemplar)
        if overlap >= best_iou:
            best_iou = overlap
            exemplar_idx = i

    ordered = sorted(
        range(len(accepted)),
        key=lambda i: (-accepted[i].score, accepted[i].box.y, accepted[i].box.x),
    )
    return tuple(
        Match(box=accepted[i].box, score=accepted[i].score, is_exemplar=(i == exemplar_idx))
        for i in ordered
    )
