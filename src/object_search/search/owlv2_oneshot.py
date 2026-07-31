"""Method 4 -- OWLv2 image-conditioned one-shot detection (the permissive learned detector).

This is the source-research **Method 4** (exemplar-conditioned open-vocabulary detectors), which
was deferred in Milestone 1 because its obvious candidates (T-Rex2, CountGD) were licence- and
ONNX-encumbered. OWLv2 is the permissive, ONNX-exportable realization of that bucket.

What it does
------------
This is the open-vocabulary *detector* baseline for the scoreboard, and the closest permissive
analogue to the visual-prompt detectors (T-Rex2 / Rex-Omni) that were rejected on licensing. It
runs OWLv2 in its **image-conditioned (one-shot) detection** mode: the exemplar crop is encoded as
a *query image*, a single query embedding is selected from it, and every patch of the scene is
scored by the **cosine similarity** of its OWLv2 class embedding to that query embedding. The
accepted patches, mapped through OWLv2's own predicted boxes, are the matches.

Its selling point over the appearance-similarity methods (Methods 3/5) is that the boxes come from
a **trained detection head**, not from a similarity blob or a class-agnostic proposer -- OWLv2 was
supervised to localize objects, so the returned boxes are detector-tight, and the same one model
does both "where is an object" and "is it the exemplar" in a single forward pass per image.

One vision graph, run twice -- there is no second ONNX input
-----------------------------------------------------------
OWLv2 image-guided detection is a "two image" task (query crop + scene), but the ONNX graph has a
**single** ``pixel_values`` input. The method encodes the exemplar crop and the scene through the
**same** :meth:`OWLv2Inferencer.embed_image` -- one call each -- and does the two-image logic
(query-embedding selection, cosine scoring) here in NumPy, in this one file. That keeps the ONNX
layer a plain single-input inferencer and the "image-guided" cleverness legible.

Query-embedding selection (distinctiveness -- a correctness requirement)
-----------------------------------------------------------------------
The user's box tightly frames the object, so the object fills the exemplar crop. Among the crop
patches whose **predicted box covers the crop** (IoU with the full ``[0, 1]`` box within
``query_iou_frac`` of the maximum), we take the single patch whose embedding is **least similar to
the mean patch embedding** -- the most *distinctive* one, which is the actual object rather than the
generic "whole-frame / background" direction. This is HuggingFace's ``embed_image_query`` heuristic,
and it is **load-bearing, not a refinement**: mean-pooling the covering patches instead (the naive
first cut) yields exactly the generic embedding, which then matches scene patches predicting
whole-image boxes -- the method scored ~0 F1 until this was fixed (see the improvement report).

Pre-processing (exact)
----------------------
OWLv2's preprocessing is **not** re-derived here; it is written once in the
:class:`~object_search.inference.owlv2.OWLv2Inferencer` docstring and reused: ``pixel_values`` f32
NCHW, **RGB**, rescale ``1/255``, **pad bottom-right to a square** of side ``max(H, W)`` with grey
``0.5``, resize to ``960x960`` **bilinear**, then CLIP mean ``[0.48145466, 0.4578275, 0.40821073]``
/ std ``[0.26862954, 0.26130258, 0.27577711]``. Because OWLv2 pads bottom-right (not centred), a
normalized ``pred_box`` maps to scene pixels by a plain multiply by ``max(H, W)`` and a clip -- no
pad offset. These numbers are the documented HF constants, asserted at export
(``scripts/export_owlv2.py``), not yet runtime-verified in ``.planning/research/MODELS.md``.

Post-processing (exact)
-----------------------
- **L2-normalize both sides**, then cosine = a plain NumPy matmul of the scene's ``(P, D)``
  normalized class-embed matrix against the ``(D,)`` normalized query embedding, in ``[-1, 1]``.
- **Recalibrate with OWLv2's own learned logit_shift/logit_scale** (see the floor-plans improvement
  report): ``calibrated = (cosine + logit_shift) * logit_scale``, HF's own
  ``Owlv2ClassPredictionHead`` formula, using this SCENE's own per-patch shift/scale (computed with
  no query, so it is query-independent). This is not a monotonic rescaling -- ``logit_shift``/
  ``logit_scale`` vary per patch, so it CAN re-rank patches (unlike a single global scale/shift),
  which is the property that motivated exporting it: on floor-plan line-art, raw cosine was
  observed ranking generic wall/room-corner rectangles above the true door/window instances.
  Measured (micro F1 @ IoU 0.5, retain_frac re-tuned to 0.85 for the new score scale): beats the
  raw-cosine baseline on every one of the six regimes measured (chipset EASY, TEXTURED, VARIED,
  CLUTTERED, and the floor-plan DOOR/WINDOW target domain) -- the two floor-plan regimes it was
  built for improve the most (DOOR F1 0.115 -> 0.154, WINDOW 0.022 -> 0.027, still weak in absolute
  terms but a real gain). Every downstream score (threshold, self-similarity anchor, NMS ordering,
  candidates) uses the calibrated score, never raw cosine.
- **Drop the generic whole-frame boxes** (area above ``max_box_area_frac`` of the image): OWLv2
  emits one that scores highest but is never a valid instance in a multi-instance scene, and left
  in it anchors the threshold and dominates NMS.
- **Threshold** via ``common.calibration``, default **``self-similarity``**: cut at
  ``self_score * retain_frac`` where ``self_score`` is the exemplar's own self-match score (the top
  calibrated score among boxes overlapping the exemplar box). ``gmm`` still degenerates badly even
  against the calibrated score (measured, not just theorized) -- it is not a viable alternative. A
  fixed ``score_threshold`` overrides the calibrator and is compared against the calibrated score.
- **NMS at ``nms_iou``**: OWLv2 emits one box per patch, so one object spans several neighbouring
  patches that each score well; NMS collapses those into a single detection. Sort ties by
  ``(-score, y, x)`` (never score alone), the project's reproducibility rule.
- **Sub-threshold candidates are retained** (EVAL-08) and **every accepted box survives** into
  matches after NMS -- there is no single-best short-circuit (METHOD-12).

Latency (EVAL-11)
-----------------
Two forward passes dominate: the query crop encode and the scene encode. Both are the same model,
so ``inference_ms`` carries their sum, but ``diagnostics.metrics`` reports ``query_ms`` and
``target_ms`` separately (and a note states which dominates -- the scene encode does), so the
finding is legible rather than buried in one total.

Known failure modes
-------------------
- **The exemplar self-match.** The exemplar is part of the scene, so one accepted patch overlaps
  it; that patch is labelled ``is_exemplar=True`` rather than dropped or silently counted.
- **Fixed 960 input caps small-object recall on large canvases** -- OWLv2's position embeddings
  pin the input to 960, so a small instance in a large scene occupies few patches. ``config.
  tile_large_scenes`` (see below) exists for this, but MEASURED across six regimes it did not
  net-help -- see the floor-plans improvement report and the ROBUSTNESS BACKLOG note below.
- **Weights absent.** OWLv2 (Apache-2.0) weights are gitignored and fetched by
  ``pixi run -e export export-owlv2``. Absent, the method returns ``outcome=error`` with a
  ``model_unavailable`` note rather than raising, so the renderer and API degrade honestly.

Licence
-------
OWLv2 is **Apache-2.0** (Google) -- the same permissive tier as DINOv2 and SuperPoint's code, with
**no** AGPL §13 or non-commercial constraint. Adopting it (unlike T-Rex2 / Rex-Omni, both
non-commercial IDEA-licensed) does not touch how this repo may be shared. See
``docs/library-reviews/owlv2.md``.

ROBUSTNESS BACKLOG
------------------
Mirrored in ``docs/methods/owlv2-oneshot.md`` and ``docs/ROBUSTNESS-BACKLOG.md``.

- **``config.tile_large_scenes`` -- BUILT AND MEASURED, off by default, not recommended.** Tiling
  a large scene into overlapping 960px windows was hypothesized to lift small-object recall (the
  EASY/chipset regime's known weakness). Measured across six regimes (see the floor-plans
  improvement report): it regressed 5 of 6, INCLUDING EASY itself (F1 -20%, recall completely
  unchanged -- every extra tile added false positives, not one new true positive). Kept as a
  documented opt-in for further investigation, not as a recommendation.
- **``config.rotation_invariant`` -- BUILT AND MEASURED, off by default, not recommended.** Scoring
  on the max cosine across 0/90/180/270-degree query rotations was hypothesized to help mirrored/
  rotated floor-plan symbols. Measured: it helped VARIED (+5%) and WINDOW (+12%, still near-zero
  absolute) but regressed DOOR badly (-26%, one of the two target-domain regimes this exploration
  cares about) and EASY (-20%). Kept as a documented opt-in, not as a recommendation.
- **Text-prompt fusion** -- OWLv2 also takes text queries; combining the drawn exemplar with an
  optional label would use both modalities (the exploration's Milestone 2 seam).
- **Query embedding from multiple exemplars** -- average several drawn boxes for a robust query.
- **owlv2-large** for accuracy at higher latency, gated behind the same export path.
"""

from __future__ import annotations

from time import perf_counter
from typing import Literal

import numpy as np
import numpy.typing as npt
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from object_search.inference import models, resolve_providers
from object_search.inference.owlv2 import OWLV2_IMAGE_SIZE, OWLv2Inferencer
from object_search.schemas import (
    BBox,
    Candidate,
    Diagnostics,
    ExemplarBox,
    LatencyBreakdown,
    Match,
    MethodError,
    SearchOutcome,
    SearchResult,
)
from object_search.search.common import calibration, nms
from object_search.search.registry import register_method

# -- Method-level constants (properties of the METHOD, not of a query, so not config fields) --

_METHOD_VERSION = "1.0.0"
_OWLV2_KEY = "owlv2-base-patch16"  # the MODEL_REGISTRY key for the OWLv2 vision graph
_PROVIDERS = resolve_providers()  # CPU by default (bit-identical); OS_ONNX_PROVIDERS opts into GPU
_EXEMPLAR_IOU = 0.5  # a match overlapping the exemplar by >= this is the exemplar's own region
_EXEMPLAR_SELF_IOU = 0.3  # looser overlap used to read off the exemplar's own self-match score
_TILE_SIZE = OWLV2_IMAGE_SIZE  # tile side matches OWLv2's native input -- no extra downscaling
_TILE_OVERLAP_FRAC = 0.2  # stride = _TILE_SIZE * (1 - this); keeps a symbol near a seam intact
_EPS = 1e-12  # guards a zero-norm division; a genuinely zero embedding is background, not a match


class Owlv2OneshotConfig(BaseModel):
    """Frozen config for :func:`search`; its JSON Schema drives the UI form (one source).

    Every field carries a ``description`` because that text becomes the form's help string --
    this is the single place it is written, so it must be written here.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    score_threshold: float | None = Field(
        default=None,
        description=(
            "Fixed accept threshold on the cosine similarity between a scene patch's OWLv2 class "
            "embedding and the query embedding. None => calibrate with the `calibration` strategy "
            "(absolute cosine cuts do not transfer across images for deep features)."
        ),
    )
    calibration: Literal["fixed", "self-similarity", "ratio", "gmm"] = Field(
        default="self-similarity",
        description=(
            "How the accept threshold is chosen when score_threshold is None. self-similarity cuts "
            "at self_score * retain_frac, anchored to the exemplar's own self-match score. OWLv2 "
            "cosine scores are compressed near 1.0 and not cleanly bimodal, so gmm degenerates to "
            "ratio and thresholds unstably (flooding some scenes, starving others)."
        ),
    )
    retain_frac: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description=(
            "self-similarity accepts scene patches scoring above self_score * retain_frac. Higher "
            "is stricter (fewer, more confident matches). 0.85 is the robust sweet spot against "
            "the CALIBRATED score (see the floor-plans improvement report) -- it beat the prior "
            "0.94/raw-cosine baseline's F1 on every regime measured, including the two floor-plan "
            "target-domain regimes that motivated re-tuning it."
        ),
    )
    query_iou_frac: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description=(
            "Query-embedding selection: consider the exemplar-crop patches whose predicted box IoU "
            "with the full crop is at least this fraction of the maximum, then pick the single "
            "most distinctive (least similar to the mean) of them. Lower widens the candidate set."
        ),
    )
    rotation_invariant: bool = Field(
        default=False,
        description=(
            "Also encode the exemplar crop rotated 90/180/270 degrees, select a query embedding "
            "per rotation, and score every scene patch by the MAX calibrated score across all four "
            "-- floor-plan door/window symbols are frequently mirrored/rotated relative to the "
            "drawn exemplar, and OWLv2 patch embeddings are not rotation-equivariant. Costs ~4x "
            "the query-encode latency (query_ms), never the scene encode."
        ),
    )
    tile_large_scenes: bool = Field(
        default=False,
        description=(
            "Split a scene wider or taller than OWLv2's native 960px input into overlapping "
            "960px tiles, encode each separately, and merge candidates before thresholding/NMS -- "
            "instead of downscaling the whole scene into one 960px pass. A scene that already fits "
            "within 960x960 is encoded once, identically to the untiled path (no-op). Costs one "
            "extra forward pass per tile (scene_ms scales with tile count); never affects the "
            "query encode."
        ),
    )
    max_box_area_frac: float = Field(
        default=0.25,
        gt=0.0,
        le=1.0,
        description=(
            "Drop any predicted box whose area exceeds this fraction of the image. OWLv2 emits a "
            "generic whole-frame box that scores highest but is never a valid instance in a "
            "multi-instance scene; discarding it stops it anchoring the threshold and biasing NMS."
        ),
    )
    nms_iou: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description=(
            "Post-detection NMS IoU. A later accepted box overlapping a kept one by MORE than this "
            "is suppressed -- OWLv2 fires several overlapping patches on one object, so a tight "
            "0.3 collapses those duplicates (a big precision win) while distinct instances, which "
            "rarely overlap that much, survive."
        ),
    )
    max_candidates: int = Field(
        default=50,
        ge=1,
        description=(
            "How many top-scoring patches (with raw scores) to keep as sub-threshold candidates "
            "for an offline PR sweep (EVAL-08), regardless of the threshold."
        ),
    )
    seed: int = Field(
        default=0,
        ge=0,
        description="random_state for the gmm calibrator (its only genuinely stochastic step).",
    )


# -- the OWLv2 backbone (module-level lazy singleton, mirrors dino_dense/propose_retrieve) -------
# Cached so every query (API, CLI, sample renderer) reuses one loaded session and pays init once.
_inferencer: OWLv2Inferencer | None = None
_inferencer_loaded = False


def _get_inferencer() -> OWLv2Inferencer | None:
    """Return the shared OWLv2 inferencer, or ``None`` when its weight is absent.

    Loads once and caches (including the absent result, so a missing weight is not re-probed on
    every call). Absence is a legitimate state -- the weight is gitignored -- so this returns
    ``None`` and lets :func:`search` degrade to an honest ``outcome=error`` rather than raising.
    """
    global _inferencer, _inferencer_loaded
    if _inferencer_loaded:
        return _inferencer
    _inferencer_loaded = True
    path = models.models_dir() / models.MODEL_REGISTRY[_OWLV2_KEY].dest
    if not path.is_file():
        logger.info(
            "owlv2-oneshot: {!r} weight absent at {}; returning model_unavailable "
            "(run `pixi run -e export export-owlv2`)",
            _OWLV2_KEY,
            path,
        )
        _inferencer = None
        return None
    _inferencer = OWLv2Inferencer(path, providers=_PROVIDERS)
    return _inferencer


def reset_inferencer_cache() -> None:
    """Drop the cached OWLv2 inferencer so the next call re-probes disk. Test isolation only."""
    global _inferencer, _inferencer_loaded
    _inferencer = None
    _inferencer_loaded = False


# -- pure-numpy helpers (model-free, so CI gates them without the gitignored weight) ------------


def _l2_normalize(vectors: npt.NDArray[np.floating], axis: int) -> npt.NDArray[np.float32]:
    """L2-normalize along ``axis``; a zero vector stays zero (guarded, never NaN)."""
    arr = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=axis, keepdims=True)
    normalized: npt.NDArray[np.float32] = (arr / np.maximum(norms, _EPS)).astype(np.float32)
    return normalized


def _iou_with_unit_box(boxes_cxcywh: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """IoU of each normalized ``(cx, cy, w, h)`` box with the full ``[0, 1] x [0, 1]`` box.

    Vectorized and pure. The full box has area ``1``; each box's intersection with it is the box
    clipped to ``[0, 1]``. Used to find the exemplar-crop patches that cover the whole object.
    """
    cx, cy, w, h = boxes_cxcywh[:, 0], boxes_cxcywh[:, 1], boxes_cxcywh[:, 2], boxes_cxcywh[:, 3]
    x1, y1, x2, y2 = cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0
    ix1 = np.maximum(x1, 0.0)
    iy1 = np.maximum(y1, 0.0)
    ix2 = np.minimum(x2, 1.0)
    iy2 = np.minimum(y2, 1.0)
    inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
    area = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    union = 1.0 + area - inter
    return np.asarray(inter / np.maximum(union, _EPS), dtype=np.float32)


def select_query_embedding(
    class_embeds: npt.NDArray[np.float32],
    boxes_cxcywh: npt.NDArray[np.float32],
    iou_frac: float,
) -> npt.NDArray[np.float32]:
    """Select one ``(D,)`` L2-normalized query embedding: HuggingFace's distinctiveness heuristic.

    This mirrors ``Owlv2ForObjectDetection.embed_image_query``, and it is a **correctness
    requirement**, not a refinement. Among the crop patches whose predicted box covers the crop
    (IoU with the full ``[0, 1]`` box at least ``iou_frac`` of the maximum), it picks the single
    patch whose embedding is **least similar to the mean patch embedding** -- the most
    *distinctive* one, which is the actual object rather than the generic "whole-frame /
    background" direction. Mean-pooling those patches instead (the naive first cut) yields exactly
    that generic embedding, which then matches scene patches predicting whole-image boxes -- the
    method scored ~0 F1 until this selection was fixed (see docs/reports/owlv2-improvement.md).

    Falls back to the single largest-area patch if no box overlaps the unit box at all.

    This is an independently callable, model-free unit -- tests exercise it with synthetic tensors.
    """
    normed = _l2_normalize(class_embeds, axis=1)
    ious = _iou_with_unit_box(boxes_cxcywh)
    max_iou = float(ious.max()) if ious.size else 0.0
    if max_iou <= 0.0:
        best = int(np.argmax(boxes_cxcywh[:, 2] * boxes_cxcywh[:, 3]))
        return np.asarray(normed[best], dtype=np.float32)
    selected = np.nonzero(ious >= iou_frac * max_iou)[0]
    # Distinctiveness: the covering patch LEAST similar to the mean patch embedding is the object,
    # not the background. sims is minimized -> the most distinctive covering patch is chosen.
    mean_embed = _l2_normalize(normed.mean(axis=0)[None, :], axis=1)[0]
    sims = np.asarray(normed[selected] @ mean_embed, dtype=np.float32)
    best = int(selected[int(np.argmin(sims))])
    return np.asarray(normed[best], dtype=np.float32)


def boxes_to_pixels(
    boxes_cxcywh: npt.NDArray[np.float32],
    orig_w: int,
    orig_h: int,
) -> list[BBox | None]:
    """Map normalized ``(cx, cy, w, h)`` patch boxes to scene-pixel :class:`BBox` (or ``None``).

    OWLv2 pads bottom-right to a square of side ``max(orig_w, orig_h)`` before resizing, so a
    normalized coordinate maps to a scene pixel by a plain multiply by that side -- no pad offset.
    A box that clips to sub-pixel size is returned as ``None`` (a degenerate 0-area detection is
    not a box), so the caller can drop it while keeping per-patch alignment by index.
    """
    side = float(max(orig_w, orig_h))
    out: list[BBox | None] = []
    for cx, cy, w, h in boxes_cxcywh:
        x1 = round((float(cx) - float(w) / 2.0) * side)
        y1 = round((float(cy) - float(h) / 2.0) * side)
        x2 = round((float(cx) + float(w) / 2.0) * side)
        y2 = round((float(cy) + float(h) / 2.0) * side)
        x1 = max(0, min(x1, orig_w))
        y1 = max(0, min(y1, orig_h))
        x2 = max(0, min(x2, orig_w))
        y2 = max(0, min(y2, orig_h))
        if x2 - x1 < 1 or y2 - y1 < 1:
            out.append(None)
        else:
            out.append(BBox(x=x1, y=y1, w=x2 - x1, h=y2 - y1))
    return out


def tile_boxes(orig_w: int, orig_h: int, tile_size: int, overlap_frac: float) -> list[BBox]:
    """Cover ``(orig_w, orig_h)`` with ``tile_size``-square windows overlapping by ``overlap_frac``.

    Returns a **single whole-image tile** when the image already fits within ``tile_size`` on both
    axes -- the tiled and untiled code paths are then identical, so tiling is a strict no-op below
    the threshold. Otherwise, tiles stride at ``tile_size * (1 - overlap_frac)`` along each axis (a
    last tile is always flush with the far edge, even if that means a bigger-than-stride overlap
    with its neighbour, so no scene pixel is ever left uncovered). Pure arithmetic -- no ONNX
    session, no weight -- so CI gates it.
    """
    if orig_w <= tile_size and orig_h <= tile_size:
        return [BBox(x=0, y=0, w=orig_w, h=orig_h)]
    stride = max(1, round(tile_size * (1.0 - overlap_frac)))

    def _starts(extent: int) -> list[int]:
        if extent <= tile_size:
            return [0]
        starts = list(range(0, extent - tile_size + 1, stride))
        if starts[-1] != extent - tile_size:
            starts.append(extent - tile_size)
        return starts

    return [
        BBox(x=x, y=y, w=min(tile_size, orig_w - x), h=min(tile_size, orig_h - y))
        for y in _starts(orig_h)
        for x in _starts(orig_w)
    ]


def _empty_or_error(
    outcome: SearchOutcome,
    note: str,
    *,
    threshold: float | None,
    latency: LatencyBreakdown,
    metrics: dict[str, float],
    proposals: tuple[BBox, ...] = (),
    error: MethodError | None = None,
    candidates: tuple[Candidate, ...] = (),
) -> SearchResult:
    """Build an honest zero-match result (``EMPTY`` or ``ERROR``) carrying a note saying why."""
    return SearchResult(
        method="owlv2-oneshot",
        method_version=_METHOD_VERSION,
        outcome=outcome,
        matches=(),
        latency=latency,
        threshold_applied=threshold,
        candidates=candidates,
        diagnostics=Diagnostics(
            notes=(note,),
            metrics=metrics,
            proposals=proposals if proposals else None,
        ),
        error=error,
    )


@register_method(
    name="owlv2-oneshot",
    description="OWLv2 image-conditioned one-shot detection: scene patches scored by cosine to the "
    "exemplar query embedding.",
    version=_METHOD_VERSION,
    config_model=Owlv2OneshotConfig,
)
def search(
    image: npt.NDArray[np.uint8],
    exemplar: ExemplarBox,
    config: BaseModel,
) -> SearchResult:
    """Find every instance of ``exemplar`` by OWLv2 image-conditioned detection."""
    # The registry types config as BaseModel (method-agnostic); the registered config_model
    # guarantees the concrete type. Narrow once here and fail loudly if the contract is violated.
    if not isinstance(config, Owlv2OneshotConfig):
        raise TypeError(
            f"owlv2-oneshot.search requires an Owlv2OneshotConfig, got {type(config).__name__}"
        )

    t_start = perf_counter()

    # An absent weight => an honest model_unavailable error (outcome=error), never a raise -- so
    # the sample renderer and the API degrade rather than crash.
    inferencer = _get_inferencer()
    if inferencer is None:
        zero = LatencyBreakdown(preprocess_ms=0.0, inference_ms=0.0, postprocess_ms=0.0)
        return _empty_or_error(
            SearchOutcome.ERROR,
            f"weight absent: {_OWLV2_KEY}; run `pixi run -e export export-owlv2`. No search ran.",
            threshold=None,
            latency=zero,
            metrics={},
            error=MethodError(kind="model_unavailable", message=f"missing weight: {_OWLV2_KEY}"),
        )

    orig_h, orig_w = int(image.shape[0]), int(image.shape[1])

    # 1. Encode the exemplar crop as a query image (same vision graph as the scene). If
    #    config.rotation_invariant, also encode the crop rotated 90/180/270 degrees -- OWLv2 patch
    #    embeddings are not rotation-equivariant, so a mirrored/rotated scene instance (common on
    #    floor plans) may not match the exemplar's own orientation.
    crop_box = exemplar.box.clipped_to(orig_w, orig_h)
    crop = np.ascontiguousarray(
        image[crop_box.y : crop_box.y2, crop_box.x : crop_box.x2], dtype=np.uint8
    )
    rotated_crops = (
        (crop, np.rot90(crop, 1), np.rot90(crop, 2), np.rot90(crop, 3))
        if config.rotation_invariant
        else (crop,)
    )

    # 2. Select one query embedding PER rotation: the most-distinctive covering patch (HF
    #    heuristic) run independently on each rotated crop.
    query_ms = 0.0
    query_embeddings: list[npt.NDArray[np.float32]] = []
    for rotated_crop in rotated_crops:
        t_query = perf_counter()
        query = inferencer.embed_image(np.ascontiguousarray(rotated_crop, dtype=np.uint8))
        query_ms += (perf_counter() - t_query) * 1000.0
        query_embeddings.append(
            select_query_embedding(query.class_embeds, query.boxes_cxcywh, config.query_iou_frac)
        )

    # 3. Encode the scene. If config.tile_large_scenes, split a scene wider or taller than OWLv2's
    #    native 960px into overlapping 960px tiles and encode each separately -- OWLv2's fixed 960
    #    input otherwise downscales a large canvas below its effective resolution (a small symbol
    #    occupies very few of the 60x60 patches). A scene that already fits within 960x960 is a
    #    single "tile" == the whole image, so this is byte-identical to the untiled path below that
    #    size (tile_boxes returns one whole-image tile). Each tile's boxes are mapped to scene
    #    pixels HERE, offset by the tile origin, so later steps can treat all tiles as one flat
    #    candidate set; tile_area tracks each patch's SOURCE tile area (not the whole scene) for
    #    the per-tile whole-frame filter in step 5.
    scene_tiles = (
        tile_boxes(orig_w, orig_h, _TILE_SIZE, _TILE_OVERLAP_FRAC)
        if config.tile_large_scenes
        else [BBox(x=0, y=0, w=orig_w, h=orig_h)]
    )
    t_target = perf_counter()
    tile_class_embeds: list[npt.NDArray[np.float32]] = []
    tile_logit_shift: list[npt.NDArray[np.float32]] = []
    tile_logit_scale: list[npt.NDArray[np.float32]] = []
    pixel_boxes: list[BBox | None] = []
    tile_area: list[float] = []
    for tile in scene_tiles:
        tile_image = np.ascontiguousarray(image[tile.y : tile.y2, tile.x : tile.x2], dtype=np.uint8)
        tile_embed = inferencer.embed_image(tile_image)
        tile_class_embeds.append(tile_embed.class_embeds)
        tile_logit_shift.append(tile_embed.logit_shift)
        tile_logit_scale.append(tile_embed.logit_scale)
        for local_box in boxes_to_pixels(tile_embed.boxes_cxcywh, tile.w, tile.h):
            global_box = (
                None
                if local_box is None
                else BBox(
                    x=local_box.x + tile.x, y=local_box.y + tile.y, w=local_box.w, h=local_box.h
                )
            )
            pixel_boxes.append(global_box)
            tile_area.append(float(tile.area))
    target_ms = (perf_counter() - t_target) * 1000.0
    class_embeds_all = np.concatenate(tile_class_embeds, axis=0)
    logit_shift_all = np.concatenate(tile_logit_shift, axis=0)
    logit_scale_all = np.concatenate(tile_logit_scale, axis=0)

    # 4. Cosine similarity of every scene patch to EACH query embedding (both sides L2-normalized,
    #    so the matmul IS cosine in [-1, 1]; a plain NumPy matmul -- no FAISS, one image), then take
    #    the per-patch MAX across rotations (a no-op max over one embedding when rotation_invariant
    #    is off). Then recalibrate with OWLv2's own learned, query-independent per-patch
    #    logit_shift/logit_scale (HF's Owlv2ClassPredictionHead formula) -- unlike a single global
    #    scale/shift, this CAN re-rank patches, which is the point: it is meant to suppress generic
    #    high-cosine patches that are not actually the object (see the module docstring / the
    #    floor-plans improvement report). Every score used downstream is calibrated, never raw
    #    cosine.
    target_norm = _l2_normalize(class_embeds_all, axis=1)
    per_rotation_cosine = [
        np.asarray(target_norm @ qe, dtype=np.float32) for qe in query_embeddings
    ]
    cosine_all = np.maximum.reduce(per_rotation_cosine).astype(np.float32)
    scores_all = np.asarray((cosine_all + logit_shift_all) * logit_scale_all, dtype=np.float32)

    # 5. Drop degenerate boxes AND the generic whole-frame boxes (area > max_box_area_frac of the
    #    box's SOURCE TILE, not the whole scene -- OWLv2 emits one whole-frame box per forward pass,
    #    so with tiling that is one per tile, each sized to ITS tile, not the scene). Left in, it
    #    scores highest and anchors the threshold / dominates NMS. Keep alignment with scores_all.
    boxes: list[BBox] = []
    kept_scores: list[float] = []
    for i, pixel_box in enumerate(pixel_boxes):
        if pixel_box is not None and pixel_box.area <= config.max_box_area_frac * tile_area[i]:
            boxes.append(pixel_box)
            kept_scores.append(float(scores_all[i]))
    if not boxes:
        latency = _split_latency(t_start, query_ms, target_ms)
        return _empty_or_error(
            SearchOutcome.EMPTY,
            "OWLv2 returned no non-degenerate boxes for this scene.",
            threshold=None,
            latency=latency,
            metrics={"n_patches": float(scores_all.size), "n_valid": 0.0},
        )
    scores = np.asarray(kept_scores, dtype=np.float32)

    # 6. Calibrate/threshold. self-similarity anchors the cut to the exemplar's OWN self-match
    #    score (self_score * retain_frac): OWLv2 cosine is compressed near 1.0 and not bimodal, so
    #    gmm degenerates to an unstable ratio cut. self_score is the top score among boxes
    #    overlapping the exemplar, falling back to the global max. "fixed" pins score_threshold.
    self_overlap = [
        float(scores[i])
        for i, box in enumerate(boxes)
        if box.iou(exemplar.box) >= _EXEMPLAR_SELF_IOU
    ]
    self_score = max(self_overlap) if self_overlap else float(scores.max())
    strategy: calibration.CalibrationStrategy = (
        "fixed" if config.score_threshold is not None else config.calibration
    )
    calib = calibration.calibrate(
        scores.astype(np.float64),
        strategy=strategy,
        fixed_threshold=config.score_threshold,
        self_score=self_score,
        retain_frac=config.retain_frac,
        seed=config.seed,
    )
    threshold = calib.threshold

    # 7. Split into accepted (matches) and sub-threshold candidates (EVAL-08), then NMS the accepted
    #    set to collapse the several neighbouring patches OWLv2 fires on one object. METHOD-12:
    #    every accepted box survives NMS -- there is no single-best cut. Ties sort (-score, y, x).
    ordered = sorted(
        range(len(boxes)),
        key=lambda i: (-float(scores[i]), boxes[i].y, boxes[i].x),
    )
    candidates = tuple(
        Candidate(box=boxes[i], score=float(scores[i])) for i in ordered[: config.max_candidates]
    )
    accepted = [i for i in ordered if float(scores[i]) > threshold]
    kept_local = nms.nms(
        [boxes[i] for i in accepted],
        [float(scores[i]) for i in accepted],
        config.nms_iou,
    )
    kept = [accepted[j] for j in kept_local]
    matches = _build_matches(
        [boxes[i] for i in kept],
        [float(scores[i]) for i in kept],
        exemplar.box,
    )

    # 8. Diagnostics carry the top candidate boxes (the UI's debug overlay) and the pre-NMS accepted
    #    count; latency attributes the query encode vs the scene encode SEPARATELY (EVAL-11).
    latency = _split_latency(t_start, query_ms, target_ms)
    proposal_boxes = tuple(c.box for c in candidates)
    metrics: dict[str, float] = {
        "threshold": threshold,
        "n_patches": float(scores_all.size),
        "n_valid": float(len(boxes)),
        "n_accepted_pre_nms": float(len(accepted)),
        "n_matches": float(len(matches)),
        "n_candidates": float(len(candidates)),
        "collapsed_by_nms": float(len(accepted) - len(matches)),
        "score_max": float(scores.max()),
        "score_mean": float(scores.mean()),
        "query_ms": query_ms,
        "target_ms": target_ms,
    }
    notes = (
        f"calibration[{calib.strategy}]: {calib.reason}",
        (
            f"{len(boxes)} valid patch box(es); {len(accepted)} cleared cosine threshold "
            f"{threshold:.4f}; NMS(iou={config.nms_iou}) collapsed {len(accepted) - len(matches)} "
            f"to {len(matches)} match(es)"
        ),
        (
            f"latency: scene encode {target_ms:.1f}ms "
            f"{'dominates' if target_ms >= query_ms else 'trails'} "
            f"query encode {query_ms:.1f}ms (EVAL-11)"
        ),
    )

    if not matches:
        return _empty_or_error(
            SearchOutcome.EMPTY,
            (
                f"no scene patch cleared the calibrated threshold {threshold:.4f} "
                f"(best {float(scores.max()):.4f}); {calib.strategy}: {calib.reason}"
            ),
            threshold=threshold,
            latency=latency,
            metrics=metrics,
            proposals=proposal_boxes,
            candidates=candidates,
        )

    return SearchResult(
        method="owlv2-oneshot",
        method_version=_METHOD_VERSION,
        outcome=SearchOutcome.OK,
        matches=matches,
        latency=latency,
        threshold_applied=threshold,
        candidates=candidates,
        diagnostics=Diagnostics(notes=notes, metrics=metrics, proposals=proposal_boxes),
    )


def _split_latency(t_start: float, query_ms: float, target_ms: float) -> LatencyBreakdown:
    """Build the latency breakdown: the two encodes are inference; everything else postprocess."""
    inference_ms = query_ms + target_ms
    postprocess_ms = max(0.0, (perf_counter() - t_start) * 1000.0 - inference_ms)
    return LatencyBreakdown(
        preprocess_ms=0.0, inference_ms=inference_ms, postprocess_ms=postprocess_ms
    )


def _build_matches(
    boxes: list[BBox],
    scores: list[float],
    exemplar: BBox,
) -> tuple[Match, ...]:
    """Turn kept detections into Matches, labelling the exemplar's own region is_exemplar.

    The exemplar is part of the scene it is searched in, so one kept detection overlaps it; that
    detection is labelled rather than dropped (which understates recall) or counted silently as a
    discovery (which overstates the method), per METHOD-04c. Matches are returned in the canonical
    ``(-score, y, x)`` order. OWLv2 is an appearance detector, so ``transform`` stays ``None``.
    """
    exemplar_idx: int | None = None
    best_iou = _EXEMPLAR_IOU
    for i, box in enumerate(boxes):
        overlap = box.iou(exemplar)
        if overlap >= best_iou:
            best_iou = overlap
            exemplar_idx = i

    order = sorted(range(len(boxes)), key=lambda i: (-scores[i], boxes[i].y, boxes[i].x))
    return tuple(
        Match(box=boxes[i], score=scores[i], is_exemplar=(i == exemplar_idx)) for i in order
    )
