"""Method 3 -- DINOv2 dense-token prototype matching (the general-purpose default).

What it does
------------
Embed the exemplar crop and the whole scene into DINOv2 dense patch tokens, **mean-pool** the
crop tokens into a single prototype vector, and score every scene location by the **cosine
similarity** of its token to that prototype. Threshold the resulting similarity map, run
connected components, and turn each component into a box. Where NCC (Method 1) correlates raw
intensities and so misses instances under pose or lighting change, DINOv2 features are
appearance-robust, so this is the method that finds "the same object, differently posed".

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
- **The exemplar crop is taken from the original BGR scene** and embedded on its own; its patch
  tokens are mean-pooled into the prototype. Mean-pooling loses part structure (a known
  limitation for articulated objects -- see the backlog), but it is the readable v1.

Post-processing (exact)
-----------------------
- **Both the prototype and every scene token are L2-normalized before the dot product.** The dot
  of two L2-normalized vectors is cosine similarity; skipping the normalization yields an
  unnormalized dot dominated by token *magnitude* (DINOv2's high-norm background artifact tokens
  in particular), which is a different, wrong quantity. Order matters: normalize, then dot.
- **The similarity MAP is bilinearly upsampled, not the tokens.** Upsampling the ``(gh, gw)``
  cosine map to pixel resolution -- using the scale factors :meth:`DINOv2Inferencer.dense_tokens`
  returns so token centres land on their true pixels -- is cheap and correct; upsampling the
  384-d tokens first would be 384x the work for the same map.
- **``connectedComponentsWithStats`` label 0 is the background and is skipped explicitly.**
  Emitting label 0 would be a single full-image false positive. Each remaining component above
  ``min_component_area`` becomes one box; its score is the peak similarity inside it.
- **Threshold via ``common.calibration``** (default ``gmm``): absolute cosine thresholds do not
  transfer across images for deep features, which is exactly what the calibration layer is for.
  The score distribution handed to the calibrator is the token-resolution similarity map.
- **Sub-threshold candidates are retained** (EVAL-08): components found at a floor
  ``candidate_margin`` below the accept threshold are logged with their raw similarity so an
  offline threshold sweep can rebuild a PR curve. Components clearing the threshold become
  matches; there is **no single-best short-circuit** (METHOD-12) -- connected components returns
  as many instances as the image contains.

Known failure modes
--------------------
- **Stride-14 coarseness.** Even with high-res inference and upsampling, boxes are quantised to
  ~14 px patches; a small object a few pixels across is a single fuzzy token. This is the
  headline limitation and the reason for the sliding-window / FeatUp backlog items.
- **Very large scenes hit the resolution cap.** Above ``scene_max_side`` the scene is downscaled
  (and it is logged), so effective localisation on a 6000 px image is coarser than the cap
  suggests.
- **A single mean-pooled prototype loses part structure.** For articulated or non-compact
  objects (the basketball-player frames) a many-to-many token similarity would do better; this
  is a deliberate v1 simplification, not a bug.
- **Weights absent.** DINOv2 weights are gitignored and fetched by ``pixi run fetch-models``.
  With no weight present the method returns ``outcome=error`` with a ``model_unavailable`` note
  rather than raising, so the sample renderer and the API degrade honestly.

ROBUSTNESS BACKLOG
------------------
Deferred deliberately (mirrored in ``docs/methods/dino-dense.md`` and
``docs/ROBUSTNESS-BACKLOG.md``); none is built in this phase:

- **Sliding-window backbone inference** for very large scenes, so localisation no longer
  degrades at the resolution cap.
- **Learned feature upsampling (FeatUp)** to recover sub-patch localisation from the stride-14
  grid without a full high-res forward pass.
- **SAM-based box refinement** -- snap each coarse component box to the nearest segment mask.
- **Many-to-many token similarity with spatial aggregation** instead of a single mean-pooled
  prototype -- measurably better for articulated objects like the basketball frames.
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
from pydantic import BaseModel, ConfigDict, Field

from object_search.inference import DINOv2Inferencer, models
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
_PROVIDERS = ["CPUExecutionProvider"]
# The exemplar's own prototype is L2-normalized, so its cosine similarity with itself is exactly
# 1.0. That is the self-match score the "self-similarity" calibrator anchors on.
_SELF_MATCH_SCORE = 1.0
# A component whose box overlaps the exemplar by at least this is the exemplar's own region,
# labelled is_exemplar rather than dropped or counted as a fresh discovery (METHOD-04c).
_EXEMPLAR_IOU = 0.5
_EPS = 1e-12  # guards a zero-norm division; a genuinely zero token is background, not a match


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
    calibration: Literal["fixed", "self-similarity", "ratio", "gmm"] = Field(
        default="gmm",
        description=(
            "How the accept threshold is chosen when `threshold` is None. gmm fits two modes "
            "(foreground/background) on the similarity map; absolute cosine thresholds do not "
            "transfer across images for deep features, which is why calibration is the default."
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
            "Minimum connected-component area in pixels; smaller blobs are dropped as noise."
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


def _extract_components(
    sim_full: npt.NDArray[np.float32],
    floor: float,
    min_area: int,
    cap_scale: float,
    orig_w: int,
    orig_h: int,
) -> list[_Component]:
    """Threshold at ``floor``, run connected components, and return boxes in ORIGINAL pixels.

    ``cv2.connectedComponentsWithStats`` numbers the background 0 and each foreground blob 1..N.
    **Label 0 is skipped explicitly** -- it is the whole non-matching background and emitting it
    would be one image-sized false positive. Each remaining component's stats bbox is scaled from
    the (capped) inference resolution back to original scene pixels by dividing out ``cap_scale``.
    """
    mask = (sim_full >= floor).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    components: list[_Component] = []
    for label in range(1, count):  # 1.. : SKIP label 0, the background component
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
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

    # 2. Embed the exemplar crop, mean-pool its patch tokens into a prototype, L2-normalize it.
    ex = exemplar.box
    crop = np.ascontiguousarray(image[ex.y : ex.y2, ex.x : ex.x2], dtype=np.uint8)
    crop_grid, _, _ = inferencer.dense_tokens(crop)
    prototype = _prototype_from_grid(crop_grid)

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

    t_infer = perf_counter()
    grid, scale_x, scale_y = inferencer.dense_tokens(capped)
    inference_ms = (perf_counter() - t_infer) * 1000.0
    gh, gw = grid.shape[0], grid.shape[1]

    # 4. Cosine similarity = normalized prototype . normalized token grid -> a (gh, gw) map.
    sim_token = _similarity_map(grid, prototype)

    # 5. Bilinearly upsample the MAP (not the tokens) to pixel resolution using the scale factors.
    sim_full = _upsample_similarity(sim_token, scale_x, scale_y)

    # 6. Calibrate the accept threshold from the token-resolution similarity distribution.
    #    self_score is 1.0 (the L2-normalized prototype's self-cosine); "fixed" is used whenever
    #    the caller pinned config.threshold. The calibrator returns its reasoning too.
    strategy: calibration.CalibrationStrategy = (
        "fixed" if config.threshold is not None else config.calibration
    )
    calib = calibration.calibrate(
        sim_token.reshape(-1),
        strategy=strategy,
        fixed_threshold=config.threshold,
        self_score=_SELF_MATCH_SCORE,
        retain_frac=config.retain_frac,
        seed=config.seed,
    )
    threshold = calib.threshold

    # 7. Threshold -> connected components (label 0 skipped). Components are found at a floor
    #    candidate_margin BELOW the accept threshold so sub-threshold near-misses are captured;
    #    the split into matches vs candidates happens on the raw score in step 8.
    candidate_floor = max(-1.0, threshold - config.candidate_margin)
    components = _extract_components(
        sim_full, candidate_floor, config.min_component_area, cap_scale, orig_w, orig_h
    )

    # 8. Split into matches and sub-threshold candidates (EVAL-08). The top max_candidates
    #    components (ranked by raw similarity) are kept as Candidates regardless of the threshold,
    #    so an offline sweep can rebuild a PR curve. Components whose score clears the threshold
    #    become Matches. METHOD-12: EVERY clearing component survives -- there is no single-best
    #    short-circuit; connected components returns as many instances as the image contains.
    ordered = sorted(components, key=lambda c: (-c.score, c.box.y, c.box.x))
    candidates = tuple(
        Candidate(box=c.box, score=c.score) for c in ordered[: config.max_candidates]
    )
    accepted = [c for c in ordered if c.score > threshold]
    matches = _build_matches(accepted, ex)

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
        "n_components": float(len(components)),
        "n_candidates": float(len(candidates)),
        "n_matches": float(len(matches)),
        "sim_max": float(sim_token.max()),
        "sim_mean": float(sim_token.mean()),
    }
    notes = (
        f"calibration[{calib.strategy}]: {calib.reason}",
        (
            f"kept {len(matches)} match(es) from {len(components)} component(s) on a {gh}x{gw} "
            f"token grid; threshold {threshold:.4f} on cosine similarity"
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
                f"(best {float(sim_token.max()):.4f}); {calib.strategy}: {calib.reason}"
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
