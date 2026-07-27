"""Method 5 -- propose-then-retrieve (class-agnostic proposals + DINOv2 region embeddings).

What it does
------------
Rather than scoring a dense similarity map (Method 3), this method first asks a class-agnostic
segmenter (**FastSAM**, "everything mode") for a few hundred region proposals, then embeds each
proposed region with the **same DINOv2 backbone as Method 3**, and ranks those regions by the
**cosine similarity** of their embedding to the exemplar's embedding. The accepted regions are the
matches. Its selling point over ``dino-dense`` is **boundary alignment**: FastSAM proposals hug
object edges, so the returned boxes are tight rectangles around real objects instead of the blobby
connected components a stride-14 similarity map produces. That claim is a *measured number* -- a
mean-IoU-against-ground-truth test on the chipset, not an impression (Phase 7 success criterion 1).

Two independently callable units -- the Milestone 2 seam
-------------------------------------------------------
This is the defining constraint of Phase 7, and the reason Milestone 2 (marker-conditioned
proposals) can add an exploration rather than fork the app:

- :func:`~object_search.search.proposals.propose` -- the proposal stage (built in 07-01), and
- :func:`embed_regions` -- the embedding stage, built here.

``search()`` **composes** the two and does nothing they cannot do alone. Neither unit reaches into
the other's internals; a seam test calls each directly, not through ``search()``.

One DINOv2 model, shared with Method 3
--------------------------------------
The embedding stage deliberately reuses Method 3's module-level :class:`DINOv2Inferencer` singleton
(``dino_dense._get_inferencer``) rather than constructing a second one. That is CONTEXT locked
decision 6: **one model download, one preprocessing contract**. There is structurally no second
DINOv2 loader in the codebase, and a test asserts ``embed_regions`` routes through that one
singleton. Reaching across to a sibling method's accessor is the single intentional cross-method
reference in the repo, made here because the backbone genuinely is shared state, not a per-method
concern.

Pre-processing (exact)
----------------------
Neither backbone's preprocessing is re-derived here; both are written down once in their
inferencer docstrings and reused verbatim:

- **FastSAM proposals** -- ``images`` f32 NCHW, **RGB**, scale ``1/255``, **NO mean/std**,
  **letterbox** to 1024x1024 with fill 114. The YOLOv8-seg output decoding (transpose
  ``[37, anchors]`` -> ``[anchors, 37]``, split into 4 box + 1 conf + 32 mask coeffs, confidence
  filter, box NMS at the loose ``iou_thres=0.9``, undo the letterbox) lives in
  :func:`~object_search.inference.fastsam.decode_fastsam`. "Everything mode" wants overlapping
  proposals, so the FastSAM-internal NMS is deliberately loose; over-segmentation is collapsed by
  a **second** NMS *after* retrieval, below.
- **DINOv2 region embeddings** -- ``pixel_values`` f32 NCHW, **RGB**, scale ``1/255``, mean
  ``[0.485, 0.456, 0.406]``, std ``[0.229, 0.224, 0.225]``, **bicubic** resize with
  **snap-to-multiple(14)** and **NO centre-crop**. Each proposal's box crop is embedded on its own.

Post-processing (exact)
-----------------------
- **Mean-pool then L2-normalize**, in that order, per region: the region embedding is the mean of
  its patch tokens, normalized *once* so its self-cosine is exactly ``1.0``. Normalizing per token
  before pooling would weight every token equally regardless of magnitude -- a different, wrong
  quantity (the same DINOv2 high-norm-artifact trap Method 3 documents).
- **Cosine nearest-neighbour is a plain NumPy matmul** of the (N, D) normalized proposal matrix
  against the normalized exemplar vector -- **no FAISS**. For a few hundred proposals in one image
  a FAISS index is pure dependency cost; the embedding matrix is shaped ``(N, D)`` so a FAISS index
  slots in unchanged when corpus-scale search arrives (backlog).
- **Threshold** via ``common.calibration`` (``gmm`` by default, or a fixed
  ``retrieval_threshold``), **clamped to an absolute ``similarity_floor``**: the gmm gives a
  per-image adaptive cut between the "matches" mode (the exemplar's own region scores ~1.0) and the
  background mode, but the cut is never allowed to sink below ``similarity_floor`` (a cosine to the
  exemplar, whose self-cosine is 1.0). The floor does two things a bare two-mode fit cannot. (1) It
  stops a *low* gmm cut from admitting moderate-cosine background -- the dominant precision leak on
  cluttered scenes. (2) It rescues the **degenerate single-mode** case: a uniform lattice of
  identical instances scores ~1.0 with no second mode, where the gmm's ``ratio`` fallback lands the
  cut *at* the max score and the strict ``> threshold`` then rejects **every true match** (recall 0
  -- the worst possible failure for a repeated-instance finder). In that case the floor alone
  decides, so all the near-1.0 regions are accepted; an image with no other instances scores below
  the floor and is correctly rejected. The floor is a distribution-independent *anchor*, not a
  label-fit cut: the same value runs on every image and AP stays threshold-free (it sweeps the full
  candidate log).
- **Post-retrieval NMS at ``nms_iou`` (0.3)** collapses SAM over-segmentation -- one object that
  FastSAM split into several *partially*-overlapping proposals (shifted/partial boxes that each
  embed well) would otherwise produce duplicate detections. Tighter than the classical 0.5 because
  "everything mode" emits many such partial proposals; 0.3 folds them in without merging genuinely
  distinct instances. The proposal count is recorded in diagnostics so the over-segmentation is
  visible, not hidden.
- **Sub-threshold candidates are retained** (EVAL-08) and **every accepted region survives** into
  matches after NMS -- there is **no single-best short-circuit** (METHOD-12).

Latency (EVAL-11)
-----------------
The proposal stage dominates -- that is a *finding*, not a defect, and it is exactly why the
latency breakdown must attribute **proposal time and embedding time separately**. Both are model
forward passes, so ``inference_ms`` carries their sum, but ``diagnostics.metrics`` reports
``proposal_ms`` and ``embedding_ms`` as distinct numbers (and a note states which dominates), so
the finding is legible rather than buried in one total.

Known failure modes
-------------------
- **SAM over-segmentation.** One object becomes several proposals; each embeds well, so without the
  post-retrieval NMS they surface as duplicate detections. NMS at ``nms_iou`` is what collapses
  them, and the pre-NMS proposal count in diagnostics is how the practitioner sees it happening.
- **The raw box crop includes background.** A tight box around a non-convex object still embeds
  some surrounding pixels; the FastSAM mask is available (``return_masks``) to mask that background
  out, which is a cheap, likely-real win deferred to the backlog.
- **Weights absent.** FastSAM (AGPL-3.0) and DINOv2 weights are gitignored and fetched by
  ``pixi run fetch-models``. With either absent the method returns ``outcome=error`` with a
  ``model_unavailable`` note rather than raising, so the sample renderer and the API degrade
  honestly.
- **MobileSAM is not a working second backend** (documented deviation -- see the method doc and
  CONTEXT decision 5): the ONNX SAM decoder takes one prompt per call, so "everything mode" is
  ~1024 sequential calls plus a ported automatic-mask generator -- a phase of work, not a config
  swap. FastSAM is the single Milestone 1 backend; the ``ProposalBackend`` protocol keeps the seam
  open for MobileSAM later.

Licence
-------
FastSAM is **AGPL-3.0** and the exported ``.onnx`` embeds that licence string. Private local use
triggers nothing; **publishing this repo or network-exposing the FastAPI app fires AGPL §13.** This
is a real constraint on how the repo may later be shared (LICENSES.md, the model spec, and the
method doc all record it), not a footnote.

ROBUSTNESS BACKLOG
------------------
Deferred deliberately (mirrored in ``docs/methods/propose-retrieve.md`` and
``docs/ROBUSTNESS-BACKLOG.md``); none is built in this phase:

- **FAISS index for corpus-scale retrieval** -- unnecessary for a few hundred proposals in one
  image; the ``(N, D)`` embedding matrix is shaped so it slots in when corpus search arrives.
- **Background-masked region embedding** -- embed the FastSAM mask interior rather than the raw box
  crop. **Measured (2026-07-25), deferred.** *Pixel*-masking (fill background with the ImageNet
  mean) HURT: it crashed synthetic recall 0.94 -> 0.65, because objects that fill their box gain
  artificial fill edges and coarse-mask errors corrupt the descriptor. *Token*-masking (pool only
  DINOv2 tokens whose patch centre is inside the mask) gave a real but small gain (+0.006 macro-F1,
  helping the weakest cluttered/varied regimes) at ~2x latency (the per-proposal mask upsample from
  ``return_masks=True``) and it erodes the clean ``embed_regions`` seam (which by design knows
  nothing of masks). Not worth the cost yet; revisit if cluttered precision becomes the priority.
- **Proposal filtering by an exemplar size/aspect prior** -- drop proposals whose shape cannot
  match the exemplar before embedding. **Measured (2026-07-25), rejected.** An area-ratio gate of
  [0.25, 4]x exemplar-area crashed textured recall (varied 0.93 -> 0.59): the true instances
  legitimately span a range of scales (that is the regime's whole point), so a size prior discards
  them along with the clutter. A size prior fights the scale-invariance this method exists to
  provide; do not add it.
- **Multi-crop / test-time augmentation embeddings** for pose-robust region descriptors.
- **Alternative proposal sources (RPN, selective search)** for images where SAM over-segments.
- **MobileSAM everything-mode** with a ported ``SamAutomaticMaskGenerator`` as a second backend.
"""

from __future__ import annotations

from collections.abc import Sequence
from time import perf_counter
from typing import Literal

import cv2
import numpy as np
import numpy.typing as npt
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from object_search.inference import FastSAMConfig, models
from object_search.inference.dinov2 import DINOV2_EMBED_DIM, DINOV2_PATCH, DINOv2Inferencer
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
from object_search.search import dino_dense
from object_search.search.common import calibration, nms
from object_search.search.proposals import Proposal, default_backend, propose
from object_search.search.registry import register_method

# -- Method-level constants (properties of the METHOD, not of a query, so not config fields) --

_METHOD_VERSION = "1.0.0"
_FASTSAM_KEY = "fastsam-s"  # the MODEL_REGISTRY key for the proposal backend
# DINOv2 is reused from Method 3 via dino_dense._get_inferencer(); its model key ("dinov2-small")
# is defined ONCE, there. This method deliberately does not name a second dinov2 model.
_PROVIDERS = ["CPUExecutionProvider"]  # pin the CPU EP so a run is bit-identical machine to machine
_EXEMPLAR_IOU = 0.5  # a match overlapping the exemplar by >= this is the exemplar's own region
_EPS = 1e-12  # guards a zero-norm division; a genuinely zero embedding is background, not a match


class ProposeRetrieveConfig(BaseModel):
    """Frozen config for :func:`search`; its JSON Schema drives the UI form (one source).

    Every field carries a ``description`` because that text becomes the form's help string --
    this is the single place it is written, so it must be written here.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    proposal_backend: Literal["fastsam"] = Field(
        default="fastsam",
        description=(
            "Which class-agnostic proposal backend to use. Only 'fastsam' is implemented in "
            "Milestone 1; MobileSAM is a documented deviation (its ONNX decoder takes one prompt "
            "per call, so everything-mode is ~1024 calls plus a ported mask generator)."
        ),
    )
    proposal_conf: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description=(
            "FastSAM objectness threshold: keep proposals whose class-0 confidence exceeds this. "
            "FastSAM's default is 0.4. Lower surfaces more (and smaller) regions."
        ),
    )
    retrieval_threshold: float | None = Field(
        default=None,
        description=(
            "Fixed accept threshold on the cosine similarity between a proposal embedding and the "
            "exemplar embedding. None => calibrate with a two-mode gmm (absolute cosine cuts do "
            "not transfer across images for deep features)."
        ),
    )
    nms_iou: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description=(
            "Post-retrieval NMS IoU. A later accepted box overlapping a kept one by MORE than this "
            "is suppressed -- this is what collapses FastSAM over-segmentation (one object split "
            "into several partially-overlapping proposals) into a single detection. Tighter than a "
            "classical 0.5 because 'everything mode' emits many shifted/partial proposals of the "
            "same object that only partly overlap the true box; 0.3 folds those in without merging "
            "genuinely distinct instances."
        ),
    )
    similarity_floor: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description=(
            "Absolute cosine floor on the calibrated accept threshold (ignored when a fixed "
            "retrieval_threshold is given). The gmm cut may raise the threshold ABOVE this but "
            "never below it: the floor stops a low gmm cut from admitting background, AND rescues "
            "the degenerate single-mode case -- a uniform lattice of identical instances scores "
            "~1.0 with no second mode, where the bare gmm/ratio fallback lands the cut at the max "
            "and rejects every true match. Anchored on the exemplar self-cosine (=1.0 for an "
            "L2-normalized embedding), so a proposal is accepted when it is at least this fraction "
            "as similar to the exemplar as the exemplar is to itself."
        ),
    )
    max_candidates: int = Field(
        default=50,
        ge=1,
        description=(
            "How many top-scoring proposals (with raw scores) to keep as sub-threshold candidates "
            "for an offline PR sweep (EVAL-08), regardless of the threshold."
        ),
    )
    seed: int = Field(
        default=0,
        ge=0,
        description="random_state for the gmm calibrator (its only genuinely stochastic step).",
    )


# -- the reused DINOv2 backbone, and the FastSAM proposal backend (module-level lazy singletons) --
# DINOv2 is Method 3's singleton, reached via dino_dense._get_inferencer() -- see the module
# docstring: one model, shared. FastSAM is this method's own backend, cached the same way so every
# query (API, CLI, sample renderer) reuses one loaded session and pays the init cost once.
_backend: object | None = None
_backend_loaded = False


def _get_backend() -> object | None:
    """Return the shared FastSAM proposal backend, or ``None`` when its weight is absent.

    Loads once and caches (including the absent result, so a missing weight is not re-probed on
    every call). Absence is a legitimate state -- the AGPL weight is gitignored -- so this returns
    ``None`` and lets :func:`search` degrade to an honest ``outcome=error`` rather than raising.
    """
    global _backend, _backend_loaded
    if _backend_loaded:
        return _backend
    _backend_loaded = True
    path = models.models_dir() / models.MODEL_REGISTRY[_FASTSAM_KEY].dest
    if not path.is_file():
        logger.info(
            "propose-retrieve: {!r} weight absent at {}; returning model_unavailable "
            "(run `pixi run -e export fetch-models --only {}`)",
            _FASTSAM_KEY,
            path,
            _FASTSAM_KEY,
        )
        _backend = None
        return None
    _backend = default_backend(path, providers=_PROVIDERS)
    return _backend


def reset_backend_cache() -> None:
    """Drop the cached FastSAM backend so the next call re-probes disk. For test isolation only."""
    global _backend, _backend_loaded
    _backend = None
    _backend_loaded = False


# -- pure-numpy helpers (model-free, so CI gates them without the gitignored weights) -----------


def _l2_normalize(vectors: npt.NDArray[np.floating], axis: int) -> npt.NDArray[np.float32]:
    """L2-normalize along ``axis``; a zero vector stays zero (guarded, never NaN)."""
    arr = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=axis, keepdims=True)
    normalized: npt.NDArray[np.float32] = (arr / np.maximum(norms, _EPS)).astype(np.float32)
    return normalized


def _mean_pool_region(
    inferencer: DINOv2Inferencer,
    image: npt.NDArray[np.uint8],
    box: BBox,
) -> npt.NDArray[np.float32]:
    """Embed one box crop into DINOv2 tokens and **mean-pool** them into a single vector.

    Pooling only; the L2-normalization happens once over the whole (N, D) matrix in
    :func:`embed_regions`, so a caller cannot accidentally normalize per token before pooling.
    A crop smaller than one patch is up-sized to a single 14x14 patch first -- a sub-patch region
    yields no tokens otherwise, and FastSAM can emit a small proposal.
    """
    crop = np.ascontiguousarray(image[box.y : box.y2, box.x : box.x2], dtype=np.uint8)
    ch, cw = int(crop.shape[0]), int(crop.shape[1])
    if ch < DINOV2_PATCH or cw < DINOV2_PATCH:
        crop = np.ascontiguousarray(
            cv2.resize(
                crop,
                (max(DINOV2_PATCH, cw), max(DINOV2_PATCH, ch)),
                interpolation=cv2.INTER_LINEAR,
            ),
            dtype=np.uint8,
        )
    grid, _, _ = inferencer.dense_tokens(crop)
    tokens = np.asarray(grid, dtype=np.float32).reshape(-1, grid.shape[-1])
    return np.asarray(tokens.mean(axis=0), dtype=np.float32)


def embed_regions(
    image: npt.NDArray[np.uint8],
    boxes: Sequence[BBox],
    config: BaseModel,
    *,
    inferencer: DINOv2Inferencer | None = None,
) -> npt.NDArray[np.float32]:
    """Embed each region into one **L2-normalized** vector -- the independently callable unit.

    This is the second half of the Milestone 2 seam. It crops each box, embeds the crop with the
    **same DINOv2 backbone as Method 3**, mean-pools the patch tokens, and L2-normalizes -- one
    ``(D,)`` row per box. It knows **nothing** about proposals, exemplars, or retrieval; ``search``
    composes it with :func:`~object_search.search.proposals.propose`.

    Args:
        image: The BGR scene the boxes index into.
        boxes: The regions to embed. May be empty -- an empty ``(0, D)`` matrix comes back.
        config: The method config. Narrowed to :class:`ProposeRetrieveConfig` to enforce the unit
            contract (the same idiom ``propose`` and the registered ``search`` functions use);
            v1 reads no embedding-time field off it.
        inferencer: The DINOv2 backbone to embed with. ``None`` reuses Method 3's shared singleton
            (``dino_dense._get_inferencer``) -- one model, no second download. Tests inject a stub
            here to exercise the callable-unit contract without the gitignored weight.

    Returns:
        A ``(len(boxes), D)`` float32 matrix, each row an L2-normalized region embedding.

    Raises:
        TypeError: If ``config`` is not a :class:`ProposeRetrieveConfig`.
        RuntimeError: If no inferencer is supplied and the shared DINOv2 weight is absent -- the
            standalone unit cannot embed without a backbone, and a silent empty result would hide
            the missing model.
    """
    if not isinstance(config, ProposeRetrieveConfig):
        raise TypeError(
            f"embed_regions requires a ProposeRetrieveConfig, got {type(config).__name__}"
        )
    inf = inferencer if inferencer is not None else dino_dense._get_inferencer()
    if inf is None:
        raise RuntimeError(
            "embed_regions needs the dinov2-small weight (shared with Method 3); it is absent. "
            "Run `pixi run fetch-models --only dinov2-small`."
        )
    if not boxes:
        return np.zeros((0, DINOV2_EMBED_DIM), dtype=np.float32)
    pooled = np.stack([_mean_pool_region(inf, image, box) for box in boxes], axis=0)
    return _l2_normalize(pooled, axis=1)


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
        method="propose-retrieve",
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
    name="propose-retrieve",
    description="FastSAM class-agnostic proposals ranked by DINOv2 region-embedding cosine NN.",
    version=_METHOD_VERSION,
    config_model=ProposeRetrieveConfig,
)
def search(
    image: npt.NDArray[np.uint8],
    exemplar: ExemplarBox,
    config: BaseModel,
) -> SearchResult:
    """Find every instance of ``exemplar`` by proposing regions and ranking by DINOv2 cosine."""
    # The registry types config as BaseModel (method-agnostic); the registered config_model
    # guarantees the concrete type. Narrow once here and fail loudly if the contract is violated.
    if not isinstance(config, ProposeRetrieveConfig):
        raise TypeError(
            f"propose-retrieve.search requires a ProposeRetrieveConfig, got {type(config).__name__}"
        )

    t_start = perf_counter()

    # Both weights are required: FastSAM proposes, DINOv2 embeds. An absent weight => an honest
    # model_unavailable error (outcome=error), never a raise -- so the renderer and API degrade.
    backend = _get_backend()
    inferencer = dino_dense._get_inferencer()
    missing = [
        key
        for key, present in ((_FASTSAM_KEY, backend), ("dinov2-small", inferencer))
        if present is None
    ]
    if missing:
        zero = LatencyBreakdown(preprocess_ms=0.0, inference_ms=0.0, postprocess_ms=0.0)
        return _empty_or_error(
            SearchOutcome.ERROR,
            f"weights absent: {', '.join(missing)}; run `pixi run fetch-models`. No search ran.",
            threshold=None,
            latency=zero,
            metrics={},
            error=MethodError(kind="model_unavailable", message=f"missing weights: {missing}"),
        )

    # 1. propose(image, config) -> proposals. The proposal unit (07-01); loose FastSAM-internal NMS
    #    keeps overlapping "everything-mode" regions -- over-segmentation is collapsed in step 6.
    fastsam_config = FastSAMConfig(conf_thres=config.proposal_conf)
    t_propose = perf_counter()
    proposals: list[Proposal] = propose(image, fastsam_config, backend=backend)  # type: ignore[arg-type]
    proposal_ms = (perf_counter() - t_propose) * 1000.0
    proposal_boxes = tuple(p.box for p in proposals)

    if not proposals:
        latency = LatencyBreakdown(
            preprocess_ms=0.0,
            inference_ms=proposal_ms,
            postprocess_ms=max(0.0, (perf_counter() - t_start) * 1000.0 - proposal_ms),
        )
        return _empty_or_error(
            SearchOutcome.EMPTY,
            f"FastSAM returned no proposals above conf {config.proposal_conf:.2f}.",
            threshold=None,
            latency=latency,
            metrics={"n_proposals": 0.0, "proposal_ms": proposal_ms, "embedding_ms": 0.0},
        )

    # 2. embed_regions(image, proposal boxes) -> (N, D) normalized proposal embeddings.
    # 3. embed_regions(image, [exemplar box]) -> the (D,) normalized exemplar embedding.
    #    Both go through the ONE shared DINOv2 backbone (step 2 and 3 use the same unit).
    t_embed = perf_counter()
    proposal_embeddings = embed_regions(image, list(proposal_boxes), config, inferencer=inferencer)
    exemplar_embedding = embed_regions(image, [exemplar.box], config, inferencer=inferencer)[0]
    embedding_ms = (perf_counter() - t_embed) * 1000.0

    # 4. Cosine nearest-neighbour = normalized proposals . normalized exemplar. Plain NumPy matmul
    #    -- NO FAISS (a few hundred rows in one image; FAISS is pure dependency cost here). Both
    #    sides are already L2-normalized, so the dot IS cosine similarity in [-1, 1].
    scores = np.asarray(proposal_embeddings @ exemplar_embedding, dtype=np.float32)

    # 5. Calibrate/threshold. A fixed retrieval_threshold passes straight through. Otherwise a
    #    two-mode gmm cuts between the "matches" mode (the exemplar's own region scores ~1.0) and
    #    the background mode, but the cut is clamped to an absolute similarity_floor (a cosine to
    #    the exemplar, whose self-cosine is 1.0). The floor does two things a bare gmm cannot:
    #    it stops a low gmm cut from admitting background, and it rescues the DEGENERATE single-mode
    #    case -- a uniform lattice of identical instances scores ~1.0 with no second mode, where the
    #    gmm/ratio fallback lands the cut at the max score and rejects every true match (recall 0,
    #    the worst failure for a repeated-instance finder). There the floor accepts everything above
    #    it; an image with no other instances scores below the floor and is correctly rejected.
    if config.retrieval_threshold is not None:
        calib = calibration.calibrate(
            scores.astype(np.float64),
            strategy="fixed",
            fixed_threshold=config.retrieval_threshold,
            seed=config.seed,
        )
        threshold = calib.threshold
    else:
        calib = calibration.calibrate(scores.astype(np.float64), strategy="gmm", seed=config.seed)
        # Degenerate (single mode) => the gmm/ratio cut is meaningless; use the floor alone.
        # Otherwise let the gmm raise the cut above the floor, but never sink it below.
        threshold = (
            config.similarity_floor
            if calib.degenerate
            else max(calib.threshold, config.similarity_floor)
        )

    # 6. Split into accepted (matches) and sub-threshold candidates (EVAL-08), then POST-RETRIEVAL
    #    NMS the accepted set to collapse SAM over-segmentation. The candidate log keeps the top
    #    max_candidates by raw score regardless of the threshold, so an offline sweep can rebuild a
    #    PR curve. METHOD-12: every accepted region survives NMS -- there is no single-best cut.
    ordered = sorted(
        range(len(proposals)),
        key=lambda i: (-float(scores[i]), proposal_boxes[i].y, proposal_boxes[i].x),
    )
    candidates = tuple(
        Candidate(box=proposal_boxes[i], score=float(scores[i]))
        for i in ordered[: config.max_candidates]
    )
    accepted = [i for i in ordered if float(scores[i]) > threshold]
    kept_local = nms.nms(
        [proposal_boxes[i] for i in accepted],
        [float(scores[i]) for i in accepted],
        config.nms_iou,
    )
    kept = [accepted[j] for j in kept_local]
    matches = _build_matches(
        [proposal_boxes[i] for i in kept], [float(scores[i]) for i in kept], exemplar.box
    )

    # 7. Diagnostics carry the FULL proposal set (the UI's debug overlay) and the pre-NMS accepted
    #    count so over-segmentation is visible; latency attributes proposal vs embedding SEPARATELY.
    inference_ms = proposal_ms + embedding_ms
    postprocess_ms = max(0.0, (perf_counter() - t_start) * 1000.0 - inference_ms)
    latency = LatencyBreakdown(
        preprocess_ms=0.0, inference_ms=inference_ms, postprocess_ms=postprocess_ms
    )
    metrics: dict[str, float] = {
        "threshold": threshold,
        "similarity_floor": float(config.similarity_floor),
        "n_proposals": float(len(proposals)),
        "n_accepted_pre_nms": float(len(accepted)),
        "n_matches": float(len(matches)),
        "n_candidates": float(len(candidates)),
        "collapsed_by_nms": float(len(accepted) - len(matches)),
        "score_max": float(scores.max()),
        "score_mean": float(scores.mean()),
        "proposal_ms": proposal_ms,
        "embedding_ms": embedding_ms,
    }
    floor_note = (
        "fixed threshold (floor not applied)"
        if config.retrieval_threshold is not None
        else f"clamped to similarity_floor {config.similarity_floor:.2f}"
        + (" (degenerate single mode)" if calib.degenerate else "")
    )
    notes = (
        f"calibration[{calib.strategy}]: {calib.reason}; {floor_note}",
        (
            f"{len(proposals)} FastSAM proposal(s); {len(accepted)} cleared cosine threshold "
            f"{threshold:.4f}; NMS(iou={config.nms_iou}) collapsed {len(accepted) - len(matches)} "
            f"to {len(matches)} match(es)"
        ),
        (
            f"latency: proposal {proposal_ms:.1f}ms "
            f"{'dominates' if proposal_ms >= embedding_ms else 'trails'} "
            f"embedding {embedding_ms:.1f}ms (EVAL-11)"
        ),
    )

    if not matches:
        return _empty_or_error(
            SearchOutcome.EMPTY,
            (
                f"no proposal cleared the calibrated threshold {threshold:.4f} "
                f"(best {float(scores.max()):.4f}); {calib.strategy}: {calib.reason}"
            ),
            threshold=threshold,
            latency=latency,
            metrics=metrics,
            proposals=proposal_boxes,
            candidates=candidates,
        )

    return SearchResult(
        method="propose-retrieve",
        method_version=_METHOD_VERSION,
        outcome=SearchOutcome.OK,
        matches=matches,
        latency=latency,
        threshold_applied=threshold,
        candidates=candidates,
        diagnostics=Diagnostics(notes=notes, metrics=metrics, proposals=proposal_boxes),
    )


def _build_matches(
    boxes: list[BBox],
    scores: list[float],
    exemplar: BBox,
) -> tuple[Match, ...]:
    """Turn kept proposals into Matches, labelling the exemplar's own region is_exemplar.

    The exemplar is part of the scene it is searched in, so one kept proposal overlaps it; that
    proposal is labelled rather than dropped (which understates recall) or counted silently as a
    discovery (which overstates the method), per METHOD-04c. Matches are returned in the canonical
    ``(-score, y, x)`` order.
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
