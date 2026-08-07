"""The Hydra benchmark runner: method x image x config, per-slice, with a model-free CI subset.

This is the **one** place Hydra is used (a locked decision): sweeping method x image x config is
exactly the multi-run problem Hydra exists for, whereas the API path keeps plain frozen Pydantic
configs. ``@hydra.main`` seizes ``sys.argv``, which is why the benchmark is its own module entry
point (``pixi run bench``) and cannot be a Typer subcommand -- the latent defect this plan fixed.

What it produces
----------------
``docs/benchmark/results.json`` with, per method: pooled precision/recall/F1, mean AP, and a
latency summary, **plus per-slice breakdowns** by true instance count, by canvas size (the
chipset ramps 320x240 -> 6000x4000, and a single pooled latency would hide where each method's
cost goes), and by scale variation. Per-slice is the deliverable (EVAL-10): "Method 1 wins on the
fixed-scale chipset, Method 3 wins once scale varies" is a statement the pooled number cannot make.

The crossover, on purpose
-------------------------
The default sweep includes the chipset (near-identical fixed-scale repeats -- NCC-favourable) and
the scale/pose-varied synthetic scenes (learned-favourable). Reporting per slice is what makes the
NCC-vs-sparse-geo crossover *observable* rather than averaged away (a locked decision, EVAL-19).

The model-free CI subset
------------------------
``ci=true`` runs only ``ncc`` and classical ``sparse-geo`` (SIFT backend, no ONNX weights) over
the chipset, which regenerates deterministically from seeds. That is the subset CI runs without
``fetch-models``; the full sweep (which needs DINOv2 / FastSAM weights) is gated behind fetched
models and run locally. A method whose weights are absent records an ``error`` outcome for that
image rather than aborting the whole sweep, so a partial environment still yields a partial report.

AP reads the candidate log
--------------------------
Per-image AP is computed over the method's matches **and** its sub-threshold candidates (EVAL-08)
combined into one ranked ``(box, score)`` list, so a full precision/recall curve comes from a
single operating point's worth of data (:func:`object_search.eval.metrics.average_precision`).
"""

from __future__ import annotations

import json
import statistics
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import cv2
import hydra
import numpy as np
import numpy.typing as npt
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ConfigDict, Field

from object_search.eval.labels import (
    GroundTruth,
    chipset_image_ids,
    load_ground_truth,
    load_research_ground_truth,
    real_objects_image_ids,
    scene_path,
    textured_image_ids,
)
from object_search.eval.metrics import (
    average_precision,
    average_precision_coco,
    counting_errors,
    match_predictions,
    match_predictions_detailed,
    precision_recall_f1,
)
from object_search.eval.sampling import ExemplarSelection, sample_exemplars
from object_search.eval.splits import load_split_manifest, research_image_ids
from object_search.provenance import current_git_sha, repo_root
from object_search.schemas.geometry import BBox, ExemplarBox
from object_search.schemas.search import (
    Candidate,
    LatencyBreakdown,
    Match,
    SearchOutcome,
    SearchResult,
)
from object_search.search import get_method
from object_search.search.common.nms import nms
from object_search.search.registry import SearchFn

# The classical, weight-free methods. The CI subset is exactly these two, because they need no
# ONNX weights and so run without `fetch-models` (EVAL-19).
_MODEL_FREE_METHODS: tuple[str, ...] = ("ncc", "sparse-geo")

# The scale-varied synthetic scenes to include in the full sweep so the crossover has a
# learned-favourable side to show against the fixed-scale chipset.
_SYNTHETIC_IMAGE_IDS: tuple[str, ...] = ("scatter-scaled", "cluttered-distractors")

# The IoU above which two detections fused from DIFFERENT exemplar runs are treated as the same
# instance and deduped (k-shot late fusion, Task 2). A repeated instance re-detected from each of
# the k exemplars yields near-identical boxes (high IoU), so this collapses the union back to one
# detection per instance rather than counting it k times. Uses the shared deterministic NMS
# (tie-break `(-score, y, x)`), the same reproducibility guarantee the rest of the harness uses.
_FUSION_NMS_IOU: float = 0.5

# -- research per-slice analysis (EVAL-10 applied to floor plans) ------------------------------
# SYMBOL-SIZE buckets cut a GT box by its area as a fraction of the plan area (box.area /
# (plan_width * plan_height)). Floor-plan door/window symbols are TINY relative to the whole plan,
# so the interesting spread is at the small end: a 30x30 door on an 800x600 plan is ~0.0019 of the
# canvas, a chunky 60x40 window ~0.005. The two cuts below split "tiny" from "typical" from
# "unusually large / merged-annotation" and are area FRACTIONS, not absolute pixels, so they hold
# across the varied plan resolutions without per-image tuning. Recall is reported per bucket so a
# method that only finds the big symbols is visibly distinguished from one that finds small ones.
_SYMBOL_SIZE_SMALL_MAX: float = 0.004
_SYMBOL_SIZE_MEDIUM_MAX: float = 0.016
# The fixed bucket order, always emitted (recall None on an empty bucket) so the table is stable.
_SYMBOL_SIZE_BUCKETS: tuple[str, ...] = ("small", "medium", "large")

# CROWDING buckets group a plan by how many target instances it holds (instances-per-plan): a plan
# with one door is a different retrieval problem from one wall of twelve. Coarse on purpose.
# PLAN-RESOLUTION buckets group by the canvas long side, since a method's localisation degrades
# with plan size independently of crowding. Both are reported at F1 (the operating-point metric).


class GtBoxRecord(BaseModel):
    """One ground-truth box's per-slice record for the research path: its size bucket + matched.

    Additive and JSON-serialisable; :class:`ImageResult` carries a tuple of these only on the
    research path (default empty everywhere else), so the chipset/CI path is unaffected. The
    ``matched`` flag comes from :func:`object_search.eval.metrics.match_predictions_detailed`, so a
    GT box is ``matched`` exactly when some prediction claimed it under the EVAL-16 duplicate rule.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    size_bucket: str
    matched: bool


class BenchmarkConfig(BaseModel):
    """Validated benchmark configuration -- the Pydantic gate over Hydra's untyped ``DictConfig``.

    Hydra composes the YAML into a ``DictConfig``; this model is what turns it into something with
    real types and defaults, so a typo in ``conf/benchmark.yaml`` fails loudly here rather than
    silently sweeping the wrong set.

    Attributes:
        methods: Registry keys to run. Ignored when ``ci`` is set (the CI subset is fixed).
        image_ids: Scene ids to sweep. Ignored when ``ci`` or ``real_objects_only`` is set.
        ci: Model-free subset -- ``ncc`` + classical ``sparse-geo`` over the chipset, no weights.
        real_objects_only: Run every configured ``methods`` over exactly
            :func:`object_search.eval.labels.real_objects_image_ids` -- no chipset, textured, or
            configured synthetic ids unioned in. Checked after ``ci`` (``ci`` wins if both are set,
            same precedence as the CI subset always taking priority over the full sweep). Feeds the
            dedicated ``real-objects-report.html`` (``pixi run bench-real-objects``), kept separate
            from the default full sweep so a reader can compare "real photographic pixels only"
            against "synthetic only" without hand-filtering one pooled ``results.json``.
        iou_threshold: IoU for a prediction to count as a true positive.
        out: Output path for ``results.json``; resolved against the repo root when relative.
        ci_image_limit: Cap on chipset images in the CI subset, keeping CI runtime bounded while
            still spanning several canvas sizes.
        datasets: Research dataset keys to sweep (e.g. ``("rpine", "carpk")``). Empty by default:
            the research sweep is a separate path (:func:`run_research_sweep`) and is **never** part
            of the CI subset, which stays chipset-only (RESEARCH risk note). The learned methods
            need fetched weights and every dataset needs fetched (licence-gated) archives, so the
            sweep is gated exactly like the full sweep gates on ``fetch-models``.
        splits: Which splits to report per dataset -- ``val`` and ``test`` by default. A test-only
            dataset (CARPK/PUCPR+) has no val ids in its manifest, so no val cell is ever emitted
            for it (D-04); tuning is confined to val (D-02).
        exemplar_counts: Operating points to score every method at -- ``1`` (the product's one-box
            point) and ``3`` (the literature convention) by default (D-05).
        seed: Config seed for the exemplar sampler's non-native draw (D-11), passed to
            :func:`object_search.eval.sampling.sample_exemplars` -- never ``cv2.setRNGSeed``.
        research_root: Base directory of converted research sidecars + co-located scenes
            (``datasets/`` once fetched); each cell reads ``<research_root>/<dataset>/<split>/``.
            Required to run the research sweep.
        research_out: Output path for the research results file; gitignored like ``results.json``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    methods: tuple[str, ...] = (
        "ncc",
        "mosse",
        "sparse-geo",
        "dino-dense",
        "propose-retrieve",
        "owlv2-oneshot",
    )
    image_ids: tuple[str, ...] = _SYNTHETIC_IMAGE_IDS
    ci: bool = False
    real_objects_only: bool = False
    iou_threshold: float = Field(default=0.5, gt=0.0, le=1.0)
    out: str = "docs/benchmark/results.json"
    ci_image_limit: int = Field(default=6, ge=1)
    # -- research sweep dimensions (EVAL-23/24); kept out of the CI subset entirely --------
    datasets: tuple[str, ...] = ()
    splits: tuple[str, ...] = ("val", "test")
    exemplar_counts: tuple[int, ...] = (1, 3)
    seed: int = 0
    research_root: str | None = None
    research_out: str = "docs/benchmark/research-results.json"

    def resolve_run_set(self) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """Return the ``(methods, image_ids)`` actually swept, applying the CI subset rule."""
        if self.ci:
            images = chipset_image_ids()[: self.ci_image_limit]
            return _MODEL_FREE_METHODS, images
        if self.real_objects_only:
            # Exactly the real-objects set, every configured method -- no chipset/textured/
            # synthetic ids unioned in, so this sweep is comparable one-for-one against the
            # real-objects rows the full sweep also produces (same cells, independent artifact).
            return self.methods, real_objects_image_ids()
        # The full sweep includes the chipset (NCC-favourable), the textured regimes (EVAL-20,
        # keypoint- and deep-feature-favourable), the real-object-insertion set (real photographic
        # texture/lighting, no synthetic render), and the configured synthetic scenes, so the
        # per-slice crossover has every side present.
        images = tuple(
            dict.fromkeys(
                (
                    *chipset_image_ids(),
                    *textured_image_ids(),
                    *real_objects_image_ids(),
                    *self.image_ids,
                )
            )
        )
        return self.methods, images


class ImageResult(BaseModel):
    """One method's result on one image: the metrics plus the slice keys it is grouped by.

    The research fields (``dataset``/``split``/``exemplar_count`` and the literature metrics
    ``ap50``/``ap75``/``predicted_count``/``true_count``) are additive and default ``None``, so the
    committed chipset/CI path constructs this exactly as before and stays byte-identical; they are
    populated only on the research path (:func:`run_research_benchmark`).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    image_id: str
    outcome: str
    canvas_size: str | None
    instance_count: int | None
    scale_bucket: str
    tp: int | None
    fp: int | None
    fn: int | None
    precision: float | None
    recall: float | None
    f1: float | None
    ap: float | None
    latency_ms: float | None
    n_matches: int
    # -- research-dataset additive fields (EVAL-22/24) --------------------------------
    dataset: str | None = None
    split: str | None = None
    exemplar_count: int | None = None
    ap50: float | None = None
    ap75: float | None = None
    predicted_count: int | None = None
    true_count: int | None = None
    # -- research per-slice records (Task 260729-dh6); default empty keeps chipset byte-identical --
    gt_records: tuple[GtBoxRecord, ...] = ()


def _scale_bucket(gt: GroundTruth) -> str:
    """Coarse label for how much instance scale varies -- the crossover axis.

    ``fixed`` (chipset repeats and the like) is where NCC is strongest; ``varied`` (>1.5x spread)
    is where a scale-invariant method should pull ahead. Bucketing rather than reporting the raw
    ratio keeps the per-slice table skimmable.
    """
    lo = gt.slice_metadata.instance_scale_min
    hi = gt.slice_metadata.instance_scale_max
    if lo is None or hi is None or lo <= 0.0:
        return "fixed"
    return "varied" if (hi / lo) > 1.5 else "fixed"


def _symbol_size_bucket(box_area: int, plan_area: int) -> str:
    """Bucket a GT box by its area as a fraction of the plan area -- small / medium / large.

    The cuts (:data:`_SYMBOL_SIZE_SMALL_MAX`, :data:`_SYMBOL_SIZE_MEDIUM_MAX`) are area fractions,
    so a door on a small plan and the same door on a large plan land in the same bucket -- the
    grouping tracks the symbol's relative footprint, not raw pixels.
    """
    frac = box_area / plan_area
    if frac < _SYMBOL_SIZE_SMALL_MAX:
        return "small"
    if frac < _SYMBOL_SIZE_MEDIUM_MAX:
        return "medium"
    return "large"


def _build_gt_records(gt: GroundTruth, matched: tuple[bool, ...]) -> tuple[GtBoxRecord, ...]:
    """Pair each GT box with its symbol-size bucket and whether it was matched (research path).

    The plan area is ``gt.width * gt.height``. When either dimension is missing (a sidecar that did
    not record the canvas) the size fraction is undefined, so **every box is skipped** from the size
    aggregation rather than defaulting to a fabricated plan area -- an unknown must never read as a
    concrete bucket. ``matched`` is aligned index-for-index to ``gt.boxes`` (the detailed matcher's
    contract), so ``zip(..., strict=True)`` is safe and a length mismatch would raise loudly.
    """
    if gt.width is None or gt.height is None:
        return ()
    plan_area = gt.width * gt.height
    if plan_area <= 0:
        return ()
    return tuple(
        GtBoxRecord(size_bucket=_symbol_size_bucket(box.area, plan_area), matched=is_matched)
        for box, is_matched in zip(gt.boxes, matched, strict=True)
    )


def _crowding_bucket(record: ImageResult) -> str | None:
    """Coarse instances-per-plan bucket, or ``None`` when the count is unknown (skipped from slice).

    ``1`` is the single-symbol plan; ``2-5`` a small room; ``6-15`` a busy plan; ``16+`` a dense
    wall of repeats. Coarse on purpose so the F1-per-crowding table stays skimmable.
    """
    count = record.instance_count
    if count is None:
        return None
    if count <= 1:
        return "1"
    if count <= 5:
        return "2-5"
    if count <= 15:
        return "6-15"
    return "16+"


def _plan_resolution_bucket(record: ImageResult) -> str | None:
    """Coarse canvas-long-side bucket from ``canvas_size`` (``"WxH"``), or ``None`` if unparseable.

    Localisation degrades with plan size independently of crowding, so the report breaks F1 down by
    the long side: ``<=800`` small, ``<=1600`` medium, ``>1600`` large. A missing or malformed
    ``canvas_size`` is skipped from the slice rather than bucketed to a guessed resolution.
    """
    canvas = record.canvas_size
    if canvas is None or "x" not in canvas:
        return None
    width_str, _, height_str = canvas.partition("x")
    try:
        long_side = max(int(width_str), int(height_str))
    except ValueError:
        return None
    if long_side <= 800:
        return "<=800"
    if long_side <= 1600:
        return "<=1600"
    return ">1600"


def _recall_by_size(records: Sequence[GtBoxRecord]) -> dict[str, Any]:
    """Pool matched/total across every GT record and emit per-symbol-size-bucket RECALL.

    A GT-box-level aggregation (not :func:`_slice_by`, which is per-image): recall is
    ``sum(matched) / n_gt`` within each bucket, with the abstention convention -- a bucket with
    **zero** GT boxes reports ``recall = None`` (undefined, never ``0.0``). All three fixed buckets
    are always present so the table is stable across cells.
    """
    totals: dict[str, list[int]] = {bucket: [0, 0] for bucket in _SYMBOL_SIZE_BUCKETS}
    for record in records:
        pair = totals.setdefault(record.size_bucket, [0, 0])
        pair[0] += int(record.matched)
        pair[1] += 1
    return {
        bucket: {
            "n_gt": total,
            "n_matched": matched,
            "recall": (matched / total if total > 0 else None),
        }
        for bucket, (matched, total) in totals.items()
    }


def _load_scene(image_id: str) -> npt.NDArray[np.uint8]:
    """Read the committed scene image as BGR uint8, raising if it is missing."""
    path = scene_path(image_id)
    if path is None:
        raise FileNotFoundError(f"no scene image on disk for {image_id!r}")
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise OSError(f"failed to read scene image {path}")
    return np.asarray(image, dtype=np.uint8)


def _run_one(method: str, image_id: str, gt: GroundTruth, iou_threshold: float) -> ImageResult:
    """Run one method on one image and score it against ground truth.

    A raised exception (most often absent ONNX weights for a learned method) is caught and
    recorded as an ``error`` outcome, so one unavailable model does not abort the whole sweep.
    """
    canvas = f"{gt.width}x{gt.height}" if gt.width and gt.height else None
    spec = get_method(method)
    try:
        scene = _load_scene(image_id)
        result = spec.fn(scene, gt.exemplar, spec.config_model())
    except Exception as exc:
        # A broad catch is deliberate: a missing ONNX weight or an unavailable backend must
        # degrade one (method, image) cell to an error outcome, never abort the whole sweep.
        logger.warning("{} on {} failed: {}", method, image_id, exc)
        return ImageResult(
            image_id=image_id,
            outcome="error",
            canvas_size=canvas,
            instance_count=gt.achieved_count,
            scale_bucket=_scale_bucket(gt),
            tp=None,
            fp=None,
            fn=None,
            precision=None,
            recall=None,
            f1=None,
            ap=None,
            latency_ms=None,
            n_matches=0,
        )

    pred_boxes = [match.box for match in result.matches]
    tp, fp, fn = match_predictions(pred_boxes, gt.boxes, iou_threshold)
    precision, recall, f1 = precision_recall_f1(tp, fp, fn)
    # AP over the full candidate log: above-threshold matches AND sub-threshold candidates.
    candidate_log: list[tuple[BBox, float]] = [(m.box, m.score) for m in result.matches]
    candidate_log += [(c.box, c.score) for c in result.candidates]
    ap = average_precision(candidate_log, gt.boxes, iou_threshold) if candidate_log else 0.0

    return ImageResult(
        image_id=image_id,
        outcome=result.outcome.value,
        canvas_size=canvas,
        instance_count=gt.achieved_count,
        scale_bucket=_scale_bucket(gt),
        tp=tp,
        fp=fp,
        fn=fn,
        precision=precision,
        recall=recall,
        f1=f1,
        ap=ap,
        latency_ms=result.latency.total_ms,
        n_matches=len(result.matches),
    )


def _aggregate(records: list[ImageResult]) -> dict[str, Any]:
    """Pool a group of per-image results into precision/recall/F1, mean AP, and latency stats.

    Precision/recall/F1 are **micro-averaged** (sum tp/fp/fn, then divide) so the abstention
    convention propagates: a group where nothing was ever returned pools to precision ``None``,
    not ``0`` (EVAL-17). AP is macro-averaged (mean of per-image AP), the standard mAP.
    """
    scored = [r for r in records if r.tp is not None]
    total_tp = sum(r.tp for r in scored if r.tp is not None)
    total_fp = sum(r.fp for r in scored if r.fp is not None)
    total_fn = sum(r.fn for r in scored if r.fn is not None)
    precision, recall, f1 = (
        precision_recall_f1(total_tp, total_fp, total_fn)
        if scored
        else (
            None,
            None,
            None,
        )
    )
    aps = [r.ap for r in scored if r.ap is not None]
    latencies = [r.latency_ms for r in records if r.latency_ms is not None]
    return {
        "n_images": len(records),
        "n_scored": len(scored),
        "n_errors": sum(1 for r in records if r.outcome == "error"),
        "n_abstentions": sum(1 for r in records if r.outcome == SearchOutcome.EMPTY.value),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_ap": statistics.fmean(aps) if aps else None,
        "latency_ms": {
            "p50": statistics.median(latencies) if latencies else None,
            "mean": statistics.fmean(latencies) if latencies else None,
            "max": max(latencies) if latencies else None,
        },
    }


def _slice_by(
    records: list[ImageResult],
    key: Callable[[ImageResult], str | int | None],
    aggregate: Callable[[list[ImageResult]], dict[str, Any]] = _aggregate,
) -> dict[str, Any]:
    """Group records by ``key(record)`` (skipping ``None`` keys) and aggregate each group.

    ``aggregate`` defaults to the chipset :func:`_aggregate` so the existing per-slice reporting is
    byte-identical; the research path passes :func:`_aggregate_research` to share the same grouping
    with the literature-metric (P/R/F1 + AP + counting) pooling.
    """
    groups: dict[str, list[ImageResult]] = {}
    for record in records:
        bucket = key(record)
        if bucket is None:
            continue
        groups.setdefault(str(bucket), []).append(record)
    return {bucket: aggregate(group) for bucket, group in sorted(groups.items())}


def run_benchmark(config: BenchmarkConfig) -> dict[str, Any]:
    """Run the sweep and write ``results.json``; return the results dict.

    This is the pure, Hydra-free core so tests can drive it with a constructed config rather than
    through ``@hydra.main`` (which would seize ``sys.argv``).

    Args:
        config: The validated benchmark configuration.

    Returns:
        The results mapping that was serialised to ``config.out``.
    """
    methods, image_ids = config.resolve_run_set()
    logger.info(
        "benchmark: {} method(s) x {} image(s){}",
        len(methods),
        len(image_ids),
        " [CI model-free subset]" if config.ci else "",
    )

    # Coverage is reported honestly: an image with no sidecar is counted as unlabelled, never
    # silently dropped from the denominator (EVAL-02).
    labelled: dict[str, GroundTruth] = {}
    unlabelled: list[str] = []
    for image_id in image_ids:
        gt = load_ground_truth(image_id)
        if gt is None:
            unlabelled.append(image_id)
        else:
            labelled[image_id] = gt

    methods_out: dict[str, Any] = {}
    for method in methods:
        records = [
            _run_one(method, image_id, gt, config.iou_threshold)
            for image_id, gt in labelled.items()
        ]
        methods_out[method] = {
            "overall": _aggregate(records),
            "slices": {
                "by_instance_count": _slice_by(records, lambda r: r.instance_count),
                "by_canvas_size": _slice_by(records, lambda r: r.canvas_size),
                "by_scale_bucket": _slice_by(records, lambda r: r.scale_bucket),
            },
            # gt_records is a research-only internal carrier for the per-slice aggregation; it is
            # excluded from per_image so the chipset output stays byte-identical (and JSON-stable).
            "per_image": [r.model_dump(exclude={"gt_records"}) for r in records],
        }

    results: dict[str, Any] = {
        "git_sha": current_git_sha(),
        "ci_subset": config.ci,
        "iou_threshold": config.iou_threshold,
        "ap_convention": "all-point interpolation (COCO-style), from the EVAL-08 candidate log",
        "coverage": {
            "images_requested": len(image_ids),
            "images_labelled": len(labelled),
            "images_unlabelled": sorted(unlabelled),
        },
        "methods": methods_out,
    }

    out_path = Path(config.out)
    if not out_path.is_absolute():
        out_path = repo_root() / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    logger.info("benchmark: wrote {}", out_path)
    return results


# --------------------------------------------------------------------------- research path
# The research sweep reuses the SAME _run_one scoring shape and the SAME micro-averaged
# aggregation as the chipset path (D-10 one-loader philosophy applied to the runner): it differs
# only in where scenes/labels come from (the gitignored datasets/ tree, addressed by explicit
# path) and in the extra literature metrics it records (COCO AP50/AP75 and the per-image counts the
# aggregate turns into MAE/RMSE/NAE). It is deliberately NOT in the Hydra CI subset, which stays
# chipset-only (D per RESEARCH risk note); this is the 11-01 tracer proving one ncc x carpk x 1 x
# test path, not the full method x dataset x {1,3} x {val,test} sweep (that is 11-03).


def _load_research_scene(research_root: Path, image_id: str) -> npt.NDArray[np.uint8]:
    """Read a converted research scene (co-located with its sidecar) as BGR uint8."""
    path = research_root / f"{image_id}.png"
    if not path.is_file():
        raise FileNotFoundError(f"no research scene image at {path}")
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise OSError(f"failed to read research scene image {path}")
    return np.asarray(image, dtype=np.uint8)


def _dedupe_matches(matches: list[Match], fusion_iou: float) -> tuple[Match, ...]:
    """NMS-dedupe fused matches so one instance is not counted once per exemplar run."""
    if not matches:
        return ()
    kept = nms([m.box for m in matches], [m.score for m in matches], fusion_iou)
    return tuple(matches[i] for i in kept)


def _dedupe_candidates(candidates: list[Candidate], fusion_iou: float) -> tuple[Candidate, ...]:
    """NMS-dedupe the fused sub-threshold candidate log the same way as the matches."""
    if not candidates:
        return ()
    kept = nms([c.box for c in candidates], [c.score for c in candidates], fusion_iou)
    return tuple(candidates[i] for i in kept)


def run_multi_exemplar(
    spec_fn: SearchFn,
    image: npt.NDArray[np.uint8],
    exemplars: Sequence[ExemplarBox],
    config: BaseModel,
    *,
    fusion_iou: float = _FUSION_NMS_IOU,
) -> SearchResult:
    """Run a method at ``k`` exemplars by **late fusion** in the eval layer (Task 2, EVAL-23).

    This is the single mechanism ratified in the Task 2 checkpoint, and it is the *only* place the
    harness knows how to query a method with more than one exemplar. The method's ``SearchFn`` and
    all four method files are UNCHANGED: each still takes exactly one :class:`ExemplarBox`. Here we
    call ``spec_fn`` **once per exemplar**, UNION the resulting matches and sub-threshold candidates
    across the ``k`` runs, then dedupe overlapping detections with the shared deterministic NMS
    (tie-break ``(-score, y, x)``), so a repeated instance re-found from each exemplar collapses to
    a single detection rather than being counted ``k`` times. The ``k = 1`` case is a pass-through
    of the single call, so 1 and 3 exemplars share this one code path.

    Args:
        spec_fn: The method's search callable (``MethodSpec.fn``). Called once per exemplar; not
            wrapped or modified.
        image: The BGR scene, shared across the ``k`` runs.
        exemplars: The sampled exemplar boxes (from
            :func:`object_search.eval.sampling.sample_exemplars`). Must be non-empty.
        config: The method's config instance, shared across the ``k`` runs.
        fusion_iou: IoU above which two detections from different runs are the same instance.

    Returns:
        One fused :class:`SearchResult`. ``k = 1`` returns the single call's result unchanged. For
        ``k > 1`` the fused result carries the NMS-deduped matches and candidates, a latency that is
        the sum across the runs (the fusion genuinely ran the method ``k`` times), and
        ``outcome = OK`` when any match survived else ``EMPTY``. If every run errored, the first
        error result is returned unchanged so the error payload is preserved.

    Raises:
        ValueError: If ``exemplars`` is empty -- a search needs at least one exemplar.
    """
    if not exemplars:
        raise ValueError("run_multi_exemplar needs at least one exemplar")

    results = [spec_fn(image, exemplar, config) for exemplar in exemplars]
    if len(results) == 1:
        return results[0]

    non_error = [r for r in results if r.outcome is not SearchOutcome.ERROR]
    if not non_error:
        # Every run errored: preserve the first error result (and its payload) verbatim.
        return results[0]

    matches = [m for r in non_error for m in r.matches]
    candidates = [c for r in non_error for c in r.candidates]
    kept_matches = _dedupe_matches(matches, fusion_iou)
    kept_candidates = _dedupe_candidates(candidates, fusion_iou)

    template = non_error[0]
    latency = LatencyBreakdown(
        preprocess_ms=sum(r.latency.preprocess_ms for r in non_error),
        inference_ms=sum(r.latency.inference_ms for r in non_error),
        postprocess_ms=sum(r.latency.postprocess_ms for r in non_error),
    )
    outcome = SearchOutcome.OK if kept_matches else SearchOutcome.EMPTY
    return SearchResult(
        method=template.method,
        method_version=template.method_version,
        outcome=outcome,
        matches=kept_matches,
        latency=latency,
        threshold_applied=template.threshold_applied,
        candidates=kept_candidates,
    )


def _run_one_research(
    method: str,
    image_id: str,
    research_root: Path,
    gt: GroundTruth,
    iou_threshold: float,
    *,
    dataset: str,
    split: str,
    exemplar_count: int,
    seed: int = 0,
    config: BaseModel | None = None,
    exemplar_selection: ExemplarSelection = "seeded-random",
) -> ImageResult:
    """Run one method on one research image and score it with the full literature metric set.

    Mirrors :func:`_run_one` (same match/precision/recall/candidate-log logic, same broad
    error-catch so a missing weight degrades one cell rather than aborting the sweep) and adds the
    COCO AP sweep plus the per-image predicted/true counts. The per-image ``ap`` field is the COCO
    mean here (the literature's headline AP), with ``ap50``/``ap75`` alongside it.

    ``config`` is the method config instance to run with; ``None`` uses the method's defaults
    (``spec.config_model()``), the committed behaviour. A non-default config is how domain threshold
    tuning (:mod:`object_search.eval.tuning`) evaluates a candidate on val/test without touching the
    method files -- the config must be an instance of this method's own ``config_model``.
    """
    canvas = f"{gt.width}x{gt.height}" if gt.width and gt.height else None
    true_count = gt.achieved_count
    spec = get_method(method)
    try:
        scene = _load_research_scene(research_root, image_id)
        # k-shot LATE FUSION (Task 2): sample `exemplar_count` boxes and run the method once per
        # box, then fuse. The sampled exemplars REMAIN in gt.boxes and are scored like any other
        # instance, so the recall denominator (len(gt.boxes)) is identical at count=1 and count=3.
        exemplars = sample_exemplars(
            gt, count=exemplar_count, seed=seed, exemplar_selection=exemplar_selection
        )
        run_config = config if config is not None else spec.config_model()
        result = run_multi_exemplar(spec.fn, scene, exemplars, run_config)
    except Exception as exc:
        logger.warning("{} on research {}/{} failed: {}", method, dataset, image_id, exc)
        return ImageResult(
            image_id=image_id,
            outcome="error",
            canvas_size=canvas,
            instance_count=true_count,
            scale_bucket=_scale_bucket(gt),
            tp=None,
            fp=None,
            fn=None,
            precision=None,
            recall=None,
            f1=None,
            ap=None,
            latency_ms=None,
            n_matches=0,
            dataset=dataset,
            split=split,
            exemplar_count=exemplar_count,
            ap50=None,
            ap75=None,
            predicted_count=None,
            true_count=true_count,
        )

    pred_boxes = [match.box for match in result.matches]
    # The DETAILED matcher (Task 1) also returns which GT boxes were matched, aligned to gt.boxes,
    # so the per-symbol-size recall slice can be built without re-running the match. `tp` still
    # equals sum(matched), so the counts here are identical to the plain match_predictions form.
    tp, fp, fn, matched = match_predictions_detailed(pred_boxes, gt.boxes, iou_threshold)
    precision, recall, f1 = precision_recall_f1(tp, fp, fn)
    candidate_log: list[tuple[BBox, float]] = [(m.box, m.score) for m in result.matches]
    candidate_log += [(c.box, c.score) for c in result.candidates]
    if candidate_log:
        ap, ap50, ap75 = average_precision_coco(candidate_log, gt.boxes)
    else:
        ap, ap50, ap75 = 0.0, 0.0, 0.0

    return ImageResult(
        image_id=image_id,
        outcome=result.outcome.value,
        canvas_size=canvas,
        instance_count=true_count,
        scale_bucket=_scale_bucket(gt),
        tp=tp,
        fp=fp,
        fn=fn,
        precision=precision,
        recall=recall,
        f1=f1,
        ap=ap,
        latency_ms=result.latency.total_ms,
        n_matches=len(result.matches),
        dataset=dataset,
        split=split,
        exemplar_count=exemplar_count,
        ap50=ap50,
        ap75=ap75,
        predicted_count=len(result.matches),
        true_count=true_count,
        gt_records=_build_gt_records(gt, matched),
    )


def _aggregate_research(records: list[ImageResult]) -> dict[str, Any]:
    """Pool research per-image results: micro P/R/F1, macro AP/AP50/AP75, and MAE/RMSE/NAE.

    Precision/recall/F1 are micro-averaged (same abstention-propagating convention as
    :func:`_aggregate`); AP/AP50/AP75 are macro-averaged (mean of per-image AP, the standard mAP);
    MAE/RMSE/NAE come from :func:`object_search.eval.metrics.counting_errors` over the per-image
    predicted/true counts. All are ``None`` when nothing was scored.
    """
    scored = [r for r in records if r.tp is not None]
    total_tp = sum(r.tp for r in scored if r.tp is not None)
    total_fp = sum(r.fp for r in scored if r.fp is not None)
    total_fn = sum(r.fn for r in scored if r.fn is not None)
    precision, recall, f1 = (
        precision_recall_f1(total_tp, total_fp, total_fn) if scored else (None, None, None)
    )

    aps = [r.ap for r in scored if r.ap is not None]
    ap50s = [r.ap50 for r in scored if r.ap50 is not None]
    ap75s = [r.ap75 for r in scored if r.ap75 is not None]

    # Counting metrics over the images that carry both counts (narrow to int lists so the
    # NULL-safe guard in counting_errors sees exactly the assessed images).
    preds: list[int] = []
    trues: list[int] = []
    for r in scored:
        if r.predicted_count is not None and r.true_count is not None:
            preds.append(r.predicted_count)
            trues.append(r.true_count)
    mae: float | None
    rmse: float | None
    nae: float | None
    if preds:
        mae, rmse, nae = counting_errors(preds, trues)
    else:
        mae = rmse = nae = None

    latencies = [r.latency_ms for r in records if r.latency_ms is not None]
    return {
        "n_images": len(records),
        "n_scored": len(scored),
        "n_errors": sum(1 for r in records if r.outcome == "error"),
        "n_abstentions": sum(1 for r in records if r.outcome == SearchOutcome.EMPTY.value),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "ap": statistics.fmean(aps) if aps else None,
        "ap50": statistics.fmean(ap50s) if ap50s else None,
        "ap75": statistics.fmean(ap75s) if ap75s else None,
        "mae": mae,
        "rmse": rmse,
        "nae": nae,
        "latency_ms": {
            "p50": statistics.median(latencies) if latencies else None,
            "mean": statistics.fmean(latencies) if latencies else None,
            "max": max(latencies) if latencies else None,
        },
    }


def run_research_benchmark(
    method: str,
    dataset: str,
    split: str,
    research_root: Path,
    *,
    exemplar_count: int = 1,
    iou_threshold: float = 0.5,
    seed: int = 0,
    manifest_root: Path | None = None,
    config: BaseModel | None = None,
    exemplar_selection: ExemplarSelection = "seeded-random",
) -> dict[str, Any]:
    """Run one method over a research dataset split, returning the report block (11-01 tracer).

    The image ids come from the committed split manifest (:func:`research_image_ids`); each label is
    the converted ``*.gt.json`` under ``research_root`` (tagged ``source="research"``), and each
    scene is co-located beside it. The return carries one per-image row (each with
    precision/recall/f1/ap/ap50/ap75 and the per-image counts) plus one pooled ``overall`` block
    with the full literature-metric column set (P/R/F1 + AP/AP50/AP75 + MAE/RMSE/NAE).

    Args:
        method: Registry key, e.g. ``"ncc"``.
        dataset: Dataset key, e.g. ``"carpk"``.
        split: ``"train"`` / ``"val"`` / ``"test"``.
        research_root: Directory of converted sidecars + co-located scenes (``datasets/<d>/<s>``).
        exemplar_count: Exemplars to seed the run with (1 = the product operating point). At >1 the
            method is run once per sampled exemplar and the results are fused by
            :func:`run_multi_exemplar`.
        iou_threshold: IoU for a prediction to count as a true positive at the P/R/F1 level.
        seed: Config seed for the exemplar sampler's non-native draw (D-11).
        manifest_root: Optional base dir for the committed split manifest (tests use ``tmp_path``).
        config: Method config instance to run with; ``None`` uses the method defaults. A tuned
            config (an instance of the method's ``config_model``) is how domain threshold tuning
            evaluates a frozen operating point on val/test.
        exemplar_selection: Exemplar-ordering mode passed to
            :func:`object_search.eval.sampling.sample_exemplars`; default ``"seeded-random"``
            preserves the committed draw, ``"size-representative"`` seeds from the median-area box.

    Returns:
        A report block: ``method``/``dataset``/``split``/``exemplar_count``, ``coverage``,
        ``overall`` (pooled metrics), ``slices`` (``by_symbol_size`` recall, ``by_crowding`` F1,
        ``by_plan_resolution`` F1), and ``per_image`` (one row per labelled image).
    """
    if split not in ("train", "val", "test"):
        raise ValueError(f"unknown split {split!r}; expected train/val/test")
    ids = research_image_ids(dataset, split, manifest_root)  # type: ignore[arg-type]
    logger.info(
        "research benchmark: {} x {}/{} at {} exemplar(s) over {} image(s)",
        method,
        dataset,
        split,
        exemplar_count,
        len(ids),
    )

    records: list[ImageResult] = []
    unlabelled: list[str] = []
    for image_id in ids:
        gt = load_research_ground_truth(research_root / f"{image_id}.gt.json")
        if gt is None:
            unlabelled.append(image_id)
            continue
        records.append(
            _run_one_research(
                method,
                image_id,
                research_root,
                gt,
                iou_threshold,
                dataset=dataset,
                split=split,
                exemplar_count=exemplar_count,
                seed=seed,
                config=config,
                exemplar_selection=exemplar_selection,
            )
        )

    # Per-slice analysis (EVAL-10 applied to floor plans): symbol-size RECALL is a GT-box-level
    # pooling over every labelled box; crowding and plan-resolution reuse the per-image _slice_by
    # with the research aggregator so they carry the full literature-metric column set at F1.
    all_gt_records = [rec for r in records for rec in r.gt_records]
    slices = {
        "by_symbol_size": _recall_by_size(all_gt_records),
        "by_crowding": _slice_by(records, _crowding_bucket, _aggregate_research),
        "by_plan_resolution": _slice_by(records, _plan_resolution_bucket, _aggregate_research),
    }

    return {
        "method": method,
        "dataset": dataset,
        "split": split,
        "exemplar_count": exemplar_count,
        "iou_threshold": iou_threshold,
        "coverage": {
            "images_requested": len(ids),
            "images_labelled": len(records),
            "images_unlabelled": sorted(unlabelled),
        },
        "overall": _aggregate_research(records),
        "slices": slices,
        # gt_records feeds `slices` above; it is excluded from per_image so the row shape is
        # unchanged and JSON-stable (a tuple field would otherwise round-trip to a list).
        "per_image": [r.model_dump(exclude={"gt_records"}) for r in records],
    }


def run_research_sweep(config: BenchmarkConfig) -> dict[str, Any]:
    """Sweep every method x dataset x {1,3 exemplars} x {val,test} and write the results file.

    This is the 11-03 consuming layer: it reuses the proven :func:`run_research_benchmark` cell
    (same ``_run_one_research`` scoring, same ``_aggregate_research`` literature-metric pooling) and
    adds only the three swept dimensions. For every configured dataset it reads the committed split
    manifest, and for each requested split that has ids it runs each ``exemplar_count`` through each
    method, reading converted sidecars + scenes from ``<research_root>/<dataset>/<split>/``.

    Two protocol rules are enforced structurally, not by hope:

    * **D-04 -- CARPK/PUCPR+ are test-only.** A test-only dataset's manifest has empty ``val`` ids,
      so ``ids_for("val")`` is empty and that split is skipped: **no val cell is ever emitted** for
      it, and tuning cannot touch it.
    * **The CI subset is never here.** This function is separate from :func:`run_benchmark`; the CI
      model-free chipset subset (``ci=true``) never reaches it, so the research sweep is gated on
      fetched archives/weights exactly like the full sweep gates on ``fetch-models``.

    The per-cell block carries the full literature column set (P/R/F1 + AP/AP50/AP75 + MAE/RMSE/NAE)
    via ``run_research_benchmark(...)["overall"]``, plus the per-slice ``slices`` block
    (``by_symbol_size`` recall, ``by_crowding`` F1, ``by_plan_resolution`` F1). Results are written
    to ``config.research_out`` (gitignored) -- distinct from the committed report the render emits.

    Args:
        config: A :class:`BenchmarkConfig` with ``datasets``/``splits``/``exemplar_counts``/``seed``
            set and ``research_root`` pointing at the converted dataset tree.

    Returns:
        The results mapping written to ``config.research_out``: provenance (``git_sha``), the swept
        dimensions, and a ``cells`` list of one block per (method, dataset, split, exemplar_count).

    Raises:
        ValueError: If ``research_root`` is unset -- there is nowhere to read scenes/labels from.
    """
    if config.research_root is None:
        raise ValueError("run_research_sweep needs research_root set (the converted datasets tree)")
    base = Path(config.research_root)
    if not base.is_absolute():
        base = repo_root() / base

    cells: list[dict[str, Any]] = []
    for dataset in config.datasets:
        manifest = load_split_manifest(dataset)
        for split in config.splits:
            if not manifest.ids_for(split):  # type: ignore[arg-type]
                # Test-only datasets (CARPK/PUCPR+) have empty val ids -> never a val cell (D-04).
                logger.info("research sweep: {}/{} has no ids; skipping", dataset, split)
                continue
            cell_root = base / dataset / split
            for count in config.exemplar_counts:
                for method in config.methods:
                    block = run_research_benchmark(
                        method,
                        dataset,
                        split,
                        cell_root,
                        exemplar_count=count,
                        iou_threshold=config.iou_threshold,
                        seed=config.seed,
                    )
                    cells.append(
                        {
                            "method": method,
                            "dataset": dataset,
                            "split": split,
                            "exemplar_count": count,
                            "coverage": block["coverage"],
                            "overall": block["overall"],
                            "slices": block["slices"],
                        }
                    )

    results: dict[str, Any] = {
        "git_sha": current_git_sha(),
        "iou_threshold": config.iou_threshold,
        "seed": config.seed,
        # Names how the 3-exemplar numbers were produced, for the report caption and the doc.
        "fusion": "k-shot late fusion",
        "datasets": list(config.datasets),
        "splits": list(config.splits),
        "exemplar_counts": list(config.exemplar_counts),
        "cells": cells,
    }

    out_path = Path(config.research_out)
    if not out_path.is_absolute():
        out_path = repo_root() / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    logger.info("research sweep: wrote {} ({} cell(s))", out_path, len(cells))
    return results


@hydra.main(version_base=None, config_path="../../../conf", config_name="benchmark")
def main(cfg: DictConfig) -> None:
    """Hydra entry point (``pixi run bench`` / ``pixi run bench-ci``).

    Hydra composes ``conf/benchmark.yaml`` and CLI overrides into ``cfg``; it is validated into a
    :class:`BenchmarkConfig` before anything runs, so an unknown key is a loud error, not a silent
    no-op.
    """
    raw = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(raw, dict):
        raise TypeError(f"benchmark config resolved to {type(raw).__name__}, expected a mapping")
    config = BenchmarkConfig.model_validate(raw)
    run_benchmark(config)


@hydra.main(version_base=None, config_path="../../../conf", config_name="benchmark")
def main_research(cfg: DictConfig) -> None:
    """Hydra entry point for the RESEARCH sweep (``pixi run bench-research``).

    Composes the same ``conf/benchmark.yaml`` (validated into :class:`BenchmarkConfig`) but runs
    :func:`run_research_sweep` over the ``datasets`` / ``splits`` / ``exemplar_counts`` dimensions
    rather than the chipset :func:`run_benchmark`. It is a separate entry precisely so the research
    sweep stays OUT of the default/CI chipset path; it needs ``research_root`` (the converted,
    fetched ``datasets/`` tree) on the CLI and so is gated on fetched archives, like ``bench`` on
    models.
    """
    raw = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(raw, dict):
        raise TypeError(f"benchmark config resolved to {type(raw).__name__}, expected a mapping")
    config = BenchmarkConfig.model_validate(raw)
    run_research_sweep(config)


if __name__ == "__main__":  # pragma: no cover - CLI dispatch, exercised via subprocess not import
    import sys

    # `bench-research` passes a leading `--research` sentinel selecting the research sweep. It is
    # stripped here BEFORE @hydra.main seizes argv, so `main_research` runs with THIS file as the
    # `__main__` module -- which is what lets Hydra resolve its file-relative `config_path`. A
    # `python -c` call or a separate entry module leaves `main_research`'s module non-`__main__`,
    # so Hydra falls back to a package-style `conf` lookup and dies with "Primary config module
    # 'conf' not found" (conf/ is a plain directory with no __init__.py, by design).
    if "--research" in sys.argv[1:]:
        sys.argv.remove("--research")
        main_research()
    else:
        main()
