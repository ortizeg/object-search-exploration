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
from collections.abc import Callable
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
    scene_path,
    textured_image_ids,
)
from object_search.eval.metrics import (
    average_precision,
    average_precision_coco,
    counting_errors,
    match_predictions,
    precision_recall_f1,
)
from object_search.eval.splits import research_image_ids
from object_search.provenance import current_git_sha, repo_root
from object_search.schemas.geometry import BBox
from object_search.schemas.search import SearchOutcome
from object_search.search import get_method

# The classical, weight-free methods. The CI subset is exactly these two, because they need no
# ONNX weights and so run without `fetch-models` (EVAL-19).
_MODEL_FREE_METHODS: tuple[str, ...] = ("ncc", "sparse-geo")

# The scale-varied synthetic scenes to include in the full sweep so the crossover has a
# learned-favourable side to show against the fixed-scale chipset.
_SYNTHETIC_IMAGE_IDS: tuple[str, ...] = ("scatter-scaled", "cluttered-distractors")


class BenchmarkConfig(BaseModel):
    """Validated benchmark configuration -- the Pydantic gate over Hydra's untyped ``DictConfig``.

    Hydra composes the YAML into a ``DictConfig``; this model is what turns it into something with
    real types and defaults, so a typo in ``conf/benchmark.yaml`` fails loudly here rather than
    silently sweeping the wrong set.

    Attributes:
        methods: Registry keys to run. Ignored when ``ci`` is set (the CI subset is fixed).
        image_ids: Scene ids to sweep. Ignored when ``ci`` is set (chipset only).
        ci: Model-free subset -- ``ncc`` + classical ``sparse-geo`` over the chipset, no weights.
        iou_threshold: IoU for a prediction to count as a true positive.
        out: Output path for ``results.json``; resolved against the repo root when relative.
        ci_image_limit: Cap on chipset images in the CI subset, keeping CI runtime bounded while
            still spanning several canvas sizes.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    methods: tuple[str, ...] = ("ncc", "sparse-geo", "dino-dense", "propose-retrieve")
    image_ids: tuple[str, ...] = _SYNTHETIC_IMAGE_IDS
    ci: bool = False
    iou_threshold: float = Field(default=0.5, gt=0.0, le=1.0)
    out: str = "docs/benchmark/results.json"
    ci_image_limit: int = Field(default=6, ge=1)

    def resolve_run_set(self) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """Return the ``(methods, image_ids)`` actually swept, applying the CI subset rule."""
        if self.ci:
            images = chipset_image_ids()[: self.ci_image_limit]
            return _MODEL_FREE_METHODS, images
        # The full sweep includes the chipset (NCC-favourable), the textured regimes (EVAL-20,
        # keypoint- and deep-feature-favourable), and the configured synthetic scenes, so the
        # per-slice crossover has every side present.
        images = tuple(
            dict.fromkeys((*chipset_image_ids(), *textured_image_ids(), *self.image_ids))
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
    records: list[ImageResult], key: Callable[[ImageResult], str | int | None]
) -> dict[str, Any]:
    """Group records by ``key(record)`` (skipping ``None`` keys) and aggregate each group."""
    groups: dict[str, list[ImageResult]] = {}
    for record in records:
        bucket = key(record)
        if bucket is None:
            continue
        groups.setdefault(str(bucket), []).append(record)
    return {bucket: _aggregate(group) for bucket, group in sorted(groups.items())}


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
            "per_image": [r.model_dump() for r in records],
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
) -> ImageResult:
    """Run one method on one research image and score it with the full literature metric set.

    Mirrors :func:`_run_one` (same match/precision/recall/candidate-log logic, same broad
    error-catch so a missing weight degrades one cell rather than aborting the sweep) and adds the
    COCO AP sweep plus the per-image predicted/true counts. The per-image ``ap`` field is the COCO
    mean here (the literature's headline AP), with ``ap50``/``ap75`` alongside it.
    """
    canvas = f"{gt.width}x{gt.height}" if gt.width and gt.height else None
    true_count = gt.achieved_count
    spec = get_method(method)
    try:
        scene = _load_research_scene(research_root, image_id)
        exemplar = gt.exemplar_at(exemplar_count)
        result = spec.fn(scene, exemplar, spec.config_model())
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
    tp, fp, fn = match_predictions(pred_boxes, gt.boxes, iou_threshold)
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
    manifest_root: Path | None = None,
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
        exemplar_count: Exemplars to seed the run with (1 = the product operating point).
        iou_threshold: IoU for a prediction to count as a true positive at the P/R/F1 level.
        manifest_root: Optional base dir for the committed split manifest (tests use ``tmp_path``).

    Returns:
        A report block: ``method``/``dataset``/``split``/``exemplar_count``, ``coverage``,
        ``overall`` (pooled metrics), and ``per_image`` (one row per labelled image).
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
            )
        )

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
        "per_image": [r.model_dump() for r in records],
    }


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


if __name__ == "__main__":
    main()
