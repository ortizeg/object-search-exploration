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
    scene_path,
)
from object_search.eval.metrics import average_precision, match_predictions, precision_recall_f1
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
        # The full sweep includes the chipset (NCC-favourable) alongside the configured synthetic
        # scenes (learned-favourable) so the per-slice crossover has both sides present.
        images = tuple(dict.fromkeys((*chipset_image_ids(), *self.image_ids)))
        return self.methods, images


class ImageResult(BaseModel):
    """One method's result on one image: the metrics plus the slice keys it is grouped by."""

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
