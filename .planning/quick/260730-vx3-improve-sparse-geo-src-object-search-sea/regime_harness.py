"""Throwaway synthetic-regime harness for quick task 260730-vx3 (NOT shipped surface).

Lives under the quick-task directory on purpose: it must not enter ``src/`` (the shipped surface)
or ``tests/`` (the coverage denominator). Its only job is the **regression guard** for the
sparse-geo improvement loop -- per-regime P/R/F1 + AP + p50 latency, so a floor-plan win that
quietly breaks the synthetic regimes is caught.

Regimes (grouping the committed benchmark image sets):

* ``EASY``      -- ``chipset-*``            (NCC-favourable chip insertions, EVAL-19)
* ``TEXTURED``  -- ``textured-plain-*``     (keypoint-favourable, EVAL-20)
* ``VARIED``    -- ``textured-varied-*``    (scale/rotation/brightness variation)
* ``CLUTTERED`` -- ``textured-cluttered-*`` (noise + distractors)

With **no overrides** the run delegates to :func:`object_search.eval.benchmark._run_one`, so the
default-config numbers are produced by the exact shipped code path. With overrides it mirrors that
function's scoring with a non-default config (the only thing ``_run_one`` cannot do).

Usage::

    pixi run python .planning/quick/<dir>/regime_harness.py
    pixi run python .planning/quick/<dir>/regime_harness.py --overrides '{"allow_mirror": true}'
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from loguru import logger

from object_search.eval.benchmark import ImageResult, _load_scene, _run_one, _scale_bucket
from object_search.eval.labels import (
    GroundTruth,
    chipset_image_ids,
    load_ground_truth,
    textured_image_ids,
)
from object_search.eval.metrics import (
    average_precision,
    match_predictions,
    precision_recall_f1,
)
from object_search.log import setup_logging
from object_search.schemas.geometry import BBox
from object_search.search import get_method

_REGIMES: tuple[tuple[str, str], ...] = (
    ("EASY", "chipset-"),
    ("TEXTURED", "textured-plain-"),
    ("VARIED", "textured-varied-"),
    ("CLUTTERED", "textured-cluttered-"),
)


def _regime_of(image_id: str) -> str | None:
    for name, prefix in _REGIMES:
        if image_id.startswith(prefix):
            return name
    return None


def _run_one_with_config(
    method: str, image_id: str, gt: GroundTruth, iou_threshold: float, overrides: dict[str, Any]
) -> ImageResult:
    """``_run_one`` with a non-default config -- the same scoring, a different config object."""
    canvas = f"{gt.width}x{gt.height}" if gt.width and gt.height else None
    spec = get_method(method)
    try:
        scene = _load_scene(image_id)
        result = spec.fn(scene, gt.exemplar, spec.config_model(**overrides))
    except Exception as exc:
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


def _pool(records: list[ImageResult]) -> dict[str, Any]:
    """Micro-average tp/fp/fn into P/R/F1, mean AP, and p50 latency over one regime."""
    tp = sum(r.tp or 0 for r in records)
    fp = sum(r.fp or 0 for r in records)
    fn = sum(r.fn or 0 for r in records)
    precision, recall, f1 = precision_recall_f1(tp, fp, fn)
    aps = [r.ap for r in records if r.ap is not None]
    lats = [r.latency_ms for r in records if r.latency_ms is not None]
    return {
        "n_images": len(records),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "ap": (sum(aps) / len(aps)) if aps else None,
        "p50_latency_ms": statistics.median(lats) if lats else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", default="sparse-geo")
    parser.add_argument("--overrides", default="{}", help="JSON config overrides.")
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--label", default="", help="Label echoed into the JSON output.")
    parser.add_argument("--out", default="", help="Optional path to write the JSON report.")
    args = parser.parse_args()
    setup_logging("INFO")

    overrides: dict[str, Any] = json.loads(args.overrides)
    image_ids = (*chipset_image_ids(), *textured_image_ids())

    per_regime: dict[str, list[ImageResult]] = {name: [] for name, _ in _REGIMES}
    for image_id in image_ids:
        regime = _regime_of(image_id)
        if regime is None:
            continue
        gt = load_ground_truth(image_id)
        if gt is None:
            continue
        record = (
            _run_one(args.method, image_id, gt, args.iou)
            if not overrides
            else _run_one_with_config(args.method, image_id, gt, args.iou, overrides)
        )
        per_regime[regime].append(record)

    report = {
        "label": args.label,
        "method": args.method,
        "overrides": overrides,
        "iou_threshold": args.iou,
        "regimes": {name: _pool(records) for name, records in per_regime.items()},
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    logger.info("regime harness [{}]:\n{}", args.label or "default", text)
    if args.out:
        with Path(args.out).open("w", encoding="utf-8") as handle:
            handle.write(text + "\n")


if __name__ == "__main__":
    main()
