"""Offline lab bench for the `propose-retrieve` retrieval/calibration-stage investigation.

Why this exists
----------------
`docs/reports/propose-retrieve-floorplans-improvement.md` fixed the PROPOSAL stage on floor-plan
doors (`proposal_conf` 0.4 -> 0.10) and, in doing so, exposed a second bottleneck: at the finalist
config the proposal stage puts 0.6395 of the crowded-bucket (11+ doors) GT on the table (T2), but
the pipeline only returns end-to-end recall 0.2615 in that bucket (T3-final) -- a ~41% transfer,
against ~0.82 pooled overall. That gap is attributed to "DINOv2 embedding + gmm calibration", but
nobody has looked INSIDE the gap yet: is it the gmm cut degrading as candidate count grows, is it
the embedding itself failing to discriminate near-identical small symbols, or is it NMS collapsing
genuine matches once the gate lets more overlapping proposals through? This script answers that by
tracing every GT box through the full pipeline (propose -> embed -> gmm calibrate -> threshold ->
NMS) and recording exactly where each one is lost.

This is a research harness, NOT part of the shipped package: it lives in `scripts/`, writes only
into this session's quick-task directory, and never touches `docs/benchmark/`. It calls the
library's own `propose`, `embed_regions`, `calibration.calibrate`, and `nms.nms` -- the identical
functions `propose_retrieve.search` composes -- so nothing here re-implements the method; it only
adds instrumentation between the steps `search()` runs opaquely.

Per-GT-box failure taxonomy
----------------------------
For each GT box in a plan, find every proposal overlapping it at IoU >= 0.5 (the eval TP
threshold) and take the one with the highest cosine score (`best`). Then:

* ``no_proposal`` -- no proposal covers the box at all. A proposal-stage miss, already
  characterized in the prior report; out of scope for a calibration-stage fix.
* ``below_threshold`` -- a covering proposal exists but `best`'s score does not clear the
  calibrated threshold (gmm cut, clamped to `similarity_floor`). A genuine calibration-stage loss.
* ``nms_suppressed`` -- `best` clears the threshold but is not in the post-threshold NMS survivor
  set (a higher-scoring, more-overlapping proposal for a DIFFERENT box won the suppression). A
  localisation-collision loss, not a threshold loss -- kept as a separate bucket because a floor
  fix cannot repair it.
* ``matched`` -- `best` clears the threshold and survives NMS. A true positive.

Alongside the taxonomy, every plan's full score distribution is logged: the gmm fit (means,
weights, degenerate flag, chosen threshold) and the mean/max score of proposals that do NOT cover
any GT box ("background" proposals) versus the mean score of each GT box's `best` proposal. This
is what lets the three candidate hypotheses in the task brief be told apart:

1. **gmm degrades with candidate count** -- shows up as the chosen threshold or the degenerate
   rate rising with `n_proposals` / crowding, independent of what the embeddings actually say.
2. **embedding discriminability is the ceiling** -- shows up as true-match scores themselves
   dropping in the crowded bucket (the embedding is not confident about the crop, regardless of
   the cut) and/or background scores climbing toward the floor (near-identical repeated symbols
   read as almost-matches).
3. **noise from the opened gate reaches the embedding stage** -- shows up as a `below_threshold`
   rate that grows with `n_proposals` even when `n_gt` is held fixed, because the flood of
   low-objectness proposals feeding the gmm shifts its background mode.

Usage:

    pixi run python scripts/propose_retrieve_calibration_experiment.py trace --split val
    pixi run python scripts/propose_retrieve_calibration_experiment.py trace --split val \\
        --conf 0.10 --floor 0.70 --only-crowded
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt
from loguru import logger

from object_search.eval.benchmark import _symbol_size_bucket  # imported, never restated
from object_search.eval.labels import load_research_ground_truth
from object_search.eval.sampling import sample_exemplars
from object_search.eval.splits import research_image_ids
from object_search.inference import FastSAMConfig
from object_search.provenance import current_git_sha, repo_root
from object_search.search import dino_dense, propose_retrieve
from object_search.search.common import calibration, nms
from object_search.search.proposals import propose

_OUT_DIR = Path(".planning/quick/260824-calibration-stage-propose-retrieve-floorplans/runs")
_RESEARCH_ROOT = Path("datasets")
_DATASETS = ("floorplans-door", "floorplans-window")
_SPLITS = ("val", "test")
_IOU = 0.5
_NMS_IOU = 0.3  # matches ProposeRetrieveConfig.nms_iou's shipped default

# The B0/T2 diagnostic crowding cuts (NOT benchmark._crowding_bucket's committed cuts), reused so
# this trace is directly comparable to the proposal-stage recall numbers already in the report.
_CROWDING_BUCKETS = ("1-3", "4-10", "11+")


def _crowding_bucket(n_gt: int) -> str:
    if n_gt <= 3:
        return "1-3"
    if n_gt <= 10:
        return "4-10"
    return "11+"


def _out_path(name: str) -> Path:
    out_dir = repo_root() / _OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / name


def _write(name: str, payload: dict[str, Any]) -> Path:
    path = _out_path(name)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    logger.info("wrote {}", path)
    return path


def _fmt(value: float | None) -> str:
    return f"{value:.3f}" if isinstance(value, int | float) else "  -  "


def _mean(values: Sequence[float]) -> float | None:
    values = [v for v in values if v is not None]
    return statistics.fmean(values) if values else None


def _load_scene(dataset: str, split: str, image_id: str) -> npt.NDArray[np.uint8] | None:
    for suffix in (".png", ".jpg", ".jpeg"):
        path = repo_root() / _RESEARCH_ROOT / dataset / split / f"{image_id}{suffix}"
        if path.is_file():
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is not None:
                return np.asarray(image, dtype=np.uint8)
    return None


def _trace_one_plan(
    image_id: str,
    dataset: str,
    split: str,
    *,
    conf: float,
    floor: float,
    seed: int,
    backend: object,
    inferencer: object,
) -> dict[str, Any] | None:
    """Trace every GT box in one plan through propose -> embed -> gmm calibrate -> NMS.

    Returns ``None`` when the plan has no sidecar/boxes/scene -- skipped, not a zero row.
    """
    sidecar = repo_root() / _RESEARCH_ROOT / dataset / split / f"{image_id}.gt.json"
    gt = load_research_ground_truth(sidecar)
    if gt is None or not gt.boxes:
        return None
    image = _load_scene(dataset, split, image_id)
    if image is None:
        return None

    exemplar = sample_exemplars(gt, count=1, seed=seed)[0]
    fastsam_config = FastSAMConfig(conf_thres=conf)
    proposals = propose(image, fastsam_config, backend=backend)  # type: ignore[arg-type]
    proposal_boxes = [p.box for p in proposals]

    config = propose_retrieve.ProposeRetrieveConfig(
        proposal_conf=conf, similarity_floor=floor, seed=seed
    )
    if not proposal_boxes:
        proposal_embeddings = np.zeros((0, 384), dtype=np.float32)
    else:
        proposal_embeddings = propose_retrieve.embed_regions(
            image,
            proposal_boxes,
            config,
            inferencer=inferencer,  # type: ignore[arg-type]
        )
    exemplar_embedding = propose_retrieve.embed_regions(
        image,
        [exemplar.box],
        config,
        inferencer=inferencer,  # type: ignore[arg-type]
    )[0]
    scores = (
        np.asarray(proposal_embeddings @ exemplar_embedding, dtype=np.float32)
        if len(proposal_boxes)
        else np.zeros((0,), dtype=np.float32)
    )

    if scores.size:
        calib = calibration.calibrate(scores.astype(np.float64), strategy="gmm", seed=seed)
        threshold = floor if calib.degenerate else max(calib.threshold, floor)
    else:
        calib = calibration.CalibrationResult(
            threshold=floor, strategy="gmm", reason="no proposals"
        )
        threshold = floor

    accepted_idx = [i for i, s in enumerate(scores) if float(s) > threshold]
    kept_local = nms.nms(
        [proposal_boxes[i] for i in accepted_idx],
        [float(scores[i]) for i in accepted_idx],
        _NMS_IOU,
    )
    kept_idx = {accepted_idx[j] for j in kept_local}

    plan_area = int(image.shape[0]) * int(image.shape[1])
    n_gt = len(gt.boxes)
    crowding_bucket = _crowding_bucket(n_gt)

    covering_prop_idx: set[int] = set()
    box_traces: list[dict[str, Any]] = []
    for gt_box in gt.boxes:
        candidates = [
            (i, float(scores[i])) for i, box in enumerate(proposal_boxes) if box.iou(gt_box) >= _IOU
        ]
        covering_prop_idx.update(i for i, _ in candidates)
        if not candidates:
            reason = "no_proposal"
            best_score = None
        else:
            best_i, best_score = max(candidates, key=lambda c: c[1])
            if best_i in kept_idx:
                reason = "matched"
            elif best_score > threshold:
                reason = "nms_suppressed"
            else:
                reason = "below_threshold"
        box_traces.append(
            {
                "reason": reason,
                "best_score": best_score,
                "n_candidates": len(candidates),
                "size_bucket": _symbol_size_bucket(gt_box.area, plan_area),
            }
        )

    background_scores = [float(scores[i]) for i in range(len(scores)) if i not in covering_prop_idx]
    # "covered" = ANY covering proposal regardless of accept/reject/NMS outcome; "true_positive" =
    # strictly the reason=="matched" subset. Keeping these separate matters: a plan-level "matched
    # score" that silently pooled both would understate how low the ACCEPTED matches actually score.
    covered_scores = [bt["best_score"] for bt in box_traces if bt["best_score"] is not None]
    true_positive_scores = [bt["best_score"] for bt in box_traces if bt["reason"] == "matched"]

    return {
        "image_id": image_id,
        "split": split,
        "n_gt": n_gt,
        "crowding_bucket": crowding_bucket,
        "n_proposals": len(proposal_boxes),
        "gmm_strategy": calib.strategy,
        "gmm_degenerate": calib.degenerate,
        "gmm_reason": calib.reason,
        "threshold": threshold,
        "n_background_proposals": len(background_scores),
        "mean_background_score": _mean(background_scores),
        "max_background_score": max(background_scores) if background_scores else None,
        "mean_covered_best_score": _mean([float(v) for v in covered_scores]),
        "mean_true_positive_score": _mean([float(v) for v in true_positive_scores]),
        "box_traces": box_traces,
        "reason_counts": {
            reason: sum(1 for bt in box_traces if bt["reason"] == reason)
            for reason in ("matched", "below_threshold", "nms_suppressed", "no_proposal")
        },
    }


def calibration_trace(
    dataset: str = "floorplans-door",
    split: str = "val",
    *,
    conf: float = 0.10,
    floor: float = 0.70,
    seed: int = 0,
    only_crowded: bool = False,
    limit: int | None = None,
) -> dict[str, Any]:
    """Trace every plan in ``split``, return per-plan rows plus a by-crowding-bucket rollup."""
    backend = propose_retrieve._get_backend()
    inferencer = dino_dense._get_inferencer()
    if backend is None or inferencer is None:
        raise RuntimeError(
            "fastsam-s and/or dinov2-small weight absent; run `pixi run fetch-models`."
        )

    ids = research_image_ids(dataset, split)  # type: ignore[arg-type]
    rows: list[dict[str, Any]] = []
    for image_id in ids if limit is None else ids[:limit]:
        row = _trace_one_plan(
            image_id,
            dataset,
            split,
            conf=conf,
            floor=floor,
            seed=seed,
            backend=backend,
            inferencer=inferencer,
        )
        if row is None:
            logger.warning("skipping {}/{}: no sidecar/scene", split, image_id)
            continue
        if only_crowded and row["crowding_bucket"] != "11+":
            continue
        rows.append(row)
        logger.info(
            "{} n_gt={} crowd={} n_prop={} thresh={} degenerate={} | matched={} below_thr={} "
            "nms_supp={} no_prop={}",
            image_id,
            row["n_gt"],
            row["crowding_bucket"],
            row["n_proposals"],
            _fmt(row["threshold"]),
            row["gmm_degenerate"],
            row["reason_counts"]["matched"],
            row["reason_counts"]["below_threshold"],
            row["reason_counts"]["nms_suppressed"],
            row["reason_counts"]["no_proposal"],
        )

    by_crowding: dict[str, Any] = {}
    for bucket in _CROWDING_BUCKETS:
        group = [r for r in rows if r["crowding_bucket"] == bucket]
        if not group:
            by_crowding[bucket] = {"n_plans": 0}
            continue
        total_gt = sum(r["n_gt"] for r in group)
        pooled_reasons = {
            reason: sum(r["reason_counts"][reason] for r in group)
            for reason in ("matched", "below_threshold", "nms_suppressed", "no_proposal")
        }
        all_covered_scores = [
            bt["best_score"]
            for r in group
            for bt in r["box_traces"]
            if bt["best_score"] is not None
        ]
        all_true_positive_scores = [
            bt["best_score"] for r in group for bt in r["box_traces"] if bt["reason"] == "matched"
        ]
        all_below_scores = [
            bt["best_score"]
            for r in group
            for bt in r["box_traces"]
            if bt["reason"] == "below_threshold"
        ]
        by_crowding[bucket] = {
            "n_plans": len(group),
            "total_gt": total_gt,
            "mean_n_proposals": _mean([float(r["n_proposals"]) for r in group]),
            "mean_threshold": _mean([float(r["threshold"]) for r in group]),
            "degenerate_rate": _mean([1.0 if r["gmm_degenerate"] else 0.0 for r in group]),
            "mean_background_score": _mean(
                [
                    r["mean_background_score"]
                    for r in group
                    if r["mean_background_score"] is not None
                ]
            ),
            "mean_covered_best_score": _mean([float(v) for v in all_covered_scores]),
            "mean_true_positive_score": _mean([float(v) for v in all_true_positive_scores]),
            "mean_below_threshold_score": _mean([float(v) for v in all_below_scores]),
            "pooled_reason_fractions": {
                reason: (count / total_gt if total_gt else None)
                for reason, count in pooled_reasons.items()
            },
            "pooled_reason_counts": pooled_reasons,
            # Attrition rate WITHIN the retrieval/calibration stage only -- denominator excludes
            # no_proposal (a proposal-stage miss, out of scope here), so this isolates exactly the
            # fraction of GT boxes the calibration stage itself loses given a covering proposal.
            "retrieval_stage_loss_rate": (
                (pooled_reasons["below_threshold"] + pooled_reasons["nms_suppressed"])
                / (total_gt - pooled_reasons["no_proposal"])
                if (total_gt - pooled_reasons["no_proposal"]) > 0
                else None
            ),
        }

    report = {
        "dataset": dataset,
        "split": split,
        "proposal_conf": conf,
        "similarity_floor": floor,
        "seed": seed,
        "git_sha": current_git_sha(),
        "rows": rows,
        "by_crowding": by_crowding,
    }
    return report


def _log_summary(report: dict[str, Any]) -> None:
    logger.info(
        "--- calibration trace: {}/{} conf={} floor={} ---",
        report["dataset"],
        report["split"],
        report["proposal_conf"],
        report["similarity_floor"],
    )
    for bucket in _CROWDING_BUCKETS:
        entry = report["by_crowding"][bucket]
        if entry.get("n_plans", 0) == 0:
            logger.info("  {:<6} (no plans)", bucket)
            continue
        logger.info(
            "  {:<6} n_plans={:<3} n_gt={:<4} mean_n_prop={} mean_thresh={} degenerate_rate={} | "
            "bg_score={} covered_score={} tp_score={} below_thr_score={} | loss_rate={} fractions={}",
            bucket,
            entry["n_plans"],
            entry["total_gt"],
            _fmt(entry["mean_n_proposals"]),
            _fmt(entry["mean_threshold"]),
            _fmt(entry["degenerate_rate"]),
            _fmt(entry["mean_background_score"]),
            _fmt(entry["mean_covered_best_score"]),
            _fmt(entry["mean_true_positive_score"]),
            _fmt(entry["mean_below_threshold_score"]),
            _fmt(entry["retrieval_stage_loss_rate"]),
            {k: _fmt(v) for k, v in entry["pooled_reason_fractions"].items()},
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=("trace",))
    parser.add_argument("--dataset", default="floorplans-door", choices=_DATASETS)
    parser.add_argument("--split", default="val", choices=_SPLITS)
    parser.add_argument("--name", default=None, help="Output stem under runs/")
    parser.add_argument("--conf", type=float, default=0.10)
    parser.add_argument("--floor", type=float, default=0.70)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--only-crowded", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    return parser


def main(argv: Sequence[str]) -> int:
    args = _parser().parse_args(argv)
    report = calibration_trace(
        args.dataset,
        args.split,
        conf=args.conf,
        floor=args.floor,
        seed=args.seed,
        only_crowded=args.only_crowded,
        limit=args.limit,
    )
    _log_summary(report)
    name = args.name or f"trace-{args.dataset}-{args.split}-c{args.conf:.2f}-f{args.floor:.2f}"
    _write(f"{name}.json", report)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
