"""Offline lab bench for the `propose-retrieve` floor-plan investigation (quick task 260812-m8m).

Why this exists
---------------
`docs/eval/floorplans-findings.md` measured `propose-retrieve` on floor-plan DOORS at test
F1 0.459 / precision 0.55 / recall 0.39. A session-local scratch diagnostic suggested the leak is
in the PROPOSAL stage rather than the DINOv2 retrieval stage -- FastSAM's everything-mode proposal
COUNT barely scales with true instance count, and its fixed 1024 letterbox shrinks a small symbol on
a large plan below detectability. That diagnostic was gitignored, so none of its numbers may be
cited. **This script is the committed re-derivation**: every number quoted in
`docs/reports/propose-retrieve-floorplans-improvement.md` and in the quick task's `EXPERIMENTS.md`
comes from a run of this file, against the committed `dataset_splits/floorplans-door.split.json`.

It is a research harness, NOT part of the shipped package: it lives in `scripts/`, writes only into
the quick task's own directory, and never touches `docs/benchmark/`. Everything it measures goes
through the library's own scorers -- `propose`, `run_research_benchmark`, `run_domain_tuning`,
`run_benchmark` -- so a number here is comparable one-for-one with a committed number.

The measurement it adds
-----------------------
The library scores FINAL detections. The question here is where recall is lost, so this script adds
one thing the library does not have: **proposal-stage recall** -- the fraction of GT boxes that have
*any* proposal over them at IoU >= 0.5, before embedding, retrieval, or thresholding runs. Final
recall can never exceed it, so it is the ceiling the retrieval stage works under.

Two conventions matter and are both reported, because they disagree:

* **eval convention (`proposal_recall`)** -- the denominator is every box in `gt.boxes`. This is what
  `run_research_benchmark` uses: the sampled exemplar REMAINS in the recall denominator and is
  scored like any other instance (see `_run_one_research`). Use this one to compare proposal-stage
  recall against final recall.
* **diagnostic convention (`proposal_recall_excl_exemplar`)** -- the denominator drops the box the
  exemplar was drawn from. The session's scratch diagnostic used this. It is reported only so the
  committed numbers can be checked against the diagnosed 0.74 / 0.51 / 0.27 pattern without
  ambiguity about which denominator produced it.

Bucketings
----------
* **crowding** -- `1-3` / `4-10` / `11+` GT boxes per plan. These cuts come from the diagnosis this
  script re-derives; they deliberately differ from `benchmark._crowding_bucket`'s `1`/`2-5`/`6-15`/
  `16+`, which is the committed *reporting* slice. Both are kept: `final_metrics` reports the
  committed cuts, `proposal_stage_recall` reports the diagnostic cuts.
* **symbol size** -- small / medium / large by box-area fraction of plan area, importing
  `benchmark._symbol_size_bucket` rather than restating 0.004 / 0.016, so the proposal-stage table
  and the committed `by_symbol_size` recall table can never drift apart.

Protocol
--------
Tune on **val** (argmax F1 @ IoU 0.5), freeze, read **test** exactly once per finalist. Precision
and recall are logged separately per trial, never F1 alone: a proposal-stage change that buys recall
by flooding the scene with low-objectness regions shows up as a precision collapse, and pooling to
F1 first would hide it.

Usage (each entry is one experiment; run them one at a time, they take minutes to hours):

    pixi run python scripts/propose_retrieve_floorplans_experiment.py b0
    pixi run python scripts/propose_retrieve_floorplans_experiment.py b1 floorplans-door
    pixi run python scripts/propose_retrieve_floorplans_experiment.py b2
    pixi run python scripts/propose_retrieve_floorplans_experiment.py b3
"""

from __future__ import annotations

import json
import statistics
import sys
from collections.abc import Sequence
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt
from loguru import logger

from object_search.eval.benchmark import (
    BenchmarkConfig,
    _symbol_size_bucket,  # imported, never restated -- see the module docstring
    run_benchmark,
    run_research_benchmark,
)
from object_search.eval.labels import load_research_ground_truth
from object_search.eval.sampling import sample_exemplars
from object_search.eval.splits import research_image_ids
from object_search.eval.tuning import run_domain_tuning
from object_search.inference import FastSAMConfig
from object_search.provenance import current_git_sha, repo_root
from object_search.schemas.geometry import BBox
from object_search.search import propose_retrieve
from object_search.search.proposals import propose

# Where every raw JSON report lands: the quick task's own directory, never docs/benchmark/.
_OUT_DIR = Path(".planning/quick/260812-m8m-improve-propose-retrieve-recall-on-floor/runs")
_RESEARCH_ROOT = Path("datasets")
_DATASETS = ("floorplans-door", "floorplans-window")
_SPLITS = ("val", "test")

# A GT box counts as PROPOSED when some proposal overlaps it at least this much -- the same IoU the
# benchmark uses for a true positive, so proposal-stage recall is an honest ceiling on final recall.
_IOU = 0.5

# The diagnosis's crowding cuts (see the module docstring; NOT benchmark._crowding_bucket's cuts).
_CROWDING_BUCKETS = ("1-3", "4-10", "11+")

# The five regimes the guardrail watches, keyed by image-id prefix. `docs/reports/
# propose-retrieve-improvement.md` records propose-retrieve at EASY 0.93 / TEXTURED 0.96 /
# VARIED 0.94 / CLUTTERED 0.82 / synthetic 0.91; any floor-plan change must leave these alone.
_REGIME_PREFIXES: tuple[tuple[str, str], ...] = (
    ("EASY (chipset)", "chipset-"),
    ("TEXTURED (plain)", "textured-plain-"),
    ("VARIED (scale/rotation)", "textured-varied-"),
    ("CLUTTERED", "textured-cluttered-"),
    ("real-objects", "real-"),
)


def _crowding_bucket(n_gt: int) -> str:
    """Bucket a plan by how many GT symbols it carries -- the axis the diagnosis split on."""
    if n_gt <= 3:
        return "1-3"
    if n_gt <= 10:
        return "4-10"
    return "11+"


def _out_path(name: str) -> Path:
    """Resolve a report path under the quick task's runs/ directory, creating it."""
    out_dir = repo_root() / _OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / name


def _write(name: str, payload: dict[str, Any]) -> Path:
    """Dump one experiment's full result to JSON so every logged number is re-readable."""
    path = _out_path(name)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    logger.info("wrote {}", path)
    return path


def _fmt(value: float | None) -> str:
    return f"{value:.3f}" if isinstance(value, int | float) else "  -  "


def _mean(values: Sequence[float]) -> float | None:
    """Mean, or ``None`` for an empty group -- an undefined mean is never reported as 0.0."""
    return statistics.fmean(values) if values else None


def _load_scene(dataset: str, split: str, image_id: str) -> npt.NDArray[np.uint8] | None:
    """Read one converted floor-plan scene as BGR uint8, or ``None`` when it is not on disk."""
    for suffix in (".png", ".jpg", ".jpeg"):
        path = repo_root() / _RESEARCH_ROOT / dataset / split / f"{image_id}{suffix}"
        if path.is_file():
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is not None:
                return np.asarray(image, dtype=np.uint8)
    return None


def _matched_flags(gt_boxes: Sequence[BBox], proposal_boxes: Sequence[BBox]) -> list[bool]:
    """One flag per GT box: is there ANY proposal over it at IoU >= 0.5?

    Deliberately NOT a one-to-one assignment. The question is whether the proposal stage put a
    usable region on each instance at all -- one proposal covering two GT boxes still means the
    retrieval stage was given something to work with for both, and counting it once would
    understate the ceiling this measurement exists to establish.
    """
    return [any(gt.iou(box) >= _IOU for box in proposal_boxes) for gt in gt_boxes]


# ------------------------------------------------------------------ (a) proposal-stage recall


def proposal_stage_recall(
    dataset: str = "floorplans-door",
    splits: Sequence[str] = _SPLITS,
    *,
    conf: float = 0.4,
    seed: int = 0,
    limit: int | None = None,
) -> dict[str, Any]:
    """Run the PROPOSAL stage alone over a split and measure how much of the GT it covers.

    For each plan: run `propose` with a `FastSAMConfig` (no embedding, no retrieval, no threshold),
    then record `n_proposals`, `n_gt`, proposal recall under both denominators, the plan's pixel
    dimensions, and the wall-clock of the proposal call. Aggregated into crowding buckets (per-plan)
    and symbol-size buckets (per-GT-box).

    Args:
        dataset: Dataset key, e.g. ``"floorplans-door"``.
        splits: Which splits to walk (both by default -- this is a diagnostic, not a tuning read).
        conf: FastSAM objectness gate (`ProposeRetrieveConfig.proposal_conf`'s default is 0.4).
        seed: Exemplar-sampler seed (D-11), so the excluded exemplar matches the eval's draw.
        limit: Stop after this many plans per split -- for the cost probe, never for a reported table.

    Returns:
        ``{"rows": [...], "by_crowding": {...}, "by_size": {...}, "overall": {...}}``.
    """
    backend = propose_retrieve._get_backend()
    if backend is None:
        raise RuntimeError(
            "the fastsam-s weight is absent; proposal-stage recall cannot be measured. "
            "Run `pixi run -e export fetch-models --only fastsam-s`."
        )

    rows: list[dict[str, Any]] = []
    for split in splits:
        ids = research_image_ids(dataset, split)  # type: ignore[arg-type]
        for image_id in ids if limit is None else ids[:limit]:
            sidecar = repo_root() / _RESEARCH_ROOT / dataset / split / f"{image_id}.gt.json"
            gt = load_research_ground_truth(sidecar)
            if gt is None or not gt.boxes:
                logger.warning("skipping {}/{}: no sidecar or no boxes", split, image_id)
                continue
            image = _load_scene(dataset, split, image_id)
            if image is None:
                logger.warning("skipping {}/{}: scene not on disk", split, image_id)
                continue

            exemplar = sample_exemplars(gt, count=1, seed=seed)[0]
            started = perf_counter()
            proposals = propose(image, FastSAMConfig(conf_thres=conf), backend=backend)
            proposal_ms = (perf_counter() - started) * 1000.0
            proposal_boxes = [p.box for p in proposals]

            matched = _matched_flags(gt.boxes, proposal_boxes)
            # The exemplar's own box is dropped only for the diagnostic-convention number; the eval
            # convention keeps it (see the module docstring).
            is_exemplar = [box.iou(exemplar.box) >= _IOU for box in gt.boxes]
            excl = [m for m, ex in zip(matched, is_exemplar, strict=True) if not ex]

            height, width = int(image.shape[0]), int(image.shape[1])
            plan_area = width * height
            rows.append(
                {
                    "split": split,
                    "image_id": image_id,
                    "width": width,
                    "height": height,
                    "n_gt": len(gt.boxes),
                    "n_gt_excl_exemplar": len(excl),
                    "n_proposals": len(proposal_boxes),
                    "n_matched": sum(matched),
                    "proposal_recall": sum(matched) / len(matched),
                    "proposal_recall_excl_exemplar": (sum(excl) / len(excl)) if excl else None,
                    "proposal_ms": proposal_ms,
                    "crowding_bucket": _crowding_bucket(len(gt.boxes)),
                    "size_buckets": [_symbol_size_bucket(box.area, plan_area) for box in gt.boxes],
                    "matched": matched,
                }
            )
            logger.info(
                "{}/{} {}x{}: n_gt={} n_prop={} proposal_recall={} ({:.0f}ms)",
                split,
                image_id,
                width,
                height,
                len(gt.boxes),
                len(proposal_boxes),
                _fmt(rows[-1]["proposal_recall"]),
                proposal_ms,
            )

    return {
        "dataset": dataset,
        "splits": list(splits),
        "proposal_conf": conf,
        "seed": seed,
        "git_sha": current_git_sha(),
        "rows": rows,
        "by_crowding": _aggregate_by_crowding(rows),
        "by_size": _aggregate_by_size(rows),
        "overall": _aggregate_by_crowding(rows, single_bucket=True).get("all"),
        "worst_plans": sorted(rows, key=lambda r: (r["proposal_recall"], -r["n_gt"]))[:5],
        "zero_recall_plans": [
            {k: r[k] for k in ("image_id", "split", "width", "height", "n_gt", "n_proposals")}
            for r in rows
            if r["proposal_recall"] == 0.0
        ],
        "attribution": _attribution(rows),
    }


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    """Pearson correlation, or ``None`` when either side has no spread (undefined, not 0.0)."""
    if len(xs) < 2:
        return None
    mean_x, mean_y = statistics.fmean(xs), statistics.fmean(ys)
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
    var = sum((x - mean_x) ** 2 for x in xs) * sum((y - mean_y) ** 2 for y in ys)
    return cov / (var**0.5) if var > 0 else None


def _attribution(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Separate the two candidate failure modes, which the raw bucket table conflates.

    The diagnosis this script re-derives named two mechanisms: (1) a proposal BUDGET that does not
    scale with instance count, and (2) FastSAM's fixed 1024 letterbox shrinking symbols on large
    plans. They make OPPOSITE predictions about plan size -- (2) says big plans do worse, (1) is
    indifferent to plan size -- so this block reports plan size and crowding as separate axes plus
    the size x crowding cross-tab that de-confounds them. Attribution matters practically, not just
    editorially: tiling buys magnification (which attacks 2) AND budget (which attacks 1), and
    knowing which lever is live decides the tile size.
    """
    recall = [float(r["proposal_recall"]) for r in rows]
    n_gt = [float(r["n_gt"]) for r in rows]
    n_prop = [float(r["n_proposals"]) for r in rows]
    long_side = [float(max(r["width"], r["height"])) for r in rows]
    area = [float(r["width"] * r["height"]) for r in rows]

    cross: dict[str, Any] = {}
    for bucket in _CROWDING_BUCKETS:
        for label, keep in (
            ("<=1024", lambda r: max(r["width"], r["height"]) <= 1024),
            (">1024", lambda r: max(r["width"], r["height"]) > 1024),
        ):
            group = [r for r in rows if r["crowding_bucket"] == bucket and keep(r)]
            cross[f"{bucket} / {label}"] = {
                "n_plans": len(group),
                "mean_n_gt": _mean([float(r["n_gt"]) for r in group]),
                "mean_n_proposals": _mean([float(r["n_proposals"]) for r in group]),
                "mean_proposal_recall": _mean([float(r["proposal_recall"]) for r in group]),
            }
    return {
        "correlations": {
            "recall_vs_n_gt": _pearson(n_gt, recall),
            "recall_vs_plan_long_side": _pearson(long_side, recall),
            "recall_vs_plan_area": _pearson(area, recall),
            "n_proposals_vs_n_gt": _pearson(n_gt, n_prop),
            "n_proposals_vs_plan_area": _pearson(area, n_prop),
        },
        "crowding_x_plan_size": cross,
    }


def _aggregate_by_crowding(
    rows: Sequence[dict[str, Any]], *, single_bucket: bool = False
) -> dict[str, Any]:
    """Pool per-plan rows into crowding buckets.

    Both a MEAN-of-per-plan recall (the form the diagnosis reported, so the reproduction check is
    apples-to-apples) and a POOLED recall (matched/total, the micro-average the benchmark uses) are
    emitted, because on a set with wildly different instance counts per plan they differ a lot and
    quoting only one invites a false comparison.
    """
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = "all" if single_bucket else str(row["crowding_bucket"])
        groups.setdefault(key, []).append(row)

    out: dict[str, Any] = {}
    order = ("all",) if single_bucket else _CROWDING_BUCKETS
    for bucket in order:
        items = groups.get(bucket, [])
        if not items:
            out[bucket] = {
                "n_plans": 0,
                "mean_proposal_recall": None,
                "pooled_proposal_recall": None,
            }
            continue
        total_gt = sum(int(r["n_gt"]) for r in items)
        total_matched = sum(int(r["n_matched"]) for r in items)
        excl = [
            r["proposal_recall_excl_exemplar"]
            for r in items
            if r["proposal_recall_excl_exemplar"] is not None
        ]
        out[bucket] = {
            "n_plans": len(items),
            "mean_n_gt": _mean([float(r["n_gt"]) for r in items]),
            "mean_n_proposals": _mean([float(r["n_proposals"]) for r in items]),
            "mean_proposal_recall": _mean([float(r["proposal_recall"]) for r in items]),
            "mean_proposal_recall_excl_exemplar": _mean([float(v) for v in excl]),
            "pooled_proposal_recall": (total_matched / total_gt) if total_gt else None,
            "mean_proposal_ms": _mean([float(r["proposal_ms"]) for r in items]),
        }
    return out


def _aggregate_by_size(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Pool per-GT-box matched/total into symbol-size buckets -- the same cuts as by_symbol_size."""
    totals: dict[str, list[int]] = {b: [0, 0] for b in ("small", "medium", "large")}
    for row in rows:
        for bucket, matched in zip(row["size_buckets"], row["matched"], strict=True):
            pair = totals.setdefault(str(bucket), [0, 0])
            pair[0] += int(bool(matched))
            pair[1] += 1
    return {
        bucket: {
            "n_gt": total,
            "n_matched": matched,
            "proposal_recall": (matched / total) if total else None,
        }
        for bucket, (matched, total) in totals.items()
    }


def _log_proposal_stage(report: dict[str, Any]) -> None:
    """Log the two bucket tables -- the shape the report quotes."""
    logger.info("--- proposal-stage recall by CROWDING bucket ({}) ---", report["dataset"])
    for bucket in _CROWDING_BUCKETS:
        entry = report["by_crowding"][bucket]
        logger.info(
            "  {:<6} n_plans={:<3} mean_n_gt={} mean_n_prop={} | mean_recall={} "
            "(excl_exemplar={}) pooled={}",
            bucket,
            entry["n_plans"],
            _fmt(entry.get("mean_n_gt")),
            _fmt(entry.get("mean_n_proposals")),
            _fmt(entry.get("mean_proposal_recall")),
            _fmt(entry.get("mean_proposal_recall_excl_exemplar")),
            _fmt(entry.get("pooled_proposal_recall")),
        )
    logger.info("--- proposal-stage recall by SYMBOL-SIZE bucket ---")
    for bucket, entry in report["by_size"].items():
        logger.info(
            "  {:<7} n_gt={:<5} recall={}", bucket, entry["n_gt"], _fmt(entry["proposal_recall"])
        )
    logger.info("--- attribution: which failure mode does the data support? ---")
    for label, value in report["attribution"]["correlations"].items():
        logger.info("  pearson {:<28} = {}", label, _fmt(value))
    for label, entry in report["attribution"]["crowding_x_plan_size"].items():
        logger.info(
            "  {:<16} n={:<3} mean_n_gt={} mean_n_prop={} mean_recall={}",
            label,
            entry["n_plans"],
            _fmt(entry["mean_n_gt"]),
            _fmt(entry["mean_n_proposals"]),
            _fmt(entry["mean_proposal_recall"]),
        )
    logger.info("--- worst plans (lowest proposal recall) ---")
    for row in report["worst_plans"]:
        logger.info(
            "  {} {}x{} n_gt={} n_prop={} recall={}",
            row["image_id"],
            row["width"],
            row["height"],
            row["n_gt"],
            row["n_proposals"],
            _fmt(row["proposal_recall"]),
        )


# ------------------------------------------------------------------------- (b) final metrics


def final_metrics(
    dataset: str = "floorplans-door",
    split: str = "test",
    *,
    config: propose_retrieve.ProposeRetrieveConfig | None = None,
    exemplar_count: int = 1,
    seed: int = 0,
) -> dict[str, Any]:
    """Pooled P/R/F1 @ IoU 0.5 plus recall-by-symbol-size, straight from the committed scorer.

    A thin wrapper over `run_research_benchmark` so final recall is produced by exactly the code
    that produced `docs/eval/floorplans-findings.md`, and is therefore directly comparable with the
    proposal-stage ceiling measured above.
    """
    block = run_research_benchmark(
        "propose-retrieve",
        dataset,
        split,
        repo_root() / _RESEARCH_ROOT / dataset / split,
        exemplar_count=exemplar_count,
        iou_threshold=_IOU,
        seed=seed,
        config=config,
    )
    overall = block["overall"]
    logger.info(
        "final[{}/{}]: P={} R={} F1={} | recall by size {}",
        dataset,
        split,
        _fmt(overall.get("precision")),
        _fmt(overall.get("recall")),
        _fmt(overall.get("f1")),
        {b: _fmt(e["recall"]) for b, e in block["slices"]["by_symbol_size"].items()},
    )
    return block


# --------------------------------------------------------------------- (c) tuned vs default


def tuned_vs_default(
    dataset: str = "floorplans-door",
    *,
    grid: Sequence[dict[str, object]] | None = None,
    name: str = "b1",
    exemplar_count: int = 1,
) -> dict[str, Any]:
    """Tune `propose-retrieve` on val, freeze, report tuned-vs-default on test.

    `grid=None` uses the COMMITTED `_TUNING_GRIDS["propose-retrieve"]`, i.e. today's numbers -- that
    is what makes this a baseline rather than a variant. Every val trial's precision and recall are
    logged separately, never F1 alone (see the module docstring).
    """
    out = _out_path(f"{name}--{dataset}.json")
    started = perf_counter()
    report = run_domain_tuning(
        dataset,
        _RESEARCH_ROOT,
        methods=("propose-retrieve",),
        exemplar_count=exemplar_count,
        grids={"propose-retrieve": grid} if grid is not None else None,
        out=str(out),
    )
    (entry,) = report["methods"]
    logger.info("--- {} val trials (P / R / F1) ---", dataset)
    for trial in entry["trials"]:
        logger.info(
            "  {} | P={} R={} F1={}",
            trial["overrides"],
            _fmt(trial["precision"]),
            _fmt(trial["recall"]),
            _fmt(trial["f1"]),
        )
    tuned, default = entry["tuned_test"], entry["default_test"]
    logger.info(
        "{}: frozen {} | val F1={} | TEST tuned P={} R={} F1={} vs default P={} R={} F1={}",
        dataset,
        entry["tuned_overrides"],
        _fmt(entry["val_f1"]),
        _fmt(tuned.get("precision")),
        _fmt(tuned.get("recall")),
        _fmt(tuned.get("f1")),
        _fmt(default.get("precision")),
        _fmt(default.get("recall")),
        _fmt(default.get("f1")),
    )
    logger.info("tuned_vs_default[{}]: done in {:.1f}s", dataset, perf_counter() - started)
    report["wall_clock_s"] = perf_counter() - started
    return report


# ---------------------------------------------------------------------- (d) regime guardrail


def regime_check(*, name: str = "b2") -> dict[str, Any]:
    """`propose-retrieve` at DEFAULT config over the chipset / textured / synthetic regimes.

    This is the guardrail baseline for the numbers recorded in
    `docs/reports/propose-retrieve-improvement.md` (EASY 0.93, TEXTURED 0.96, VARIED 0.94,
    CLUTTERED 0.82, synthetic 0.91). It runs the committed `run_benchmark` sweep restricted to one
    method, then regroups its per-image rows into the report's five regimes by image-id prefix --
    the same grouping the report table uses. Any regime the box cannot produce shows up as a group
    with `n_images = 0` rather than being silently dropped from the guardrail.
    """
    started = perf_counter()
    results = run_benchmark(
        BenchmarkConfig(
            methods=("propose-retrieve",),
            iou_threshold=_IOU,
            out=str(_out_path(f"{name}--regimes-raw.json")),
        )
    )
    rows = results["methods"]["propose-retrieve"]["per_image"]
    by_regime: dict[str, Any] = {}
    claimed: set[str] = set()
    for label, prefix in _REGIME_PREFIXES:
        group = [r for r in rows if str(r["image_id"]).startswith(prefix)]
        claimed.update(str(r["image_id"]) for r in group)
        by_regime[label] = _pool(group)
    # Everything the prefixes did not claim is the configured synthetic set (scatter-scaled,
    # cluttered-distractors, the lattices) -- named last so a new asset family cannot vanish.
    by_regime["synthetic"] = _pool([r for r in rows if str(r["image_id"]) not in claimed])

    logger.info("--- regime guardrail (propose-retrieve, DEFAULT config) ---")
    for label, entry in by_regime.items():
        logger.info(
            "  {:<26} n={:<3} P={} R={} F1={}",
            label,
            entry["n_images"],
            _fmt(entry["precision"]),
            _fmt(entry["recall"]),
            _fmt(entry["f1"]),
        )
    payload = {
        "git_sha": current_git_sha(),
        "overall": results["methods"]["propose-retrieve"]["overall"],
        "by_regime": by_regime,
        "wall_clock_s": perf_counter() - started,
    }
    _write(f"{name}--regimes.json", payload)
    return payload


def _pool(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Micro-average tp/fp/fn over a regime group; an unscored group reports ``None``, never 0."""
    scored = [r for r in rows if r.get("tp") is not None]
    tp = sum(int(r["tp"]) for r in scored)
    fp = sum(int(r["fp"]) for r in scored)
    fn = sum(int(r["fn"]) for r in scored)
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and (precision + recall) > 0
        else None
    )
    return {
        "n_images": len(rows),
        "n_scored": len(scored),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "image_ids": sorted(str(r["image_id"]) for r in rows),
    }


# ------------------------------------------------------------------------------ (e) cost probe


def tile_count_forecast(
    dataset: str = "floorplans-door",
    splits: Sequence[str] = _SPLITS,
    *,
    tile_sides: Sequence[int] = (512, 768, 1024),
    overlaps: Sequence[float] = (0.2, 0.3),
) -> dict[str, Any]:
    """How many tiles each candidate geometry would produce, from the committed plan dimensions.

    Pure arithmetic over the GT sidecars' recorded canvas sizes -- no model, no image decode. It
    exists because "tiling costs 4-16x" is a guess, while tile count is actually a FUNCTION of plan
    area: a 725x697 plan (this dataset's median) yields 4 tiles at side 512 but exactly ONE at side
    1024 -- i.e. a 1024 tile is a no-op on the median plan -- whereas the 4000x1685 outlier yields
    40. Forecasting it here turns the CPU-vs-GPU cost question into a computed number instead of a
    round multiplier.

    Mirrors the tile geometry specified for `_tile_origins` (step = round(side * (1 - overlap)),
    final tile clamped to the image edge, a single full-image tile when the plan already fits), so
    the forecast and the implementation cannot drift on the count. It does NOT include the
    SAHI + FI full-image pass; that adds exactly one more pass per plan when enabled.
    """

    def _n_tiles(extent: int, side: int, overlap: float) -> int:
        if extent <= side:
            return 1
        step = max(1, round(side * (1.0 - overlap)))
        return int(-(-(extent - side) // step)) + 1

    dims: list[tuple[int, int]] = []
    for split in splits:
        for image_id in research_image_ids(dataset, split):  # type: ignore[arg-type]
            gt = load_research_ground_truth(
                repo_root() / _RESEARCH_ROOT / dataset / split / f"{image_id}.gt.json"
            )
            if gt is None or gt.width is None or gt.height is None:
                continue
            dims.append((int(gt.width), int(gt.height)))

    forecast: dict[str, Any] = {}
    for side in tile_sides:
        for overlap in overlaps:
            counts = [_n_tiles(w, side, overlap) * _n_tiles(h, side, overlap) for w, h in dims]
            untiled = sum(1 for c in counts if c == 1)
            forecast[f"side{side}_overlap{overlap}"] = {
                "tile_side": side,
                "overlap": overlap,
                "n_plans": len(counts),
                "mean_n_tiles": _mean([float(c) for c in counts]),
                "median_n_tiles": statistics.median(counts) if counts else None,
                "max_n_tiles": max(counts) if counts else None,
                "n_plans_untiled": untiled,
                "frac_plans_untiled": untiled / len(counts) if counts else None,
            }
            logger.info(
                "  side={:<5} overlap={} | mean_tiles={} median={} max={} | {}/{} plans untouched",
                side,
                overlap,
                _fmt(forecast[f"side{side}_overlap{overlap}"]["mean_n_tiles"]),
                forecast[f"side{side}_overlap{overlap}"]["median_n_tiles"],
                forecast[f"side{side}_overlap{overlap}"]["max_n_tiles"],
                untiled,
                len(counts),
            )
    return forecast


def cost_probe(
    dataset: str = "floorplans-door",
    split: str = "val",
    *,
    n_plans: int = 5,
    tile_multipliers: Sequence[int] = (4, 9, 16),
) -> dict[str, Any]:
    """Wall-clock the proposal stage and a full search on this box, then extrapolate a tiled sweep.

    The measured half is real: `n_plans` plans, each timed for (i) the proposal stage alone and
    (ii) a full `propose_retrieve.search`. The extrapolated half is ARITHMETIC -- proposal cost
    multiplied by a hypothetical tile count -- and is labelled as such in the payload
    (`extrapolated_tiled_sweep`), because nothing tiled has been built or run yet. It is a planning
    number for the CPU-vs-GPU decision, not a measurement, and must never be quoted as one.

    **The probe samples by AREA QUANTILE, not by manifest order.** The first revision of this
    function took `ids[:n_plans]`, which on `floorplans-door` val drew three plans whose mean area
    is **0.54x the val mean** — so it under-estimated a full val pass by a large factor (7.0 min
    predicted vs ~54 min measured; see EXPERIMENTS.md B3). Cost here is super-linear in plan area
    (a bigger plan yields more proposals AND bigger crops, and every proposal is embedded by its own
    DINOv2 forward pass), so a small-plan sample is not merely noisy, it is biased low. Sampling
    evenly across the area distribution is the fix; a cost probe that cannot be trusted to size a
    sweep is worse than none, because it is quoted as if it could.
    """
    backend = propose_retrieve._get_backend()
    if backend is None:
        raise RuntimeError("the fastsam-s weight is absent; cannot probe cost.")
    config = propose_retrieve.ProposeRetrieveConfig()

    # Order the split by plan area, then take evenly spaced quantiles, so the sample spans the
    # distribution (smallest -> largest) instead of whatever happens to sort first by filename.
    sized: list[tuple[int, str]] = []
    for image_id in research_image_ids(dataset, split):  # type: ignore[arg-type]
        gt = load_research_ground_truth(
            repo_root() / _RESEARCH_ROOT / dataset / split / f"{image_id}.gt.json"
        )
        if gt is None or gt.width is None or gt.height is None:
            continue
        sized.append((int(gt.width) * int(gt.height), image_id))
    sized.sort()
    picks = (
        [sized[round(i * (len(sized) - 1) / max(1, n_plans - 1))][1] for i in range(n_plans)]
        if sized
        else []
    )

    per_plan: list[dict[str, Any]] = []
    for image_id in picks:
        sidecar = repo_root() / _RESEARCH_ROOT / dataset / split / f"{image_id}.gt.json"
        gt = load_research_ground_truth(sidecar)
        image = _load_scene(dataset, split, image_id)
        if gt is None or image is None:
            continue
        exemplar = sample_exemplars(gt, count=1, seed=0)[0]

        started = perf_counter()
        proposals = propose(image, FastSAMConfig(conf_thres=config.proposal_conf), backend=backend)
        proposal_s = perf_counter() - started

        started = perf_counter()
        result = propose_retrieve.search(image, exemplar, config)
        search_s = perf_counter() - started

        metrics = dict(result.diagnostics.metrics) if result.diagnostics else {}
        per_plan.append(
            {
                "image_id": image_id,
                "width": int(image.shape[1]),
                "height": int(image.shape[0]),
                "n_gt": len(gt.boxes),
                "n_proposals": len(proposals),
                "proposal_stage_s": proposal_s,
                "full_search_s": search_s,
                "proposal_ms": metrics.get("proposal_ms"),
                "embedding_ms": metrics.get("embedding_ms"),
            }
        )
        logger.info(
            "cost[{}]: {}x{} n_prop={} | proposal {:.2f}s, full search {:.2f}s "
            "(proposal_ms={} embedding_ms={})",
            image_id,
            per_plan[-1]["width"],
            per_plan[-1]["height"],
            len(proposals),
            proposal_s,
            search_s,
            _fmt(metrics.get("proposal_ms")),
            _fmt(metrics.get("embedding_ms")),
        )

    mean_proposal_s = _mean([float(p["proposal_stage_s"]) for p in per_plan]) or 0.0
    mean_search_s = _mean([float(p["full_search_s"]) for p in per_plan]) or 0.0
    n_val = len(research_image_ids(dataset, "val"))
    # EXTRAPOLATION, not a measurement: a tiled search pays its proposal stage n_tiles times over
    # (each tile is a separate 1024-letterboxed FastSAM forward pass) while the embedding stage
    # scales with the merged proposal count, which is unknown until tiling exists. So this holds the
    # embedding stage FIXED and multiplies only the proposal stage -- a deliberate LOWER bound.
    extrapolated = {
        str(mult): {
            "per_plan_s": mean_proposal_s * mult + (mean_search_s - mean_proposal_s),
            "val_sweep_s_per_trial": (mean_proposal_s * mult + (mean_search_s - mean_proposal_s))
            * n_val,
        }
        for mult in tile_multipliers
    }

    logger.info("--- tile-count forecast (arithmetic over committed plan dimensions) ---")
    forecast = tile_count_forecast(dataset)
    # The forecast-driven cost is still an EXTRAPOLATION, but its multiplier is computed from this
    # dataset's real plan sizes rather than assumed, so it is the number the CPU-vs-GPU decision
    # should actually be read off.
    forecast_cost = {
        key: {
            "mean_n_tiles": entry["mean_n_tiles"],
            "per_plan_s": mean_proposal_s * float(entry["mean_n_tiles"] or 1.0)
            + (mean_search_s - mean_proposal_s),
            "val_sweep_s_per_trial": (
                mean_proposal_s * float(entry["mean_n_tiles"] or 1.0)
                + (mean_search_s - mean_proposal_s)
            )
            * n_val,
        }
        for key, entry in forecast.items()
    }
    for key, entry in forecast_cost.items():
        logger.info(
            "  {} -> {:.2f}s/plan, {:.1f} min per 56-plan val trial",
            key,
            entry["per_plan_s"],
            entry["val_sweep_s_per_trial"] / 60.0,
        )

    payload = {
        "git_sha": current_git_sha(),
        "runtime": "CPUExecutionProvider (onnxruntime CPU build)",
        "dataset": dataset,
        "split": split,
        "n_val_plans": n_val,
        "per_plan": per_plan,
        "mean_proposal_stage_s": mean_proposal_s,
        "mean_full_search_s": mean_search_s,
        "measured_val_pass_s": mean_search_s * n_val,
        "tile_count_forecast": forecast,
        "forecast_driven_tiled_cost": forecast_cost,
        "extrapolated_tiled_sweep": extrapolated,
        "extrapolation_note": (
            "ARITHMETIC EXTRAPOLATION, NOT A MEASUREMENT: proposal stage x n_tiles, embedding stage "
            "held fixed. Nothing tiled has been built or run. A LOWER bound -- tiling also raises "
            "the merged proposal count, which raises the embedding stage."
        ),
    }
    logger.info(
        "cost probe: mean proposal {:.2f}s, mean full search {:.2f}s -> one 56-plan val pass "
        "{:.0f}s ({:.1f} min)",
        mean_proposal_s,
        mean_search_s,
        mean_search_s * n_val,
        mean_search_s * n_val / 60.0,
    )
    for mult, entry in extrapolated.items():
        logger.info(
            "  extrapolated {}x tiles: {:.2f}s/plan -> {:.1f} min per val trial",
            mult,
            entry["per_plan_s"],
            entry["val_sweep_s_per_trial"] / 60.0,
        )
    _write("b3--cost-probe.json", payload)
    return payload


# ------------------------------------------------------------------------------------ runner


def main(argv: Sequence[str]) -> int:
    experiments = ("b0", "b1", "b2", "b3")
    if not argv or argv[0] not in experiments:
        logger.error(
            "usage: propose_retrieve_floorplans_experiment.py {{{}}} [dataset]",
            "|".join(experiments),
        )
        return 2
    experiment = argv[0]
    dataset = argv[1] if len(argv) > 1 else "floorplans-door"
    if dataset not in _DATASETS:
        logger.error("unknown dataset {!r}; expected one of {}", dataset, _DATASETS)
        return 2

    if experiment == "b0":
        report = proposal_stage_recall(dataset)
        _log_proposal_stage(report)
        _write(f"b0--{dataset}.json", report)
    elif experiment == "b1":
        tuned_vs_default(dataset)
        for split in _SPLITS:
            final_metrics(dataset, split)
    elif experiment == "b2":
        regime_check()
    else:
        cost_probe(dataset)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
