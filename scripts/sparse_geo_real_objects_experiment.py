"""Offline lab bench: does the floor-plans SuperPoint verdict survive on real photos? (260811-p0l)

Why this exists
---------------
`docs/reports/sparse-geo-improvement.md` disproved the SuperPoint backend for `sparse-geo` in 4/4
floor-plan cells. But it did so on a domain where **both** detectors nearly starve: 4 of 5 exemplar
crops fell below `min_exemplar_keypoints=8` under SIFT *and* under SuperPoint, and SuperPoint
additionally hit a hard zero-keypoint ONNX/CoreML crash that cost 2/28 window plans their coverage.
Neither condition is expected to hold on 1024px photographs of apples and hammers. Until it is
measured on rich photographic texture, "SuperPoint is worse" is an over-generalisation from one
adversarial domain.

This script measures the same backend question on the 30-image `real-objects` set (10 everyday
objects x plain/varied/cluttered) under five conditions:

    sift / single-4dof         the shipped baseline (reconciled against the published numbers)
    sift / translation-2dof    CONTROL -- same voting mode as superpoint's default
    sift / pairwise-4dof       CONTROL -- same voting mode as superpoint's other option
    superpoint / translation-2dof
    superpoint / pairwise-4dof

The two SIFT controls are the point of the design: SuperPoint is FRAMELESS, so it cannot run at
`single-4dof` (`SparseGeoConfig` raises at construction). Without a same-voting-mode SIFT control,
any SuperPoint delta would be silently credited to the voting-mode switch that SuperPoint forces.

What it does NOT touch
----------------------
Nothing in `src/`, nothing in `conf/`, nothing in `docs/benchmark/`. This is a research harness in
`scripts/`; it writes only into this quick task's own `runs/` directory. `backend="sift"` remains
the shipped default regardless of the numbers -- the MagicLeap SuperPoint weights are
non-commercial research-only and gitignored, so this backend could not become a default even if it
won.

How the scoring stays honest
----------------------------
Every cell goes through `object_search.eval.benchmark._run_one` **unmodified**, so the scoring, the
AP50 candidate-log convention, the abstention/error handling, and the per-GT-box records are
identical to `pixi run bench-real-objects` by construction. `_run_one` hardcodes
`spec.config_model()` -- there is no per-method config override on that path -- so the condition is
supplied by handing it a `MethodSpec` variant whose `config_model` is a bound `functools.partial`.
Pooling is `benchmark._aggregate` / `benchmark._slice_by`, never a re-implemented scorer.

No tuning happens here. `real-objects` has no val/test split, so there is nothing to tune on and
nothing is fit to these labels: every condition runs at method defaults with only `backend` and
`voting_mode` pinned.

Usage:

    pixi run fetch-models --only superpoint          # ~5 MiB, sha256-gated
    pixi run python scripts/sparse_geo_real_objects_experiment.py smoke
    pixi run python scripts/sparse_geo_real_objects_experiment.py sweep

Set `OS_GIT_SHA` to record provenance when running from an exported tree with no `.git` (e.g. a
rented remote box); it falls back to `object_search.provenance.current_git_sha()`.
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import json
import os
import sys
from pathlib import Path
from time import perf_counter
from typing import Any, Literal, cast

import cv2
import numpy as np
import numpy.typing as npt
from loguru import logger
from pydantic import BaseModel

from object_search.eval import benchmark
from object_search.eval.labels import GroundTruth, load_ground_truth, real_objects_image_ids
from object_search.provenance import current_git_sha
from object_search.search import get_method
from object_search.search import sparse_geo as sg
from object_search.search.registry import MethodSpec

# Where the raw JSON + warning logs land: this quick task's own directory, never docs/benchmark/.
_OUT_DIR = Path(".planning/quick/260811-p0l-spike-explore-the-superpoint-backend-for/runs")
_SUMMARY_PATH = _OUT_DIR / "summary.json"

# The IoU at which a prediction counts as a true positive -- the project-wide default, so these
# numbers line up with every other report.
_IOU_THRESHOLD = 0.5

_METHOD = "sparse-geo"

# One image, used for the `smoke` subcommand's thinnest complete path.
_SMOKE_IMAGE_ID = "real-plain-apple"

_Backend = Literal["sift", "akaze", "orb", "superpoint"]
_VotingMode = Literal["single-4dof", "translation-2dof", "pairwise-4dof"]


@dataclasses.dataclass(frozen=True)
class Condition:
    """One measured condition: a label plus the two `SparseGeoConfig` fields it pins."""

    label: str
    backend: _Backend
    voting_mode: _VotingMode
    role: str


# The five conditions. Order matters for the sweep: the shipped baseline runs FIRST and is
# reconciled against the published real-objects figures before any delta is computed, because a
# baseline that does not reproduce invalidates every comparison measured against it.
#
# superpoint + single-4dof is absent by DESIGN, not by choice: SuperPoint keypoints are frameless,
# so `SparseGeoConfig` refuses that pair at construction time (METHOD-04a).
_CONDITIONS: tuple[Condition, ...] = (
    Condition("sift/single-4dof", "sift", "single-4dof", "shipped baseline"),
    Condition("sift/translation-2dof", "sift", "translation-2dof", "voting-mode control"),
    Condition("sift/pairwise-4dof", "sift", "pairwise-4dof", "voting-mode control"),
    Condition("superpoint/translation-2dof", "superpoint", "translation-2dof", "hypothesis"),
    Condition("superpoint/pairwise-4dof", "superpoint", "pairwise-4dof", "hypothesis"),
)

# The published `sparse-geo` real-objects numbers from docs/reports/real-objects-findings.md, which
# the baseline condition must reproduce. Overall figures are quoted there to 3 dp; the per-regime
# F1s to 2 dp, so they are checked at their own stated precision rather than a spuriously tight one.
_PUBLISHED_OVERALL: dict[str, float] = {
    "precision": 0.833,
    "recall": 0.772,
    "f1": 0.786,
    "mean_ap": 0.740,
}
_PUBLISHED_BY_REGIME_F1: dict[str, float] = {"plain": 0.99, "varied": 0.67, "cluttered": 0.66}
_PUBLISHED_COVERAGE = {"n_images": 30, "n_scored": 30, "n_errors": 0, "n_abstentions": 0}


# --------------------------------------------------------------------- the MethodSpec variant


def _variant_spec(condition: Condition) -> MethodSpec:
    """Return a `sparse-geo` `MethodSpec` whose config is pinned to `condition`.

    `benchmark._run_one` only ever calls `spec.config_model()` with no arguments, so a bound
    `functools.partial` satisfies the contract exactly -- which is what justifies the cast to
    `type[BaseModel]` (a partial is a callable factory, not a class). `MethodSpec` is a frozen
    dataclass, so `dataclasses.replace` is the supported way to make a variant of it.
    """
    spec = get_method(_METHOD)
    bound = functools.partial(
        sg.SparseGeoConfig, backend=condition.backend, voting_mode=condition.voting_mode
    )
    return dataclasses.replace(spec, config_model=cast(type[BaseModel], bound))


def _run_condition_on(condition: Condition, image_ids: tuple[str, ...]) -> list[Any]:
    """Score `condition` over `image_ids` through the project's own `_run_one`, unmodified.

    The variant is installed by patching `benchmark.get_method` (which `_run_one` calls) and
    restored afterwards, so nothing about the scoring path is re-implemented or bypassed.
    """
    variant = _variant_spec(condition)
    original = benchmark.get_method

    def _patched(name: str) -> MethodSpec:
        return variant if name == _METHOD else original(name)

    benchmark.get_method = _patched  # type: ignore[assignment]
    try:
        records = []
        for image_id in image_ids:
            gt = _ground_truth(image_id)
            records.append(benchmark._run_one(_METHOD, image_id, gt, iou_threshold=_IOU_THRESHOLD))
        return records
    finally:
        benchmark.get_method = original  # type: ignore[assignment]


def _ground_truth(image_id: str) -> GroundTruth:
    """Load the committed sidecar for `image_id`, failing loudly if it is missing."""
    gt = load_ground_truth(image_id)
    if gt is None:
        raise FileNotFoundError(f"no ground-truth sidecar for {image_id!r}")
    return gt


# ------------------------------------------------------------------ the keypoint diagnostic


def _keypoint_counts(image_id: str, backend_name: _Backend) -> dict[str, int]:
    """Count exemplar-crop and full-scene keypoints for one image under one backend.

    This is the quantity the floor-plans premise was actually about (4/5 crops there fell below the
    8-keypoint floor under BOTH detectors), so it must be measured on real photographic texture too
    rather than asserted. The crop and greyscale handling mirror `sparse_geo.search` step 2 exactly.
    """
    gt = _ground_truth(image_id)
    scene = benchmark._load_scene(image_id)
    gray: npt.NDArray[np.uint8] = np.ascontiguousarray(
        cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY), dtype=np.uint8
    )
    backend = sg._make_backend(backend_name)
    box = gt.exemplar.box
    crop_gray = np.ascontiguousarray(gray[box.y : box.y2, box.x : box.x2], dtype=np.uint8)
    crop_kps = sg._detect(crop_gray, backend, origin_xy=(box.x, box.y))
    scene_kps = sg._detect(gray, backend)
    return {"crop": crop_kps.count, "scene": scene_kps.count}


# ------------------------------------------------------------------------------- formatting


def _fmt(value: float | None) -> str:
    """Three decimals, or an em dash for a `None` (an abstained/errored cell)."""
    return "—" if value is None else f"{value:.3f}"


def _log_record(condition: Condition, record: Any, kps: dict[str, dict[str, int]]) -> None:
    """Log one per-image row with its metrics and both keypoint counts."""
    logger.info(
        "  {:<28} outcome={:<9} tp/fp/fn={}/{}/{} P={} R={} F1={} AP={} {:.0f} ms "
        "| crop_kp={} scene_kp={}",
        condition.label,
        record.outcome,
        record.tp,
        record.fp,
        record.fn,
        _fmt(record.precision),
        _fmt(record.recall),
        _fmt(record.f1),
        _fmt(record.ap),
        record.latency_ms or 0.0,
        kps[condition.backend]["crop"],
        kps[condition.backend]["scene"],
    )


# ----------------------------------------------------------------------------- reconciliation


def _reconcile_baseline(overall: dict[str, Any], by_regime: dict[str, Any]) -> dict[str, Any]:
    """Check the baseline condition against the published real-objects figures.

    A baseline that does not reproduce invalidates every delta measured against it, so this runs
    BEFORE any comparison is computed and any disagreement is surfaced as a WARNING (which the
    per-condition Loguru file sink also captures).
    """
    cells: list[dict[str, Any]] = []
    for key, published in _PUBLISHED_OVERALL.items():
        measured = overall.get(key)
        delta = None if measured is None else round(measured - published, 6)
        cells.append(
            {
                "cell": f"overall.{key}",
                "published": published,
                "measured": measured,
                "delta": delta,
                # Overall figures are published to 3 dp; anything beyond half a unit in the last
                # published place is a real disagreement, not a rounding artefact.
                "agrees": delta is not None and abs(delta) <= 0.0005,
            }
        )
    for regime, published in _PUBLISHED_BY_REGIME_F1.items():
        measured = by_regime.get(regime, {}).get("f1")
        delta = None if measured is None else round(measured - published, 6)
        cells.append(
            {
                "cell": f"by_regime.{regime}.f1",
                "published": published,
                "measured": measured,
                "delta": delta,
                # Per-regime F1 is published to 2 dp only, so it is checked at that precision.
                "agrees": delta is not None and abs(delta) <= 0.005,
            }
        )
    for key, published_count in _PUBLISHED_COVERAGE.items():
        measured_count = overall.get(key)
        cells.append(
            {
                "cell": f"overall.{key}",
                "published": published_count,
                "measured": measured_count,
                "delta": None if measured_count is None else measured_count - published_count,
                "agrees": measured_count == published_count,
            }
        )

    disagreements = [c for c in cells if not c["agrees"]]
    for cell in disagreements:
        logger.warning(
            "BASELINE RECONCILIATION MISMATCH {}: published={} measured={} (delta={})",
            cell["cell"],
            cell["published"],
            cell["measured"],
            cell["delta"],
        )
    if not disagreements:
        logger.info("baseline reconciles with the published real-objects numbers in every cell")
    return {
        "source": "docs/reports/real-objects-findings.md",
        "reproduces": not disagreements,
        "n_disagreements": len(disagreements),
        "cells": cells,
    }


# ---------------------------------------------------------------------------- the subcommands


def smoke() -> int:
    """Run ONE image through the shipped baseline and through superpoint/translation-2dof.

    The thinnest complete path: proves both backends load and produce real scored outcomes on a
    real photograph before the 5-condition sweep is worth starting.
    """
    logger.info("smoke: {} under both backends", _SMOKE_IMAGE_ID)
    kps = {
        backend: _keypoint_counts(_SMOKE_IMAGE_ID, backend)
        for backend in cast(tuple[_Backend, ...], ("sift", "superpoint"))
    }
    for backend, counts in kps.items():
        logger.info("  keypoints [{}]: crop={} scene={}", backend, counts["crop"], counts["scene"])

    conditions = (_CONDITIONS[0], _CONDITIONS[3])
    failed = False
    for condition in conditions:
        records = _run_condition_on(condition, (_SMOKE_IMAGE_ID,))
        _log_record(condition, records[0], kps)
        if records[0].outcome == "error":
            logger.error("{} produced an `error` outcome -- backend failed to run", condition.label)
            failed = True
    if failed:
        return 1
    logger.info("smoke OK: both backends ran and were scored on a real photograph")
    return 0


def sweep() -> int:
    """Run all five conditions over all 30 real-objects images and write `runs/summary.json`."""
    image_ids = real_objects_image_ids()
    if len(image_ids) != 30:
        logger.warning("expected 30 real-objects ids, found {}", len(image_ids))
    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("sweep: {} conditions x {} images", len(_CONDITIONS), len(image_ids))

    # The keypoint diagnostic over every image under both detectors -- the causal evidence the
    # verdict rests on, measured rather than asserted.
    logger.info("measuring keypoint counts under both detectors on all {} images", len(image_ids))
    keypoints: dict[str, dict[str, dict[str, int]]] = {}
    for backend in cast(tuple[_Backend, ...], ("sift", "superpoint")):
        keypoints[backend] = {}
        for image_id in image_ids:
            try:
                keypoints[backend][image_id] = _keypoint_counts(image_id, backend)
            except Exception as exc:
                logger.warning("keypoint count failed for {} / {}: {}", backend, image_id, exc)
                keypoints[backend][image_id] = {"crop": -1, "scene": -1}

    conditions_out: dict[str, Any] = {}
    reconciliation: dict[str, Any] = {}

    for condition in _CONDITIONS:
        log_path = _OUT_DIR / f"warnings-{condition.label.replace('/', '-')}.log"
        # A per-condition WARNING sink: `_run_one` degrades any raised exception to an `error`
        # outcome with a `logger.warning`, so this file IS the evidence for the zero-keypoint
        # ONNX/CoreML crash question -- either the text lands here, or it demonstrably does not.
        sink_id = logger.add(log_path, level="WARNING", mode="w")
        started = perf_counter()
        try:
            logger.info("--- condition {} ({}) ---", condition.label, condition.role)
            records = _run_condition_on(condition, image_ids)
        finally:
            elapsed = perf_counter() - started
            logger.remove(sink_id)

        overall = benchmark._aggregate(records)
        by_regime = benchmark._slice_by(records, lambda r: r.image_id.split("-")[1])

        logger.info(
            "  {:<28} P={} R={} F1={} AP={} scored={} errors={} abstentions={} p50={} ms "
            "elapsed={:.1f}s",
            condition.label,
            _fmt(overall["precision"]),
            _fmt(overall["recall"]),
            _fmt(overall["f1"]),
            _fmt(overall["mean_ap"]),
            overall["n_scored"],
            overall["n_errors"],
            overall["n_abstentions"],
            _fmt(overall["latency_ms"]["p50"]),
            elapsed,
        )

        # The baseline is first in _CONDITIONS, and is reconciled BEFORE any delta is claimed.
        if condition.role == "shipped baseline":
            reconciliation = _reconcile_baseline(overall, by_regime)

        conditions_out[condition.label] = {
            "backend": condition.backend,
            "voting_mode": condition.voting_mode,
            "role": condition.role,
            "overall": overall,
            "by_regime": by_regime,
            "per_image": [r.model_dump() for r in records],
            "elapsed_s": round(elapsed, 3),
            "warning_log": log_path.name,
            "warning_log_lines": len(log_path.read_text().splitlines()) if log_path.exists() else 0,
        }

    summary = {
        "provenance": {
            "quick_task": "260811-p0l",
            "git_sha": os.environ.get("OS_GIT_SHA") or current_git_sha(),
            "method": _METHOD,
            "iou_threshold": _IOU_THRESHOLD,
            "image_ids": list(image_ids),
            "n_images": len(image_ids),
            "tuning": (
                "none -- real-objects has no val/test split, so nothing is tuned and no threshold "
                "is selected against these labels. Every condition runs at SparseGeoConfig "
                "defaults with only `backend` and `voting_mode` pinned."
            ),
            "single_4dof_for_superpoint": (
                "not measured: SparseGeoConfig raises at construction for superpoint + "
                "single-4dof (frameless keypoints cannot determine a 4-DoF similarity)."
            ),
        },
        "baseline_reconciliation": reconciliation,
        "keypoints": keypoints,
        "conditions": conditions_out,
    }
    _SUMMARY_PATH.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    logger.info("wrote {}", _SUMMARY_PATH)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("smoke", "sweep"))
    args = parser.parse_args(argv)
    return smoke() if args.command == "smoke" else sweep()


if __name__ == "__main__":
    sys.exit(main())
