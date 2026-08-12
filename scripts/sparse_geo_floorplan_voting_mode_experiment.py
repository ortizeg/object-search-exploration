"""Offline lab bench for the `sparse-geo` floor-plans voting-mode-confound spike (260812-mm3).

Why this exists
----------------
`docs/reports/sparse-geo-improvement.md` measured `backend="superpoint"` against the shipped
`backend="sift"` baseline on floor-plans-500 and found SuperPoint DISPROVEN in 4/4 cells. But
`SparseGeoConfig` refuses `voting_mode="single-4dof"` for a frameless backend at construction
(`_reject_single_4dof_for_frameless_superpoint`), so every SuperPoint condition in that table ran
at `translation-2dof` or `pairwise-4dof` while the SIFT baseline it was compared against ran at
`single-4dof` -- two variables changed at once, backend AND voting mode, with no same-voting-mode
SIFT control to isolate which one moved the numbers.

`docs/reports/sparse-geo-real-objects-superpoint-spike.md` then ran that missing control on a
DIFFERENT domain (real photographic texture) and found switching SIFT itself off `single-4dof`
costs 0.055-0.079 F1 and 0.050-0.145 AP on its own, independent of backend -- large enough to
account for a real share of the floor-plans deltas. A hypothesis about floor-plans data can only be
settled on floor-plans data, so this script runs the same SIFT controls (`translation-2dof`,
`pairwise-4dof`) on floor-plans-500 itself, at BOTH grids the published table used (the committed
grid that produced the `single-4dof` baseline, and the SuperPoint-matched grid that produced the
SuperPoint rows and the `pairwise-4dof` no-mirror control), so the new cells drop straight into the
published comparison.

It is a research harness, NOT part of the shipped package: it lives in `scripts/`, writes only into
this quick task's own directory, and never touches `docs/benchmark/`. Everything goes through the
library's own `run_domain_tuning` via the additive `grids=` override, exactly the seam
`scripts/mosse_floorplan_experiment.py` uses, so a condition swept here is validated by
`SparseGeoConfig` exactly as a committed grid entry would be. `backend` is never pinned in any
override dict -- `sift` is already `SparseGeoConfig`'s default, so leaving it unset keeps every
override minimal and makes `voting_mode` the only thing that moved. Nothing in `src/` or `conf/` is
touched by this script or by anything it measures: `backend` stays `sift` and `voting_mode` stays
`single-4dof` as shipped defaults regardless of the outcome.

Protocol (identical to the committed tuning protocol, so numbers are comparable): tune on val
(argmax F1 @ IoU 0.5) -> freeze -> report tuned AND default on test. Precision and recall are
logged per trial alongside F1, never F1 alone.

Usage:

    pixi run python scripts/sparse_geo_floorplan_voting_mode_experiment.py smoke
    pixi run python scripts/sparse_geo_floorplan_voting_mode_experiment.py sweep
"""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from pathlib import Path
from time import perf_counter
from typing import Any

from loguru import logger

from object_search.eval.tuning import run_domain_tuning
from object_search.provenance import current_git_sha

# Where the raw JSON reports land: the quick task's own directory, never docs/benchmark/.
_OUT_DIR = Path(".planning/quick/260812-mm3-follow-up-spike-add-same-voting-mode-sif/runs")
_DATASETS = ("floorplans-door", "floorplans-window")

# The three conditions this spike measures. `sift/single-4dof` is the reproduced published
# baseline; the other two are the missing same-voting-mode SIFT controls. `backend` unset ->
# SparseGeoConfig's own default (sift).
_CONDITIONS: tuple[tuple[str, str], ...] = (
    ("sift/single-4dof", "single-4dof"),
    ("sift/translation-2dof", "translation-2dof"),
    ("sift/pairwise-4dof", "pairwise-4dof"),
)

# COMMITTED grid: reproduces _TUNING_GRIDS["sparse-geo"] exactly (min_inliers 2..10 x nms_iou
# {0.3, 0.5}, 14 entries). This is the grid that produced the published sift/single-4dof baseline
# rows (door test F1 0.219, window test F1 0.309). Written out literally rather than importing the
# private grid, per the plan -- this script is meant to be read standalone.
_COMMITTED_MIN_INLIERS: tuple[int, ...] = (2, 3, 4, 5, 6, 8, 10)

# SUPERPOINT-MATCHED grid: min_inliers {5, 8, 12, 16, 20} x nms_iou {0.3, 0.5} (10 entries). This is
# the grid the published SuperPoint rows AND the published pairwise-4dof no-mirror control were
# measured on (sparse-geo-improvement.md's Hypothesis 2 sweep). Its min_inliers floor of 5 is the
# grid-floor caveat that report already disclosed; matching it here is what makes the SuperPoint
# comparison grid-fair rather than comparing SuperPoint's grid-fair numbers against a control tuned
# on a wider grid.
_SP_MATCHED_MIN_INLIERS: tuple[int, ...] = (5, 8, 12, 16, 20)

_NMS_IOU: tuple[float, ...] = (0.3, 0.5)

_GRIDS: dict[str, tuple[int, ...]] = {
    "committed": _COMMITTED_MIN_INLIERS,
    "sp-matched": _SP_MATCHED_MIN_INLIERS,
}

# Six published targets this run must reproduce before any delta is claimed, all from
# docs/reports/sparse-geo-improvement.md. The last two are the substantive check that this clean
# post-revert tree reproduces the tree the allow_mirror-era pairwise-4dof control was measured in.
_RECONCILIATION_TARGETS: tuple[dict[str, Any], ...] = (
    {
        "target": "committed grid, single-4dof, door",
        "condition": "sift/single-4dof",
        "grid": "committed",
        "dataset": "floorplans-door",
        "cell": "tuned_test",
        "published": {"f1": 0.219, "precision": 0.442, "recall": 0.146},
        "decimals": 3,
    },
    {
        "target": "committed grid, single-4dof, window",
        "condition": "sift/single-4dof",
        "grid": "committed",
        "dataset": "floorplans-window",
        "cell": "tuned_test",
        "published": {"f1": 0.309, "precision": 0.627, "recall": 0.205},
        "decimals": 3,
    },
    {
        "target": "sp-matched grid, single-4dof, door (val F1)",
        "condition": "sift/single-4dof",
        "grid": "sp-matched",
        "dataset": "floorplans-door",
        "cell": "val_f1",
        "published": {"val_f1": 0.2468},
        "decimals": 4,
    },
    {
        "target": "sp-matched grid, single-4dof, window (val F1)",
        "condition": "sift/single-4dof",
        "grid": "sp-matched",
        "dataset": "floorplans-window",
        "cell": "val_f1",
        "published": {"val_f1": 0.2143},
        "decimals": 4,
    },
    {
        "target": "sp-matched grid, pairwise-4dof, door (allow_mirror-era control)",
        "condition": "sift/pairwise-4dof",
        "grid": "sp-matched",
        "dataset": "floorplans-door",
        "cell": "both",
        "published": {"val_f1": 0.238, "test_f1": 0.231},
        "decimals": 3,
    },
    {
        "target": "sp-matched grid, pairwise-4dof, window (allow_mirror-era control)",
        "condition": "sift/pairwise-4dof",
        "grid": "sp-matched",
        "dataset": "floorplans-window",
        "cell": "both",
        "published": {"val_f1": 0.222, "test_f1": 0.256},
        "decimals": 3,
    },
)


def _fmt(value: float | None) -> str:
    return f"{value:.3f}" if isinstance(value, int | float) else "  -  "


def _grid(voting_mode: str, min_inliers: tuple[int, ...]) -> tuple[dict[str, object], ...]:
    """Cross `min_inliers` x `nms_iou`, with `voting_mode` folded into every entry."""
    return tuple(
        {"voting_mode": voting_mode, "min_inliers": mi, "nms_iou": nms}
        for mi in min_inliers
        for nms in _NMS_IOU
    )


def _log_trials(label: str, dataset: str, entry: dict[str, Any]) -> None:
    """Log every val trial's P / R / F1 -- never F1 alone (see the module docstring)."""
    logger.info("--- {} / {} val trials (P / R / F1) ---", label, dataset)
    for trial in entry["trials"]:
        overrides = trial["overrides"]
        logger.info(
            "  voting_mode={:<16} min_inliers={:<4} nms_iou={:<4} | P={} R={} F1={}",
            overrides.get("voting_mode", "-"),
            overrides.get("min_inliers", "-"),
            overrides.get("nms_iou", "-"),
            _fmt(trial["precision"]),
            _fmt(trial["recall"]),
            _fmt(trial["f1"]),
        )


def _log_frozen(label: str, dataset: str, entry: dict[str, Any]) -> None:
    """Log the frozen-on-val config's tuned-vs-default TEST block (the one test read)."""
    tuned, default = entry["tuned_test"], entry["default_test"]
    logger.info(
        "{}/{}: frozen {} | val F1={} | TEST tuned P={} R={} F1={} AP50={} vs default F1={}",
        label,
        dataset,
        entry["tuned_overrides"],
        _fmt(entry["val_f1"]),
        _fmt(tuned.get("precision")),
        _fmt(tuned.get("recall")),
        _fmt(tuned.get("f1")),
        _fmt(tuned.get("ap50")),
        _fmt(default.get("f1")),
    )


def run_experiment(name: str, dataset: str, grid: Sequence[dict[str, object]]) -> dict[str, Any]:
    """Tune `sparse-geo` on `dataset` val over `grid`, report tuned-vs-default on test."""
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = _OUT_DIR / f"{name}--{dataset}.json"
    logger.info("experiment[{}] {}: {} trial(s) -> {}", name, dataset, len(grid), out)
    started = perf_counter()
    report = run_domain_tuning(
        dataset,
        Path("datasets"),
        methods=("sparse-geo",),
        grids={"sparse-geo": grid},
        out=str(out),
    )
    (entry,) = report["methods"]
    _log_trials(name, dataset, entry)
    _log_frozen(name, dataset, entry)
    logger.info("experiment[{}] {}: done in {:.1f}s", name, dataset, perf_counter() - started)
    return report


def _cell_dict(
    condition_label: str, voting_mode: str, grid_name: str, dataset: str, elapsed_s: float
) -> dict[str, Any]:
    grid = _grid(voting_mode, _GRIDS[grid_name])
    report = run_experiment(f"{grid_name}--{condition_label.replace('/', '-')}", dataset, grid)
    (entry,) = report["methods"]
    return {
        "condition": condition_label,
        "grid": grid_name,
        "dataset": dataset,
        "tuned_overrides": entry["tuned_overrides"],
        "val_f1": entry["val_f1"],
        "tuned_test": entry["tuned_test"],
        "default_test": entry["default_test"],
        "delta_f1": entry["delta_f1"],
        "trials": entry["trials"],
        "elapsed_s": elapsed_s,
    }


def _reconcile(cells: list[dict[str, Any]]) -> dict[str, Any]:
    by_key = {(c["condition"], c["grid"], c["dataset"]): c for c in cells}
    results: list[dict[str, Any]] = []
    for target in _RECONCILIATION_TARGETS:
        cell = by_key[(target["condition"], target["grid"], target["dataset"])]
        measured: dict[str, float | None] = {}
        published: dict[str, float] = target["published"]
        if target["cell"] == "val_f1":
            measured["val_f1"] = cell["val_f1"]
        elif target["cell"] == "tuned_test":
            tt = cell["tuned_test"]
            for key in published:
                measured[key] = tt.get(key)
        else:  # "both": val_f1 + test f1
            measured["val_f1"] = cell["val_f1"]
            measured["test_f1"] = cell["tuned_test"].get("f1")
        decimals = target["decimals"]
        row_cells = []
        all_agree = True
        for key, pub_val in published.items():
            meas_val = measured.get(key)
            agrees = isinstance(meas_val, int | float) and round(meas_val, decimals) == round(
                pub_val, decimals
            )
            all_agree = all_agree and agrees
            delta = (meas_val - pub_val) if isinstance(meas_val, int | float) else None
            row_cells.append(
                {
                    "cell": key,
                    "published": pub_val,
                    "measured": meas_val,
                    "delta": delta,
                    "agrees": agrees,
                }
            )
        if not all_agree:
            logger.warning("RECONCILIATION DISAGREEMENT: {} -> {}", target["target"], row_cells)
        results.append({"target": target["target"], "cells": row_cells, "reproduces": all_agree})
    return {"n_disagreements": sum(1 for r in results if not r["reproduces"]), "targets": results}


def smoke() -> int:
    """Thinnest complete path: sift/translation-2dof on floorplans-door, a tiny 2-entry grid."""
    logger.info("smoke: sift/translation-2dof on floorplans-door, min_inliers in (5, 10)")
    grid = tuple(
        {"voting_mode": "translation-2dof", "min_inliers": mi, "nms_iou": 0.3} for mi in (5, 10)
    )
    report = run_experiment("smoke", "floorplans-door", grid)
    (entry,) = report["methods"]
    checks = {
        "method is sparse-geo": entry["method"] == "sparse-geo",
        "exactly 2 val trials": len(entry["trials"]) == 2,
        "every trial is translation-2dof": all(
            t["overrides"]["voting_mode"] == "translation-2dof" for t in entry["trials"]
        ),
        "tuned test F1 is a float": isinstance(entry["tuned_test"].get("f1"), float),
        "default test F1 is a float": isinstance(entry["default_test"].get("f1"), float),
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        logger.error("smoke FAILED: {} | entry={}", failed, entry)
        return 1
    logger.info("smoke OK: val trials scored, config frozen, tuned+default test both scored")
    return 0


def sweep() -> int:
    """Full matrix: 3 conditions x 2 grids x 2 datasets = 12 tuning runs."""
    started = perf_counter()
    cells: list[dict[str, Any]] = []
    for grid_name in ("committed", "sp-matched"):
        for condition_label, voting_mode in _CONDITIONS:
            for dataset in _DATASETS:
                cell_started = perf_counter()
                cells.append(
                    _cell_dict(
                        condition_label,
                        voting_mode,
                        grid_name,
                        dataset,
                        perf_counter() - cell_started,
                    )
                )
    reconciliation = _reconcile(cells)
    reconciliation["source"] = "docs/reports/sparse-geo-improvement.md"
    summary = {
        "provenance": {
            "git_sha": current_git_sha(),
            "method": "sparse-geo",
            "iou_threshold": 0.5,
            "seed": 0,
            "tune_split": "val",
            "eval_split": "test",
            "conditions": [label for label, _ in _CONDITIONS],
            "grids": {
                "committed": {
                    "min_inliers": list(_COMMITTED_MIN_INLIERS),
                    "nms_iou": list(_NMS_IOU),
                },
                "sp-matched": {
                    "min_inliers": list(_SP_MATCHED_MIN_INLIERS),
                    "nms_iou": list(_NMS_IOU),
                },
            },
            "note": (
                "tuning reads val only; each frozen config is scored on test exactly once. "
                "SuperPoint conditions and the committed-grid sift/single-4dof baseline are NOT "
                "re-measured here -- pulled forward from sparse-geo-improvement.md as published "
                "literals in the report."
            ),
        },
        "reconciliation": reconciliation,
        "cells": cells,
        "elapsed_s": perf_counter() - started,
    }
    out = _OUT_DIR / "summary.json"
    out.write_text(json.dumps(summary, indent=2, sort_keys=True))
    logger.info(
        "sweep: {} cells, {} reconciliation disagreement(s), wrote {} in {:.1f}s",
        len(cells),
        reconciliation["n_disagreements"],
        out,
        summary["elapsed_s"],
    )
    return 0


def main(argv: Sequence[str]) -> int:
    if len(argv) != 1 or argv[0] not in ("smoke", "sweep"):
        logger.error("usage: sparse_geo_floorplan_voting_mode_experiment.py {smoke|sweep}")
        return 2
    return smoke() if argv[0] == "smoke" else sweep()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
