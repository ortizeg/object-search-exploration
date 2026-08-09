"""Offline lab bench for the `mosse` floor-plan investigation (quick task 260730-w9s).

Why this exists
----------------
`docs/eval/floorplans-findings.md` measured `mosse` on floor-plan DOORS at test F1 0.213 with
recall 0.19 / 0.25 / 0.36 across small / medium / large symbols -- LESS flat than `ncc`'s pattern
(0.31/0.31/0.29), so this hypothesis is verified independently for `mosse` rather than assumed to
transfer. `MOSSEConfig.train_angles_deg` defaults to +/-35 deg, folded via `n_angle_groups` (default
3) into a few sharp sub-filters; a door on a perpendicular wall sits ~90 deg off the exemplar,
outside that bank. This script measures whether widening it helps, respecting the angles-per-group
invariant `ncc`'s sibling investigation flagged as a trap (widening the bank without also raising
`n_angle_groups` reproduces the already-measured-bad "one blurry filter" case).

It is a research harness, NOT part of the shipped package: it lives in `scripts/`, writes only into
this quick task's own directory, and never touches `docs/benchmark/`. Everything goes through the
library's own `run_domain_tuning` via the additive `grids=` override, so a variant swept here is
validated by `MOSSEConfig` exactly as a committed grid entry would be.

Protocol (identical to the committed tuning protocol, so numbers are comparable):
    tune on val (argmax F1 @ IoU 0.5) -> freeze -> report tuned AND default on test.
Precision and recall are logged per trial alongside F1, never F1 alone.

Usage (each variant is one experiment; run one at a time, they take minutes):

    pixi run python scripts/mosse_floorplan_experiment.py baseline floorplans-door
    pixi run python scripts/mosse_floorplan_experiment.py orientation floorplans-door
    pixi run python scripts/mosse_floorplan_experiment.py mirror floorplans-door
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path
from time import perf_counter
from typing import Any

from loguru import logger

from object_search.eval.tuning import run_domain_tuning

# Where the raw JSON reports land: the quick task's own directory, never docs/benchmark/.
_OUT_DIR = Path(".planning/quick/260730-w9s-improve-mosse-on-floor-plan-door-window-/runs")
_DATASETS = ("floorplans-door", "floorplans-window")

# ------------------------------------------------------------- Experiment A: orientation bank
# Bank x n_angle_groups combinations, respecting the angles-per-group invariant (~2-3 angles per
# sub-filter, matching the shipped default's 7 angles / 3 groups ~= 2.3 angles/group): a naive
# "28 angles, 3-4 groups" trial would reproduce the already-measured-bad blurry-filter case
# (n_angle_groups=1 measured VARIED F1 0.267) and wrongly refute the hypothesis.
_FINE_OFFSETS: tuple[float, ...] = (-35.0, -23.3, -11.7, 0.0, 11.7, 23.3, 35.0)
_CARDINALS: tuple[float, ...] = (0.0, 90.0, 180.0, 270.0)
_CARDINAL_X_FINE: tuple[float, ...] = tuple(
    round(cardinal + offset, 1) for cardinal in _CARDINALS for offset in _FINE_OFFSETS
)

# (bank, n_angle_groups) pairs -- each keeps angles-per-group in the 2-3 range except the deliberate
# "naive" control, which is included ONLY to demonstrate the trap, not as a candidate to ship.
_ORIENTATION_TRIALS: tuple[tuple[str, tuple[float, ...], int], ...] = (
    ("shipped", _FINE_OFFSETS, 3),  # 7 angles / 3 groups ~= 2.3/group (today's default)
    ("cardinal", _CARDINALS, 4),  # 4 angles / 4 groups = 1/group (each cardinal stays sharp alone)
    ("cardinal-x-fine-naive", _CARDINAL_X_FINE, 4),  # 28 angles / 4 groups = 7/group -- THE TRAP
    ("cardinal-x-fine-scaled", _CARDINAL_X_FINE, 12),  # 28 angles / 12 groups ~= 2.3/group -- fair
)

_RETAIN_FRACS: tuple[float, ...] = (0.35, 0.45, 0.55)


def _orientation_grid() -> tuple[dict[str, object], ...]:
    """Experiment A's grid: bank x n_angle_groups (paired, see above) x retain_frac."""
    return tuple(
        {
            "scales": (1.0,),
            "train_angles_deg": bank,
            "n_angle_groups": groups,
            "retain_frac": retain,
        }
        for _name, bank, groups in _ORIENTATION_TRIALS
        for retain in _RETAIN_FRACS
    )


# ------------------------------------------------------------------------------ Experiment B
# Mirror, measured SEPARATELY from orientation. Fixed on Experiment A's winning
# (bank, n_angle_groups) -- filled in once Experiment A's winner is known, see `_mirror_grid`.
_MIRROR_RETAIN_FRAC = 0.55


def _mirror_grid(
    winning_bank: tuple[float, ...], winning_groups: int
) -> tuple[dict[str, object], ...]:
    """Experiment B's grid: the Experiment-A winning (bank, groups), mirror off vs on."""
    return tuple(
        {
            "scales": (1.0,),
            "train_angles_deg": winning_bank,
            "n_angle_groups": winning_groups,
            "retain_frac": _MIRROR_RETAIN_FRAC,
            "mirror": mirror,
        }
        for mirror in (False, True)
    )


# ------------------------------------------------------------------------------------- runner


def _log_trials(dataset: str, entry: dict[str, Any]) -> None:
    """Log every val trial's P / R / F1 -- never F1 alone (see the module docstring)."""
    logger.info("--- {} val trials (P / R / F1) ---", dataset)
    for trial in entry["trials"]:
        overrides = trial["overrides"]
        angles = overrides.get("train_angles_deg")
        n_angles = len(angles) if isinstance(angles, tuple | list) else "-"
        groups = overrides.get("n_angle_groups", "-")
        logger.info(
            "  n_angles={:<4} groups={:<4} retain={:<5} mirror={:<5} | P={} R={} F1={}",
            n_angles,
            groups,
            overrides.get("retain_frac", "-"),
            overrides.get("mirror", False),
            _fmt(trial["precision"]),
            _fmt(trial["recall"]),
            _fmt(trial["f1"]),
        )


def _fmt(value: float | None) -> str:
    return f"{value:.3f}" if isinstance(value, int | float) else "  -  "


def _log_frozen(dataset: str, entry: dict[str, Any]) -> None:
    """Log the frozen-on-val config's tuned-vs-default TEST block (the one test read)."""
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


def run_experiment(
    name: str,
    dataset: str,
    grid: Sequence[dict[str, object]] | None,
) -> dict[str, Any]:
    """Tune `mosse` on `dataset` val over `grid` (None => the committed grid), report on test."""
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = _OUT_DIR / f"{name}--{dataset}.json"
    logger.info(
        "experiment[{}] {}: {} trial(s) -> {}",
        name,
        dataset,
        len(grid) if grid is not None else "committed grid",
        out,
    )
    started = perf_counter()
    report = run_domain_tuning(
        dataset,
        Path("datasets"),
        methods=("mosse",),
        grids={"mosse": grid} if grid is not None else None,
        out=str(out),
    )
    (entry,) = report["methods"]
    _log_trials(dataset, entry)
    _log_frozen(dataset, entry)
    logger.info("experiment[{}] {}: done in {:.1f}s", name, dataset, perf_counter() - started)
    return report


def main(argv: Sequence[str]) -> int:
    if len(argv) != 2 or argv[0] not in ("baseline", "orientation", "mirror"):
        logger.error("usage: mosse_floorplan_experiment.py {baseline|orientation|mirror} <dataset>")
        return 2
    experiment, dataset = argv[0], argv[1]
    if dataset not in _DATASETS:
        logger.error("unknown dataset {!r}; expected one of {}", dataset, _DATASETS)
        return 2

    if experiment == "baseline":
        # No grid override: the committed _TUNING_GRIDS["mosse"], i.e. today's numbers.
        run_experiment("baseline", dataset, None)
    elif experiment == "orientation":
        run_experiment("orientation", dataset, _orientation_grid())
    else:
        # Mirror is measured against the shipped bank by default; re-run with the actual
        # Experiment-A winner once known (see EXPERIMENTS.md) by editing the call below.
        run_experiment("mirror", dataset, _mirror_grid(_CARDINALS, 4))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
