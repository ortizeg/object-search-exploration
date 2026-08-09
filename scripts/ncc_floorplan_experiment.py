"""Offline lab bench for the `ncc` floor-plan investigation (quick task 260730-vx4).

Why this exists
---------------
`docs/eval/floorplans-findings.md` measured `ncc` on floor-plan DOORS at test F1 0.248 with recall
0.31 / 0.31 / 0.29 across small / medium / large symbols. Recall that flat across symbol size is
the signature of a POSE problem, not a scale or texture one -- a door drawn on a perpendicular wall
sits ~90 deg off the exemplar, entirely outside `NCCConfig.angles_deg`'s default +/-35 deg bank, and
a door with the opposite swing hand is a MIRROR of the exemplar that no rotation bank can ever
cover. This script measures that hypothesis instead of assuming it.

It is a research harness, NOT part of the shipped package: it lives in `scripts/`, writes only into
the quick task's own directory, and never touches `docs/benchmark/`. Everything it does goes
through the library's own `run_domain_tuning`, using the additive `grids=` override -- so a variant
swept here is validated by `NCCConfig` exactly as a committed grid entry would be.

Protocol (identical to the committed tuning protocol, so the numbers are comparable):
    tune on val (argmax F1 @ IoU 0.5) -> freeze -> report tuned AND default on test.
Precision and recall are logged per trial alongside F1, never F1 alone: a wider rotation bank buys
recall by throwing more candidate templates at a highly structured background (walls, dimension
lines, hatching), so a precision collapse is the specific failure to watch for.

Usage (each variant is one experiment; run one at a time, they take minutes to hours):

    pixi run python scripts/ncc_floorplan_experiment.py baseline floorplans-door
    pixi run python scripts/ncc_floorplan_experiment.py rotation-bank floorplans-door
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
_OUT_DIR = Path(".planning/quick/260730-vx4-improve-ncc-on-floor-plan-door-window-do/runs")
_DATASETS = ("floorplans-door", "floorplans-window")

# ---------------------------------------------------------------- the rotation-bank variants
# Experiment A. Four banks, crossed with retain_frac, at a FIXED single scale (1.0) -- the
# floor-plan eval already established `scales=[1.0]` as the selected scale set on this domain, so
# holding it fixed keeps the rotation effect attributable rather than confounded with scale.
#
#   default   -- what ships today: 7 steps over +/-35 deg.
#   cardinal  -- 0/90/180/270 only: the pure "doors sit on perpendicular walls" hypothesis.
#   cardinal-x-fine -- every cardinal, each with the shipped +/-35 sub-bank around it (28 angles):
#                      wall orientation AND the within-wall jitter the default bank was built for.
#   uniform30 -- 12 angles at a flat 30 deg spacing: a middle-ground CONTROL, so a win by
#                cardinal-x-fine can be attributed to angular COVERAGE rather than to the cardinal
#                angles specifically.
_FINE_OFFSETS: tuple[float, ...] = (-35.0, -23.3, -11.7, 0.0, 11.7, 23.3, 35.0)
_CARDINALS: tuple[float, ...] = (0.0, 90.0, 180.0, 270.0)

_ANGLE_BANKS: dict[str, tuple[float, ...]] = {
    "default": _FINE_OFFSETS,
    "cardinal": _CARDINALS,
    "cardinal-x-fine": tuple(
        round(cardinal + offset, 1) for cardinal in _CARDINALS for offset in _FINE_OFFSETS
    ),
    "uniform30": tuple(float(a) for a in range(0, 360, 30)),
}

_RETAIN_FRACS: tuple[float, ...] = (0.35, 0.45, 0.55)


def _rotation_bank_grid() -> tuple[dict[str, object], ...]:
    """Experiment A's grid: 4 banks x 3 retain_frac at a fixed single scale."""
    return tuple(
        {"scales": (1.0,), "angles_deg": bank, "retain_frac": retain}
        for bank in _ANGLE_BANKS.values()
        for retain in _RETAIN_FRACS
    )


# ------------------------------------------------------------------------------ Experiment B
# Mirror, measured SEPARATELY from the rotation-bank sweep above so the two effects are not
# conflated. Fixed on the WINNING bank from Experiment A (measured: `cardinal`, retain_frac=0.55,
# on both floorplans-door and floorplans-window val) -- mirror is toggled on top of that winner,
# never on top of the shipped default, since the question here is purely "does adding the
# horizontally-flipped template help once orientation coverage is already fixed."
_MIRROR_WINNING_BANK: tuple[float, ...] = _CARDINALS
_MIRROR_RETAIN_FRAC = 0.55


def _mirror_grid() -> tuple[dict[str, object], ...]:
    """Experiment B's grid: the Experiment-A winning bank, mirror off vs on."""
    return tuple(
        {
            "scales": (1.0,),
            "angles_deg": _MIRROR_WINNING_BANK,
            "retain_frac": _MIRROR_RETAIN_FRAC,
            "mirror": mirror,
        }
        for mirror in (False, True)
    )


# ------------------------------------------------------------------------------- the runner


def _log_trials(dataset: str, entry: dict[str, Any]) -> None:
    """Log every val trial's P / R / F1 -- never F1 alone (see the module docstring)."""
    logger.info("--- {} val trials (P / R / F1) ---", dataset)
    for trial in entry["trials"]:
        overrides = trial["overrides"]
        angles = overrides.get("angles_deg")
        label = f"{len(angles)}-angle" if isinstance(angles, tuple | list) else "default-bank"
        logger.info(
            "  {:<16} retain={:<5} mirror={:<5} | P={} R={} F1={}",
            label,
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
    """Tune `ncc` on `dataset` val over `grid` (None => the committed grid), report on test."""
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
        methods=("ncc",),
        grids={"ncc": grid} if grid is not None else None,
        out=str(out),
    )
    (entry,) = report["methods"]
    _log_trials(dataset, entry)
    _log_frozen(dataset, entry)
    logger.info("experiment[{}] {}: done in {:.1f}s", name, dataset, perf_counter() - started)
    return report


def main(argv: Sequence[str]) -> int:
    if len(argv) != 2 or argv[0] not in ("baseline", "rotation-bank", "mirror"):
        logger.error("usage: ncc_floorplan_experiment.py {baseline|rotation-bank|mirror} <dataset>")
        return 2
    experiment, dataset = argv[0], argv[1]
    if dataset not in _DATASETS:
        logger.error("unknown dataset {!r}; expected one of {}", dataset, _DATASETS)
        return 2

    if experiment == "baseline":
        # No grid override: the committed _TUNING_GRIDS["ncc"], i.e. today's numbers.
        run_experiment("baseline", dataset, None)
    elif experiment == "rotation-bank":
        run_experiment("rotation-bank", dataset, _rotation_bank_grid())
    else:
        run_experiment("mirror", dataset, _mirror_grid())
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
