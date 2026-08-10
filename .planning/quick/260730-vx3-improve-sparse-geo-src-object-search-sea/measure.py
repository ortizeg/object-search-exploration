"""Throwaway floor-plan measurement driver for quick task 260730-vx3 (NOT shipped surface).

``run_domain_tuning`` always uses the committed ``_TUNING_GRIDS`` entry for a method. During the
iterate/measure loop we need to score a CANDIDATE grid without committing it -- otherwise every
disproven hypothesis would have to be committed and then reverted out of ``src/``. This driver
reproduces the same tune-on-val / freeze / report-on-test protocol with an explicit grid passed
in, using the project's own ``tune_method`` and ``_evaluate`` so the numbers are produced by the
identical code path.

Tuning reads ``val`` only; ``test`` is scored once per frozen config (and once at the method's
defaults, as the baseline column).

Usage::

    pixi run python .../measure.py --label hyp1-single --grid '<json list>' \
        --dataset floorplans-door --out measurements/hyp1-door.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from loguru import logger

from object_search.eval.tuning import _evaluate, tune_method
from object_search.log import setup_logging
from object_search.search import get_method

_METHOD = "sparse-geo"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--grid", required=True, help="JSON list of override dicts.")
    parser.add_argument("--label", default="")
    parser.add_argument("--research-root", default="datasets")
    parser.add_argument("--exemplars", type=int, default=1)
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    setup_logging("INFO")

    grid: list[dict[str, Any]] = json.loads(args.grid)
    base = Path(args.research_root)
    spec = get_method(_METHOD)

    tuned = tune_method(
        _METHOD,
        args.dataset,
        base / args.dataset / "val",
        exemplar_count=args.exemplars,
        grid=grid,
    )
    best = tuned["best"]
    tuned_config = spec.config_model(**best["overrides"]) if best else None

    test_root = base / args.dataset / "test"
    tuned_test = _evaluate(
        _METHOD,
        args.dataset,
        "test",
        test_root,
        config=tuned_config,
        exemplar_count=args.exemplars,
        iou_threshold=0.5,
        seed=0,
        manifest_root=None,
    )
    report = {
        "label": args.label,
        "dataset": args.dataset,
        "method": _METHOD,
        "tuned_overrides": best["overrides"] if best else {},
        "val_f1": best["f1"] if best else None,
        "tuned_test": tuned_test,
        "trials": tuned["trials"],
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    logger.info(
        "{} / {}: tuned {} -> test F1 {}",
        args.label,
        args.dataset,
        report["tuned_overrides"],
        tuned_test.get("f1"),
    )
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
