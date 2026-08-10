"""Throwaway table formatter for quick task 260730-vx3 (NOT shipped surface).

Distils the gitignored tuning-result JSONs and the regime-harness JSONs into the small
markdown tables RESULTS.md carries. Aggregate metrics only -- no plan pixels, no per-image
records (T-vx3-02).

Usage::

    pixi run python .../summarize.py floorplan measurements/baseline-door.json ...
    pixi run python .../summarize.py regime measurements/baseline-regimes.json ...
"""

# ruff: noqa: T201 -- this script's entire job is printing a markdown table to stdout for
# copy-paste into RESULTS.md; Loguru would prefix every row with a timestamp/level and break it.
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any  # value is whatever json.loads() decoded -- str | float | int | None


def _fmt(value: Any, places: int = 3) -> str:  # noqa: ANN401
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{places}f}"
    return str(value)


def _floorplan(paths: list[Path]) -> None:
    print("| class | config | F1 | P | R | AP50 | coverage | p50 ms |")
    print("|---|---|---|---|---|---|---|---|")
    for path in paths:
        report = json.loads(path.read_text())
        klass = report["dataset"].replace("floorplans-", "")
        entry = report["methods"][0]
        for label, key in (("tuned", "tuned_test"), ("default", "default_test")):
            block = entry[key]
            overrides = entry["tuned_overrides"] if label == "tuned" else {}
            desc = ", ".join(f"{k}={v}" for k, v in sorted(overrides.items())) or "defaults"
            coverage = f"{block['n_scored']}/{block['n_images']}"
            latency = block["latency_ms"]
            p50 = latency.get("p50") if isinstance(latency, dict) else latency
            print(
                f"| {klass} | {label} ({desc}) | {_fmt(block['f1'])} | {_fmt(block['precision'])} "
                f"| {_fmt(block['recall'])} | {_fmt(block['ap50'])} | {coverage} "
                f"| {_fmt(p50, 1)} |"
            )
        print(
            f"|  | val F1 (selection) | {_fmt(entry['val_f1'])} |  |  |  |  |  |",
        )


def _regime(paths: list[Path]) -> None:
    print("| run | regime | P | R | F1 | AP | p50 ms |")
    print("|---|---|---|---|---|---|---|")
    for path in paths:
        report = json.loads(path.read_text())
        label = report.get("label") or path.stem
        for name in ("EASY", "TEXTURED", "VARIED", "CLUTTERED"):
            block = report["regimes"][name]
            print(
                f"| {label} | {name} | {_fmt(block['precision'])} | {_fmt(block['recall'])} "
                f"| {_fmt(block['f1'])} | {_fmt(block['ap'])} "
                f"| {_fmt(block['p50_latency_ms'], 1)} |"
            )


def main() -> None:
    kind, *rest = sys.argv[1:]
    paths = [Path(p) for p in rest]
    if kind == "floorplan":
        _floorplan(paths)
    elif kind == "regime":
        _regime(paths)
    else:
        raise SystemExit(f"unknown kind {kind!r}; expected 'floorplan' or 'regime'")


if __name__ == "__main__":
    main()
