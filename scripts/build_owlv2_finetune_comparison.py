#!/usr/bin/env python
"""Assemble the baseline-vs-fine-tuned OWLv2 comparison from the twelve pulled-back result files.

Reads ``docs/benchmark/owlv2-finetune/{dataset}-{arm}.json`` (written by ``run_domain_tuning`` on
the vast.ai box, see ``scripts/gpu_finetune.sh``) for every (dataset, arm) pair, and emits:

* ``docs/benchmark/owlv2-finetune-comparison.json`` (gitignored, regenerable) -- one row per
  (dataset, arm) carrying both the default-config and the tuned-config precision/recall/F1 @ IoU
  0.5, plus coverage and the tuned overrides.
* A markdown table on stdout, for pasting into ``docs/reports/owlv2-floorplans-finetune.md``.

Each result file's schema is ``run_domain_tuning``'s report shape with a single method
(``methods=("owlv2-oneshot",)``): ``report["methods"][0]`` carries ``tuned_overrides``, ``val_f1``,
``tuned_test`` and ``default_test`` (each an ``_aggregate_research`` overall block: precision,
recall, f1, n_images, n_scored, ...).

Run after ``scripts/gpu_finetune.sh`` has pulled the six JSONs back into the worktree:

    pixi run python scripts/build_owlv2_finetune_comparison.py
"""

from __future__ import annotations

import json
from typing import Any

from loguru import logger

from object_search.provenance import repo_root

_DATASETS = ("floorplans-door", "floorplans-window")
_ARMS = ("baseline", "headonly", "full", "contrastive", "contrastive-crop", "contrastive-crop-v2")
_RESULTS_DIR = repo_root() / "docs" / "benchmark" / "owlv2-finetune"
_OUT_PATH = repo_root() / "docs" / "benchmark" / "owlv2-finetune-comparison.json"


def _metric_block(overall: dict[str, Any]) -> dict[str, Any]:
    return {
        "precision": overall.get("precision"),
        "recall": overall.get("recall"),
        "f1": overall.get("f1"),
        "n_scored": overall.get("n_scored"),
        "n_images": overall.get("n_images"),
    }


def _load_row(dataset: str, arm: str) -> dict[str, Any]:
    path = _RESULTS_DIR / f"{dataset}-{arm}.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"missing result file for ({dataset}, {arm}): {path}\n"
            "Run scripts/gpu_finetune.sh and pull docs/benchmark/owlv2-finetune/*.json back "
            "into the worktree first."
        )
    report = json.loads(path.read_text())
    methods = report.get("methods") or []
    if len(methods) != 1 or methods[0].get("method") != "owlv2-oneshot":
        raise ValueError(f"{path} does not carry exactly one owlv2-oneshot method block")
    block = methods[0]
    tuned_f1 = block["tuned_test"].get("f1")
    if tuned_f1 is None:
        raise ValueError(f"{path}: tuned_test f1 is null (nothing scored) -- not a usable result")
    return {
        "dataset": dataset,
        "arm": arm,
        "tuned_overrides": block.get("tuned_overrides", {}),
        "val_f1": block.get("val_f1"),
        "default": _metric_block(block["default_test"]),
        "tuned": _metric_block(block["tuned_test"]),
        "f1": float(tuned_f1),
    }


def build_comparison() -> dict[str, Any]:
    """Read all six result files and return the comparison report (also written to ``_OUT_PATH``)."""
    rows = [_load_row(dataset, arm) for dataset in _DATASETS for arm in _ARMS]
    report = {"rows": rows}
    _OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _OUT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    logger.info("wrote {} rows to {}", len(rows), _OUT_PATH)
    return report


def _fmt(value: float | None) -> str:
    return f"{value:.3f}" if isinstance(value, int | float) else "-"


def render_markdown(report: dict[str, Any]) -> str:
    """Render the comparison as a markdown table, one row per (dataset, arm)."""
    header = (
        "| dataset | arm | default P | default R | default F1 "
        "| tuned P | tuned R | tuned F1 | tuned overrides |"
    )
    sep = "|---|---|---|---|---|---|---|---|---|"
    lines = [header, sep]
    for row in report["rows"]:
        default, tuned = row["default"], row["tuned"]
        lines.append(
            f"| {row['dataset']} | {row['arm']} "
            f"| {_fmt(default['precision'])} | {_fmt(default['recall'])} | {_fmt(default['f1'])} "
            f"| {_fmt(tuned['precision'])} | {_fmt(tuned['recall'])} | {_fmt(tuned['f1'])} "
            f"| {row['tuned_overrides']} |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    comparison = build_comparison()
    print(render_markdown(comparison))
