"""Throwaway SuperPoint feasibility probe for quick task 260730-vx3 (NOT shipped surface).

Hypothesis 2 is ``backend="superpoint"`` + ``voting_mode="pairwise-4dof"``. Before committing a
10-entry grid x 56 val plans to it, this probe answers the three things the plan says to watch,
on a handful of plans:

* **Latency** -- pairwise-4dof is O(n^2) in correspondences and SuperPoint yields far more
  keypoints than SIFT on line art. This sizes the sweep (or proves it impractical on CPU).
* **Abstentions** -- ``min_exemplar_keypoints=8`` with SuperPoint's 8 px effective border may push
  small door crops to ``outcome=EMPTY``.
* **Keypoint yield** -- the actual exemplar/scene keypoint counts, SIFT vs SuperPoint, which is the
  whole premise of the hypothesis.

Usage::

    pixi run python .planning/quick/<dir>/probe_superpoint.py --dataset floorplans-door -n 6
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from loguru import logger

from object_search.eval.benchmark import _load_research_scene
from object_search.eval.labels import load_research_ground_truth
from object_search.eval.sampling import sample_exemplars
from object_search.eval.splits import research_image_ids
from object_search.log import setup_logging
from object_search.search import get_method

_METHOD = "sparse-geo"


def _probe(label: str, overrides: dict[str, Any], root: Path, ids: list[str]) -> dict[str, Any]:
    spec = get_method(_METHOD)
    config = spec.config_model(**overrides)
    rows: list[dict[str, Any]] = []
    for image_id in ids:
        gt = load_research_ground_truth(root / f"{image_id}.gt.json")
        if gt is None:
            continue
        scene = _load_research_scene(root, image_id)
        exemplars = sample_exemplars(gt, count=1, seed=0)
        started = time.perf_counter()
        result = spec.fn(scene, exemplars[0], config)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        metrics = dict(result.diagnostics.metrics) if result.diagnostics else {}
        rows.append(
            {
                "image_id": image_id,
                "outcome": result.outcome.value,
                "n_matches": len(result.matches),
                "gt_instances": len(gt.boxes),
                "latency_ms": round(elapsed_ms, 1),
                "n_crop_keypoints": metrics.get("n_crop_keypoints"),
                "n_scene_keypoints": metrics.get("n_scene_keypoints"),
                "n_correspondences": metrics.get("n_correspondences"),
                "n_peaks": metrics.get("n_peaks"),
            }
        )
    lats = sorted(r["latency_ms"] for r in rows)
    return {
        "label": label,
        "overrides": overrides,
        "n_images": len(rows),
        "n_empty": sum(1 for r in rows if r["outcome"] == "empty"),
        "n_error": sum(1 for r in rows if r["outcome"] == "error"),
        "p50_latency_ms": lats[len(lats) // 2] if lats else None,
        "max_latency_ms": lats[-1] if lats else None,
        "total_s": round(sum(lats) / 1000.0, 1),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="floorplans-door")
    parser.add_argument("--split", default="val")
    parser.add_argument("--research-root", default="datasets")
    parser.add_argument("-n", type=int, default=6)
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    setup_logging("WARNING")

    root = Path(args.research_root) / args.dataset / args.split
    ids = list(research_image_ids(args.dataset, args.split))[: args.n]

    report = {
        "dataset": args.dataset,
        "split": args.split,
        "probes": [
            _probe("sift-single4dof (baseline)", {}, root, ids),
            _probe(
                "sift-pairwise4dof", {"voting_mode": "pairwise-4dof", "min_inliers": 5}, root, ids
            ),
            _probe(
                "superpoint-translation2dof",
                {"backend": "superpoint", "min_inliers": 5},
                root,
                ids,
            ),
            _probe(
                "superpoint-pairwise4dof",
                {"backend": "superpoint", "voting_mode": "pairwise-4dof", "min_inliers": 5},
                root,
                ids,
            ),
        ],
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    for probe in report["probes"]:
        assert isinstance(probe, dict)  # noqa: S101
        logger.warning(
            "{}: p50 {} ms / max {} ms / total {} s / empty {} / err {}",
            probe["label"],
            probe["p50_latency_ms"],
            probe["max_latency_ms"],
            probe["total_s"],
            probe["n_empty"],
            probe["n_error"],
        )
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
