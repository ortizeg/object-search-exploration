"""Step-A1 diagnostic for quick task 260730-vx3 (NOT shipped surface).

Answers ONE question with evidence, before any tuning: **where does a mirrored door instance
actually die** in the sparse-geo pipeline?

Three candidate stages, in pipeline order:

1. **Descriptors** -- SIFT descriptors are not mirror-invariant, so a mirrored door may produce
   no correspondences at all.
2. **Voting** -- ``_vote_single_4dof`` and ``_proper_similarity_2pt`` compute only the
   ORIENTATION-PRESERVING branch, so a mirrored instance's correspondences predict a *wrong*
   pose and never accumulate into a peak.
3. **``_is_degenerate``** -- the ``det < 0`` gate discards a reflected fit that did survive to
   RANSAC.

If stage 3's mirror branch almost never fires, then relaxing it alone cannot move the numbers,
because mirrored instances are already gone before they get there. That is the finding this
script is here to establish or refute.

Instrumentation is by monkeypatch on the private helpers -- deliberately, so ``sparse_geo.py``
carries no diagnostic scaffolding.

Usage::

    pixi run python .../diagnose_mirror.py --split val --n 12
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from loguru import logger

from object_search.eval.benchmark import _load_research_scene
from object_search.eval.labels import load_research_ground_truth
from object_search.eval.sampling import sample_exemplars
from object_search.eval.splits import research_image_ids
from object_search.log import setup_logging
from object_search.search import sparse_geo
from object_search.search.sparse_geo import SparseGeoConfig, _SimilarityModel

_REJECT = Counter[str]()
_RANSAC_WINNERS = Counter[str]()

_real_is_degenerate = sparse_geo._is_degenerate
_real_ransac = sparse_geo._ransac_similarity


def _tap_is_degenerate(
    model: _SimilarityModel, min_scale: float, max_scale: float
) -> tuple[bool, str]:
    degenerate, reason = _real_is_degenerate(model, min_scale, max_scale)
    if not degenerate:
        _REJECT["accepted"] += 1
    elif "mirror" in reason:
        _REJECT["rejected: mirror (det<0)"] += 1
    else:
        _REJECT["rejected: scale"] += 1
    return degenerate, reason


def _tap_ransac(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401 -- generic monkeypatch wrapper
    result = _real_ransac(*args, **kwargs)
    if result.model is None:
        _RANSAC_WINNERS["no model"] += 1
    elif result.model.det < 0.0:
        _RANSAC_WINNERS["winner was REFLECTED (det<0)"] += 1
    else:
        _RANSAC_WINNERS["winner was proper (det>0)"] += 1
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="floorplans-door")
    parser.add_argument("--split", default="val")
    parser.add_argument("--n", type=int, default=12)
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    setup_logging("INFO")

    sparse_geo._is_degenerate = _tap_is_degenerate  # type: ignore[assignment]
    sparse_geo._ransac_similarity = _tap_ransac  # type: ignore[assignment]

    root = Path("datasets") / args.dataset / args.split
    image_ids = research_image_ids(args.dataset, args.split)[: args.n]  # type: ignore[arg-type]

    config = SparseGeoConfig()
    totals = Counter[str]()
    per_plan: list[dict[str, Any]] = []
    for image_id in image_ids:
        gt = load_research_ground_truth(root / f"{image_id}.gt.json")
        if gt is None or not gt.boxes:
            continue
        image = _load_research_scene(root, image_id)
        # Same seeded draw the scored runs use (D-11), so the diagnostic looks at the same
        # queries the measured F1 came from.
        exemplar = sample_exemplars(gt, count=1, seed=0)[0]
        result = sparse_geo.search(image, exemplar, config)
        metrics = result.diagnostics.metrics
        per_plan.append(
            {
                "plan": image_id,
                "n_gt": len(gt.boxes),
                "n_crop_keypoints": metrics.get("n_crop_keypoints"),
                "n_correspondences": metrics.get("n_correspondences"),
                "n_crop_matched": metrics.get("n_crop_matched"),
                "n_peaks": metrics.get("n_peaks"),
                "n_instances": metrics.get("n_instances"),
            }
        )
        for key in (
            "n_crop_keypoints",
            "n_correspondences",
            "n_crop_matched",
            "n_peaks",
            "n_instances",
        ):
            totals[key] += int(metrics.get(key, 0.0))
        totals["n_gt"] += len(gt.boxes)

    report = {
        "dataset": args.dataset,
        "split": args.split,
        "n_plans": len(per_plan),
        "totals": dict(totals),
        "degeneracy_gate": dict(_REJECT),
        "ransac_winner_chirality": dict(_RANSAC_WINNERS),
        "per_plan": per_plan,
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    logger.info("mirror diagnosis:\n{}", text)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
