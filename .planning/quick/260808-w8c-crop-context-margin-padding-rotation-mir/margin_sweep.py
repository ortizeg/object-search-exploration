"""Crop context-margin sweep against the exported contrastive-crop checkpoint, no retraining.

Per D-w8c-03/04: this is the cheapest possible test of the crop-margin hypothesis -- no training
run, just re-scoring an existing checkpoint. It sweeps ``crop_context_margin_frac`` in
``(0.0, 0.15, 0.3, 0.5)`` crossed with owlv2-oneshot's EXISTING tuning grid (``max_box_area_frac`` x
``query_iou_frac``), through the SAME ``run_domain_tuning`` tune-on-val/freeze/report-on-test
methodology every other arm's numbers came from, using the new ``grids`` override added to
``object_search.eval.tuning`` for exactly this purpose. The margin=0.0 cell is a live regression
check: it must reproduce 260808-dla's committed ``contrastive-crop`` tuned F1 exactly on both
datasets, or the ``grids`` plumbing has a bug and no nonzero-margin number should be trusted.

Written for local CPU execution:
    pixi run python .planning/quick/260808-w8c-crop-context-margin-padding-rotation-mir/\
margin_sweep.py

In practice this is heavier than it looks -- each of the 8 (margin, dataset) cells re-runs the full
9-entry grid over the whole 56-image val split plus tuned+default test evaluation, unlike the
lightweight single-exemplar diagnostics elsewhere in this repo. A local CPU run was tried first and
killed after 35+ minutes with the first cell still incomplete; the numbers this script actually
produced were measured on a vast.ai RTX 3090 with ``OS_ONNX_PROVIDERS=CUDAExecutionProvider,
CPUExecutionProvider`` set, which finished all 8 cells in ~34 minutes. The script itself is
provider-agnostic (:mod:`object_search.inference.onnx_inferencer` already respects
``OS_ONNX_PROVIDERS``); only the environment it happened to run in differed from "local CPU."
"""

import json
import os

from loguru import logger

from object_search.eval.tuning import _TUNING_GRIDS, run_domain_tuning
from object_search.provenance import repo_root

_MARGINS = (0.0, 0.15, 0.3, 0.5)
_DATASETS = ("floorplans-door", "floorplans-window")
_OUT_DIR = repo_root() / "docs" / "benchmark" / "owlv2-margin-sweep"
_COMMITTED_DIR = repo_root() / "docs" / "benchmark" / "owlv2-finetune"


def main() -> None:
    os.environ["OS_OWLV2_MODEL"] = "owlv2_base_patch16_floorplans_ft_contrastive_crop.onnx"
    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    summary: dict[str, float | None] = {}
    for margin in _MARGINS:
        margin_grid = [
            {**dict(g), "crop_context_margin_frac": margin} for g in _TUNING_GRIDS["owlv2-oneshot"]
        ]
        for dataset in _DATASETS:
            tag = str(margin)
            out = _OUT_DIR / f"{dataset}-margin-{tag}.json"
            report = run_domain_tuning(
                dataset,
                "datasets",
                methods=("owlv2-oneshot",),
                exemplar_count=1,
                grids={"owlv2-oneshot": margin_grid},
                out=str(out),
            )
            f1 = report["methods"][0]["tuned_test"]["f1"]
            summary[f"{dataset}-{tag}"] = f1
            logger.info("margin={} {}: tuned F1={}", margin, dataset, f1)

    (_OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    logger.info("wrote {}", _OUT_DIR / "summary.json")

    # D-w8c-04's built-in regression check: margin=0.0 must reproduce the committed numbers exactly.
    for dataset in _DATASETS:
        committed = json.loads((_COMMITTED_DIR / f"{dataset}-contrastive-crop.json").read_text())
        committed_f1 = committed["methods"][0]["tuned_test"]["f1"]
        swept_f1 = summary[f"{dataset}-0.0"]
        if swept_f1 != committed_f1:
            raise RuntimeError(
                f"{dataset}: margin=0.0 cell {swept_f1} != committed {committed_f1} -- "
                "grids plumbing bug, do not trust any nonzero-margin number"
            )
        logger.info("{}: margin=0.0 regression check passed ({})", dataset, swept_f1)


if __name__ == "__main__":
    main()
