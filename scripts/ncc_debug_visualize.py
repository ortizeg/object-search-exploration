"""Per-step debug visualization for `ncc` on a single image (quick task 260730-vx4).

Why this exists
----------------
The floor-plan investigation found a real, measured F1 gain on doors (0.164 -> 0.358 test) but a
weaker, more mixed result on windows (the argmax-on-val pick generalizes to test F1 0.350, slightly
BELOW what the pre-existing default-bank grid entry already achieved: 0.401). Aggregate F1/P/R
numbers cannot say *why* -- which instances are missed, which false positives appear, whether the
cardinal bank's extra candidate templates are throwing false peaks on structured wall/dimension-line
background. This script renders exactly that, image by image, so a practitioner can look.

It is a research/debug tool, not part of the shipped package: it lives in `scripts/`, only ever
reads committed or fetched datasets, and never writes into `docs/benchmark/` or the shipped config.
Debug output generation is itself the "flag" -- this script only runs, and only writes debug images,
when explicitly invoked; `ncc.search()` itself is untouched and carries no debug branch (the
method-module convention: no hidden control flow inside a method).

What it renders, per step of the pipeline that matters for debugging a miss or a false positive:
    01_query.png          -- the exemplar box drawn on the full scene (what was asked for).
    02_matches_vs_gt.png   -- accepted matches (green), sub-threshold candidates (red, scored),
                              and every ground-truth box, colored by whether some match claimed it
                              (green outline = matched, yellow = missed) -- step 8's output judged
                              directly against the labels.
    03_heatmap.png         -- the representative (scale~=1, unmirrored) similarity heatmap -- step 4
                              /5's output, before peak extraction and calibration.
Console output prints the calibration reasoning, threshold, self-score, and per-level peak counts
(steps 5-9) so the numeric side of the pipeline is visible alongside the images.

Usage:
    pixi run python scripts/ncc_debug_visualize.py \
        --image datasets/floorplans-window/test/<stem>.png \
        --config tuned-window \
        --out .planning/quick/260730-vx4-improve-ncc-on-floor-plan-door-window-do/debug/window-101

    pixi run python scripts/ncc_debug_visualize.py --image <path> --config default --out <dir>
    pixi run python scripts/ncc_debug_visualize.py --image <path> --exemplar-index 2 --out <dir>
"""

from __future__ import annotations

import argparse
import base64
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt
from loguru import logger

from object_search.eval.labels import load_research_ground_truth
from object_search.eval.metrics import match_predictions_detailed
from object_search.schemas import BBox, ExemplarBox
from object_search.search.common import viz
from object_search.search.ncc import NCCConfig
from object_search.search.ncc import search as ncc_search

# Config presets: "default" is NCCConfig()'s shipped defaults (untouched by this investigation).
# "tuned-door" / "tuned-window" are the argmax-on-val winners measured in EXPERIMENTS.md for each
# floor-plan class -- fixed here (not re-derived) so a debug run always reflects the exact config
# the tuning grid actually picked, not a re-guess.
_CONFIG_PRESETS: dict[str, dict[str, object]] = {
    "default": {},
    "tuned-door": {
        "scales": (1.0,),
        "angles_deg": (0.0, 90.0, 180.0, 270.0),
        "mirror": True,
        "retain_frac": 0.65,
        "nms_iou": 0.3,
    },
    "tuned-window": {
        "scales": (1.0,),
        "angles_deg": (0.0, 90.0, 180.0, 270.0),
        "mirror": False,
        "retain_frac": 0.65,
        "nms_iou": 0.3,
    },
}

_MATCHED_GT_COLOR = (0, 200, 0)  # green -- a GT box some match claimed
_MISSED_GT_COLOR = (0, 220, 255)  # yellow -- a GT box no match claimed (a false negative)
_CANDIDATE_COLOR = (0, 0, 220)  # red -- a sub-threshold candidate, scored but rejected


def _decode_heatmap(png_b64: str) -> npt.NDArray[np.uint8] | None:
    """Decode the diagnostics heatmap PNG back to a BGR image (mirrors samples.py's helper)."""
    raw = np.frombuffer(base64.b64decode(png_b64), dtype=np.uint8)
    decoded = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    return np.asarray(decoded, dtype=np.uint8) if decoded is not None else None


def _draw_gt_and_candidates(
    canvas: npt.NDArray[np.uint8],
    gt_boxes: tuple[BBox, ...],
    matched: tuple[bool, ...],
    candidates: tuple[Any, ...],
) -> npt.NDArray[np.uint8]:
    """Overlay every GT box (matched=green, missed=yellow) and every candidate (red, scored)."""
    out = canvas.copy()
    for box, is_matched in zip(gt_boxes, matched, strict=True):
        color = _MATCHED_GT_COLOR if is_matched else _MISSED_GT_COLOR
        x, y, x2, y2 = box.xyxy
        cv2.rectangle(out, (x, y), (x2 - 1, y2 - 1), color, 1)
    for candidate in candidates:
        x, y, x2, y2 = candidate.box.xyxy
        cv2.rectangle(out, (x, y), (x2 - 1, y2 - 1), _CANDIDATE_COLOR, 1)
        cv2.putText(
            out,
            f"{candidate.score:.2f}",
            (x, min(out.shape[0] - 1, y2 + 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            _CANDIDATE_COLOR,
            1,
            cv2.LINE_AA,
        )
    return out


def _sidecar_for(image_path: Path) -> Path:
    """The converted-dataset sidecar path for an image: ``<name>.png`` -> ``<name>.gt.json``.

    Plain string suffix-stripping, not ``Path.with_suffix`` -- the converted stems are dotted
    (e.g. ``16_png.rf.P5wwuwazuoIXFvGYThYe.png``), and ``with_suffix`` treats everything after
    the *last* dot as the suffix, so it would corrupt the stem instead of just dropping ``.png``.
    """
    return image_path.parent / f"{image_path.name.removesuffix('.png')}.gt.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True, help="Path to a scene PNG.")
    parser.add_argument(
        "--exemplar-index",
        type=int,
        default=None,
        help="GT box index to query with; default: the sidecar's own exemplar_index.",
    )
    parser.add_argument(
        "--config",
        choices=sorted(_CONFIG_PRESETS),
        default="default",
        help="Which NCCConfig to run.",
    )
    parser.add_argument("--out", type=Path, required=True, help="Directory to write debug images.")
    args = parser.parse_args(argv)

    raw = cv2.imread(str(args.image))
    if raw is None:
        logger.error("could not read image: {}", args.image)
        return 2
    image = np.ascontiguousarray(raw, dtype=np.uint8)

    sidecar = _sidecar_for(args.image)
    gt = load_research_ground_truth(sidecar)
    if gt is None:
        logger.error("no ground-truth sidecar found at {}", sidecar)
        return 2

    exemplar_index = args.exemplar_index if args.exemplar_index is not None else gt.exemplar_index
    exemplar = ExemplarBox(box=gt.boxes[exemplar_index])
    config = NCCConfig.model_validate(_CONFIG_PRESETS[args.config])

    result = ncc_search(image, exemplar, config)

    args.out.mkdir(parents=True, exist_ok=True)

    # 01 -- the query: what was asked for.
    query_tile = viz.draw_matches(image, [], exemplar=exemplar)
    cv2.imwrite(str(args.out / "01_query.png"), query_tile)

    # 02 -- matches, candidates, and GT (matched vs missed) overlaid together.
    pred_boxes = tuple(m.box for m in result.matches)
    _tp, _fp, _fn, matched_gt = match_predictions_detailed(pred_boxes, gt.boxes, iou_threshold=0.5)
    matches_tile = viz.draw_matches(image, result.matches)
    matches_vs_gt = _draw_gt_and_candidates(matches_tile, gt.boxes, matched_gt, result.candidates)
    cv2.imwrite(str(args.out / "02_matches_vs_gt.png"), matches_vs_gt)

    # 03 -- the representative similarity heatmap, if the method produced one.
    heatmap_payload = result.diagnostics.similarity_heatmap
    if heatmap_payload is not None:
        heatmap = _decode_heatmap(heatmap_payload.png_b64)
        if heatmap is not None:
            cv2.imwrite(str(args.out / "03_heatmap.png"), heatmap)

    tp = sum(matched_gt)
    fp = len(pred_boxes) - tp
    fn = len(gt.boxes) - tp

    logger.info(
        "image={} config={} exemplar_index={}", args.image.name, args.config, exemplar_index
    )
    logger.info(
        "outcome={} tp={} fp={} fn={} (P={:.3f} R={:.3f})",
        result.outcome.value,
        tp,
        fp,
        fn,
        tp / (tp + fp) if (tp + fp) else 0.0,
        tp / (tp + fn) if (tp + fn) else 0.0,
    )
    logger.info(
        "threshold={} self_score={} n_matches={} n_candidates={}",
        result.threshold_applied,
        result.diagnostics.metrics.get("self_score"),
        len(result.matches),
        len(result.candidates),
    )
    for note in result.diagnostics.notes:
        logger.info("note: {}", note)
    logger.info("wrote debug images to {}", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
