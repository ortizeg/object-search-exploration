"""Ground-truth loading (EVAL-02) and detection metrics (EVAL-04/16), including edge cases.

The abstention edge case (R = 0 -> precision None, not 0) and a hand-computed all-point AP are
the two assertions this file exists to make load-bearing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from object_search.eval.labels import GroundTruth, load_ground_truth
from object_search.eval.metrics import (
    average_precision,
    match_predictions,
    precision_recall_f1,
)
from object_search.schemas.geometry import BBox

# --------------------------------------------------------------------------- labels (Task 1)


def _write_sidecar(directory: Path, image_id: str, payload: dict[str, object]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{image_id}.gt.json").write_text(json.dumps(payload), encoding="utf-8")


def test_labels_reads_real_chipset_sidecar_with_achieved_count() -> None:
    # The committed chipset-01 sidecar has achieved_n = 5. The loader must surface exactly those
    # five boxes and the achieved (not requested) count.
    gt = load_ground_truth("chipset-01")
    assert gt is not None
    assert gt.source == "chipset"
    assert gt.achieved_count == 5
    assert len(gt.boxes) == 5
    assert gt.width == 320 and gt.height == 240
    assert gt.exemplar_index == 0
    # Every chip is 24x24 by construction.
    assert all(box.w == 24 and box.h == 24 for box in gt.boxes)


def test_labels_reads_synthetic_sidecar_slice_metadata() -> None:
    gt = load_ground_truth("scatter-scaled")
    assert gt is not None
    assert gt.source == "synthetic"
    assert gt.achieved_count == 10
    assert gt.slice_metadata.true_instance_count == 10
    # The synthetic spec carries the canvas size.
    assert gt.width == 960 and gt.height == 640


def test_labels_returns_none_for_unlabelled_image() -> None:
    # None means "nobody labelled this", NOT "zero instances". Reported honestly, never [].
    assert load_ground_truth("no-such-image-anywhere") is None


def test_labels_reads_achieved_count_from_tmp_sidecar(tmp_path: Path) -> None:
    load_ground_truth.cache_clear()
    payload = {
        "image": "toy.png",
        "width": 100,
        "height": 80,
        "requested_n": 9,  # requested 9 ...
        "achieved_n": 2,  # ... but only 2 placed; the loader must use 2, never 9.
        "exemplar_index": 1,
        "boxes": [
            {"x": 0, "y": 0, "w": 10, "h": 10},
            {"x": 40, "y": 40, "w": 10, "h": 10},
        ],
    }
    _write_sidecar(tmp_path, "toy", payload)
    gt = load_ground_truth("toy", root=tmp_path)
    assert gt is not None
    assert gt.achieved_count == 2  # the achieved count, not the requested 9
    assert gt.exemplar_index == 1
    assert gt.exemplar.box == BBox(x=40, y=40, w=10, h=10)


def test_labels_rejects_corrupt_achieved_count(tmp_path: Path) -> None:
    load_ground_truth.cache_clear()
    payload = {
        "achieved_n": 5,  # lies: only one box present
        "boxes": [{"x": 0, "y": 0, "w": 10, "h": 10}],
    }
    _write_sidecar(tmp_path, "bad", payload)
    with pytest.raises(ValueError, match="disagrees"):
        load_ground_truth("bad", root=tmp_path)


def test_labels_rejects_empty_box_list(tmp_path: Path) -> None:
    load_ground_truth.cache_clear()
    _write_sidecar(tmp_path, "empty", {"boxes": []})
    with pytest.raises(ValueError, match="empty box list"):
        load_ground_truth("empty", root=tmp_path)


def test_labels_rejects_out_of_range_exemplar_index(tmp_path: Path) -> None:
    load_ground_truth.cache_clear()
    payload = {"exemplar_index": 7, "boxes": [{"x": 0, "y": 0, "w": 10, "h": 10}]}
    _write_sidecar(tmp_path, "oor", payload)
    with pytest.raises(ValueError, match="exemplar_index"):
        load_ground_truth("oor", root=tmp_path)


def test_groundtruth_is_frozen() -> None:
    gt = GroundTruth(
        image_id="x",
        boxes=(BBox(x=0, y=0, w=1, h=1),),
        exemplar_index=0,
        source="hand",
    )
    with pytest.raises(ValidationError):
        gt.image_id = "y"  # type: ignore[misc]


# --------------------------------------------------------------------------- matching (Task 2)


def test_match_predictions_perfect() -> None:
    gt = [BBox(x=0, y=0, w=10, h=10), BBox(x=50, y=50, w=10, h=10)]
    pred = [BBox(x=0, y=0, w=10, h=10), BBox(x=50, y=50, w=10, h=10)]
    assert match_predictions(pred, gt) == (2, 0, 0)


def test_match_predictions_duplicate_is_tp_plus_fp() -> None:
    # EVAL-16: two boxes on one instance = 1 TP + 1 FP, never 2 TP.
    gt = [BBox(x=0, y=0, w=10, h=10)]
    pred = [BBox(x=0, y=0, w=10, h=10), BBox(x=1, y=0, w=10, h=10)]
    tp, fp, fn = match_predictions(pred, gt)
    assert (tp, fp, fn) == (1, 1, 0)


def test_match_predictions_miss_is_fn() -> None:
    gt = [BBox(x=0, y=0, w=10, h=10), BBox(x=50, y=50, w=10, h=10)]
    pred = [BBox(x=0, y=0, w=10, h=10)]
    assert match_predictions(pred, gt) == (1, 0, 1)


def test_match_predictions_below_threshold_is_fp() -> None:
    gt = [BBox(x=0, y=0, w=10, h=10)]
    # IoU well under 0.5 -- a near miss is a false positive plus a false negative.
    pred = [BBox(x=8, y=0, w=10, h=10)]
    tp, fp, fn = match_predictions(pred, gt)
    assert (tp, fp, fn) == (0, 1, 1)


def test_match_predictions_higher_iou_gt_wins() -> None:
    # One prediction near two GT: it must claim the better-overlapping one.
    gt = [BBox(x=0, y=0, w=10, h=10), BBox(x=3, y=0, w=10, h=10)]
    pred = [BBox(x=3, y=0, w=10, h=10)]
    tp, fp, fn = match_predictions(pred, gt)
    assert (tp, fp, fn) == (1, 0, 1)


# --------------------------------------------------------------------------- p/r/f1 (Task 2)


def test_precision_recall_f1_known() -> None:
    p, r, f1 = precision_recall_f1(tp=3, fp=1, fn=1)
    assert p == pytest.approx(0.75)
    assert r == pytest.approx(0.75)
    assert f1 == pytest.approx(0.75)


def test_precision_none_on_abstention() -> None:
    # tp + fp == 0: nothing returned. Precision is UNDEFINED (None), never 0.
    p, r, f1 = precision_recall_f1(tp=0, fp=0, fn=4)
    assert p is None
    assert r == pytest.approx(0.0)
    assert f1 is None


def test_recall_none_when_nothing_to_find() -> None:
    p, r, f1 = precision_recall_f1(tp=0, fp=3, fn=0)
    assert p == pytest.approx(0.0)
    assert r is None
    assert f1 is None


def test_f1_none_when_p_and_r_both_zero() -> None:
    p, r, f1 = precision_recall_f1(tp=0, fp=2, fn=2)
    assert p == pytest.approx(0.0)
    assert r == pytest.approx(0.0)
    assert f1 is None


# --------------------------------------------------------------------------- AP (Task 2)


def test_average_precision_hand_computed_all_point() -> None:
    # Two GT. Ranked candidate log: TP(g1), FP, TP(g2).
    #   after #1: p=1.0,   r=0.5
    #   after #2: p=0.5,   r=0.5
    #   after #3: p=2/3,   r=1.0
    # Envelope (monotone from right): [1.0, 2/3, 2/3].
    # AP = (0.5-0.0)*1.0 + (1.0-0.5)*(2/3) = 0.5 + 1/3 = 5/6.
    gt = [BBox(x=0, y=0, w=10, h=10), BBox(x=50, y=50, w=10, h=10)]
    candidates = [
        (BBox(x=0, y=0, w=10, h=10), 0.9),
        (BBox(x=200, y=200, w=10, h=10), 0.8),
        (BBox(x=50, y=50, w=10, h=10), 0.7),
    ]
    ap = average_precision(candidates, gt)
    assert ap == pytest.approx(5.0 / 6.0)


def test_average_precision_perfect_is_one() -> None:
    gt = [BBox(x=0, y=0, w=10, h=10)]
    candidates = [(BBox(x=0, y=0, w=10, h=10), 0.9)]
    assert average_precision(candidates, gt) == pytest.approx(1.0)


def test_average_precision_empty_candidate_log_is_zero() -> None:
    gt = [BBox(x=0, y=0, w=10, h=10)]
    assert average_precision([], gt) == pytest.approx(0.0)


def test_average_precision_raises_without_ground_truth() -> None:
    with pytest.raises(ValueError, match="zero ground-truth"):
        average_precision([(BBox(x=0, y=0, w=10, h=10), 0.9)], [])
