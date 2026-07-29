"""COCO AP sweep and counting metrics (EVAL-24).

Two constraints this file makes load-bearing:

* **AP50 cannot drift from the pre-existing number.** ``average_precision_coco`` must return an
  ``AP50`` byte-equal to ``average_precision(..., 0.5)`` -- it calls the same function, so old and
  new reports reconcile. A regression that reimplemented AP would silently break that.
* **NAE guards ``true == 0``.** A zero true count has no normalised error; the image is skipped
  from the NAE average rather than dividing by zero.
"""

from __future__ import annotations

import math

import pytest

from object_search.eval.metrics import (
    average_precision,
    average_precision_coco,
    counting_errors,
    match_predictions,
    match_predictions_detailed,
)
from object_search.schemas.geometry import BBox

# ------------------------------------------------------------------ matched-GT-index sibling


def test_detailed_matcher_projects_to_the_tuple_form_unchanged() -> None:
    # Two GT, two perfect preds + one spurious: (tp, fp, fn) must equal match_predictions exactly.
    gt = [BBox(x=0, y=0, w=10, h=10), BBox(x=50, y=50, w=10, h=10)]
    preds = [
        BBox(x=0, y=0, w=10, h=10),  # hits gt[0]
        BBox(x=50, y=50, w=10, h=10),  # hits gt[1]
        BBox(x=200, y=200, w=10, h=10),  # hits nothing
    ]
    tp, fp, fn, matched = match_predictions_detailed(preds, gt)
    assert (tp, fp, fn) == match_predictions(preds, gt)  # sibling reconciles with the tuple form
    assert (tp, fp, fn) == (2, 1, 0)  # both GT found, one spurious pred, nothing missed
    assert matched == (True, True)  # aligned to gt; both GT found
    assert sum(matched) == tp  # the documented invariant


def test_detailed_matcher_reports_which_gt_matched() -> None:
    # Only the second GT is found; the first is missed -> matched flags pinpoint which.
    gt = [BBox(x=0, y=0, w=10, h=10), BBox(x=50, y=50, w=10, h=10)]
    preds = [BBox(x=50, y=50, w=10, h=10)]
    tp, fp, fn, matched = match_predictions_detailed(preds, gt)
    assert (tp, fp, fn) == (1, 0, 1)
    assert matched == (False, True)
    assert sum(matched) == tp


def test_detailed_matcher_duplicate_rule_credits_a_gt_once() -> None:
    # EVAL-16: two preds on ONE gt -> that gt matched once, the second pred is a false positive.
    gt = [BBox(x=0, y=0, w=10, h=10)]
    preds = [BBox(x=0, y=0, w=10, h=10), BBox(x=0, y=0, w=10, h=10)]
    tp, fp, fn, matched = match_predictions_detailed(preds, gt)
    assert (tp, fp, fn) == (1, 1, 0)  # 1 TP + 1 duplicate FP, not 2 TP
    assert matched == (True,)  # the single GT is matched exactly once
    assert sum(matched) == tp


def test_detailed_matcher_empty_predictions_matches_nothing() -> None:
    gt = [BBox(x=0, y=0, w=10, h=10), BBox(x=50, y=50, w=10, h=10)]
    tp, fp, fn, matched = match_predictions_detailed([], gt)
    assert (tp, fp, fn) == (0, 0, 2)
    assert matched == (False, False)
    assert sum(matched) == tp


# --------------------------------------------------------------------------- COCO AP sweep


def test_ap50_equals_single_iou_average_precision_exactly() -> None:
    # The same hand-computed 5/6 case as test_eval_metrics: AP50 MUST reconcile with the existing
    # single-IoU-0.5 AP, because average_precision_coco calls average_precision(..., 0.5).
    gt = [BBox(x=0, y=0, w=10, h=10), BBox(x=50, y=50, w=10, h=10)]
    candidates = [
        (BBox(x=0, y=0, w=10, h=10), 0.9),
        (BBox(x=200, y=200, w=10, h=10), 0.8),
        (BBox(x=50, y=50, w=10, h=10), 0.7),
    ]
    ap, ap50, ap75 = average_precision_coco(candidates, gt)
    assert ap50 == pytest.approx(average_precision(candidates, gt, 0.5))
    assert ap50 == pytest.approx(5.0 / 6.0)
    # Perfect boxes are TPs at every IoU up to 1.0, so the sweep mean is also 5/6 here.
    assert ap == pytest.approx(5.0 / 6.0)
    assert ap75 == pytest.approx(average_precision(candidates, gt, 0.75))


def test_ap75_differs_from_ap50_on_a_loose_box() -> None:
    # A single prediction with IoU 0.64: a TP at IoU {0.50, 0.55, 0.60} and a FP at {0.65..0.95}.
    #   pred 8x8 over a 10x10 gt: intersection 64, union 100 -> IoU 0.64.
    # AP50 = 1.0 (0.64 >= 0.5); AP75 = 0.0 (0.64 < 0.75); sweep = 3 hits / 10 thresholds = 0.3.
    gt = [BBox(x=0, y=0, w=10, h=10)]
    candidates = [(BBox(x=0, y=0, w=8, h=8), 0.9)]
    ap, ap50, ap75 = average_precision_coco(candidates, gt)
    assert ap50 == pytest.approx(1.0)
    assert ap75 == pytest.approx(0.0)
    assert ap == pytest.approx(0.3)


def test_average_precision_coco_raises_without_ground_truth() -> None:
    with pytest.raises(ValueError, match="zero ground-truth"):
        average_precision_coco([(BBox(x=0, y=0, w=10, h=10), 0.9)], [])


# --------------------------------------------------------------------------- counting errors


def test_counting_errors_hand_computed() -> None:
    # pred=[2,4,4], true=[3,4,5] -> deltas [-1,0,-1].
    #   MAE  = mean(|Δ|)        = (1+0+1)/3 = 2/3
    #   RMSE = sqrt(mean(Δ^2))  = sqrt((1+0+1)/3) = sqrt(2/3)
    #   NAE  = mean(|Δ|/true)   = (1/3 + 0/4 + 1/5)/3
    mae, rmse, nae = counting_errors([2, 4, 4], [3, 4, 5])
    assert mae == pytest.approx(2.0 / 3.0)
    assert rmse == pytest.approx(math.sqrt(2.0 / 3.0))
    assert nae == pytest.approx((1.0 / 3.0 + 0.0 + 1.0 / 5.0) / 3.0)


def test_counting_errors_nae_skips_zero_true() -> None:
    # The middle image has true==0: it contributes to MAE/RMSE but is skipped from the NAE mean
    # (no normalised error is defined for it), never a divide-by-zero.
    mae, rmse, nae = counting_errors([1, 2, 0], [0, 2, 0])
    # deltas [1, 0, 0]: MAE = 1/3, RMSE = sqrt(1/3).
    assert mae == pytest.approx(1.0 / 3.0)
    assert rmse == pytest.approx(math.sqrt(1.0 / 3.0))
    # Only the true==2 image is normalisable: |0|/2 = 0.0 -> NAE mean over one image = 0.0.
    assert nae == pytest.approx(0.0)


def test_counting_errors_all_zero_true_gives_zero_nae() -> None:
    mae, _rmse, nae = counting_errors([1, 2], [0, 0])
    assert mae == pytest.approx(1.5)
    assert nae == pytest.approx(0.0)  # no normalisable image -> 0.0, not an error


def test_counting_errors_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="same length"):
        counting_errors([1, 2], [1])


def test_counting_errors_rejects_empty() -> None:
    with pytest.raises(ValueError, match="empty set"):
        counting_errors([], [])
