"""Detection metrics on ground-truth-scored predictions: precision, recall, F1, AP.

These mirror the store's derived-metric semantics (:mod:`object_search.store.schema` views)
but score against *ground truth* rather than human ratings. Where the two overlap the
definitions are identical on purpose -- a precision computed here and a precision computed in
``run_metrics`` must mean the same thing, or the objective chipset numbers and the human-rated
scoreboard could not be read side by side.

Two conventions are load-bearing and easy to regress:

* **Abstention is not zero.** ``precision`` is ``None`` when nothing was returned
  (``tp + fp == 0``), never ``0.0``; ``recall`` is ``None`` when there is nothing to find
  (``tp + fn == 0``). A method that honestly returns no boxes has *undefined* precision, and
  scoring it ``0`` would punish an abstention as though it were a wrong answer -- the same
  NULL-versus-zero distinction the store enforces one layer up (EVAL-17).
* **The EVAL-16 duplicate rule.** Each ground-truth instance is matched **at most once**. Two
  predicted boxes on one true instance count as **1 TP + 1 FP**, not 2 TP: the second box is a
  duplicate detection, which is a false positive, not a bonus.

**AP is all-point interpolation** (the modern COCO-style convention, as opposed to the older
11-point PASCAL VOC average), and it is computed from the **sub-threshold candidate log**
(EVAL-08): the caller assembles every scored observation it kept -- the above-threshold matches
*and* the below-threshold candidates -- into one ranked list, so a full precision/recall curve,
and hence AP, is recoverable from a single operating point's worth of data.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from object_search.schemas.geometry import BBox

# The COCO IoU sweep: 0.50, 0.55, ..., 0.95 (ten thresholds). Written as scaled integers and
# divided so the thresholds are exact (0.05 has no finite binary representation, and a naive
# ``arange`` drifts by ~1e-16, which would make AP50/AP75 miss the sweep points they must land on).
_COCO_IOU_THRESHOLDS: tuple[float, ...] = tuple(t / 100 for t in range(50, 100, 5))


def match_predictions(
    pred_boxes: Sequence[BBox],
    gt_boxes: Sequence[BBox],
    iou_threshold: float = 0.5,
) -> tuple[int, int, int]:
    """Greedily match predicted boxes to ground truth and count TP / FP / FN (EVAL-16).

    Predictions are consumed in the given order; each claims the highest-IoU **still-unmatched**
    ground-truth box whose IoU meets ``iou_threshold``. A prediction that finds no such box is a
    false positive, which is what makes a second box on an already-matched instance a duplicate
    (1 TP + 1 FP) rather than a second true positive. Unmatched ground truth at the end is the
    false-negative (missed) count.

    Args:
        pred_boxes: The boxes a method claimed. Order is respected but does not change the
            counts for the non-overlapping benchmark sets this primarily scores.
        gt_boxes: The exact ground-truth instances.
        iou_threshold: Minimum IoU for a match. ``0.5`` is the default detection convention.

    Returns:
        ``(tp, fp, fn)``. ``tp + fp == len(pred_boxes)`` and ``tp + fn == len(gt_boxes)`` always.
    """
    matched: list[bool] = [False] * len(gt_boxes)
    tp = 0
    for pred in pred_boxes:
        best_gt = _best_unmatched_gt(pred, gt_boxes, matched, iou_threshold)
        if best_gt >= 0:
            matched[best_gt] = True
            tp += 1
    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - tp
    return tp, fp, fn


def _best_unmatched_gt(
    pred: BBox,
    gt_boxes: Sequence[BBox],
    matched: Sequence[bool],
    iou_threshold: float,
) -> int:
    """Index of the highest-IoU still-unmatched GT at or above ``iou_threshold``, else ``-1``.

    A strict ``>`` on the running best means an exact IoU tie keeps the first (lowest-index)
    ground-truth box, so matching is deterministic regardless of GT ordering.
    """
    best_iou = iou_threshold
    best_gt = -1
    for gt_index, gt in enumerate(gt_boxes):
        if matched[gt_index]:
            continue
        iou = pred.iou(gt)
        if iou >= iou_threshold and (best_gt == -1 or iou > best_iou):
            best_iou = iou
            best_gt = gt_index
    return best_gt


def precision_recall_f1(
    tp: int, fp: int, fn: int
) -> tuple[float | None, float | None, float | None]:
    """Precision, recall and F1 with the abstention convention (None, never 0).

    Args:
        tp: True positives.
        fp: False positives.
        fn: False negatives (missed instances).

    Returns:
        ``(precision, recall, f1)``. ``precision`` is ``None`` when ``tp + fp == 0`` (nothing was
        returned, so precision is undefined -- an abstention, not a zero). ``recall`` is ``None``
        when ``tp + fn == 0`` (there was nothing to find). ``f1`` is ``None`` when either input
        rate is ``None`` or both are exactly ``0`` (an undefined harmonic mean).
    """
    precision = tp / (tp + fp) if (tp + fp) > 0 else None
    recall = tp / (tp + fn) if (tp + fn) > 0 else None
    if precision is None or recall is None or (precision + recall) == 0.0:
        f1 = None
    else:
        f1 = 2.0 * precision * recall / (precision + recall)
    return precision, recall, f1


def average_precision(
    candidates_with_scores: Sequence[tuple[BBox, float]],
    gt_boxes: Sequence[BBox],
    iou_threshold: float = 0.5,
) -> float:
    """All-point-interpolation average precision from a ranked candidate log (EVAL-08).

    The candidate log is every scored observation the method kept at one operating point -- the
    above-threshold matches *and* the sub-threshold candidates -- so ranking it by score sweeps
    the full precision/recall curve without re-running anything. AP is the area under the
    **all-point-interpolated** curve: precision is made monotonically non-increasing from the
    right (``p_interp(r) = max{p(r') : r' >= r}``) and integrated exactly over every recall step,
    which is the modern COCO convention rather than the coarser 11-point PASCAL VOC average.

    Args:
        candidates_with_scores: ``(box, score)`` pairs; higher score = more confident. Ties are
            broken by ``(y, x)`` so the ranking is reproducible (PITFALLS §6.3), never by score
            alone.
        gt_boxes: The exact ground-truth instances.
        iou_threshold: Minimum IoU for a candidate to count as a true positive.

    Returns:
        AP in ``[0.0, 1.0]``. ``0.0`` when the candidate log is empty (nothing was found, recall
        never rises).

    Raises:
        ValueError: If ``gt_boxes`` is empty. AP over zero ground truth is undefined (recall has
            no denominator); the caller must guard on missing ground truth rather than receive a
            fabricated ``0.0``.
    """
    n_gt = len(gt_boxes)
    if n_gt == 0:
        raise ValueError("average_precision is undefined with zero ground-truth boxes")
    if not candidates_with_scores:
        return 0.0

    # Rank by descending score, tie-broken by (y, x) -- score alone would leave tied candidates
    # in an order that varies between runs and quietly perturbs the curve (PITFALLS §6.3).
    ranked = sorted(
        candidates_with_scores,
        key=lambda item: (-item[1], item[0].y, item[0].x),
    )

    matched: list[bool] = [False] * n_gt
    cum_tp = 0
    cum_fp = 0
    precisions: list[float] = []
    recalls: list[float] = []
    for box, _score in ranked:
        best_gt = _best_unmatched_gt(box, gt_boxes, matched, iou_threshold)
        if best_gt >= 0:
            matched[best_gt] = True
            cum_tp += 1
        else:
            cum_fp += 1
        precisions.append(cum_tp / (cum_tp + cum_fp))
        recalls.append(cum_tp / n_gt)

    # All-point interpolation: envelope the precision curve (monotone non-increasing from the
    # right), then integrate over the recall axis. Prepend the (recall=0, precision=envelope[0])
    # segment so a curve that reaches full precision immediately is credited for it.
    interp: list[float] = precisions[:]
    for i in range(len(interp) - 2, -1, -1):
        interp[i] = max(interp[i], interp[i + 1])

    ap = 0.0
    prev_recall = 0.0
    for i, recall in enumerate(recalls):
        if recall > prev_recall:
            ap += (recall - prev_recall) * interp[i]
            prev_recall = recall
    return ap


def average_precision_coco(
    candidates_with_scores: Sequence[tuple[BBox, float]],
    gt_boxes: Sequence[BBox],
) -> tuple[float, float, float]:
    """COCO-style AP over the IoU sweep ``[0.50, 0.55, ..., 0.95]`` (EVAL-24).

    This is the literature's detection metric (FSCD-147 / Counting-DETR), reported alongside the
    project's existing single-IoU number so our results sit next to published work. It is a thin
    generalisation, **not** a reimplementation: it calls :func:`average_precision` once per IoU
    threshold, so by construction ``AP50`` is *exactly* the pre-existing single-IoU-0.5 AP and old
    and new reports reconcile with no drift (the AP-convention constraint from ``08-CONTEXT.md``).

    Args:
        candidates_with_scores: The ranked ``(box, score)`` candidate log, identical to what
            :func:`average_precision` consumes.
        gt_boxes: The exact ground-truth instances.

    Returns:
        ``(AP, AP50, AP75)`` where ``AP`` is the mean over the ten COCO IoU thresholds, ``AP50``
        is AP at IoU 0.5, and ``AP75`` is AP at IoU 0.75. All in ``[0.0, 1.0]``.

    Raises:
        ValueError: If ``gt_boxes`` is empty -- inherited from :func:`average_precision`, the same
            guard, so the two functions refuse zero-ground-truth identically.
    """
    per_threshold = [
        average_precision(candidates_with_scores, gt_boxes, iou) for iou in _COCO_IOU_THRESHOLDS
    ]
    ap = sum(per_threshold) / len(per_threshold)
    ap50 = average_precision(candidates_with_scores, gt_boxes, 0.5)
    ap75 = average_precision(candidates_with_scores, gt_boxes, 0.75)
    return ap, ap50, ap75


def counting_errors(
    pred_counts: Sequence[int], true_counts: Sequence[int]
) -> tuple[float, float, float]:
    """Counting-comparability errors MAE, RMSE and NAE over predicted-vs-true counts (EVAL-24).

    These are the class-agnostic-counting literature's headline numbers, added so a per-method
    count error can be read next to published counting work. They are aggregate errors over a set
    of images, not a per-image score.

    * **MAE** ``= mean(|pred - true|)`` -- the primary error.
    * **RMSE** ``= sqrt(mean((pred - true)^2))`` -- penalises large misses more.
    * **NAE** ``= mean(|pred - true| / true)`` -- normalised, but **only over images with
      ``true > 0``**. Dividing by a zero true count is undefined, so those images are skipped from
      the NAE average explicitly (documented) rather than silently contributing an infinity or a
      divide-by-zero; NAE is ``0.0`` when *every* image has ``true == 0``.

    Args:
        pred_counts: Predicted instance count per image.
        true_counts: True instance count per image, aligned by index with ``pred_counts``.

    Returns:
        ``(MAE, RMSE, NAE)``.

    Raises:
        ValueError: If the two sequences differ in length, or are empty. An empty set has no mean.
    """
    if len(pred_counts) != len(true_counts):
        raise ValueError(
            f"pred_counts ({len(pred_counts)}) and true_counts ({len(true_counts)}) "
            f"must be the same length"
        )
    if not pred_counts:
        raise ValueError("counting_errors is undefined over an empty set of images")

    deltas = [pred - true for pred, true in zip(pred_counts, true_counts, strict=True)]
    mae = sum(abs(d) for d in deltas) / len(deltas)
    rmse = math.sqrt(sum(d * d for d in deltas) / len(deltas))

    # NAE guards true == 0 explicitly: a zero denominator is undefined, so those images are left
    # out of the normalised average rather than fabricating an error term for them.
    normalised = [
        abs(pred - true) / true
        for pred, true in zip(pred_counts, true_counts, strict=True)
        if true > 0
    ]
    nae = sum(normalised) / len(normalised) if normalised else 0.0
    return mae, rmse, nae
