"""Tests for the deterministic greedy IoU NMS offering.

The load-bearing test here is not "does suppression work" but "does it produce byte-identical
output when scores tie" -- that is the reproducibility property (PITFALLS.md 6.3) the shared
module exists to guarantee.
"""

import numpy as np

from object_search.schemas import BBox
from object_search.search.common.nms import nms


def test_nms_returns_indices_not_boxes():
    # Two heavily overlapping boxes, one clearly stronger -> the weak one is suppressed and we
    # get back the INDEX of the survivor, so a caller can look up its own payload.
    boxes = [BBox(x=0, y=0, w=10, h=10), BBox(x=1, y=1, w=10, h=10)]
    scores = [0.9, 0.5]
    kept = nms(boxes, scores, iou_threshold=0.3)
    assert kept == [0]
    assert all(isinstance(i, int) for i in kept)


def test_nms_clear_cut_suppression():
    # Three boxes: 0 and 1 overlap strongly (1 loses to 0); 2 is far away and survives.
    boxes = [
        BBox(x=0, y=0, w=10, h=10),
        BBox(x=2, y=2, w=10, h=10),
        BBox(x=100, y=100, w=10, h=10),
    ]
    scores = [0.9, 0.8, 0.7]
    kept = nms(boxes, scores, iou_threshold=0.3)
    assert kept == [0, 2]


def test_nms_empty_input_returns_empty_list():
    assert nms([], [], iou_threshold=0.3) == []


def test_nms_tie_break_is_deterministic_across_shuffles():
    # THE load-bearing test. Four boxes all scoring exactly 1.0 -- the synthetic-lattice case.
    # Two of them (A, B) overlap enough to suppress each other; the tie must break on geometry
    # so the SAME one always survives no matter what order the caller built the list in.
    a = BBox(x=0, y=0, w=10, h=10)
    b = BBox(x=3, y=0, w=10, h=10)  # overlaps a; same y, larger x -> a wins the (-score, y, x) tie
    c = BBox(x=100, y=0, w=10, h=10)
    d = BBox(x=200, y=0, w=10, h=10)
    tied_score = 1.0

    reference = nms([a, b, c, d], [tied_score] * 4, iou_threshold=0.3)

    rng = np.random.default_rng(1234)
    for _ in range(50):
        order = [int(i) for i in rng.permutation(4)]
        boxes = [[a, b, c, d][i] for i in order]
        result_in_shuffled = nms(boxes, [tied_score] * 4, iou_threshold=0.3)
        # Map the shuffled-index result back to the original boxes and compare as a set of
        # identities: the SAME boxes survive regardless of input order.
        survivors = {boxes[i] for i in result_in_shuffled}
        assert survivors == {[a, b, c, d][i] for i in reference}


def test_nms_output_is_byte_identical_on_repeated_runs():
    boxes = [BBox(x=i * 3, y=0, w=10, h=10) for i in range(6)]
    scores = [1.0] * 6
    first = nms(boxes, scores, iou_threshold=0.3)
    second = nms(boxes, scores, iou_threshold=0.3)
    assert first == second  # exact list equality, not just same-set


def test_nms_boundary_is_strict_greater_than():
    # Construct two boxes whose IoU is EXACTLY the threshold, and assert the documented
    # boundary: at exactly iou_threshold the box is KEPT (suppression is strict `>`).
    # Two 10x10 boxes offset so intersection = 50, union = 150 -> IoU = 1/3.
    a = BBox(x=0, y=0, w=10, h=10)
    b = BBox(x=5, y=0, w=10, h=10)  # intersection 5x10=50, union 100+100-50=150 -> IoU=0.3333
    iou = a.iou(b)
    # At a threshold equal to that exact IoU, b must survive (kept), proving `>` not `>=`.
    kept = nms([a, b], [1.0, 1.0], iou_threshold=iou)
    assert set(kept) == {0, 1}
    # And just below it, b is suppressed.
    kept_below = nms([a, b], [1.0, 1.0], iou_threshold=iou - 1e-6)
    assert kept_below == [0]
