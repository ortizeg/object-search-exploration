"""The seeded exemplar sampler: 1- and 3-exemplar draws, native-honouring, reproducible (EVAL-23).

Every method is scored at BOTH 1 exemplar (the product's real one-box operating point) and 3
exemplars (the published-benchmark convention). This module proves the four properties the sweep
depends on:

* **native honouring** -- a dataset that ships native exemplar boxes (FSCD-* provide three) has
  those boxes returned as-is at ``count=3``;
* **the prefix property** -- the 1-exemplar set is the *first* of the 3-exemplar set, so 1 and 3
  are two questions over one selection, never two independent draws;
* **seed determinism** -- the same ``(gt, count, seed)`` yields byte-identical boxes, and the seed
  moves only the *sampled* (non-native) tail, never the native prefix (D-11);
* **the out-of-range guard** -- ``count`` larger than the number of GT boxes returns all boxes
  rather than indexing off the end.
"""

from __future__ import annotations

import pytest

from object_search.eval.labels import GroundTruth
from object_search.eval.sampling import sample_exemplars
from object_search.schemas.geometry import BBox


def _gt(n_boxes: int, *, exemplar_indices: tuple[int, ...] = ()) -> GroundTruth:
    """A GroundTruth with ``n_boxes`` distinct boxes and an optional native exemplar list."""
    boxes = tuple(BBox(x=10 * i, y=0, w=8, h=8) for i in range(n_boxes))
    return GroundTruth(
        image_id=f"img-{n_boxes}",
        boxes=boxes,
        exemplar_index=exemplar_indices[0] if exemplar_indices else 0,
        exemplar_indices=exemplar_indices,
        source="research",
    )


def test_native_exemplars_are_honoured_for_count_three() -> None:
    # FSCD-* ship three native exemplar boxes; count=3 returns exactly those, order preserved.
    gt = _gt(6, exemplar_indices=(2, 4, 5))
    sampled = sample_exemplars(gt, count=3, seed=0)
    assert tuple(e.box for e in sampled) == (gt.boxes[2], gt.boxes[4], gt.boxes[5])


def test_count_one_is_the_first_native_exemplar() -> None:
    gt = _gt(6, exemplar_indices=(2, 4, 5))
    sampled = sample_exemplars(gt, count=1, seed=0)
    assert len(sampled) == 1
    assert sampled[0].box == gt.boxes[2]


def test_prefix_property_native() -> None:
    # The 1-exemplar set is the first of the 3-exemplar set -- a different question, not a re-draw.
    gt = _gt(6, exemplar_indices=(2, 4, 5))
    one = sample_exemplars(gt, count=1, seed=7)
    three = sample_exemplars(gt, count=3, seed=7)
    assert one == three[:1]


def test_prefix_property_sampled() -> None:
    # With no native exemplars, the draw is seeded -- but the prefix property still holds.
    gt = _gt(8)
    one = sample_exemplars(gt, count=1, seed=7)
    three = sample_exemplars(gt, count=3, seed=7)
    assert one == three[:1]
    assert len({e.box.x for e in three}) == 3  # three DISTINCT boxes


def test_seed_determinism_same_seed_identical() -> None:
    gt = _gt(8)
    a = sample_exemplars(gt, count=3, seed=3)
    b = sample_exemplars(gt, count=3, seed=3)
    assert a == b


def test_different_seed_changes_sampled_selection() -> None:
    # A larger box set makes a coincidental collision astronomically unlikely.
    gt = _gt(30)
    a = sample_exemplars(gt, count=3, seed=1)
    b = sample_exemplars(gt, count=3, seed=2)
    assert a != b


def test_seed_never_moves_the_native_prefix() -> None:
    # The native prefix is seed-independent; only the sampled tail (index 3+) can move.
    gt = _gt(30, exemplar_indices=(0, 1, 2))
    a = sample_exemplars(gt, count=3, seed=1)
    b = sample_exemplars(gt, count=3, seed=2)
    assert a == b  # fully native -> seed changes nothing at count=3


def test_partial_native_prefix_stable_tail_sampled() -> None:
    # One native exemplar: the first box is fixed, the other two are seeded and seed-sensitive.
    gt = _gt(30, exemplar_indices=(5,))
    a = sample_exemplars(gt, count=3, seed=1)
    b = sample_exemplars(gt, count=3, seed=2)
    assert a[0].box == gt.boxes[5] == b[0].box  # native prefix fixed
    assert a != b  # sampled tail moved


def test_count_exceeding_boxes_returns_all_boxes() -> None:
    # Out-of-range is handled explicitly (return all boxes), never an index error.
    gt = _gt(2)
    sampled = sample_exemplars(gt, count=3, seed=0)
    assert len(sampled) == 2
    assert {e.box for e in sampled} == set(gt.boxes)


def test_count_below_one_is_rejected() -> None:
    # A zero-exemplar query is meaningless: every method needs at least one positive example.
    gt = _gt(4)
    with pytest.raises(ValueError, match="at least one exemplar"):
        sample_exemplars(gt, count=0, seed=0)
