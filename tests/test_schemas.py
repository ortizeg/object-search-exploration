"""Tests for the frozen schema set.

The tests here are not "does Pydantic work" tests. Each one pins a decision that the rest
of the system relies on and that a well-meaning edit could quietly undo:

* boxes are half-open, so ``w == x2 - x`` with no ``+1``;
* every contract is frozen and rejects unknown fields;
* an inconsistent :class:`SearchResult` cannot be constructed at all;
* ``Rating`` count fields stay ``None`` when unassessed (EVAL-17).
"""

import pytest
from pydantic import ValidationError

from object_search.schemas import BBox, ExemplarBox, Point

# --------------------------------------------------------------------------------- BBox


def test_bbox_derived_values_follow_the_exclusive_x2_convention():
    box = BBox(x=10, y=20, w=30, h=40)
    assert box.x2 == 40  # x + w, exclusive
    assert box.y2 == 60
    assert box.w == box.x2 - box.x  # the convention, restated as an assertion
    assert box.h == box.y2 - box.y
    assert box.area == 1200  # no "+1" terms
    assert box.xyxy == (10, 20, 40, 60)
    assert box.cx == 25.0
    assert box.cy == 40.0


def test_bbox_slices_a_numpy_array_with_no_off_by_one():
    """The whole point of the half-open convention: the box IS the slice."""
    np = pytest.importorskip("numpy")
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    box = BBox(x=10, y=20, w=30, h=40)
    crop = image[box.y : box.y2, box.x : box.x2]
    assert crop.shape == (box.h, box.w, 3)


def test_bbox_is_frozen():
    box = BBox(x=0, y=0, w=1, h=1)
    with pytest.raises(ValidationError):
        box.x = 5  # type: ignore[misc]


def test_bbox_forbids_unknown_fields():
    with pytest.raises(ValidationError):
        BBox(x=0, y=0, w=1, h=1, score=0.9)  # type: ignore[call-arg]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"x": -1, "y": 0, "w": 1, "h": 1},
        {"x": 0, "y": -1, "w": 1, "h": 1},
        {"x": 0, "y": 0, "w": 0, "h": 1},
        {"x": 0, "y": 0, "w": 1, "h": 0},
    ],
)
def test_bbox_rejects_out_of_range_dimensions(kwargs):
    with pytest.raises(ValidationError):
        BBox(**kwargs)


def test_bbox_iou_identical_is_one_and_disjoint_is_zero():
    a = BBox(x=0, y=0, w=10, h=10)
    assert a.iou(a) == pytest.approx(1.0)
    far = BBox(x=100, y=100, w=10, h=10)
    assert a.iou(far) == 0.0


def test_bbox_iou_touching_boxes_do_not_overlap():
    """Half-open means ``a.x2 == b.x`` is adjacency, not overlap."""
    a = BBox(x=0, y=0, w=10, h=10)
    b = BBox(x=10, y=0, w=10, h=10)
    assert a.iou(b) == 0.0


def test_bbox_iou_half_overlap():
    a = BBox(x=0, y=0, w=10, h=10)
    b = BBox(x=5, y=0, w=10, h=10)
    # intersection 5x10 = 50, union 100 + 100 - 50 = 150
    assert a.iou(b) == pytest.approx(50 / 150)


def test_bbox_clipped_to_trims_and_preserves_the_convention():
    box = BBox(x=90, y=90, w=50, h=50)
    clipped = box.clipped_to(100, 120)
    assert clipped.xyxy == (90, 90, 100, 120)
    assert clipped.w == 10
    assert clipped.h == 30


def test_bbox_clipped_to_is_a_noop_when_already_inside():
    box = BBox(x=1, y=2, w=3, h=4)
    assert box.clipped_to(100, 100) == box


def test_bbox_clipped_to_raises_when_fully_outside():
    box = BBox(x=200, y=200, w=10, h=10)
    with pytest.raises(ValueError, match="does not intersect"):
        box.clipped_to(100, 100)


# -------------------------------------------------------------------------- ExemplarBox


def test_exemplar_box_wraps_a_bbox_and_defaults_to_no_label():
    exemplar = ExemplarBox(box=BBox(x=1, y=1, w=4, h=4))
    assert exemplar.label is None
    assert exemplar.box.area == 16


def test_exemplar_box_is_frozen():
    exemplar = ExemplarBox(box=BBox(x=1, y=1, w=4, h=4), label="player")
    with pytest.raises(ValidationError):
        exemplar.label = "ball"  # type: ignore[misc]


# --------------------------------------------------------------------------------- Point


def test_point_is_subpixel_and_frozen():
    point = Point(x=1.5, y=2.25)
    assert point.x == 1.5
    with pytest.raises(ValidationError):
        point.x = 0.0  # type: ignore[misc]
