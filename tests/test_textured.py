"""Tests for the textured benchmark regimes (EVAL-20).

The load-bearing test is that every emblem clears the 20-SIFT-keypoint floor -- without it
``sparse-geo`` abstains and the whole set proves nothing. The rest enforce the same exact-ground-
truth invariants as the chipset: strict non-overlap, achieved-not-requested counts, determinism,
and that the varied regime actually varies.
"""

from __future__ import annotations

import numpy as np

from object_search.schemas.geometry import BBox
from object_search.synthetic.textured import (
    _KEYPOINT_FLOOR,
    TEXTURED_SPECS,
    generate_textured_image,
    render_emblem,
    sift_keypoint_count,
)


def _iou(a: BBox, b: BBox) -> float:
    ix, iy = max(a.x, b.x), max(a.y, b.y)
    ix2, iy2 = min(a.x + a.w, b.x + b.w), min(a.y + a.h, b.y + b.h)
    iw, ih = max(0, ix2 - ix), max(0, iy2 - iy)
    inter = iw * ih
    return inter / (a.w * a.h + b.w * b.h - inter) if inter else 0.0


def test_set_has_three_regimes_and_expected_size() -> None:
    assert len(TEXTURED_SPECS) == 48
    regimes = sorted({s.regime for s in TEXTURED_SPECS})
    assert regimes == ["cluttered", "plain", "varied"]


def test_every_emblem_clears_the_sift_keypoint_floor() -> None:
    """The reason the set exists: sparse-geo must find keypoints, not abstain."""
    counts = [sift_keypoint_count(render_emblem(s.emblem)) for s in TEXTURED_SPECS]
    assert min(counts) >= _KEYPOINT_FLOOR, f"an emblem yielded {min(counts)} < {_KEYPOINT_FLOOR}"


def test_ground_truth_boxes_are_pairwise_non_overlapping() -> None:
    # One image per regime is enough to exercise every placement path (plain/varied/cluttered).
    for regime in ("plain", "varied", "cluttered"):
        spec = next(s for s in TEXTURED_SPECS if s.regime == regime)
        boxes = generate_textured_image(spec).boxes
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                assert _iou(boxes[i], boxes[j]) == 0.0, f"{regime}: boxes {i},{j} overlap"


def test_recorded_count_is_the_achieved_count() -> None:
    for spec in (s for s in TEXTURED_SPECS if s.regime == "varied"):
        result = generate_textured_image(spec)
        assert len(result.boxes) == result.slice_metadata.true_instance_count
        assert len(result.boxes) <= spec.n_instances  # never MORE than requested


def test_same_seed_is_byte_identical_and_different_seed_differs() -> None:
    spec = TEXTURED_SPECS[0]
    a = generate_textured_image(spec).image
    b = generate_textured_image(spec).image
    assert np.array_equal(a, b)
    other = generate_textured_image(TEXTURED_SPECS[1]).image
    # Different spec (different emblem + layout) -> different pixels (shapes may differ, so guard).
    assert a.shape != other.shape or not np.array_equal(a, other)


def test_varied_regime_actually_varies_scale_and_rotation() -> None:
    spec = next(s for s in TEXTURED_SPECS if s.regime == "varied")
    meta = generate_textured_image(spec).slice_metadata
    assert meta.instance_scale_min is not None and meta.instance_scale_max is not None
    assert meta.instance_scale_max > meta.instance_scale_min  # scale varied across instances
    assert meta.rotation_min_deg is not None and meta.rotation_max_deg is not None
    assert meta.rotation_max_deg - meta.rotation_min_deg > 0.0  # rotation varied


def test_plain_regime_is_fixed_scale_and_upright() -> None:
    spec = next(s for s in TEXTURED_SPECS if s.regime == "plain")
    assert spec.scale_min == spec.scale_max == 1.0
    assert spec.rotation_deg == 0.0
    assert spec.n_distractors == 0


def test_cluttered_regime_has_distractors_excluded_from_ground_truth() -> None:
    spec = next(s for s in TEXTURED_SPECS if s.regime == "cluttered")
    assert spec.n_distractors > 0
    # The recorded boxes are the real instances only; distractors are drawn but never counted, so
    # the count matches the achieved instance count (not instances + distractors).
    result = generate_textured_image(spec)
    assert len(result.boxes) == result.slice_metadata.true_instance_count
    assert result.slice_metadata.clutter_level == 1.0
