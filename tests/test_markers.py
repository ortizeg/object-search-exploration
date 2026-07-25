"""Marker synthesis (Task 1) and orientation estimation (Task 2).

The synthetic markers here are the *exact* oracles the orientation estimator is tested
against: each arrow's tip and unit direction are known by construction, so a recovered
angle can be checked to within a few degrees rather than eyeballed.
"""

from __future__ import annotations

import math

import numpy as np

from object_search.synthetic.generator import (
    MARKER_DEMO_SPECS,
    MarkerSpec,
    synthesize_markers,
)

# ------------------------------------------------------------------- Task 1: synthesis


def test_same_seed_is_byte_identical() -> None:
    spec = MarkerSpec(seed=7, marker="arrow", n_markers=4)
    a = synthesize_markers(spec)
    b = synthesize_markers(spec)
    assert np.array_equal(a.image, b.image)
    assert a.markers == b.markers


def test_different_seed_differs() -> None:
    a = synthesize_markers(MarkerSpec(seed=1, marker="arrow", n_markers=3))
    b = synthesize_markers(MarkerSpec(seed=2, marker="arrow", n_markers=3))
    assert not np.array_equal(a.image, b.image)


def test_drawn_tip_pixel_is_foreground_and_matches_gt() -> None:
    spec = MarkerSpec(seed=3, marker="arrow", n_markers=3)
    out = synthesize_markers(spec)
    for marker in out.markers:
        px = round(marker.tip.x)
        py = round(marker.tip.y)
        pixel = out.image[py, px]
        assert tuple(int(c) for c in pixel) != spec.bg_color  # tip pixel was drawn


def test_arrow_reports_unit_direction() -> None:
    out = synthesize_markers(MarkerSpec(seed=4, marker="arrow", n_markers=3))
    for marker in out.markers:
        assert marker.direction is not None
        dx, dy = marker.direction
        assert math.isclose(math.hypot(dx, dy), 1.0, abs_tol=1e-6)


def test_markers_do_not_overlap() -> None:
    spec = MarkerSpec(seed=5, marker="arrow", n_markers=4)
    out = synthesize_markers(spec)
    boxes = [m.box for m in out.markers]
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            assert boxes[i].iou(boxes[j]) == 0.0


def test_dot_has_no_direction() -> None:
    out = synthesize_markers(MarkerSpec(seed=6, marker="dot", n_markers=4))
    assert out.markers
    for marker in out.markers:
        assert marker.direction is None


def test_targets_recorded_when_requested() -> None:
    out = synthesize_markers(
        MarkerSpec(seed=8, marker="arrow", n_markers=3, with_targets=True, target_gap=30)
    )
    assert out.markers
    for marker in out.markers:
        assert marker.target is not None
        # The target sits ahead of the tip along the pointing direction.
        assert marker.direction is not None
        dx, dy = marker.direction
        to_target_x = marker.target.cx - marker.tip.x
        to_target_y = marker.target.cy - marker.tip.y
        assert dx * to_target_x + dy * to_target_y > 0.0


def test_save_marker_image_writes_sidecar(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from object_search.synthetic.generator import save_marker_image

    out = synthesize_markers(MarkerSpec(seed=9, marker="arrow", n_markers=2))
    path = save_marker_image(out, tmp_path / "markers.png")
    assert path.exists()
    assert (tmp_path / "markers.markers.json").exists()


def test_demo_specs_cover_arrow_dot() -> None:
    assert "arrows" in MARKER_DEMO_SPECS
    assert "dots" in MARKER_DEMO_SPECS
    dots = synthesize_markers(MARKER_DEMO_SPECS["dots"])
    assert all(m.direction is None for m in dots.markers)
