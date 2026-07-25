"""Marker synthesis (Task 1) and orientation estimation (Task 2).

The synthetic markers here are the *exact* oracles the orientation estimator is tested
against: each arrow's tip and unit direction are known by construction, so a recovered
angle can be checked to within a few degrees rather than eyeballed.
"""

from __future__ import annotations

import math

import numpy as np

from object_search.explorations.markers import (
    estimate_geometry,
    foreground_mask,
    theta_from_transform,
)
from object_search.schemas.geometry import BBox, Point
from object_search.synthetic.generator import (
    MARKER_DEMO_SPECS,
    MarkerSpec,
    synthesize_markers,
)


def _angle_deg(v: tuple[float, float]) -> float:
    return math.degrees(math.atan2(v[1], v[0]))


def _angle_gap_deg(a: float, b: float) -> float:
    """Smallest absolute difference between two angles in degrees, wrapped to [0, 180]."""
    return abs((a - b + 180.0) % 360.0 - 180.0)


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


# --------------------------------------------------------------- Task 2: orientation


def _crop(image: np.ndarray, box: BBox) -> np.ndarray:
    return image[box.y : box.y2, box.x : box.x2]


def test_pca_recovers_arrow_direction_within_10_degrees() -> None:
    out = synthesize_markers(MarkerSpec(seed=21, marker="arrow", n_markers=5, arrow_len=72))
    assert out.markers
    for marker in out.markers:
        assert marker.direction is not None
        geom = estimate_geometry(_crop(out.image, marker.box), marker.box)
        assert geom.direction is not None
        gap = _angle_gap_deg(_angle_deg(geom.direction), _angle_deg(marker.direction))
        assert gap <= 10.0, f"direction off by {gap:.1f} deg"


def test_pca_recovers_arrow_tip_within_a_few_pixels() -> None:
    out = synthesize_markers(MarkerSpec(seed=22, marker="arrow", n_markers=4, arrow_len=72))
    for marker in out.markers:
        geom = estimate_geometry(_crop(out.image, marker.box), marker.box)
        err = math.hypot(
            geom.reference_point.x - marker.tip.x, geom.reference_point.y - marker.tip.y
        )
        assert err <= 6.0, f"tip off by {err:.1f} px"


def test_dot_returns_no_direction() -> None:
    out = synthesize_markers(MarkerSpec(seed=23, marker="dot", n_markers=4))
    for marker in out.markers:
        geom = estimate_geometry(_crop(out.image, marker.box), marker.box)
        assert geom.direction is None
        # The reference point falls back to the (near-)centroid of the dot.
        err = math.hypot(
            geom.reference_point.x - marker.centroid.x,
            geom.reference_point.y - marker.centroid.y,
        )
        assert err <= 4.0


def test_pca_geometry_is_deterministic() -> None:
    out = synthesize_markers(MarkerSpec(seed=24, marker="arrow", n_markers=3))
    marker = out.markers[0]
    g1 = estimate_geometry(_crop(out.image, marker.box), marker.box)
    g2 = estimate_geometry(_crop(out.image, marker.box), marker.box)
    assert g1 == g2


def test_theta_from_transform_matches_known_rotation() -> None:
    theta = math.radians(37.0)
    a, c = math.cos(theta), math.sin(theta)
    transform = (a, -c, 5.0, c, a, -3.0)  # [a, b, tx, c, d, ty], a pure rotation + translation
    assert math.isclose(theta_from_transform(transform), theta, abs_tol=1e-9)


def test_transform_path_resolves_flip_toward_mapped_exemplar_tip() -> None:
    theta = math.radians(37.0)
    a, c = math.cos(theta), math.sin(theta)
    tx, ty = 100.0, 80.0
    transform = (a, -c, tx, c, a, ty)
    exemplar_tip = Point(x=10.0, y=0.0)
    # Instance box centred on the transform's translation (the exemplar origin maps to (tx, ty)).
    box = BBox(x=int(tx) - 20, y=int(ty) - 20, w=40, h=40)
    dummy_crop = np.zeros((box.h, box.w, 3), dtype=np.uint8)

    geom = estimate_geometry(dummy_crop, box, transform=transform, exemplar_tip=exemplar_tip)
    assert geom.direction is not None
    assert geom.confidence == 1.0

    mapped_tip = (
        a * exemplar_tip.x - c * exemplar_tip.y + tx,
        c * exemplar_tip.x + a * exemplar_tip.y + ty,
    )
    expected = (mapped_tip[0] - box.cx, mapped_tip[1] - box.cy)
    gap = _angle_gap_deg(_angle_deg(geom.direction), _angle_deg(expected))
    assert gap <= 1.0
    assert math.isclose(geom.reference_point.x, mapped_tip[0], abs_tol=1e-6)
    assert math.isclose(geom.reference_point.y, mapped_tip[1], abs_tol=1e-6)


def test_transform_path_without_exemplar_tip_uses_raw_angle() -> None:
    theta = math.radians(-52.0)
    a, c = math.cos(theta), math.sin(theta)
    transform = (a, -c, 0.0, c, a, 0.0)
    box = BBox(x=0, y=0, w=20, h=20)
    geom = estimate_geometry(np.zeros((20, 20, 3), dtype=np.uint8), box, transform=transform)
    assert geom.direction is not None
    gap = _angle_gap_deg(_angle_deg(geom.direction), math.degrees(theta))
    assert gap <= 1e-6


def test_foreground_mask_flags_the_marker() -> None:
    out = synthesize_markers(MarkerSpec(seed=25, marker="arrow", n_markers=1))
    marker = out.markers[0]
    mask = foreground_mask(_crop(out.image, marker.box))
    assert mask.any()
    # Foreground is a genuine minority of the (thin-arrow) box, not the whole crop.
    assert mask.mean() < 0.9
