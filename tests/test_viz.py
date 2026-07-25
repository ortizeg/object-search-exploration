"""Tests for the visualization offering.

These assert output **shape, dtype, and non-blankness** -- never a pixel comparison against a
golden figure. cv2 and matplotlib rendering shift subtly across versions, so a golden-image
test would be brittle noise; "the render is the right size and not uniformly empty" is the
honest, stable invariant (CONTEXT risk summary).
"""

import base64

import cv2
import matplotlib
import numpy as np

from object_search.schemas import BBox, ExemplarBox, Match, Point
from object_search.search.common import viz
from object_search.search.common.viz import (
    compose_panel,
    draw_correspondences,
    draw_keypoints,
    draw_matches,
    heatmap_png_b64,
)


def _scene() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 60, size=(80, 100, 3), dtype=np.uint8)


def test_matplotlib_backend_is_headless():
    # The module forces Agg at import; a display-backed backend in CI would crash on render.
    assert matplotlib.get_backend().lower() == "agg"


def test_draw_matches_returns_uint8_same_size_and_non_blank():
    scene = _scene()
    matches = [Match(box=BBox(x=10, y=10, w=20, h=20), score=0.8)]
    out = draw_matches(scene, matches)
    assert out.dtype == np.uint8
    assert out.shape == scene.shape
    # A box was drawn, so the output cannot be identical to the input.
    assert not np.array_equal(out, scene)


def test_draw_matches_renders_exemplar_distinctly_from_ordinary_matches():
    scene = _scene()
    ordinary = draw_matches(scene, [Match(box=BBox(x=10, y=10, w=20, h=20), score=0.8)])
    exemplar_box = ExemplarBox(box=BBox(x=40, y=40, w=20, h=20))
    with_exemplar = draw_matches(scene, [], exemplar=exemplar_box)

    # The exemplar is drawn in its distinct magenta; ordinary matches are green. So the
    # exemplar render contains magenta pixels the ordinary render does not.
    magenta = np.all(with_exemplar == np.array(viz._EXEMPLAR_COLOR, dtype=np.uint8), axis=-1)
    green = np.all(ordinary == np.array(viz._MATCH_COLOR, dtype=np.uint8), axis=-1)
    assert magenta.any(), "exemplar box should be drawn in the distinct exemplar colour"
    assert green.any(), "ordinary match should be drawn in the match colour"
    # And the exemplar render does NOT use the ordinary-match colour for its box.
    assert not np.all(with_exemplar == np.array(viz._MATCH_COLOR, dtype=np.uint8), axis=-1).any()


def test_grayscale_input_is_accepted():
    gray = np.random.default_rng(1).integers(0, 255, size=(50, 50), dtype=np.uint8)
    out = draw_matches(gray, [Match(box=BBox(x=5, y=5, w=10, h=10), score=0.5)])
    assert out.ndim == 3 and out.shape[2] == 3
    assert out.dtype == np.uint8


def test_heatmap_payload_carries_true_vmin_vmax_and_decodes_non_blank():
    # A response spanning a known, non-0..1 range: the payload must report THAT range.
    response = np.linspace(0.2, 0.9, 60 * 80, dtype=np.float32).reshape(60, 80)
    payload = heatmap_png_b64(response)

    assert payload.width == 80
    assert payload.height == 60
    assert abs(payload.vmin - 0.2) < 1e-6
    assert abs(payload.vmax - 0.9) < 1e-6
    assert len(payload.png_b64) > 0

    raw = base64.b64decode(payload.png_b64)
    decoded = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    assert decoded is not None
    assert decoded.shape[:2] == (60, 80)
    assert decoded.var() > 0.0  # a gradient colour-maps to a non-uniform image


def test_heatmap_explicit_vmin_vmax_is_preserved():
    response = np.full((10, 10), 0.5, dtype=np.float32)
    payload = heatmap_png_b64(response, vmin=0.0, vmax=1.0)
    assert payload.vmin == 0.0
    assert payload.vmax == 1.0


def test_heatmap_constant_map_does_not_divide_by_zero():
    response = np.full((8, 8), 0.3, dtype=np.float32)
    payload = heatmap_png_b64(response)  # vmin == vmax; must not raise
    assert payload.vmin == payload.vmax
    assert len(payload.png_b64) > 0


def test_draw_keypoints_non_blank():
    scene = _scene()
    points = [Point(x=20.0, y=30.0), Point(x=60.0, y=50.0)]
    out = draw_keypoints(scene, points)
    assert out.dtype == np.uint8
    assert not np.array_equal(out, scene)


def test_draw_correspondences_side_by_side_shape_and_non_blank():
    crop = np.random.default_rng(2).integers(0, 40, size=(30, 30, 3), dtype=np.uint8)
    scene = _scene()
    corrs = [(Point(x=10.0, y=10.0), Point(x=50.0, y=40.0))]
    out = draw_correspondences(crop, scene, corrs)
    assert out.dtype == np.uint8
    assert out.shape[1] == crop.shape[1] + scene.shape[1]
    assert out.var() > 0.0


def test_compose_panel_non_blank_and_uint8():
    tiles = [("scene", _scene()), ("crop", np.zeros((30, 40, 3), dtype=np.uint8))]
    out = compose_panel(tiles)
    assert out.dtype == np.uint8
    assert out.ndim == 3 and out.shape[2] == 3
    assert out.var() > 0.0


def test_compose_panel_empty_raises():
    import pytest

    with pytest.raises(ValueError, match="at least one tile"):
        compose_panel([])
