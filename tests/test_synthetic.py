"""Tests for the synthetic generator: determinism and exact ground truth (EVAL-03)."""

from __future__ import annotations

import numpy as np

from object_search.schemas.geometry import BBox
from object_search.synthetic import DEMO_SPECS, SyntheticSpec, save, synthesize


def _box_contains_color(image: np.ndarray, box: BBox, color: tuple[int, int, int]) -> bool:
    region = image[box.y : box.y2, box.x : box.x2]
    return bool(np.any(np.all(region == np.array(color, dtype=np.uint8), axis=-1)))


def test_same_seed_is_byte_identical() -> None:
    spec = SyntheticSpec(seed=7, mode="scatter", n_instances=8, scale_jitter=0.3, clutter=0.4)
    first = synthesize(spec)
    second = synthesize(spec)
    assert np.array_equal(first.image, second.image)
    assert first.boxes == second.boxes


def test_different_seeds_differ() -> None:
    a = synthesize(SyntheticSpec(seed=1, mode="scatter", n_instances=8, clutter=0.3))
    b = synthesize(SyntheticSpec(seed=2, mode="scatter", n_instances=8, clutter=0.3))
    assert not np.array_equal(a.image, b.image)


def test_lattice_no_jitter_places_exactly_n() -> None:
    spec = SyntheticSpec(seed=0, mode="lattice", shape="plus", n_instances=12)
    out = synthesize(spec)
    assert len(out.boxes) == 12
    assert out.slice_metadata.true_instance_count == 12


def test_every_box_contains_foreground_pixels() -> None:
    spec = SyntheticSpec(seed=3, mode="lattice", shape="rect", n_instances=9)
    out = synthesize(spec)
    for box in out.boxes:
        assert _box_contains_color(out.image, box, spec.fg_color)


def test_rotated_aabb_exceeds_instance_size() -> None:
    """A rotated square's AABB is strictly larger than its side length -- ground truth must
    reflect the *drawn* extent, not the nominal size."""
    spec = SyntheticSpec(
        seed=5, mode="lattice", shape="rect", n_instances=9, rotation_jitter_deg=45.0
    )
    out = synthesize(spec)
    assert any(box.w > spec.instance_size or box.h > spec.instance_size for box in out.boxes)


def test_distractors_are_not_in_boxes() -> None:
    spec = SyntheticSpec(seed=6, mode="scatter", shape="plus", n_instances=6, n_distractors=6)
    out = synthesize(spec)
    # Only the real instances are ground truth; the six distractors are excluded.
    assert len(out.boxes) == out.slice_metadata.true_instance_count
    assert len(out.boxes) <= 6


def test_scatter_records_achieved_count_not_requested() -> None:
    # A tiny canvas cannot fit many instances; the achieved count must match len(boxes).
    spec = SyntheticSpec(seed=9, mode="scatter", width=200, height=200, n_instances=40)
    out = synthesize(spec)
    assert out.slice_metadata.true_instance_count == len(out.boxes)
    assert len(out.boxes) <= 40


def test_save_writes_png_and_sidecar(tmp_path) -> None:
    out = synthesize(DEMO_SPECS["lattice-plain"])
    image_path = save(out, tmp_path / "demo.png")
    assert image_path.is_file()
    sidecar = tmp_path / "demo.gt.json"
    assert sidecar.is_file()
    assert '"boxes"' in sidecar.read_text(encoding="utf-8")


def test_demo_specs_all_generate() -> None:
    for name, spec in DEMO_SPECS.items():
        out = synthesize(spec)
        assert out.image.shape == (spec.height, spec.width, 3), name
        assert len(out.boxes) >= 1, name
