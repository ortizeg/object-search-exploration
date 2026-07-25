"""Tests for the chip-insertion benchmark set (EVAL-19)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from object_search.synthetic import chipset
from object_search.synthetic.chipset import (
    CHIP_CANVAS_SIZES,
    CHIPSET_SPECS,
    generate_chipset_image,
    render_chip,
    write_chipset,
)

# The two largest canvases (4096x3072, 6000x4000) are slow to generate; exclude them from the
# per-pixel tests but keep 3200x2400 (index 7) covered.
_PIXEL_TEST_SPECS = CHIPSET_SPECS[:8]


def test_exactly_ten_specs_with_increasing_area_largest_6000x4000() -> None:
    assert len(CHIPSET_SPECS) == 10
    areas = [w * h for (w, h) in CHIP_CANVAS_SIZES]
    assert areas == sorted(areas)
    assert len(set(CHIP_CANVAS_SIZES)) == 10
    assert CHIP_CANVAS_SIZES[-1] == (6000, 4000)


def test_every_n_instances_is_5_10_or_15() -> None:
    for spec in CHIPSET_SPECS:
        assert spec.n_instances in (5, 10, 15)


def test_pairwise_iou_is_zero_across_all_images() -> None:
    for spec in _PIXEL_TEST_SPECS:
        boxes = generate_chipset_image(spec).boxes
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                assert boxes[i].iou(boxes[j]) == 0.0, spec.image_id
                # No containment either (equal-size boxes: identical origin would contain).
                assert boxes[i].xyxy != boxes[j].xyxy


def test_len_boxes_equals_recorded_achieved_count() -> None:
    for spec in _PIXEL_TEST_SPECS:
        result = generate_chipset_image(spec)
        assert len(result.boxes) == result.slice_metadata.true_instance_count
        assert len(result.boxes) <= spec.n_instances


def test_determinism_is_byte_identical() -> None:
    first = generate_chipset_image(CHIPSET_SPECS[0])
    second = generate_chipset_image(CHIPSET_SPECS[0])
    assert np.array_equal(first.image, second.image)
    assert first.boxes == second.boxes


def test_each_box_region_differs_from_white_background() -> None:
    spec = CHIPSET_SPECS[2]
    result = generate_chipset_image(spec)
    for box in result.boxes:
        region = result.image[box.y : box.y2, box.x : box.x2]
        assert not np.all(region == 255), "chip region should not be blank white"


def test_chip_has_non_trivial_variance() -> None:
    # A flat chip would trip Method 1's low-variance template guard.
    chip = render_chip(CHIPSET_SPECS[0].chip)
    assert float(np.var(chip)) > 100.0


def test_large_canvas_is_covered() -> None:
    # Index 7 is 3200x2400 -- prove the pipeline works at a large size without the two biggest.
    spec = CHIPSET_SPECS[7]
    assert (spec.width, spec.height) == (3200, 2400)
    result = generate_chipset_image(spec)
    assert len(result.boxes) == result.slice_metadata.true_instance_count


def test_write_chipset_emits_sidecars(tmp_path: Path) -> None:
    # Only generate the small canvases here to keep the test fast.
    import object_search.synthetic.chipset as chip_mod

    original = chip_mod.CHIPSET_SPECS
    chip_mod.CHIPSET_SPECS = original[:3]
    try:
        written = write_chipset(tmp_path)
    finally:
        chip_mod.CHIPSET_SPECS = original

    assert len(written) == 3
    sidecar = tmp_path / "chipset-01.gt.json"
    assert sidecar.is_file()
    text = sidecar.read_text(encoding="utf-8")
    assert '"achieved_n"' in text
    assert '"requested_n"' in text
    assert '"exemplar_index"' in text


def test_module_import_alias_is_stable() -> None:
    assert chipset.CHIPSET_SPECS[0].image_id == "chipset-01"
