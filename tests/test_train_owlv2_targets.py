"""Tests for the torch-free OWLv2 fine-tuning glue (``object_search.train.owlv2_targets``).

These run in CI with **no torch, no weights, and no GPU** -- which is the entire reason the target
conversion lives in ``src/`` and the training loop lives in ``scripts/``.

The load-bearing test is :func:`test_boxes_normalize_by_the_padded_square_side_not_the_width`. If
the normalization denominator ever silently becomes ``(W, H)`` per-axis instead of the padded-square
side ``max(H, W)``, training still runs and the loss still falls -- it just learns systematically
skewed boxes and is then evaluated against un-skewed ground truth. Exact floats are asserted so that
regression cannot pass.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from object_search.train.owlv2_targets import (
    FLOORPLAN_CLASSES,
    OWLV2_NUM_PATCHES,
    FinetuneConfig,
    coco_to_owlv2_targets,
    cosine_warmup_factor,
    deterministic_batches,
    warmup_steps_for,
)

_CATEGORIES: list[dict[str, Any]] = [
    # The Roboflow export's real ids and names, including the non-class supercategory row and the
    # deliberately "wrong" numbering (door is 2, window is 5) that name-matching has to survive.
    {"id": 0, "name": "floorplans"},
    {"id": 1, "name": "bathroom"},
    {"id": 2, "name": "door"},
    {"id": 3, "name": "perimeter"},
    {"id": 4, "name": "stairs"},
    {"id": 5, "name": "window"},
]


def _coco(
    images: list[dict[str, Any]],
    annotations: list[dict[str, Any]],
) -> dict[str, Any]:
    """A minimal COCO split dict in the Roboflow export's shape."""
    return {"images": images, "annotations": annotations, "categories": _CATEGORIES}


def _one_box_coco(
    width: int, height: int, bbox: list[float], category_id: int = 2
) -> dict[str, Any]:
    return _coco(
        [{"id": 7, "file_name": "plan.png", "width": width, "height": height}],
        [{"id": 1, "image_id": 7, "category_id": category_id, "bbox": bbox}],
    )


# ------------------------------------------------------------------ the normalization denominator


def test_boxes_normalize_by_the_padded_square_side_not_the_width() -> None:
    """A 200x100 image's box normalizes over max(H, W) == 200 -- the key link of this task.

    COCO box ``[0, 0, 100, 50]`` on a 200-wide, 100-high image gives ``cx = 0.25`` over the padded
    square. Normalizing by the width per-axis would give ``cx = 0.5`` and ``h = 0.5`` -- the exact
    silent bug this asserts against.
    """
    targets = coco_to_owlv2_targets(_one_box_coco(200, 100, [0.0, 0.0, 100.0, 50.0]))

    assert len(targets) == 1
    box = targets[0].boxes[0]
    assert box.tolist() == pytest.approx([0.25, 0.125, 0.5, 0.25])
    # ... and emphatically NOT the per-axis (W, H) normalization.
    assert box[0] != pytest.approx(0.5)


def test_landscape_normalizes_by_width_and_portrait_by_height() -> None:
    """The denominator follows the LONG side either way round (the padded square is square)."""
    # Landscape: W=400 > H=100, so the denominator is 400.
    landscape = coco_to_owlv2_targets(_one_box_coco(400, 100, [100.0, 0.0, 200.0, 100.0]))
    assert landscape[0].boxes[0].tolist() == pytest.approx([0.5, 0.125, 0.5, 0.25])

    # Portrait: H=400 > W=100, so the denominator is again 400.
    portrait = coco_to_owlv2_targets(_one_box_coco(100, 400, [0.0, 100.0, 100.0, 200.0]))
    assert portrait[0].boxes[0].tolist() == pytest.approx([0.125, 0.5, 0.25, 0.5])


# ------------------------------------------------------------------------------ the class mapping


def test_classes_map_by_name_regardless_of_coco_category_ids() -> None:
    """``door`` -> index 0 and ``window`` -> index 1 by NAME, though COCO numbers them 2 and 5."""
    coco = _coco(
        [{"id": 1, "file_name": "a.png", "width": 100, "height": 100}],
        [
            {"id": 1, "image_id": 1, "category_id": 2, "bbox": [0.0, 0.0, 10.0, 10.0]},  # door
            {"id": 2, "image_id": 1, "category_id": 5, "bbox": [20.0, 0.0, 10.0, 10.0]},  # window
            {"id": 3, "image_id": 1, "category_id": 4, "bbox": [40.0, 0.0, 10.0, 10.0]},  # stairs
        ],
    )
    targets = coco_to_owlv2_targets(coco)

    assert FLOORPLAN_CLASSES == ("door", "window", "bathroom", "perimeter", "stairs")
    assert isinstance(FLOORPLAN_CLASSES, tuple)  # order-stable; a set would not be
    assert targets[0].class_labels.tolist() == [0, 1, 4]
    assert targets[0].class_labels.dtype == np.int64


def test_annotations_of_an_unlisted_category_are_skipped() -> None:
    """The export's ``floorplans`` supercategory row is not an object class -- its boxes go."""
    coco = _coco(
        [{"id": 1, "file_name": "a.png", "width": 100, "height": 100}],
        [
            {"id": 1, "image_id": 1, "category_id": 0, "bbox": [0.0, 0.0, 90.0, 90.0]},  # supercat
            {"id": 2, "image_id": 1, "category_id": 2, "bbox": [0.0, 0.0, 10.0, 10.0]},  # door
        ],
    )
    targets = coco_to_owlv2_targets(coco)
    assert targets[0].class_labels.tolist() == [0]


# --------------------------------------------------------------------------- the two drop guards


def test_degenerate_box_is_dropped_not_raised_on() -> None:
    """A sub-pixel box is annotation noise: it is dropped, and the rest of the image survives."""
    coco = _coco(
        [{"id": 1, "file_name": "a.png", "width": 100, "height": 100}],
        [
            {"id": 1, "image_id": 1, "category_id": 2, "bbox": [0.0, 0.0, 0.4, 10.0]},  # w rounds 0
            {"id": 2, "image_id": 1, "category_id": 2, "bbox": [0.0, 0.0, 10.0, 0.2]},  # h rounds 0
            {"id": 3, "image_id": 1, "category_id": 2, "bbox": [20.0, 20.0, 10.0, 10.0]},  # good
        ],
    )
    targets = coco_to_owlv2_targets(coco)
    assert targets[0].boxes.shape == (1, 4)


def test_out_of_range_box_is_dropped_rather_than_clamped() -> None:
    """A box outside the image is DROPPED -- clamping would fabricate a plausible target."""
    coco = _coco(
        [{"id": 1, "file_name": "a.png", "width": 100, "height": 100}],
        [
            {"id": 1, "image_id": 1, "category_id": 2, "bbox": [-5.0, 0.0, 10.0, 10.0]},  # x < 0
            {"id": 2, "image_id": 1, "category_id": 2, "bbox": [95.0, 0.0, 10.0, 10.0]},  # x2 > W
            {"id": 3, "image_id": 1, "category_id": 2, "bbox": [0.0, 95.0, 10.0, 10.0]},  # y2 > H
            {"id": 4, "image_id": 1, "category_id": 2, "bbox": [20.0, 20.0, 10.0, 10.0]},  # good
        ],
    )
    targets = coco_to_owlv2_targets(coco)
    assert targets[0].boxes.shape == (1, 4)
    assert targets[0].boxes[0].tolist() == pytest.approx([0.25, 0.25, 0.1, 0.1])


def test_annotation_pointing_at_a_missing_image_is_skipped() -> None:
    """A dangling image_id is corrupt input, not a target -- it must not KeyError the split."""
    coco = _coco(
        [{"id": 1, "file_name": "a.png", "width": 100, "height": 100}],
        [
            {"id": 1, "image_id": 999, "category_id": 2, "bbox": [0.0, 0.0, 10.0, 10.0]},
            {"id": 2, "image_id": 1, "category_id": 2, "bbox": [20.0, 20.0, 10.0, 10.0]},
        ],
    )
    targets = coco_to_owlv2_targets(coco)
    assert len(targets) == 1
    assert targets[0].boxes.shape == (1, 4)


# ------------------------------------------------------------------- image-level shape guarantees


def test_image_with_no_surviving_annotation_is_dropped_from_the_list() -> None:
    """A zero-box target contributes nothing to a Hungarian match, so it is not emitted at all."""
    coco = _coco(
        [
            {"id": 1, "file_name": "empty.png", "width": 100, "height": 100},
            {"id": 2, "file_name": "has-a-door.png", "width": 100, "height": 100},
        ],
        [{"id": 1, "image_id": 2, "category_id": 2, "bbox": [10.0, 10.0, 10.0, 10.0]}],
    )
    targets = coco_to_owlv2_targets(coco)
    assert [target.file_name for target in targets] == ["has-a-door.png"]


def test_targets_are_sorted_by_file_name_so_batching_is_deterministic() -> None:
    """Ordering is a property of the data, not of the JSON's insertion order (D-11)."""
    coco = _coco(
        [
            {"id": 3, "file_name": "c.png", "width": 100, "height": 100},
            {"id": 1, "file_name": "a.png", "width": 100, "height": 100},
            {"id": 2, "file_name": "b.png", "width": 100, "height": 100},
        ],
        [
            {"id": i, "image_id": i, "category_id": 2, "bbox": [10.0, 10.0, 10.0, 10.0]}
            for i in (1, 2, 3)
        ],
    )
    targets = coco_to_owlv2_targets(coco)
    assert [target.file_name for target in targets] == ["a.png", "b.png", "c.png"]
    assert [target.image_id for target in targets] == [1, 2, 3]


def test_every_emitted_box_is_inside_the_unit_square_with_positive_sides() -> None:
    """The output invariant the training loop relies on (threat T-8zy-04)."""
    coco = _coco(
        [
            {"id": 1, "file_name": "a.png", "width": 1173, "height": 817},
            {"id": 2, "file_name": "b.png", "width": 649, "height": 791},
        ],
        [
            {"id": 1, "image_id": 1, "category_id": 2, "bbox": [752.45, 451.11, 123.86, 114.76]},
            {"id": 2, "image_id": 1, "category_id": 5, "bbox": [0.0, 0.0, 1173.0, 817.0]},
            {"id": 3, "image_id": 2, "category_id": 5, "bbox": [15.32, 30.83, 19.8, 169.42]},
        ],
    )
    for target in coco_to_owlv2_targets(coco):
        cx, cy, w, h = (target.boxes[:, i] for i in range(4))
        assert np.all(w > 0.0) and np.all(h > 0.0)
        assert np.all(cx - w / 2.0 >= -1e-6) and np.all(cy - h / 2.0 >= -1e-6)
        assert np.all(cx + w / 2.0 <= 1.0 + 1e-6) and np.all(cy + h / 2.0 <= 1.0 + 1e-6)
        assert target.boxes.dtype == np.float32


def test_empty_coco_yields_no_targets() -> None:
    assert coco_to_owlv2_targets(_coco([], [])) == []


# ----------------------------------------------------------------------------- derived constants


def test_num_patches_is_derived_from_the_pinned_owlv2_operating_point() -> None:
    """3600 is 960/16 squared -- derived, never restated, so a re-export cannot desync it."""
    assert OWLV2_NUM_PATCHES == 3600


# ----------------------------------------------------------------------------- the config schema


def test_finetune_config_is_frozen_forbids_extras_and_documents_every_field() -> None:
    config = FinetuneConfig()
    with pytest.raises(ValidationError):
        FinetuneConfig(nonesuch=1)  # type: ignore[call-arg]
    with pytest.raises(ValidationError):
        config.seed = 3  # type: ignore[misc]

    schema = FinetuneConfig.model_json_schema()["properties"]
    missing = [name for name, field in schema.items() if not field.get("description")]
    assert missing == []


def test_finetune_config_defaults_are_the_owl_vit_recipe() -> None:
    """The matching costs and loss weights are the OWL-ViT paper's 1 / 5 / 2."""
    config = FinetuneConfig()
    assert (config.class_cost, config.bbox_cost, config.giou_cost) == (1.0, 5.0, 2.0)
    assert (config.w_class, config.w_bbox, config.w_giou) == (1.0, 5.0, 2.0)
    assert (config.focal_alpha, config.focal_gamma) == (0.25, 2.0)
    assert config.unfreeze_all is False and config.unfreeze_last_n == 0  # arm A is the default


def test_the_default_loss_mode_is_focal_the_already_measured_recipe() -> None:
    """D-hg1-05: adding the contrastive variant must not change what a bare run does.

    The three arms measured by quick task 260801-8zy were produced by this default; if it ever
    flips, their numbers silently stop describing the shipped recipe.
    """
    assert FinetuneConfig().loss_mode == "focal"


def test_the_supcon_defaults_are_the_khosla_recipe() -> None:
    """tau = 0.07 (D-hg1-02), an unweighted contrastive term, and 64 background negatives."""
    config = FinetuneConfig()
    assert config.supcon_temperature == pytest.approx(0.07)
    assert config.w_contrast == pytest.approx(1.0)
    assert config.supcon_background_negatives == 64


@pytest.mark.parametrize("loss_mode", ["focal", "contrastive", "both"])
def test_every_advertised_loss_mode_is_accepted(loss_mode: str) -> None:
    assert FinetuneConfig(loss_mode=loss_mode).loss_mode == loss_mode  # type: ignore[arg-type]


def test_the_crop_context_defaults_are_off_and_match_inference() -> None:
    """D-dla-03/05: off by default, and the IoU threshold is pinned to the inference default."""
    config = FinetuneConfig()
    assert config.supcon_crop_context is False
    assert config.supcon_query_iou_frac == pytest.approx(0.8)


@pytest.mark.parametrize("value", [True, False])
def test_supcon_crop_context_accepts_both_bools(value: bool) -> None:
    assert FinetuneConfig(supcon_crop_context=value).supcon_crop_context is value


def test_the_crop_margin_and_augment_defaults_are_off() -> None:
    """D-w8c-02/06: both new fields default to today's crop-context-alone behavior."""
    config = FinetuneConfig()
    assert config.supcon_crop_margin_frac == pytest.approx(0.0)
    assert config.supcon_crop_augment is False


@pytest.mark.parametrize("value", [True, False])
def test_supcon_crop_augment_accepts_both_bools(value: bool) -> None:
    assert FinetuneConfig(supcon_crop_augment=value).supcon_crop_augment is value


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("epochs", 0),
        ("batch_size", 0),
        ("lr_head", 0.0),
        ("lr_backbone", -1e-5),
        ("unfreeze_last_n", -1),
        ("weight_decay", -0.1),
        ("grad_clip", 0.0),
        ("grad_accum", 0),
        ("focal_alpha", 1.5),
        ("max_steps", 0),
        ("limit_images", 0),
        ("seed", -1),
        ("loss_mode", "supcon"),  # a plausible-looking name that is not one of the three
        ("supcon_temperature", 0.0),  # tau divides the similarities
        ("supcon_temperature", -0.07),
        ("w_contrast", -1.0),
        ("supcon_background_negatives", -1),
        ("supcon_query_iou_frac", -0.01),
        ("supcon_query_iou_frac", 1.01),
        ("supcon_crop_margin_frac", -0.01),
        ("supcon_crop_margin_frac", 1.01),
    ],
)
def test_finetune_config_rejects_out_of_range_values(field: str, value: float | str) -> None:
    with pytest.raises(ValidationError):
        FinetuneConfig(**{field: value})


# ------------------------------------------------------------------------ the deterministic order


def test_deterministic_batches_covers_every_index_exactly_once() -> None:
    batches = list(deterministic_batches(7, 3, np.random.default_rng(0)))
    assert [len(batch) for batch in batches] == [3, 3, 1]  # the short tail is KEPT, not dropped
    assert sorted(index for batch in batches for index in batch) == list(range(7))


def test_deterministic_batches_is_identical_for_the_same_seed_and_differs_across_seeds() -> None:
    """The repo's reproducibility rule, at the one place the training loop is stochastic."""
    first = list(deterministic_batches(20, 4, np.random.default_rng(0)))
    second = list(deterministic_batches(20, 4, np.random.default_rng(0)))
    other = list(deterministic_batches(20, 4, np.random.default_rng(1)))
    assert first == second
    assert first != other


def test_deterministic_batches_of_an_empty_split_yields_nothing() -> None:
    assert list(deterministic_batches(0, 4, np.random.default_rng(0))) == []


# --------------------------------------------------------------------- the learning-rate schedule


def test_cosine_warmup_rises_linearly_then_decays_to_zero() -> None:
    """The three landmarks of the schedule: first step, end of warmup, end of run."""
    factors = [cosine_warmup_factor(step, total_steps=100, warmup_steps=10) for step in range(100)]

    assert factors[0] == pytest.approx(0.1)  # counts from 1, so the first step is NOT a no-op
    assert factors[9] == pytest.approx(1.0)  # last warmup step reaches the base learning rate
    assert factors[10] == pytest.approx(1.0)  # ... and the cosine branch meets it continuously
    assert factors[-1] < 0.001  # the run ends at (essentially) zero
    assert all(0.0 <= factor <= 1.0 for factor in factors)


def test_cosine_warmup_is_monotone_up_then_monotone_down() -> None:
    factors = [cosine_warmup_factor(step, total_steps=50, warmup_steps=5) for step in range(50)]
    warmup, decay = factors[:5], factors[4:]
    assert warmup == sorted(warmup)
    assert decay == sorted(decay, reverse=True)


def test_cosine_warmup_without_warmup_starts_at_the_base_rate() -> None:
    assert cosine_warmup_factor(0, total_steps=20, warmup_steps=0) == pytest.approx(1.0)


def test_cosine_warmup_respects_a_nonzero_floor() -> None:
    """A `min_factor` floor keeps the tail of the run learning instead of stalling at zero."""
    last = cosine_warmup_factor(19, total_steps=20, warmup_steps=0, min_factor=0.1)
    assert 0.1 <= last < 0.11


@pytest.mark.parametrize(
    "kwargs",
    [
        {"step": -1, "total_steps": 10, "warmup_steps": 1},
        {"step": 0, "total_steps": 0, "warmup_steps": 0},
        {"step": 0, "total_steps": 10, "warmup_steps": -1},
        {"step": 0, "total_steps": 10, "warmup_steps": 10},  # would make the schedule constant
        {"step": 0, "total_steps": 10, "warmup_steps": 1, "min_factor": 1.5},
    ],
)
def test_cosine_warmup_rejects_nonsensical_schedules(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        cosine_warmup_factor(**kwargs)


@pytest.mark.parametrize(
    ("total_steps", "warmup_frac", "expected"),
    [
        (100, 0.1, 10),
        (100, 0.0, 0),
        (3, 0.1, 0),  # rounds to nothing on a smoke run rather than swallowing it
        (3, 0.9, 2),  # clamped to total_steps - 1, so the schedule never becomes constant
        (1, 0.5, 0),
    ],
)
def test_warmup_steps_for_rounds_and_clamps(
    total_steps: int, warmup_frac: float, expected: int
) -> None:
    resolved = warmup_steps_for(total_steps, warmup_frac)
    assert resolved == expected
    # Whatever it returns is, by construction, accepted by the schedule it feeds.
    assert 0.0 <= cosine_warmup_factor(0, total_steps, resolved) <= 1.0


def test_warmup_steps_for_rejects_an_empty_run() -> None:
    with pytest.raises(ValueError):
        warmup_steps_for(0, 0.1)


def test_finetune_config_carries_the_warmup_fraction() -> None:
    assert FinetuneConfig().warmup_frac == pytest.approx(0.1)
    with pytest.raises(ValidationError):
        FinetuneConfig(warmup_frac=1.0)  # a whole run of warmup is not a schedule
