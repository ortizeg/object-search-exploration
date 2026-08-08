"""Tests for the torch-free SupCon specification (``object_search.train.supcon``).

These run in CI with **no torch, no weights, and no GPU** -- the same reason the target conversion
lives in ``src/`` and the training loop lives in ``scripts/``. The torch mirror in
``scripts/finetune_owlv2.py`` is gated separately, by ``finetune-owlv2 --self-check``, which asserts
it agrees with this module numerically.

Two tests carry the weight of the file:

* :func:`test_planted_same_class_structure_scores_lower_than_shuffled_labels` is the
  **non-tautological** one. It does not re-derive the formula; it asserts the loss is *lower* when
  the labels genuinely describe the clusters than when they are shuffled. A constant-return stub, an
  ``L_in`` misimplementation that ignores the label structure, or a sign error all fail it.
* :func:`test_background_mask_of_a_quarter_square_box_is_the_hand_computed_index_list` pins the
  raster order and the coordinate frame against a list computed by hand. If patch index ``i`` ever
  stops meaning ``divmod(i, grid)`` row-major in the padded-square frame, background sampling starts
  drawing "negatives" from inside the objects and the loss trains the opposite of what it claims
  while the curve still falls (threat T-hg1-01).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from object_search.train.supcon import (
    background_patch_mask,
    cosine_gap_report,
    crop_scene_agreement,
    patch_grid_size,
    sample_background_indices,
    supcon_loss,
)


def _clustered_pool(
    classes: int = 3,
    per_class: int = 4,
    dim: int = 8,
    spread: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Embeddings genuinely clustered by label: ``class_direction + a little noise``."""
    rng = np.random.default_rng(0)
    centres = rng.normal(size=(classes, dim))
    centres /= np.linalg.norm(centres, axis=1, keepdims=True)
    labels = np.repeat(np.arange(classes), per_class)
    noise = rng.normal(scale=spread, size=(classes * per_class, dim))
    return centres[labels] + noise, labels


# ------------------------------------------------------------------- the non-tautological check


def test_planted_same_class_structure_scores_lower_than_shuffled_labels() -> None:
    """Labels that describe the clusters must score STRICTLY below labels that do not.

    This is the test that a constant-return stub cannot pass: it compares two calls whose only
    difference is the label assignment, so any implementation ignoring ``labels`` returns the same
    number twice and fails.
    """
    embeddings, labels = _clustered_pool()
    shuffled = np.random.default_rng(1).permutation(labels)
    assert not np.array_equal(labels, shuffled)  # otherwise the comparison is vacuous

    planted = supcon_loss(embeddings, labels, temperature=0.07)
    scrambled = supcon_loss(embeddings, shuffled, temperature=0.07)

    assert planted < scrambled
    # ... and by a wide margin, not a float wobble: tight clusters are nearly separable.
    assert scrambled - planted > 1.0


def test_loss_of_a_hand_computed_three_row_pool_is_exact() -> None:
    """Two identical same-class rows plus one orthogonal other-class row at ``tau = 1``.

    Anchors 0 and 1 each see one positive (each other, cosine 1) and one negative (cosine 0), so
    each contributes ``log(e + 1) - 1``. Anchor 2's label is unique, so it contributes nothing and
    is excluded from the divisor -- the mean is over TWO anchors, not three.
    """
    embeddings = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    labels = np.array([0, 0, 1])

    expected = math.log(math.e + 1.0) - 1.0
    assert supcon_loss(embeddings, labels, temperature=1.0) == pytest.approx(expected)


# ---------------------------------------------------------------- anchors without a positive pair


def test_a_pool_of_all_distinct_labels_is_exactly_zero_not_nan() -> None:
    """No anchor has a positive, so every anchor is excluded: 0.0 exactly, never NaN."""
    embeddings, _labels = _clustered_pool(classes=6, per_class=1)
    loss = supcon_loss(embeddings, np.arange(6), temperature=0.07)

    assert loss == 0.0
    assert not math.isnan(loss)


def test_a_two_row_same_class_pool_has_nothing_to_contrast_against() -> None:
    """With one positive and no negative the denominator IS the positive: exactly 0.0."""
    embeddings = np.array([[1.0, 0.0], [0.3, 0.9]])
    assert supcon_loss(embeddings, np.array([0, 0]), temperature=0.07) == pytest.approx(0.0)


def test_a_single_row_pool_returns_zero() -> None:
    assert supcon_loss(np.array([[1.0, 0.0]]), np.array([0]), temperature=0.07) == 0.0


def test_an_anchor_without_a_positive_is_dropped_from_the_divisor() -> None:
    """A uniquely-labelled row must not dilute the mean by averaging in a fabricated 0.0.

    Asserted exactly, on the hand-computed three-row pool: row 2's label is unique, so the mean is
    over the TWO contributing anchors. Averaging the singleton in as a zero would give ``2/3`` of
    that -- a plausible-looking number that depends on the batch composition rather than on the
    embeddings, which is precisely why the divisor rule is a correctness requirement.
    """
    embeddings = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    labels = np.array([0, 0, 1])
    contributing_mean = math.log(math.e + 1.0) - 1.0

    loss = supcon_loss(embeddings, labels, temperature=1.0)

    assert loss == pytest.approx(contributing_mean)
    assert loss != pytest.approx(contributing_mean * 2.0 / 3.0)  # the diluted mean, ruled out


# ------------------------------------------------------------------------------ the invariances


def test_loss_is_invariant_to_a_permutation_of_the_pool() -> None:
    embeddings, labels = _clustered_pool()
    order = np.random.default_rng(2).permutation(len(labels))

    assert supcon_loss(embeddings[order], labels[order], temperature=0.07) == pytest.approx(
        supcon_loss(embeddings, labels, temperature=0.07)
    )


def test_loss_is_invariant_to_a_positive_rescaling_of_any_row() -> None:
    """L2 normalization happens INSIDE, so only the directions matter -- as at inference."""
    embeddings, labels = _clustered_pool()
    scaled = embeddings.copy()
    scaled[0] *= 17.5
    scaled[5] *= 0.01

    assert supcon_loss(scaled, labels, temperature=0.07) == pytest.approx(
        supcon_loss(embeddings, labels, temperature=0.07)
    )


def test_a_zero_row_is_normalized_without_producing_nan() -> None:
    """OWLv2 can emit an all-zero embedding for a padded patch; it must not poison the pool."""
    embeddings, labels = _clustered_pool(classes=2, per_class=3)
    embeddings[0] = 0.0

    loss = supcon_loss(embeddings, labels, temperature=0.07)
    assert math.isfinite(loss)


def test_the_loss_is_finite_at_the_default_temperature_that_would_overflow_float32() -> None:
    """1/0.07-scaled similarities overflow a naive float32 exp into a plausible inf (T-hg1-07)."""
    embeddings, labels = _clustered_pool(classes=4, per_class=8, dim=64, spread=0.001)
    loss = supcon_loss(embeddings, labels, temperature=0.07)

    assert math.isfinite(loss)
    assert loss >= 0.0


# --------------------------------------------------------------- denominator-only (background)


def test_appending_a_background_row_cannot_decrease_the_loss() -> None:
    """A denominator-only row adds a term to every anchor's denominator: the loss can only rise."""
    embeddings, labels = _clustered_pool(classes=2, per_class=4)
    without = supcon_loss(embeddings, labels, temperature=0.07)

    background = embeddings[0:1] * 0.5 + 0.1  # deliberately close to a real anchor
    pooled = np.vstack([embeddings, background])
    negative_only = np.zeros(len(pooled), dtype=bool)
    negative_only[-1] = True

    with_background = supcon_loss(
        pooled,
        np.concatenate([labels, [0]]),
        temperature=0.07,
        negative_only=negative_only,
    )

    assert with_background >= without
    assert with_background > without  # this one is genuinely close, so it must actually bite


def test_a_background_row_is_never_a_positive_even_when_its_label_matches() -> None:
    """The denominator-only rule is enforced HERE, not trusted to the caller.

    Row 1 shares row 0's label but is flagged ``negative_only``. If it were allowed to act as a
    positive, anchor 0 would contribute and the loss would be non-zero; the rule says anchor 0 has
    no positive, so every anchor is excluded and the result is exactly 0.0.
    """
    embeddings = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])
    labels = np.array([0, 0, 1])
    negative_only = np.array([False, True, False])

    loss = supcon_loss(embeddings, labels, temperature=0.07, negative_only=negative_only)
    assert loss == 0.0


def test_a_background_row_is_never_an_anchor() -> None:
    """Flagging the ONLY same-class pair as background leaves no anchor at all."""
    embeddings, labels = _clustered_pool(classes=2, per_class=2)
    negative_only = np.array([False, True, False, True])

    assert supcon_loss(embeddings, labels, temperature=0.5, negative_only=negative_only) == 0.0


# ------------------------------------------------------------------------------- the temperature


def test_a_lower_temperature_raises_the_loss_of_an_imperfect_configuration() -> None:
    """Each anchor's nearest neighbour is the WRONG class, so sharpening must hurt.

    ``tau`` divides the similarities, so as it falls the log-sum-exp approaches the max and the loss
    approaches ``(best_negative - mean_positive) / tau``, which diverges for this configuration.
    """
    embeddings = np.array(
        [
            [1.0, 0.0],  # label 0
            [0.0, 1.0],  # label 0 -- orthogonal to its own positive
            [0.99, 0.14],  # label 1 -- but nearest to row 0
            [0.14, 0.99],  # label 1 -- but nearest to row 1
        ]
    )
    labels = np.array([0, 0, 1, 1])

    assert supcon_loss(embeddings, labels, temperature=0.07) > supcon_loss(
        embeddings, labels, temperature=0.5
    )


# --------------------------------------------------------------------------------- input guards


@pytest.mark.parametrize(
    ("embeddings", "labels", "temperature", "negative_only"),
    [
        (np.zeros(4), np.zeros(4, dtype=int), 0.07, None),  # not 2-D
        (np.zeros((4, 3)), np.zeros(3, dtype=int), 0.07, None),  # labels mismatch
        (np.zeros((4, 3)), np.zeros(4, dtype=int), 0.0, None),  # tau must be positive
        (np.zeros((4, 3)), np.zeros(4, dtype=int), -1.0, None),
        (np.zeros((4, 3)), np.zeros(4, dtype=int), 0.07, np.zeros(3, dtype=bool)),  # mask mismatch
    ],
)
def test_supcon_loss_rejects_mismatched_inputs(
    embeddings: np.ndarray,
    labels: np.ndarray,
    temperature: float,
    negative_only: np.ndarray | None,
) -> None:
    with pytest.raises(ValueError):
        supcon_loss(embeddings, labels, temperature, negative_only=negative_only)


# ------------------------------------------------------------------------------- the patch grid


def test_patch_grid_size_is_sixty_at_the_pinned_operating_point() -> None:
    """960 / 16 = 60 per side, 3600 patches -- derived from the tensor, never restated."""
    assert patch_grid_size(3600) == 60
    assert patch_grid_size(1) == 1


@pytest.mark.parametrize("num_patches", [0, -1, 2, 3599, 3601])
def test_patch_grid_size_rejects_a_non_square_count(num_patches: int) -> None:
    with pytest.raises(ValueError):
        patch_grid_size(num_patches)


def test_background_mask_of_a_quarter_square_box_is_the_hand_computed_index_list() -> None:
    """On a 4x4 grid, a box over ``[0, 0.5] x [0, 0.5]`` covers exactly cells 0, 1, 4 and 5.

    Cell centres are at 0.125, 0.375, 0.625, 0.875. The box spans ``[0, 0.5]`` on both axes, so the
    two low centres are inside and the two high ones are not: rows 0-1 x cols 0-1, which in
    row-major order is ``{0, 1, 4, 5}``. Any transposition of the raster order would give
    ``{0, 1, 4, 5}`` too by symmetry, so the box is deliberately made ASYMMETRIC below as well.
    """
    mask = background_patch_mask(np.array([[0.25, 0.25, 0.5, 0.5]]), grid=4)

    assert mask.shape == (16,)
    assert sorted(np.flatnonzero(~mask).tolist()) == [0, 1, 4, 5]


def test_background_mask_raster_order_is_row_major_not_column_major() -> None:
    """An asymmetric box distinguishes ``divmod(i, grid)`` from its transpose.

    The box spans x in ``[0, 0.5]`` and y in ``[0, 0.25]``: columns 0-1 of row 0 only, i.e. cells
    ``{0, 1}``. A column-major reading would instead mark ``{0, 4}``, so this is the assertion that
    actually pins the order.
    """
    mask = background_patch_mask(np.array([[0.25, 0.125, 0.5, 0.25]]), grid=4)
    assert sorted(np.flatnonzero(~mask).tolist()) == [0, 1]


def test_a_box_covering_the_unit_square_leaves_no_background() -> None:
    mask = background_patch_mask(np.array([[0.5, 0.5, 1.0, 1.0]]), grid=8)
    assert not mask.any()


def test_an_empty_box_array_leaves_every_cell_background() -> None:
    for empty in (np.zeros((0, 4)), np.array([]).reshape(0, 4)):
        mask = background_patch_mask(empty, grid=5)
        assert mask.all() and mask.shape == (25,)


def test_background_mask_unions_overlapping_boxes() -> None:
    """Two boxes mask the UNION of their cells, never the intersection."""
    boxes = np.array([[0.125, 0.125, 0.25, 0.25], [0.875, 0.875, 0.25, 0.25]])
    mask = background_patch_mask(boxes, grid=4)
    assert sorted(np.flatnonzero(~mask).tolist()) == [0, 15]


@pytest.mark.parametrize(
    ("boxes", "grid"),
    [
        (np.zeros((1, 4)), 0),
        (np.zeros((1, 4)), -3),
        (np.zeros((2, 3)), 4),
    ],
)
def test_background_patch_mask_rejects_bad_shapes(boxes: np.ndarray, grid: int) -> None:
    with pytest.raises(ValueError):
        background_patch_mask(boxes, grid)


# ------------------------------------------------------------------- background negative sampling

_TWO_BOXES = np.array([[0.2, 0.2, 0.2, 0.2], [0.7, 0.6, 0.3, 0.2]])


def test_background_sampling_is_identical_for_the_same_seed_and_differs_across_seeds() -> None:
    """The repo's reproducibility rule at the one place the contrastive loss is stochastic."""
    first = sample_background_indices(_TWO_BOXES, 20, 16, np.random.default_rng(0))
    again = sample_background_indices(_TWO_BOXES, 20, 16, np.random.default_rng(0))
    other = sample_background_indices(_TWO_BOXES, 20, 16, np.random.default_rng(1))

    assert np.array_equal(first, again)
    assert not np.array_equal(first, other)


def test_sampled_indices_are_never_inside_a_box() -> None:
    """Asserted against the mask itself, which is the claim that makes the loss mean anything.

    If this ever fails, the "negatives" are patches the box loss is simultaneously supervising as
    objects, and the contrastive term trains the exact opposite of what it advertises while its
    curve still falls (threat T-hg1-01).
    """
    mask = background_patch_mask(_TWO_BOXES, 20)
    drawn = sample_background_indices(_TWO_BOXES, 20, 64, np.random.default_rng(3))

    assert drawn.size == 64
    assert mask[drawn].all()
    assert len(set(drawn.tolist())) == drawn.size  # without replacement


def test_sampled_indices_are_ascending_and_int64() -> None:
    """Sorted, so the pooled row order depends on the sampled SET, not on the draw order."""
    drawn = sample_background_indices(_TWO_BOXES, 20, 32, np.random.default_rng(4))

    assert drawn.dtype == np.int64
    assert drawn.tolist() == sorted(drawn.tolist())


def test_asking_for_more_than_exist_returns_every_background_cell_not_an_error() -> None:
    """A densely-annotated plan genuinely has few background cells; that is not a failure."""
    box = np.array([[0.5, 0.5, 0.75, 0.75]])  # leaves only the border cells on a 6x6 grid
    mask = background_patch_mask(box, 6)
    drawn = sample_background_indices(box, 6, 64, np.random.default_rng(0))

    assert 0 < drawn.size < 64
    assert drawn.tolist() == np.flatnonzero(mask).tolist()


def test_an_image_with_no_background_cell_yields_an_empty_draw() -> None:
    covered = np.array([[0.5, 0.5, 1.0, 1.0]])
    drawn = sample_background_indices(covered, 8, 32, np.random.default_rng(0))

    assert drawn.size == 0
    assert drawn.dtype == np.int64


def test_a_count_of_zero_is_a_supported_disable() -> None:
    """`supcon_background_negatives=0` must be an ordinary return, so the knob is testable."""
    drawn = sample_background_indices(_TWO_BOXES, 20, 0, np.random.default_rng(0))

    assert drawn.size == 0
    assert drawn.dtype == np.int64


def test_zero_count_does_not_consume_the_generator() -> None:
    """Disabling background negatives must not perturb the seeded stream the epoch shuffle uses."""
    rng = np.random.default_rng(7)
    sample_background_indices(_TWO_BOXES, 20, 0, rng)

    assert rng.integers(0, 1_000_000) == np.random.default_rng(7).integers(0, 1_000_000)


@pytest.mark.parametrize("count", [-1, -64])
def test_sample_background_indices_rejects_a_negative_count(count: int) -> None:
    with pytest.raises(ValueError):
        sample_background_indices(_TWO_BOXES, 20, count, np.random.default_rng(0))


def test_sample_background_indices_propagates_the_mask_shape_guards() -> None:
    with pytest.raises(ValueError):
        sample_background_indices(np.zeros((2, 3)), 4, 2, np.random.default_rng(0))


# -------------------------------------------------------------------------- the cosine diagnostic


def test_cosine_gap_of_clustered_anchors_far_from_background_is_near_its_maximum() -> None:
    """Tight clusters, mutually near-orthogonal, with background further still."""
    embeddings, labels = _clustered_pool(classes=3, per_class=4, dim=16, spread=0.01)
    background = -embeddings[:6]  # antipodal to real anchors: as far as a cosine can get

    report = cosine_gap_report(embeddings, labels, background)

    assert report["same_class_mean"] is not None and report["same_class_mean"] > 0.99
    assert report["gap_class"] is not None and report["gap_class"] > 0.9
    assert report["gap_background"] is not None and report["gap_background"] > 0.9


def test_cosine_gap_of_randomly_labelled_anchors_is_near_zero() -> None:
    """Labels that describe nothing must not produce a class gap -- the diagnostic's null case."""
    rng = np.random.default_rng(11)
    embeddings = rng.normal(size=(60, 32))
    labels = rng.integers(0, 3, size=60)

    report = cosine_gap_report(embeddings, labels, np.zeros((0, 32)))

    assert report["gap_class"] is not None
    assert abs(report["gap_class"]) < 0.05
    assert report["background_mean"] is None  # nothing to measure against
    assert report["gap_background"] is None


def test_a_component_with_no_contributing_pair_is_none_never_zero() -> None:
    """One anchor per class: there IS no same-class pair, so `None` -- not a fabricated 0.0.

    A ``0.0`` here would read in the report's cosine-gap table as "measured, no separation", when
    the truth is "not measurable". The repo's nullable-human-count rule, applied to a metric.
    """
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]])
    report = cosine_gap_report(embeddings, np.array([0, 1]), np.zeros((0, 2)))

    assert report["same_class_mean"] is None
    assert report["diff_class_mean"] == pytest.approx(0.0)  # measured, and genuinely zero
    assert report["gap_class"] is None
    assert report["gap_background"] is None


def test_a_single_class_pool_has_no_different_class_mean() -> None:
    embeddings, _labels = _clustered_pool(classes=1, per_class=4)
    report = cosine_gap_report(embeddings, np.zeros(4, dtype=int), np.zeros((0, 8)))

    assert report["same_class_mean"] is not None
    assert report["diff_class_mean"] is None
    assert report["gap_class"] is None


def test_an_empty_anchor_pool_reports_every_component_as_none() -> None:
    report = cosine_gap_report(np.zeros((0, 4)), np.zeros(0, dtype=int), np.zeros((3, 4)))

    assert set(report) == {
        "same_class_mean",
        "diff_class_mean",
        "background_mean",
        "gap_class",
        "gap_background",
    }
    assert all(value is None for value in report.values())


def test_the_background_mean_is_measured_over_every_anchor_background_pair() -> None:
    """Two anchors at 0 and 90 degrees against one background row at 0: mean cosine 0.5."""
    anchors = np.array([[1.0, 0.0], [0.0, 1.0]])
    report = cosine_gap_report(anchors, np.array([0, 1]), np.array([[3.0, 0.0]]))

    assert report["background_mean"] == pytest.approx(0.5)


def test_cosine_gap_is_invariant_to_a_positive_rescaling() -> None:
    """L2 normalization happens INSIDE, exactly as it does at inference."""
    embeddings, labels = _clustered_pool(classes=2, per_class=3)
    background = np.abs(embeddings[:2]) + 0.3
    scaled = embeddings.copy()
    scaled[0] *= 12.0

    baseline = cosine_gap_report(embeddings, labels, background)
    rescaled = cosine_gap_report(scaled, labels, background * 5.0)

    for key, value in baseline.items():
        assert rescaled[key] == pytest.approx(value)


def test_a_zero_anchor_row_does_not_produce_nan() -> None:
    embeddings, labels = _clustered_pool(classes=2, per_class=3)
    embeddings[0] = 0.0

    report = cosine_gap_report(embeddings, labels, np.zeros((1, 8)))

    assert all(value is not None and math.isfinite(value) for value in report.values())


@pytest.mark.parametrize(
    ("anchors", "labels", "background"),
    [
        (np.zeros(4), np.zeros(4, dtype=int), np.zeros((1, 4))),  # anchors not 2-D
        (np.zeros((4, 3)), np.zeros(4, dtype=int), np.zeros(3)),  # background not 2-D
        (np.zeros((4, 3)), np.zeros(3, dtype=int), np.zeros((1, 3))),  # labels misaligned
        (np.zeros((4, 3)), np.zeros(4, dtype=int), np.zeros((1, 5))),  # dimensions disagree
    ],
)
def test_cosine_gap_report_rejects_mismatched_inputs(
    anchors: np.ndarray, labels: np.ndarray, background: np.ndarray
) -> None:
    with pytest.raises(ValueError):
        cosine_gap_report(anchors, labels, background)


# ------------------------------------------------------------------- crop/scene self-score (dla)


def test_crop_scene_agreement_hand_computed_two_pair_case() -> None:
    """Pair 0: identical vectors -> cosine 1.0. Pair 1: orthogonal -> cosine 0.0."""
    crop = np.array([[1.0, 0.0], [1.0, 0.0]])
    scene = np.array([[1.0, 0.0], [0.0, 1.0]])

    report = crop_scene_agreement(crop, scene)

    assert report["n_pairs"] == 2.0
    assert report["self_score_mean"] == pytest.approx(0.5)
    assert report["self_score_min"] == pytest.approx(0.0)
    assert report["self_score_max"] == pytest.approx(1.0)


def test_crop_scene_agreement_is_invariant_to_a_permutation_of_the_pool() -> None:
    rng = np.random.default_rng(0)
    crop = rng.normal(size=(6, 8))
    scene = rng.normal(size=(6, 8))
    order = np.random.default_rng(1).permutation(6)

    baseline = crop_scene_agreement(crop, scene)
    permuted = crop_scene_agreement(crop[order], scene[order])

    assert permuted["self_score_mean"] == pytest.approx(baseline["self_score_mean"])
    assert permuted["self_score_min"] == pytest.approx(baseline["self_score_min"])
    assert permuted["self_score_max"] == pytest.approx(baseline["self_score_max"])
    assert permuted["n_pairs"] == baseline["n_pairs"]


def test_crop_scene_agreement_is_invariant_to_independent_positive_rescaling() -> None:
    """L2 normalization happens INSIDE, independently on each side -- as at inference."""
    rng = np.random.default_rng(2)
    crop = rng.normal(size=(5, 4))
    scene = rng.normal(size=(5, 4))

    baseline = crop_scene_agreement(crop, scene)
    rescaled = crop_scene_agreement(crop * 11.0, scene * 0.02)

    assert rescaled["self_score_mean"] == pytest.approx(baseline["self_score_mean"])
    assert rescaled["self_score_min"] == pytest.approx(baseline["self_score_min"])
    assert rescaled["self_score_max"] == pytest.approx(baseline["self_score_max"])


@pytest.mark.parametrize(
    ("crop", "scene"),
    [
        (np.zeros(4), np.zeros((1, 4))),  # crop not 2-D
        (np.zeros((1, 4)), np.zeros(4)),  # scene not 2-D
        (np.zeros((3, 4)), np.zeros((2, 4))),  # n disagrees
        (np.zeros((3, 4)), np.zeros((3, 5))),  # d disagrees
    ],
)
def test_crop_scene_agreement_rejects_mismatched_shapes(
    crop: np.ndarray, scene: np.ndarray
) -> None:
    with pytest.raises(ValueError):
        crop_scene_agreement(crop, scene)


def test_crop_scene_agreement_of_two_empty_pools_reports_every_key_none_not_zero() -> None:
    report = crop_scene_agreement(np.zeros((0, 4)), np.zeros((0, 4)))

    assert report["n_pairs"] == 0.0
    assert report["self_score_mean"] is None
    assert report["self_score_min"] is None
    assert report["self_score_max"] is None
