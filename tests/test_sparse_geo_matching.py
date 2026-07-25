"""Tests for Method 2 (`sparse-geo`) -- backends and the ratio-test-disabled matching layer.

These pin the two things easiest to get wrong in the matching step, both stated as Phase 5
success criteria:

1. **The standard Lowe ratio test is DISABLED.** The counterfactual test builds a scene where
   every crop keypoint has several near-equal neighbours (the repeated-instance signature) and
   proves the standard best/second ratio *would* suppress those correspondences while the
   many-to-many top-k path (and the optional k+1 ratio) keeps them.
2. **The descriptor distance metric is a property of the backend, not a config field** -- SIFT
   and AKAZE are float L2, ORB is binary Hamming.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from object_search.search.sparse_geo import (
    SparseGeoConfig,
    _abstain_note,
    _Keypoints,
    _make_backend,
    _match_top_k,
    _pairwise_distances,
    _topk_neighbours,
)

_SOURCE = Path(__file__).resolve().parents[1] / "src" / "object_search" / "search" / "sparse_geo.py"


# --------------------------------------------------------------------------- source contract


def test_source_documents_the_disabled_ratio_test_and_backlog() -> None:
    text = _SOURCE.read_text(encoding="utf-8")
    assert "ROBUSTNESS BACKLOG" in text
    # The disabled standard ratio test is the single most load-bearing decision; it must be
    # documented in the module the practitioner reads, not only in the plan.
    assert "ratio test is DISABLED" in text
    assert "k+1" in text


# ------------------------------------------------------------------------ backend / metric


def test_backends_fix_their_distance_metric() -> None:
    # The metric is chosen from the backend and is NOT a SparseGeoConfig field.
    assert "metric" not in SparseGeoConfig.model_fields
    assert _make_backend("sift").metric == "l2"
    assert _make_backend("akaze").metric == "l2"
    assert _make_backend("orb").metric == "hamming"
    # All three classical backends carry a keypoint frame (scale + orientation).
    assert all(_make_backend(name).has_frame for name in ("sift", "akaze", "orb"))


# ------------------------------------------------------------------------ distance helpers


def test_l2_distances_match_the_naive_formula() -> None:
    query = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    train = np.array([[0.0, 0.0], [3.0, 4.0]], dtype=np.float32)
    dist = _pairwise_distances(query, train, "l2")
    assert dist.shape == (2, 2)
    assert dist[0, 0] == pytest.approx(0.0)
    assert dist[0, 1] == pytest.approx(5.0)
    assert dist[1, 0] == pytest.approx(np.sqrt(2.0))


def test_hamming_distances_count_differing_bits() -> None:
    query = np.array([[0b0000_0000]], dtype=np.uint8)
    train = np.array([[0b0000_0000], [0b1111_1111], [0b1010_1010]], dtype=np.uint8)
    dist = _pairwise_distances(query, train, "hamming")
    assert list(dist[0]) == [0.0, 8.0, 4.0]


def test_topk_returns_sorted_nearest() -> None:
    dist = np.array([[5.0, 1.0, 3.0, 2.0, 4.0]], dtype=np.float64)
    idx, sorted_dist = _topk_neighbours(dist, 3)
    assert list(idx[0]) == [1, 3, 2]
    assert list(sorted_dist[0]) == [1.0, 2.0, 3.0]


# -------------------------------------------------------------- the counterfactual (success)


def _keypoints_from_descriptors(desc: npt.NDArray[np.float32]) -> _Keypoints:
    """Wrap descriptors in a framed _Keypoints with placeholder geometry (matching ignores it)."""
    n = desc.shape[0]
    xy = np.zeros((n, 2), dtype=np.float64)
    scale = np.ones(n, dtype=np.float64)
    angle = np.zeros(n, dtype=np.float64)
    return _Keypoints(xy, scale, angle, desc)


def _repeated_instance_scene(
    n_crop: int, n_instances: int, dim: int = 16
) -> tuple[_Keypoints, _Keypoints, npt.NDArray[np.float64]]:
    """Build a crop and a scene where each crop keypoint has ``n_instances`` near-equal matches.

    Each crop descriptor is a distinct random vector; the scene holds ``n_instances`` slightly
    perturbed copies of each (the repeated instances, all at near-equal distance) plus a block of
    far-away descriptors so a genuine (k+1)-th neighbour exists. Returns the raw crop->scene
    distance matrix too, so the standard-ratio counterfactual can be computed independently.
    """
    rng = np.random.default_rng(0)
    crop_desc = rng.normal(0.0, 1.0, size=(n_crop, dim)).astype(np.float32) * 10.0
    scene_rows: list[npt.NDArray[np.float32]] = []
    for i in range(n_crop):
        for _ in range(n_instances):
            # Near-identical copies: tiny, near-equal perturbations => near-equal distances.
            scene_rows.append(crop_desc[i] + rng.normal(0.0, 0.01, size=dim).astype(np.float32))
    # A block of far descriptors so the (k+1)-th neighbour is clearly separated.
    far = rng.normal(0.0, 1.0, size=(n_crop, dim)).astype(np.float32) + 500.0
    scene_desc = np.concatenate([np.asarray(scene_rows, dtype=np.float32), far], axis=0)
    crop = _keypoints_from_descriptors(crop_desc)
    scene = _keypoints_from_descriptors(scene_desc)
    dist = _pairwise_distances(crop_desc, scene_desc, "l2")
    return crop, scene, dist


def test_standard_ratio_would_suppress_the_repeats_the_topk_keeps() -> None:
    """Phase 5 success criterion: prove the disabled standard ratio test would kill the repeats.

    Six near-identical instances per crop keypoint. The standard Lowe ratio (best/second < 0.8)
    would reject every crop keypoint, because its best and second neighbours are near-equal. The
    many-to-many top-k path keeps all six per keypoint. This is the whole reason the standard
    ratio test is disabled for this method.
    """
    k = 6
    n_crop = 4
    crop, scene, dist = _repeated_instance_scene(n_crop=n_crop, n_instances=k)

    # What the STANDARD ratio test (best/second < 0.8) would do -- computed here, never in the
    # module (the module must contain no such test).
    kept_by_standard_ratio = 0
    for row in dist:
        ordered = np.sort(row)
        best, second = ordered[0], ordered[1]
        if second > 0.0 and best / second < 0.8:
            kept_by_standard_ratio += 1
    assert kept_by_standard_ratio == 0, "standard ratio must reject every repeated keypoint here"

    # What THIS method does: top-k unconditionally, ratio test disabled.
    result = _match_top_k(crop, scene, "l2", k=k, use_kplus1_ratio=False, kplus1_ratio=0.9)
    assert result.n_crop_matched == n_crop
    assert len(result.correspondences) == n_crop * k, "all k repeats per keypoint are kept"


def test_kplus1_ratio_keeps_repeats_but_the_ceiling_is_recorded_when_truncated() -> None:
    # k=6 with exactly 6 instances: the (k+1)-th neighbour is a FAR descriptor, so the k+1 ratio
    # does NOT drop the keypoint (the six repeats are cleanly separated from the tail).
    crop, scene, _ = _repeated_instance_scene(n_crop=3, n_instances=6)
    kept = _match_top_k(crop, scene, "l2", k=6, use_kplus1_ratio=True, kplus1_ratio=0.9)
    assert len(kept.correspondences) == 3 * 6
    assert kept.n_dropped_kplus1 == 0
    assert kept.k_ceiling_hit == 0

    # k=6 with EIGHT instances: now the (k+1)-th neighbour is itself a near-equal repeat, so the
    # ceiling truncated real instances -- k_ceiling_hit must fire, and the k+1 ratio drops them.
    crop8, scene8, _ = _repeated_instance_scene(n_crop=3, n_instances=8)
    truncated = _match_top_k(crop8, scene8, "l2", k=6, use_kplus1_ratio=False, kplus1_ratio=0.9)
    assert truncated.k_ceiling_hit == 3, "every crop keypoint hit the k ceiling"
    dropped = _match_top_k(crop8, scene8, "l2", k=6, use_kplus1_ratio=True, kplus1_ratio=0.9)
    assert dropped.n_dropped_kplus1 == 3, "the k+1 ratio drops non-discriminative keypoints"


def test_empty_inputs_match_to_nothing() -> None:
    empty = _Keypoints(np.empty((0, 2), np.float64), None, None, np.empty((0, 16), np.float32))
    some = _keypoints_from_descriptors(np.ones((3, 16), np.float32))
    assert _match_top_k(empty, some, "l2", 6, False, 0.9).correspondences == ()
    assert _match_top_k(some, empty, "l2", 6, False, 0.9).correspondences == ()


# --------------------------------------------------------------------- low-keypoint guard note


def test_abstain_note_explains_the_low_keypoint_guard() -> None:
    note = _abstain_note("sift", n_keypoints=7, minimum=20)
    assert "7" in note and "20" in note
    assert "texture" in note.lower()


# ------------------------------------------------------------------------- real-image sanity


def test_sift_yields_many_keypoints_on_texture_but_few_on_a_flat_crop() -> None:
    """The guard's premise: a textured crop clears min_exemplar_keypoints, a flat one does not."""
    rng = np.random.default_rng(0)
    textured = rng.integers(0, 256, size=(96, 96), dtype=np.uint8)
    flat = np.full((96, 96), 128, dtype=np.uint8)
    backend = _make_backend("sift")

    assert _detect_count(backend, textured) >= 20
    assert _detect_count(backend, flat) < 20


def _detect_count(backend: object, gray: npt.NDArray[np.uint8]) -> int:
    from object_search.search.sparse_geo import _detect

    return _detect(gray, backend).count  # type: ignore[arg-type]  # _Backend is module-private
