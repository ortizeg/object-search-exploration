"""Tests for Method 2 (`sparse-geo`) -- generalized Hough voting.

These pin the four voting details a naive implementation silently gets wrong:

1. Votes live in a **dict keyed by the bin tuple** (a hash table), not a dense array, because
   the location bin width is scale-dependent.
2. **Soft binning** spreads each vote across the 2 nearest bins per dimension (16 in 4-DoF).
3. **Theta wraps circularly** -- a vote near 359 degrees reaches bin 0, not the opposite end.
4. **single-4dof RAISES** on a frameless backend rather than silently degrading, and the
   O(n^2) pairwise cap is recorded in diagnostics.
"""

from __future__ import annotations

import numpy as np
import pytest

from object_search.search.sparse_geo import (
    _accumulate_votes,
    _cast_votes,
    _Correspondence,
    _enumerate_peaks,
    _soft_neighbours,
    _Vote,
)


def _corr(
    index: int,
    crop_xy: tuple[float, float],
    scene_xy: tuple[float, float],
    *,
    framed: bool = True,
) -> _Correspondence:
    return _Correspondence(
        index=index,
        crop_xy=crop_xy,
        scene_xy=scene_xy,
        crop_scale=1.0 if framed else None,
        crop_angle=0.0 if framed else None,
        scene_scale=1.0 if framed else None,
        scene_angle=0.0 if framed else None,
        distance=0.0,
        rank=0,
    )


# ------------------------------------------------------------------------------ soft bins


def test_soft_neighbours_split_across_two_bins_summing_to_one() -> None:
    (lo, w_lo), (hi, w_hi) = _soft_neighbours(25.0, 10.0)  # coord 2.5 -> bins 2 and 3
    assert (lo, hi) == (2, 3)
    assert w_lo == pytest.approx(0.5)
    assert w_hi == pytest.approx(0.5)
    assert w_lo + w_hi == pytest.approx(1.0)


def test_a_single_vote_spreads_unit_weight_over_the_hash_table() -> None:
    # A vote off every bin centre lands in 2 bins per dimension => 16 dict entries whose weights
    # sum to exactly 1. The container is a DICT keyed by the bin tuple, never a dense array.
    # The 5th key element is CHIRALITY (0 proper / 1 reflected) and is never soft-binned, so a
    # single vote still spreads over exactly 16 bins, not 32.
    vote = _Vote(px=25.0, py=35.0, log_scale=0.3, theta_deg=17.0, members=(0,))
    weight, members = _accumulate_votes((vote,), base_location_width=10.0)
    assert isinstance(weight, dict)
    assert all(isinstance(key, tuple) and len(key) == 5 for key in weight)
    assert {key[4] for key in weight} == {0}, "a proper vote never touches a reflected bin"
    assert len(weight) == 16
    assert sum(weight.values()) == pytest.approx(1.0)
    assert all(0 in idxs for idxs in members.values())


def test_a_reflected_vote_occupies_its_own_chirality_bins() -> None:
    # The same pose cast as a reflection must NOT pool with the proper one: two half-clusters
    # describing different transforms would otherwise jointly clear min_votes.
    proper = _Vote(px=25.0, py=35.0, log_scale=0.3, theta_deg=17.0, members=(0,))
    reflected = _Vote(px=25.0, py=35.0, log_scale=0.3, theta_deg=17.0, members=(1,), reflect=True)
    weight, _ = _accumulate_votes((proper, reflected), base_location_width=10.0)
    assert len(weight) == 32, "the two chiralities occupy disjoint bins"
    assert {key[4] for key in weight} == {0, 1}


# ----------------------------------------------------------------------- clustering peaks


def test_translation_votes_cluster_into_one_peak_at_the_right_offset() -> None:
    # Five correspondences sharing the translation (+30, -10); centre at (50, 50).
    centre = (50.0, 50.0)
    corrs = tuple(
        _corr(i, crop_xy=(float(i), float(2 * i)), scene_xy=(float(i) + 30.0, float(2 * i) - 10.0))
        for i in range(5)
    )
    cast = _cast_votes(
        "translation-2dof",
        corrs,
        centre,
        has_frame=True,
        pairwise_cap=1,
        rng=np.random.default_rng(0),
    )
    weight, members = _accumulate_votes(cast.votes, base_location_width=10.0)
    peaks = _enumerate_peaks(weight, members, min_votes=3, base_location_width=10.0)

    assert len(peaks) == 1
    peak = peaks[0]
    assert peak.dx == pytest.approx(80.0, abs=10.0)  # 50 + 30, within one location bin
    assert peak.dy == pytest.approx(40.0, abs=10.0)  # 50 - 10
    assert len(peak.member_indices) == 5


def test_adjacent_bins_are_de_duplicated_but_far_bins_are_not() -> None:
    # Two adjacent bins (dx differ by 1) describe one cluster -> a single peak.
    weight = {(8, 4, 0, 0, 0): 5.0, (9, 4, 0, 0, 0): 4.0}
    members = {(8, 4, 0, 0, 0): [0, 1, 2, 3, 4], (9, 4, 0, 0, 0): [1, 2, 3, 4]}
    peaks = _enumerate_peaks(weight, members, min_votes=3, base_location_width=10.0)
    assert len(peaks) == 1
    assert peaks[0].dx == pytest.approx(80.0)  # the stronger bin (index 8) wins

    # Two far-apart bins are distinct clusters -> two peaks.
    far = {(8, 4, 0, 0, 0): 5.0, (20, 4, 0, 0, 0): 4.0}
    far_members = {(8, 4, 0, 0, 0): [0, 1, 2], (20, 4, 0, 0, 0): [3, 4, 5]}
    assert len(_enumerate_peaks(far, far_members, min_votes=3, base_location_width=10.0)) == 2

    # Same bin, OPPOSITE chirality: two distinct hypotheses, never de-duplicated against
    # each other (chirality does not step in the neighbourhood).
    chiral = {(8, 4, 0, 0, 0): 5.0, (8, 4, 0, 0, 1): 4.0}
    chiral_members = {(8, 4, 0, 0, 0): [0, 1, 2], (8, 4, 0, 0, 1): [3, 4, 5]}
    assert len(_enumerate_peaks(chiral, chiral_members, min_votes=3, base_location_width=10.0)) == 2


# ---------------------------------------------------------------------- circular theta wrap


def test_theta_wraps_circularly_near_360() -> None:
    # A vote at 359 degrees must reach bin 0 (adjacent by wrap), never a linear opposite end.
    vote = _Vote(px=0.0, py=0.0, log_scale=0.0, theta_deg=359.0, members=(0,))
    weight, _ = _accumulate_votes((vote,), base_location_width=10.0)

    theta_indices = {key[3] for key in weight}
    assert theta_indices == {11, 0}, "359 degrees straddles bins 11 and 0, nothing else"
    weight_at_0 = sum(w for key, w in weight.items() if key[3] == 0)
    weight_at_11 = sum(w for key, w in weight.items() if key[3] == 11)
    assert weight_at_0 > weight_at_11, "359 is closer to the 0 bin than the 11 bin"


# ------------------------------------------------------------------------ voting-mode rules


def test_single_4dof_raises_on_a_frameless_backend() -> None:
    corrs = (_corr(0, (0.0, 0.0), (5.0, 5.0), framed=False),)
    with pytest.raises(ValueError, match="single-4dof"):
        _cast_votes(
            "single-4dof",
            corrs,
            (10.0, 10.0),
            has_frame=False,
            pairwise_cap=1,
            rng=np.random.default_rng(0),
        )


def test_single_4dof_identity_correspondence_votes_at_the_centre() -> None:
    # A correspondence whose scene keypoint equals its crop keypoint under the identity frame must
    # predict the exemplar centre itself.
    corr = _corr(0, crop_xy=(20.0, 20.0), scene_xy=(20.0, 20.0))
    cast = _cast_votes(
        "single-4dof",
        (corr,),
        (50.0, 60.0),
        has_frame=True,
        pairwise_cap=1,
        rng=np.random.default_rng(0),
    )
    assert len(cast.votes) == 1
    assert cast.votes[0].px == pytest.approx(50.0)
    assert cast.votes[0].py == pytest.approx(60.0)


def test_pairwise_cap_is_recorded_in_the_vote_cast() -> None:
    # 10 correspondences -> 45 pairs. A cap of 5 must be honoured and reported.
    corrs = tuple(
        _corr(i, crop_xy=(float(i), float(2 * i)), scene_xy=(float(i) + 5.0, float(2 * i) + 5.0))
        for i in range(10)
    )
    capped = _cast_votes(
        "pairwise-4dof",
        corrs,
        (50.0, 50.0),
        has_frame=True,
        pairwise_cap=5,
        rng=np.random.default_rng(0),
    )
    assert capped.pairwise_capped is True
    assert capped.pairwise_pairs_sampled == 5

    uncapped = _cast_votes(
        "pairwise-4dof",
        corrs,
        (50.0, 50.0),
        has_frame=True,
        pairwise_cap=1000,
        rng=np.random.default_rng(0),
    )
    assert uncapped.pairwise_capped is False
    assert uncapped.pairwise_pairs_sampled == 45
