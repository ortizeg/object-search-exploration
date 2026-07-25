"""Tests for Method 2 (`sparse-geo`) -- the sequential-RANSAC decomposition (METHOD-04b).

Sequential RANSAC is the pluggable alternative to generalized Hough voting, behind the **same**
interface: it consumes the same correspondences and returns the same list of accepted models. It
fits the dominant similarity over all correspondences, removes its inliers, and repeats until the
support falls below `min_inliers`. These pin:

1. `decomposition` switches between 'hough' and 'sequential-ransac' by **config alone**.
2. On a 6-instance scene, sequential-ransac recovers **multiple** distinct models (METHOD-12).
3. It is **deterministic** under a fixed seed (the NumPy `default_rng(config.seed)` is the real
   seed, exactly as for the per-peak Hough path), and a different seed still recovers instances.
4. It finds a **comparable** instance count to the Hough path on the same scene.
"""

from __future__ import annotations

import cv2
import numpy as np
import numpy.typing as npt

from object_search.schemas import BBox, ExemplarBox, SearchOutcome
from object_search.search.sparse_geo import SparseGeoConfig, search


def _tiled_scene() -> npt.NDArray[np.uint8]:
    """Six identical, SIFT-rich tiles on a flat background -- a deliberate multi-instance scene.

    Mirrors the fixture used by the Hough end-to-end test so the two decompositions are compared
    on the identical scene.
    """
    rng = np.random.default_rng(3)
    small = rng.integers(0, 256, size=(10, 10), dtype=np.uint8)
    tile = cv2.resize(small, (64, 64), interpolation=cv2.INTER_CUBIC)  # low-freq blobs SIFT likes
    scene = np.full((360, 600), 128, dtype=np.uint8)
    for x, y in [(20, 20), (180, 25), (340, 30), (60, 200), (240, 210), (460, 200)]:
        scene[y : y + 64, x : x + 64] = tile
    return np.ascontiguousarray(np.stack([scene] * 3, axis=-1))


_EXEMPLAR = ExemplarBox(box=BBox(x=20, y=20, w=64, h=64))


def test_decomposition_switches_by_config_alone() -> None:
    """The only difference between the two runs is the `decomposition` field."""
    hough = SparseGeoConfig(decomposition="hough")
    sequential = SparseGeoConfig(decomposition="sequential-ransac")
    assert hough.decomposition == "hough"
    assert sequential.decomposition == "sequential-ransac"
    # Same code path, same everything else -- only the strategy differs.
    assert hough.model_dump(exclude={"decomposition"}) == sequential.model_dump(
        exclude={"decomposition"}
    )


def test_sequential_ransac_recovers_multiple_models_on_a_six_instance_scene() -> None:
    scene = _tiled_scene()
    result = search(scene, _EXEMPLAR, SparseGeoConfig(decomposition="sequential-ransac"))

    assert result.outcome is SearchOutcome.OK
    # METHOD-12: multiple DISTINCT models, never a single-best short-circuit.
    assert len(result.matches) >= 2
    distinct = {m.box.xyxy for m in result.matches}
    assert len(distinct) == len(result.matches), "each recovered model is a distinct box"
    # Each match carries its fitted 2x3 similarity as a flattened 6-tuple.
    assert all(m.transform is not None and len(m.transform) == 6 for m in result.matches)


def test_sequential_ransac_is_deterministic_under_a_fixed_seed() -> None:
    scene = _tiled_scene()
    config = SparseGeoConfig(decomposition="sequential-ransac", seed=7)
    first = search(scene, _EXEMPLAR, config)
    second = search(scene, _EXEMPLAR, config)

    assert [m.box.xyxy for m in first.matches] == [m.box.xyxy for m in second.matches]
    assert [m.transform for m in first.matches] == [m.transform for m in second.matches]


def test_sequential_ransac_still_recovers_instances_under_a_different_seed() -> None:
    """A different seed changes the sampling but must still recover the repeated instances."""
    scene = _tiled_scene()
    other = search(scene, _EXEMPLAR, SparseGeoConfig(decomposition="sequential-ransac", seed=99))
    assert other.outcome is SearchOutcome.OK
    assert len(other.matches) >= 2


def test_sequential_and_hough_find_a_comparable_instance_count() -> None:
    """The two strategies are alternatives, so their counts should be in the same ballpark."""
    scene = _tiled_scene()
    hough = search(scene, _EXEMPLAR, SparseGeoConfig(decomposition="hough"))
    sequential = search(scene, _EXEMPLAR, SparseGeoConfig(decomposition="sequential-ransac"))

    assert hough.outcome is SearchOutcome.OK
    assert sequential.outcome is SearchOutcome.OK
    # Both should recover several of the six instances; neither collapses to a single box.
    assert abs(len(hough.matches) - len(sequential.matches)) <= 3
    assert len(sequential.matches) >= 2


def test_sequential_ransac_labels_the_exemplar_self_match_once() -> None:
    """The crop is part of the scene; its identity-transform instance is labelled, not dropped."""
    scene = _tiled_scene()
    result = search(scene, _EXEMPLAR, SparseGeoConfig(decomposition="sequential-ransac"))
    assert sum(m.is_exemplar for m in result.matches) == 1
