"""Tests for Method 2 (`sparse-geo`) -- NumPy per-peak RANSAC, degeneracy, and end-to-end search.

These pin the RANSAC and verification behaviours the research flagged:

- RANSAC is seeded from ``np.random.default_rng(config.seed)`` and is byte-identical for a given
  seed; a *different* seed changes the sampling (proving the seed is real, unlike
  ``cv2.setRNGSeed``, which has no effect on OpenCV RANSAC).
- Degeneracy rejection uses **scale plausibility** and **mirror rejection** (negative
  determinant), NOT shear/aspect (vacuous for a 4-DoF similarity).
- End to end: the low-keypoint guard fires (METHOD-04c), the exemplar self-match is labelled,
  and a 6-instance scene returns multiple distinct models (METHOD-12).
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt
import pytest

from object_search.schemas import BBox, ExemplarBox, SearchOutcome
from object_search.search import get_method, has_method
from object_search.search.sparse_geo import (
    SparseGeoConfig,
    _Instance,
    _is_degenerate,
    _model_from_complex,
    _model_to_box,
    _ransac_similarity,
    _suppress_overlapping_instances,
    search,
)

_SOURCE = Path(__file__).resolve().parents[1] / "src" / "object_search" / "search" / "sparse_geo.py"


# --------------------------------------------------------------------------- source contract


def test_registered_with_its_config() -> None:
    assert has_method("sparse-geo")
    spec = get_method("sparse-geo")
    assert spec.config_model is SparseGeoConfig
    assert spec.version == "1.0.0"


def test_source_carries_the_mandated_structure() -> None:
    text = _SOURCE.read_text(encoding="utf-8")
    assert "@register_method" in text
    assert "ROBUSTNESS BACKLOG" in text
    for step in range(1, 10):
        assert f"# {step}." in text, f"missing numbered step comment '# {step}.'"
    # The seed must be REAL: default_rng drives sampling, and cv2.setRNGSeed is never CALLED.
    assert "np.random.default_rng(config.seed)" in text
    assert "setRNGSeed(" not in text


# ----------------------------------------------------------------------- RANSAC helpers


def _apply_complex(
    a: complex, b: complex, pts: npt.NDArray[np.float64], *, reflect: bool = False
) -> npt.NDArray[np.float64]:
    out = []
    for x, y in pts:
        p = complex(x, y)
        q = a * (p.conjugate() if reflect else p) + b
        out.append([q.real, q.imag])
    return np.array(out, dtype=np.float64)


def test_ransac_is_seeded_deterministic_and_a_different_seed_changes_sampling() -> None:
    rng_pts = np.random.default_rng(1)
    src = rng_pts.uniform(0.0, 100.0, size=(20, 2))
    a = 1.2 * np.exp(1j * 0.3)  # scale 1.2, rotation ~17 degrees
    b = complex(30.0, -15.0)
    dst = _apply_complex(a, b, src)
    dst[15:] = rng_pts.uniform(0.0, 100.0, size=(5, 2))  # 5 gross outliers

    r0a = _ransac_similarity(src, dst, iters=200, thresh_px=2.0, rng=np.random.default_rng(0))
    r0b = _ransac_similarity(src, dst, iters=200, thresh_px=2.0, rng=np.random.default_rng(0))
    r1 = _ransac_similarity(src, dst, iters=200, thresh_px=2.0, rng=np.random.default_rng(1))

    # Same seed => byte-identical sampling AND model.
    assert r0a.sample_log == r0b.sample_log
    assert r0a.model is not None and r0b.model is not None
    assert np.array_equal(r0a.model.matrix, r0b.model.matrix)
    # Different seed => the sampling order differs (the seed genuinely controls sampling).
    assert r0a.sample_log != r1.sample_log

    # The clean similarity is recovered: >= 15 inliers, positive determinant, scale ~1.2.
    assert r0a.n_inliers >= 15
    assert r0a.model.det > 0.0
    assert r0a.model.scale == pytest.approx(1.2, abs=0.05)


def test_mirror_is_rejected_by_the_determinant_sign() -> None:
    rng = np.random.default_rng(2)
    src = rng.uniform(0.0, 100.0, size=(12, 2))
    a = 1.3 * np.exp(1j * 0.2)
    dst = _apply_complex(a, complex(10.0, 20.0), src, reflect=True)  # a reflection

    result = _ransac_similarity(src, dst, iters=200, thresh_px=1.0, rng=np.random.default_rng(0))
    assert result.model is not None
    assert result.model.det < 0.0, "mirrored data is best explained by the reflected model"
    degenerate, reason = _is_degenerate(result.model, min_scale=0.2, max_scale=5.0)
    assert degenerate and "mirror" in reason


def test_scale_plausibility_rejection() -> None:
    too_big = _model_from_complex(10.0 + 0j, complex(0.0, 0.0), reflect=False)  # scale 10
    degenerate, reason = _is_degenerate(too_big, min_scale=0.2, max_scale=5.0)
    assert degenerate and "scale" in reason

    plausible = _model_from_complex(1.0 + 0j, complex(0.0, 0.0), reflect=False)  # scale 1
    assert _is_degenerate(plausible, min_scale=0.2, max_scale=5.0)[0] is False


def test_nms_suppresses_a_duplicate_box_but_keeps_a_distinct_one() -> None:
    """Two peaks mapping to nearly the same box are one detection; a far box is a separate one.

    Hough de-duplicates in pose space, so a second peak with a slightly different fitted pose can
    still land on the same scene box -- a 1 TP + 1 FP precision leak (EVAL-16) that only the final
    IoU NMS removes. The stronger (more inliers) box survives; a spatially distinct instance is
    untouched.
    """
    model = _model_from_complex(1.0 + 0j, complex(0.0, 0.0), reflect=False)
    strong = _Instance(box=BBox(x=10, y=10, w=40, h=40), model=model, n_inliers=20, votes=20.0)
    near_dup = _Instance(box=BBox(x=12, y=11, w=40, h=40), model=model, n_inliers=8, votes=8.0)
    distinct = _Instance(box=BBox(x=200, y=200, w=40, h=40), model=model, n_inliers=12, votes=12.0)

    kept = _suppress_overlapping_instances([strong, near_dup, distinct], iou_threshold=0.4)

    kept_boxes = {inst.box.xyxy for inst in kept}
    assert strong.box.xyxy in kept_boxes, "the stronger of the overlapping pair must survive"
    assert near_dup.box.xyxy not in kept_boxes, "the weaker near-duplicate must be dropped"
    assert distinct.box.xyxy in kept_boxes, "a spatially distinct instance must be untouched"

    # nms_iou=1.0 is the escape hatch: nothing is ever suppressed (no box has IoU > 1.0).
    unfiltered = _suppress_overlapping_instances([strong, near_dup, distinct], iou_threshold=1.0)
    assert len(unfiltered) == 3


def test_identity_model_maps_the_box_to_itself() -> None:
    ex = BBox(x=10, y=20, w=30, h=40)
    identity = _model_from_complex(1.0 + 0j, complex(0.0, 0.0), reflect=False)
    box = _model_to_box(identity, ex, width=200, height=200)
    assert box is not None
    assert box.xyxy == ex.xyxy


# --------------------------------------------------------------------------- end-to-end


def _tiled_scene() -> tuple[npt.NDArray[np.uint8], list[tuple[int, int]]]:
    """Six identical, SIFT-rich tiles on a flat background -- a deliberate multi-instance scene."""
    rng = np.random.default_rng(3)
    small = rng.integers(0, 256, size=(10, 10), dtype=np.uint8)
    tile = cv2.resize(small, (64, 64), interpolation=cv2.INTER_CUBIC)  # low-freq blobs SIFT likes
    scene = np.full((360, 600), 128, dtype=np.uint8)
    positions = [(20, 20), (180, 25), (340, 30), (60, 200), (240, 210), (460, 200)]
    for x, y in positions:
        scene[y : y + 64, x : x + 64] = tile
    return np.ascontiguousarray(np.stack([scene] * 3, axis=-1)), positions


def test_low_keypoint_crop_abstains_with_a_note() -> None:
    flat = np.full((200, 300, 3), 128, np.uint8)  # a textureless scene => a low-keypoint crop
    exemplar = ExemplarBox(box=BBox(x=40, y=40, w=60, h=60))

    result = search(flat, exemplar, SparseGeoConfig())

    assert result.outcome is SearchOutcome.EMPTY
    assert result.matches == ()
    assert result.diagnostics.notes, "an abstention must say why (METHOD-04c)"
    assert "texture" in result.diagnostics.notes[0].lower()


def test_six_instance_scene_returns_multiple_distinct_models() -> None:
    scene, _positions = _tiled_scene()
    exemplar = ExemplarBox(box=BBox(x=20, y=20, w=64, h=64))

    result = search(scene, exemplar, SparseGeoConfig())

    assert result.outcome is SearchOutcome.OK
    # METHOD-12: multiple distinct models, never a single-best short-circuit.
    assert len(result.matches) >= 4
    distinct = {m.box.xyxy for m in result.matches}
    assert len(distinct) >= 4
    # The exemplar self-match is labelled exactly once, not dropped or double-counted.
    assert sum(m.is_exemplar for m in result.matches) == 1
    # Each match carries its fitted 2x3 similarity as a flattened 6-tuple.
    assert all(m.transform is not None and len(m.transform) == 6 for m in result.matches)


def test_search_is_byte_identical_across_runs_with_the_same_seed() -> None:
    scene, _ = _tiled_scene()
    exemplar = ExemplarBox(box=BBox(x=20, y=20, w=64, h=64))
    first = search(scene, exemplar, SparseGeoConfig())
    second = search(scene, exemplar, SparseGeoConfig())
    assert _boxes(first.matches) == _boxes(second.matches)
    assert [m.transform for m in first.matches] == [m.transform for m in second.matches]


def test_sequential_ransac_decomposition_also_recovers_instances() -> None:
    scene, _ = _tiled_scene()
    exemplar = ExemplarBox(box=BBox(x=20, y=20, w=64, h=64))
    result = search(scene, exemplar, SparseGeoConfig(decomposition="sequential-ransac"))
    assert result.outcome is SearchOutcome.OK
    assert len(result.matches) >= 2


def _boxes(matches: Iterable[object]) -> list[tuple[int, int, int, int]]:
    return [m.box.xyxy for m in matches]  # type: ignore[attr-defined]
