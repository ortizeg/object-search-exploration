"""FSCD-147 / FSCD-LVIS / RPINE converters + the shared seeded val-carve helper (D-03/D-06/D-11).

Every converter must translate its native format into the **single** ``*.gt.json`` sidecar schema
(D-10), so each conversion is asserted by loading it back through
:func:`object_search.eval.labels.load_research_ground_truth` -- the one ground-truth reader. All
runs are offline on the committed fixtures.
"""

from __future__ import annotations

from pathlib import Path

from object_search.eval.converters import (
    convert_fscd147,
    convert_fscd_lvis,
    convert_rpine,
)
from object_search.eval.labels import load_research_ground_truth
from object_search.eval.splits import carve_val
from object_search.provenance import repo_root
from object_search.schemas.geometry import BBox

_RESEARCH = repo_root() / "tests" / "fixtures" / "research"


# --------------------------------------------------------------------------- FSCD-147 converter


def test_convert_fscd147_emits_loadable_sidecars_with_three_exemplars(tmp_path: Path) -> None:
    out_root = tmp_path / "fscd147" / "test"
    sidecars = convert_fscd147(_RESEARCH / "fscd147", out_root)
    assert sidecars, "at least one val/test sidecar written"

    for sidecar in sidecars:
        gt = load_research_ground_truth(sidecar)
        assert gt is not None
        assert gt.source == "research"
        # Boxes round-trip through the half-open BBox convention (first box == BBox(10,10,20,20)).
        assert all(isinstance(box, BBox) for box in gt.boxes)
        assert gt.boxes[0] == BBox(x=10, y=10, w=20, h=20)
        # The three native exemplar boxes survive as exemplar_indices; the first is the exemplar.
        assert len(gt.exemplar_indices) == 3
        assert gt.exemplar_indices[0] == gt.exemplar_index
        assert gt.exemplar_at(1).box == gt.boxes[gt.exemplar_indices[0]]


def test_convert_fscd147_box_count_matches_source(tmp_path: Path) -> None:
    out_root = tmp_path / "fscd147" / "test"
    convert_fscd147(_RESEARCH / "fscd147", out_root)
    gt = load_research_ground_truth(out_root / "fscd147-fixture-val-1.gt.json")
    assert gt is not None
    assert gt.achieved_count == 4  # the fixture annotates four objects per scored image


def test_convert_fscd147_skips_train_pseudo_boxes(tmp_path: Path) -> None:
    # Train images carry pseudo boxes (D-06) and are not in the annotations => no sidecar.
    out_root = tmp_path / "fscd147" / "test"
    convert_fscd147(_RESEARCH / "fscd147", out_root)
    assert not (out_root / "fscd147-fixture-train-1.gt.json").is_file()


# --------------------------------------------------------------------------- FSCD-LVIS converter


def test_convert_fscd_lvis_emits_only_target_category_boxes(tmp_path: Path) -> None:
    out_root = tmp_path / "fscd_lvis" / "test"
    sidecars = convert_fscd_lvis(_RESEARCH / "fscd_lvis", out_root, protocol="unseen")
    assert sidecars

    gt = load_research_ground_truth(out_root / "fscd-lvis-fixture-test-1.gt.json")
    assert gt is not None
    # The fixture has 4 target-category boxes + 2 distractor boxes; only the 4 targets are GT
    # (distractor rejection is the point -- returning a distractor is a false positive, not GT).
    assert gt.achieved_count == 4
    assert len(gt.exemplar_indices) == 3
    assert all(isinstance(box, BBox) for box in gt.boxes)


def test_convert_fscd_lvis_rejects_unsupported_protocol(tmp_path: Path) -> None:
    import pytest

    with pytest.raises(NotImplementedError, match="unseen"):
        convert_fscd_lvis(_RESEARCH / "fscd_lvis", tmp_path / "out", protocol="seen")


# --------------------------------------------------------------------------- RPINE converter


def test_convert_rpine_emits_loadable_sidecars(tmp_path: Path) -> None:
    out_root = tmp_path / "rpine" / "test"
    sidecars = convert_rpine(_RESEARCH / "rpine", out_root)
    assert sidecars

    gt = load_research_ground_truth(out_root / "rpine-fixture-test-1.gt.json")
    assert gt is not None
    assert gt.source == "research"
    assert gt.achieved_count == 4
    assert all(isinstance(box, BBox) for box in gt.boxes)
    # Seeded exemplar indices are in range and non-empty (up to three from the GT boxes).
    assert 1 <= len(gt.exemplar_indices) <= 3
    assert all(0 <= i < gt.achieved_count for i in gt.exemplar_indices)


def test_convert_rpine_exemplar_sampling_is_seed_stable(tmp_path: Path) -> None:
    # Same seed => byte-identical sidecars (D-11: np.random.default_rng, not process-salted hash).
    a = convert_rpine(_RESEARCH / "rpine", tmp_path / "a", seed=0)
    b = convert_rpine(_RESEARCH / "rpine", tmp_path / "b", seed=0)
    for sidecar_a, sidecar_b in zip(a, b, strict=True):
        assert sidecar_a.read_text(encoding="utf-8") == sidecar_b.read_text(encoding="utf-8")


# --------------------------------------------------------------------------- shared carve_val


def _many_ids(n: int) -> tuple[str, ...]:
    return tuple(f"img-{i:03d}" for i in range(n))


def test_carve_val_is_byte_identical_for_the_same_seed() -> None:
    train = _many_ids(50)
    first = carve_val(train, seed=7, val_fraction=0.2)
    second = carve_val(train, seed=7, val_fraction=0.2)
    assert first == second  # exact tuple equality, both elements


def test_carve_val_differs_for_a_different_seed() -> None:
    train = _many_ids(50)
    _, val_7 = carve_val(train, seed=7, val_fraction=0.2)
    _, val_8 = carve_val(train, seed=8, val_fraction=0.2)
    assert set(val_7) != set(val_8)


def test_carve_val_partitions_train_and_never_sees_test() -> None:
    train = _many_ids(50)
    test = tuple(f"test-{i:03d}" for i in range(9))  # disjoint list, never passed to carve_val
    remainder, val = carve_val(train, seed=3, val_fraction=0.2)
    # train is exactly partitioned into (remainder, val), disjoint, and test is untouched.
    assert set(remainder) | set(val) == set(train)
    assert set(remainder) & set(val) == set()
    assert set(val) & set(test) == set()


def test_carve_val_is_order_independent() -> None:
    forward = _many_ids(20)
    shuffled = tuple(reversed(forward))
    assert carve_val(forward, seed=1, val_fraction=0.25) == carve_val(
        shuffled, seed=1, val_fraction=0.25
    )


def test_carve_val_empty_train_is_empty() -> None:
    assert carve_val((), seed=1, val_fraction=0.2) == ((), ())
