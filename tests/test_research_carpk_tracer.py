"""The CARPK research tracer, end to end and fully offline (EVAL-21/22/24).

Proves the whole vertical slice on the committed fixture with NO network and NO licence gate:
convert native CARPK annotations -> ``*.gt.json`` sidecar -> load through the single loader ->
run ``ncc`` at 1 exemplar through the benchmark research entry point -> one report row per image
carrying the full literature-metric column set. The real (licence-gated) CARPK data is a pending
user_setup step; this fixture stands in for it in the exact native format.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from object_search.eval.benchmark import run_research_benchmark
from object_search.eval.converters import convert_carpk
from object_search.eval.labels import load_research_ground_truth
from object_search.eval.splits import load_split_manifest, research_image_ids
from object_search.provenance import repo_root
from object_search.schemas.geometry import BBox

_FIXTURE_ROOT = repo_root() / "tests" / "fixtures" / "research" / "carpk"
_EXPECTED_BOX_COUNTS = {
    "carpk-fixture-01": 3,
    "carpk-fixture-02": 2,
    "carpk-fixture-03": 4,
}


def _convert_fixture(tmp_path: Path) -> Path:
    out_root = tmp_path / "carpk" / "test"
    sidecars = convert_carpk(_FIXTURE_ROOT, out_root)
    assert len(sidecars) == len(_EXPECTED_BOX_COUNTS)
    return out_root


# --------------------------------------------------------------------------- converter


def test_convert_carpk_box_counts_match_annotation_lines(tmp_path: Path) -> None:
    out_root = _convert_fixture(tmp_path)
    for image_id, expected in _EXPECTED_BOX_COUNTS.items():
        gt = load_research_ground_truth(out_root / f"{image_id}.gt.json")
        assert gt is not None
        # The box count equals the fixture annotation's line count.
        n_lines = len(
            (_FIXTURE_ROOT / "Annotations" / f"{image_id}.txt")
            .read_text(encoding="utf-8")
            .splitlines()
        )
        assert gt.achieved_count == expected == n_lines


def test_convert_carpk_boxes_round_trip_through_bbox(tmp_path: Path) -> None:
    out_root = _convert_fixture(tmp_path)
    # carpk-fixture-02 line 1 is "20 20 44 36 1": corner-inclusive -> half-open BBox(20,20,24,16).
    gt = load_research_ground_truth(out_root / "carpk-fixture-02.gt.json")
    assert gt is not None
    assert gt.boxes[0] == BBox(x=20, y=20, w=44 - 20, h=36 - 20)


def test_research_ground_truth_is_tagged_research_not_hand(tmp_path: Path) -> None:
    # Research sidecars must be tagged source="research", never the "hand" default, or the report
    # would mislabel their provenance.
    out_root = _convert_fixture(tmp_path)
    gt = load_research_ground_truth(out_root / "carpk-fixture-01.gt.json")
    assert gt is not None
    assert gt.source == "research"
    # The additive multi-exemplar field survives the round trip; the 1-exemplar box is the first.
    assert gt.effective_exemplar_indices == (0,)
    assert gt.exemplar_at(1).box == gt.boxes[0]


def test_load_research_ground_truth_missing_returns_none(tmp_path: Path) -> None:
    assert load_research_ground_truth(tmp_path / "nope.gt.json") is None


# --------------------------------------------------------------------------- split manifest


def test_committed_carpk_manifest_is_test_only() -> None:
    manifest = load_split_manifest("carpk")
    assert manifest.dataset == "carpk"
    assert manifest.val_strategy == "test-only"
    assert manifest.train == () and manifest.val == ()
    assert set(manifest.test) == set(_EXPECTED_BOX_COUNTS)
    # research_image_ids reads the same committed manifest the benchmark sweeps.
    assert set(research_image_ids("carpk", "test")) == set(_EXPECTED_BOX_COUNTS)


# --------------------------------------------------------------------------- end-to-end tracer


def test_ncc_carpk_1exemplar_test_produces_full_metric_rows(tmp_path: Path) -> None:
    out_root = _convert_fixture(tmp_path)
    report = run_research_benchmark("ncc", "carpk", "test", out_root, exemplar_count=1)

    assert report["method"] == "ncc"
    assert report["dataset"] == "carpk"
    assert report["split"] == "test"
    assert report["exemplar_count"] == 1

    # Exactly one row per fixture image, each carrying the localization column set.
    per_image = report["per_image"]
    assert len(per_image) == len(_EXPECTED_BOX_COUNTS)
    for row in per_image:
        for key in ("precision", "recall", "f1", "ap", "ap50", "ap75"):
            assert key in row, key
        assert row["dataset"] == "carpk"
        assert row["exemplar_count"] == 1
        assert row["true_count"] in set(_EXPECTED_BOX_COUNTS.values())

    # One pooled research row: P/R/F1 + AP/AP50/AP75 + MAE/RMSE/NAE (the tracer deliverable).
    overall = report["overall"]
    for key in ("precision", "recall", "f1", "ap", "ap50", "ap75", "mae", "rmse", "nae"):
        assert key in overall, key
    assert overall["n_images"] == len(_EXPECTED_BOX_COUNTS)
    # MAE/RMSE/NAE are real numbers here (every image is scored, not an error/abstention pool).
    assert overall["mae"] is not None
    assert overall["rmse"] == pytest.approx(overall["rmse"])  # not NaN
