"""Tests for the committed benchmark charts (EVAL-06).

The load-bearing test is determinism: the charts are committed to git, so re-rendering must be
**byte-identical** or every regeneration churns the repo. The rest exercises the honest
``None``-vs-zero handling (abstention renders as ``n/a`` in the tables, not ``0``) and the
thumbs empty-state panel.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from object_search.eval import charts

# A tiny but structurally complete results document: two methods, both scale buckets, two canvas
# sizes, one abstaining method -- enough to exercise every rendering branch.
_RESULTS: dict = {
    "git_sha": "deadbeef",
    "ci_subset": False,
    "iou_threshold": 0.5,
    "ap_convention": "all-point interpolation (COCO-style)",
    "coverage": {"images_labelled": 2, "images_requested": 2, "images_unlabelled": []},
    "methods": {
        "ncc": {
            "overall": {
                "precision": 0.9,
                "recall": 0.95,
                "f1": 0.924,
                "mean_ap": 0.5,
                "n_abstentions": 0,
                "n_errors": 0,
                "latency_ms": {"p50": 100.0, "mean": 120.0, "max": 200.0},
            },
            "slices": {
                "by_scale_bucket": {
                    "fixed": {"recall": 0.99, "precision": 0.95, "mean_ap": 0.5},
                    "varied": {"recall": 0.30, "precision": 0.43, "mean_ap": 0.15},
                },
                "by_canvas_size": {
                    "320x240": {"latency_ms": {"p50": 20.0}},
                    "800x600": {"latency_ms": {"p50": 75.0}},
                },
                "by_symbol_size": {
                    "small": {"n_gt": 4, "n_matched": 3, "recall": 0.75},
                    "medium": {"n_gt": 2, "n_matched": 2, "recall": 1.0},
                    "large": {"n_gt": 0, "n_matched": 0, "recall": None},
                },
            },
            # One row per regime (EASY/TEXTURED/VARIED/CLUTTERED) so the per-regime section has
            # something to pool, plus a real-objects row that must be EXCLUDED from that section.
            "per_image": [
                {"image_id": "chipset-01", "tp": 5, "fp": 0, "fn": 0, "ap": 1.0},
                {"image_id": "textured-plain-01", "tp": 4, "fp": 1, "fn": 0, "ap": 0.9},
                {"image_id": "textured-varied-01", "tp": 2, "fp": 1, "fn": 2, "ap": 0.4},
                {"image_id": "textured-cluttered-01", "tp": 3, "fp": 2, "fn": 1, "ap": 0.6},
                {"image_id": "real-plain-apple", "tp": 6, "fp": 0, "fn": 0, "ap": 1.0},
            ],
        },
        "sparse-geo": {
            "overall": {
                "precision": 0.83,
                "recall": 0.10,
                "f1": 0.17,
                "mean_ap": 0.08,
                "n_abstentions": 11,
                "n_errors": 0,
                "latency_ms": {"p50": 76.0, "mean": 80.0, "max": 233.0},
            },
            "slices": {
                "by_scale_bucket": {
                    "fixed": {"recall": 0.10, "precision": 0.83, "mean_ap": 0.09},
                    "varied": {"recall": 0.0, "precision": None, "mean_ap": 0.0},
                },
                "by_canvas_size": {
                    "320x240": {"latency_ms": {"p50": 5.0}},
                    "800x600": {"latency_ms": {"p50": 25.0}},
                },
                "by_symbol_size": {
                    "small": {"n_gt": 4, "n_matched": 0, "recall": 0.0},
                    "medium": {"n_gt": 2, "n_matched": 1, "recall": 0.5},
                    "large": {"n_gt": 0, "n_matched": 0, "recall": None},
                },
            },
            "per_image": [
                {"image_id": "chipset-01", "tp": None, "fp": None, "fn": None, "ap": None},
                {"image_id": "textured-plain-01", "tp": 1, "fp": 3, "fn": 4, "ap": 0.1},
                {"image_id": "textured-varied-01", "tp": 0, "fp": 0, "fn": 5, "ap": 0.0},
                {"image_id": "textured-cluttered-01", "tp": 1, "fp": 1, "fn": 5, "ap": 0.05},
            ],
        },
    },
}


@pytest.fixture
def results_file(tmp_path: Path) -> Path:
    """Write the sample results document to a temp file."""
    path = tmp_path / "results.json"
    path.write_text(json.dumps(_RESULTS), encoding="utf-8")
    return path


def test_render_all_writes_every_artifact(results_file: Path, tmp_path: Path) -> None:
    """All four PNGs and the Markdown table land on disk."""
    out = tmp_path / "out"
    paths = charts.render_all(results_file, out, ratings=None)
    names = {p.name for p in paths}
    assert names == {
        "metrics_by_method.png",
        "crossover_by_scale.png",
        "latency_by_canvas.png",
        "thumbs_wilson.png",
        "results.md",
    }
    for path in paths:
        assert path.is_file()
        assert path.stat().st_size > 0


def test_charts_are_byte_identical_on_rerender(results_file: Path, tmp_path: Path) -> None:
    """Two renders into separate dirs produce byte-identical PNGs (EVAL-06 determinism)."""
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    paths_a = charts.render_all(results_file, out_a, ratings=None)
    paths_b = charts.render_all(results_file, out_b, ratings=None)
    for pa, pb in zip(sorted(paths_a), sorted(paths_b), strict=True):
        assert pa.read_bytes() == pb.read_bytes(), f"{pa.name} is not reproducible"


def test_latency_chart_exists_and_is_nonempty(results_file: Path, tmp_path: Path) -> None:
    """The latency-by-canvas-size chart (the EVAL-19 scaling story) is rendered."""
    out = tmp_path / "out"
    results = charts.load_results(results_file)
    path = charts.render_latency_chart(results, out)
    assert path.name == "latency_by_canvas.png"
    assert path.stat().st_size > 0


def test_results_md_prints_abstention_as_na(results_file: Path, tmp_path: Path) -> None:
    """A ``None`` precision (abstention) is written ``n/a`` in the table, never ``0``."""
    out = tmp_path / "out"
    results = charts.load_results(results_file)
    md_path = charts.write_results_markdown(results, out)
    text = md_path.read_text(encoding="utf-8")
    # sparse-geo's varied-scale precision is None -> the crossover table must show n/a there.
    assert "n/a" in text
    assert "| `ncc` |" in text
    assert "| `sparse-geo` |" in text


def test_results_md_has_per_regime_and_size_tables(results_file: Path, tmp_path: Path) -> None:
    """The expanded results.md has per-regime scoreboards, a size-bucket table, and Insight."""
    out = tmp_path / "out"
    results = charts.load_results(results_file)
    md_path = charts.write_results_markdown(results, out)
    text = md_path.read_text(encoding="utf-8")

    for heading in (
        "## Results by regime",
        "### EASY",
        "### TEXTURED",
        "### VARIED",
        "### CLUTTERED",
    ):
        assert heading in text
    assert "## Recall by ground-truth box size" in text
    assert "## Insight" in text
    # Computed, not fixed: ncc wins every regime in the fixture, so the "unusual" branch fires.
    assert "wins every regime here" in text


def test_results_md_regime_section_excludes_real_objects(results_file: Path) -> None:
    """A real-objects row must never be pooled into the synthetic per-regime EASY table."""
    results = charts.load_results(results_file)
    grouped = charts._rows_by_regime(results, "ncc")
    easy_ids = {r["image_id"] for r in grouped.get("EASY", [])}
    assert easy_ids == {"chipset-01"}
    all_grouped_ids = {r["image_id"] for rows in grouped.values() for r in rows}
    assert "real-plain-apple" not in all_grouped_ids


def test_thumbs_empty_state_when_no_ratings(tmp_path: Path) -> None:
    """With no ratings the thumbs chart renders the honest empty-state panel, not a bar."""
    out = tmp_path / "out"
    path_none = charts.render_thumbs_chart(None, out / "none")
    path_zero = charts.render_thumbs_chart({"ncc": (0, 0)}, out / "zero")
    assert path_none.stat().st_size > 0
    # An all-zero-n mapping is treated identically to None: byte-for-byte the empty-state panel.
    assert path_none.read_bytes() == path_zero.read_bytes()


def test_thumbs_chart_with_ratings_renders_bars(tmp_path: Path) -> None:
    """When ratings exist the thumbs chart renders (Wilson interval path is exercised)."""
    out = tmp_path / "out"
    path = charts.render_thumbs_chart({"ncc": (8, 10), "sparse-geo": (1, 4)}, out)
    assert path.stat().st_size > 0


def test_missing_results_raises(tmp_path: Path) -> None:
    """A missing results file is a loud error, not an empty figure."""
    with pytest.raises(FileNotFoundError):
        charts.load_results(tmp_path / "nope.json")
