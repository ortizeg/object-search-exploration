"""The research report renderer (DOC-07 numbers) and the survey doc (DOC-07 prose).

``scripts/build_research_report.py`` is not an importable package, so it is loaded by path here --
the same way it is invoked as a script. The tests prove the report emits every literature column and
one row per cell (abstentions as n/a, never 0), guards a missing results file with an actionable
message, and that ``docs/eval/research-datasets.md`` documents every dataset, metric, and the
tune-on-val → report-on-test protocol.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

from object_search.provenance import repo_root


def _load_report_module() -> ModuleType:
    path = repo_root() / "scripts" / "build_research_report.py"
    spec = importlib.util.spec_from_file_location("build_research_report", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fixture_results() -> dict[str, object]:
    """A minimal research-results mapping with two cells (one abstaining) covering every column."""
    return {
        "git_sha": "abc1234def",
        "seed": 0,
        "iou_threshold": 0.5,
        "fusion": "k-shot late fusion",
        "cells": [
            {
                "method": "ncc",
                "dataset": "rpine",
                "split": "test",
                "exemplar_count": 1,
                "overall": {
                    "precision": 0.8,
                    "recall": 0.6,
                    "f1": 0.68,
                    "ap": 0.5,
                    "ap50": 0.7,
                    "ap75": 0.4,
                    "mae": 1.2,
                    "rmse": 1.5,
                    "nae": 0.3,
                },
            },
            {
                "method": "ncc",
                "dataset": "rpine",
                "split": "test",
                "exemplar_count": 3,
                "overall": {
                    "precision": None,  # abstention -> must render n/a, never 0
                    "recall": None,
                    "f1": None,
                    "ap": 0.55,
                    "ap50": 0.75,
                    "ap75": 0.45,
                    "mae": 1.0,
                    "rmse": 1.3,
                    "nae": 0.25,
                },
            },
        ],
    }


def test_report_has_every_column_and_one_row_per_cell(tmp_path: Path) -> None:
    module = _load_report_module()
    results_path = tmp_path / "research-results.json"
    results_path.write_text(json.dumps(_fixture_results()), encoding="utf-8")
    out_path = tmp_path / "research-report.html"

    html = module.build_research_report(results_path, out_path)

    assert out_path.is_file()
    # Every literature column header is present.
    for header in ("P", "R", "F1", "AP", "AP50", "AP75", "MAE", "RMSE", "NAE"):
        assert f"<th>{header}</th>" in html, header
    # One <tr> per cell in the tbody (plus the one header row in the thead).
    assert html.count("<tr>") == 1 + 2  # header row + two data rows
    # The 3-exemplar fusion is named so the reader knows how those numbers were produced.
    assert "k-shot late fusion" in html
    # Abstention renders as n/a, never as 0%.
    assert "n/a" in html


def test_report_missing_results_is_actionable(tmp_path: Path) -> None:
    module = _load_report_module()
    with pytest.raises(FileNotFoundError) as excinfo:
        module.build_research_report(tmp_path / "absent.json", tmp_path / "out.html")
    message = str(excinfo.value)
    assert "fetch-datasets" in message  # tells the user how to produce the data
    assert "gitignored" in message


# --------------------------------------------------------------------------- DOC-07 survey


def test_doc07_documents_every_dataset_metric_and_protocol() -> None:
    doc = (repo_root() / "docs" / "eval" / "research-datasets.md").read_text(encoding="utf-8")

    # All four datasets named.
    for name in ("RPINE", "FSCD-147", "FSCD-LVIS", "CARPK", "PUCPR+"):
        assert name in doc, name
    # A source link per dataset (five distinct http(s) references at least).
    assert doc.count("https://") >= 5
    # Metric definitions present.
    for metric in ("MAE", "RMSE", "NAE", "AP50", "AP75"):
        assert metric in doc, metric
    # The 3-exemplar fusion is named.
    assert "k-shot late fusion" in doc
    # The protocol is stated.
    assert "tune on val" in doc.lower() or "tune-on-val" in doc.lower()
    assert "report on test" in doc.lower() or "report-on-test" in doc.lower()
    # CARPK/PUCPR+ test-only rule is stated.
    assert "test-only" in doc.lower()
