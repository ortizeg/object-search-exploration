"""Paired comparison (EVAL-05): one exemplar box through several methods, winners into the DB.

Uses the two model-free methods (ncc + classical sparse-geo) on a real chipset image so it needs
no ONNX weights. On the fixed-scale chipset ncc finds the chips while sparse-geo abstains (too few
keypoints), which is a decisive, deterministic paired outcome to assert on.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt
import pytest

from object_search.eval.labels import load_ground_truth, scene_path
from object_search.eval.paired import run_paired
from object_search.schemas.geometry import BBox, ExemplarBox
from object_search.store.db import open_store


@pytest.fixture
def store(tmp_path: Path) -> sqlite3.Connection:
    return open_store(tmp_path / "runs.db")


def _chipset_scene() -> tuple[npt.NDArray[np.uint8], ExemplarBox]:
    path = scene_path("chipset-01")
    assert path is not None
    image = np.asarray(cv2.imread(str(path), cv2.IMREAD_COLOR), dtype=np.uint8)
    gt = load_ground_truth("chipset-01")
    assert gt is not None
    return image, gt.exemplar


def test_run_paired_runs_all_methods_on_one_box(store: sqlite3.Connection) -> None:
    image, exemplar = _chipset_scene()
    result = run_paired(store, image, "chipset-01", exemplar, methods=["ncc", "sparse-geo"])
    assert result.gt_available is True
    # Both methods scored on the SAME exemplar box.
    assert {s.method for s in result.scores} == {"ncc", "sparse-geo"}
    # One unordered pair -> one comparison.
    assert len(result.comparisons) == 1


def test_run_paired_records_winner_into_table(store: sqlite3.Connection) -> None:
    image, exemplar = _chipset_scene()
    run_paired(store, image, "chipset-01", exemplar, methods=["ncc", "sparse-geo"])

    rows = store.execute(
        "SELECT method_a, method_b, winner, image_id FROM paired_comparisons"
    ).fetchall()
    assert len(rows) == 1
    row = rows[0]
    assert row["image_id"] == "chipset-01"
    # Alphabetical ordering: method_a='ncc', method_b='sparse-geo'.
    assert row["method_a"] == "ncc"
    assert row["method_b"] == "sparse-geo"
    # ncc finds the chips; sparse-geo abstains on tiny low-keypoint chips -> 'a' wins.
    assert row["winner"] == "a"


def test_run_paired_without_ground_truth_records_nothing(store: sqlite3.Connection) -> None:
    # A scene with no sidecar: methods still run, but no objective winner is invented (EVAL-02).
    image = np.full((240, 320, 3), 255, dtype=np.uint8)
    exemplar = ExemplarBox(box=BBox(x=10, y=10, w=20, h=20))
    result = run_paired(store, image, "no-such-image", exemplar, methods=["ncc", "sparse-geo"])
    assert result.gt_available is False
    assert result.comparisons == ()
    count = store.execute("SELECT COUNT(*) AS n FROM paired_comparisons").fetchone()["n"]
    assert count == 0


def test_run_paired_stores_ties_as_a_distinct_outcome(store: sqlite3.Connection) -> None:
    # A blank canvas carrying chipset-01's (real) ground truth: the chips are absent, so BOTH
    # methods find nothing and score F1 -> 0.0. Equal scores are a tie, stored as the distinct
    # 'tie' outcome so the half-a-win modelling choice can be revisited later.
    gt = load_ground_truth("chipset-01")
    assert gt is not None
    blank = np.full((gt.height or 240, gt.width or 320, 3), 255, dtype=np.uint8)
    result = run_paired(store, blank, "chipset-01", gt.exemplar, methods=["ncc", "sparse-geo"])

    assert {c.winner for c in result.comparisons} == {"tie"}
    row = store.execute("SELECT winner FROM paired_comparisons").fetchone()
    assert row["winner"] == "tie"
