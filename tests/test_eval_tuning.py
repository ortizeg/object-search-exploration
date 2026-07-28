"""Domain threshold tuning: argmax-F1 selection on val, tuned-vs-default report on test.

Drives :mod:`object_search.eval.tuning` on the committed floor-plan fixture with the model-free
``ncc`` method, so it runs offline with no ONNX weights. Asserts the tuning contract (every grid
entry is tried; the frozen config is the F1 argmax; a tuned and a default test block are both
produced) rather than specific metric values, which are meaningless on the tiny noise fixture.
"""

from __future__ import annotations

from pathlib import Path

from object_search.eval.converters import convert_floorplans
from object_search.eval.splits import NativeSplits, build_manifest, write_split_manifest
from object_search.eval.tuning import _TUNING_GRIDS, run_domain_tuning, tune_method
from object_search.provenance import repo_root

_FIXTURE_ROOT = repo_root() / "tests" / "fixtures" / "research" / "floorplans"
_VAL_IDS = ("fp-val-1", "fp-val-2")
_TEST_IDS = ("fp-test-1", "fp-test-2")


def _stage(tmp_path: Path) -> Path:
    """Convert the fixture val+test door splits under tmp and write a matching manifest.

    Returns the research-root base (containing ``floorplans-door/{val,test}/``).
    """
    for split, our in (("valid", "val"), ("test", "test")):
        convert_floorplans(
            _FIXTURE_ROOT / split, tmp_path / "floorplans-door" / our, target_class="door"
        )
    manifest = build_manifest(
        "floorplans-door", NativeSplits(train=(), val=_VAL_IDS, test=_TEST_IDS)
    )
    write_split_manifest(manifest, root=tmp_path)
    return tmp_path


def test_tune_method_selects_the_f1_argmax_over_the_grid(tmp_path: Path) -> None:
    base = _stage(tmp_path)
    # A tiny, fast grid (single scale/angle) with two operating points.
    grid = [
        {"scales": (1.0,), "angles_deg": (0.0,), "retain_frac": 0.30},
        {"scales": (1.0,), "angles_deg": (0.0,), "retain_frac": 0.60},
    ]
    result = tune_method(
        "ncc", "floorplans-door", base / "floorplans-door" / "val", grid=grid, manifest_root=base
    )

    # Every grid entry is tried, and best is one of them.
    assert len(result["trials"]) == len(grid)
    assert result["best"]["overrides"] in grid

    # best is the F1 argmax (None F1 -- nothing scored -- treated as below any real F1).
    def key(f1: float | None) -> float:
        return float(f1) if isinstance(f1, int | float) else -1.0

    best_key = key(result["best"]["f1"])
    assert all(best_key >= key(t["f1"]) for t in result["trials"])


def test_run_domain_tuning_reports_tuned_and_default_on_test(tmp_path: Path) -> None:
    base = _stage(tmp_path)
    report = run_domain_tuning(
        "floorplans-door", base, methods=("ncc",), manifest_root=base, out=None
    )

    assert report["dataset"] == "floorplans-door"
    assert report["selection_metric"] == "f1@iou0.5"
    (entry,) = report["methods"]
    assert entry["method"] == "ncc"
    # The frozen overrides came from the method's real grid.
    assert entry["tuned_overrides"] in [dict(o) for o in _TUNING_GRIDS["ncc"]]
    # Both a tuned and a default test block, each carrying the full literature metric set.
    for block in (entry["tuned_test"], entry["default_test"]):
        for metric_key in ("precision", "recall", "f1", "ap50", "mae"):
            assert metric_key in block
    assert "delta_f1" in entry
    assert "val_f1" in entry


def test_run_domain_tuning_writes_report_when_out_set(tmp_path: Path) -> None:
    base = _stage(tmp_path)
    out = tmp_path / "tuning.json"
    run_domain_tuning("floorplans-door", base, methods=("ncc",), manifest_root=base, out=str(out))
    assert out.is_file()
