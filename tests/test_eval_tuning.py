"""Domain threshold tuning: argmax-F1 selection on val, tuned-vs-default report on test.

Drives :mod:`object_search.eval.tuning` on the committed floor-plan fixture with the model-free
``ncc`` method, so it runs offline with no ONNX weights. Asserts the tuning contract (every grid
entry is tried; the frozen config is the F1 argmax; a tuned and a default test block are both
produced) rather than specific metric values, which are meaningless on the tiny noise fixture.
"""

from __future__ import annotations

from pathlib import Path

from object_search.eval.converters import convert_floorplans
from object_search.eval.labels import GroundTruth
from object_search.eval.sampling import sample_exemplars
from object_search.eval.splits import NativeSplits, build_manifest, write_split_manifest
from object_search.eval.tuning import (
    _TUNING_GRIDS,
    run_domain_tuning,
    tune_method,
)
from object_search.provenance import repo_root
from object_search.schemas.geometry import BBox

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


# ------------------------------------------------ broadened multi-knob grids


def test_broadened_multi_knob_grid_runs_and_selects_an_in_grid_argmax(tmp_path: Path) -> None:
    base = _stage(tmp_path)
    # A multi-knob grid mirroring the shape of the broadened ncc grid (scales + retain + nms).
    grid = [
        {"scales": (1.0,), "retain_frac": 0.30, "nms_iou": 0.3},
        {"scales": (0.9, 1.0, 1.1), "retain_frac": 0.60, "nms_iou": 0.5},
    ]
    result = tune_method(
        "ncc", "floorplans-door", base / "floorplans-door" / "val", grid=grid, manifest_root=base
    )
    assert len(result["trials"]) == len(grid)
    # The selected best is one of the multi-key entries (validated through NCCConfig(**overrides)).
    assert result["best"]["overrides"] in grid

    def key(f1: float | None) -> float:
        return float(f1) if isinstance(f1, int | float) else -1.0

    best_key = key(result["best"]["f1"])
    assert all(best_key >= key(t["f1"]) for t in result["trials"])


def test_real_ncc_grid_is_multi_knob() -> None:
    # The committed ncc grid sweeps three knobs (scales + retain_frac + nms_iou) in its base
    # block, PLUS an additive cardinal-rotation-bank x mirror block (angles_deg + mirror on top of
    # scales/retain_frac/nms_iou) from the floor-plan domain investigation (260730-vx4) -- every
    # entry is one or the other, never a mix of unrelated keys.
    base_keys = {"scales", "retain_frac", "nms_iou"}
    cardinal_keys = base_keys | {"angles_deg", "mirror"}
    for overrides in _TUNING_GRIDS["ncc"]:
        assert set(overrides) in (base_keys, cardinal_keys)
    # The additive cardinal block is present (not accidentally dropped).
    assert any(set(overrides) == cardinal_keys for overrides in _TUNING_GRIDS["ncc"])
    # sparse-geo / propose-retrieve / owlv2 each pair a primary knob with a second one.
    for overrides in _TUNING_GRIDS["sparse-geo"]:
        assert set(overrides) == {"min_inliers", "nms_iou"}
    for overrides in _TUNING_GRIDS["propose-retrieve"]:
        assert set(overrides) == {"similarity_floor", "nms_iou"}
    for overrides in _TUNING_GRIDS["owlv2-oneshot"]:
        assert set(overrides) == {"max_box_area_frac", "query_iou_frac"}


# ------------------------------------------------ per-method grid override (grids=)


def test_ncc_and_mosse_grids_are_independent_objects() -> None:
    """The two correlation grids started aliased and must not be the same object.

    Both config models are ``extra="forbid"``, so a method-only key added to a shared grid would
    be fed into the other method's validator and raise -- this pins the split. ``ncc``'s grid is
    now a strict superset of ``mosse``'s (the base scales x retain_frac x nms_iou block is
    byte-identical, plus ``ncc``'s additive cardinal-rotation-bank x mirror block from the
    floor-plan investigation, which has no ``mosse`` equivalent -- ``mosse``'s own floor-plan
    grid entries are added independently by a sibling quick task).
    """
    ncc_grid, mosse_grid = _TUNING_GRIDS["ncc"], _TUNING_GRIDS["mosse"]
    assert ncc_grid is not mosse_grid
    assert len(ncc_grid) > len(mosse_grid)
    # The shared base block (mosse's whole grid) is byte-identical and present verbatim in ncc's.
    base = [dict(o) for o in mosse_grid]
    assert [dict(o) for o in ncc_grid[: len(mosse_grid)]] == base
    # ...and no individual override dict in the shared prefix is aliased either.
    assert all(a is not b for a, b in zip(ncc_grid[: len(mosse_grid)], mosse_grid, strict=True))


def test_grids_override_replaces_the_committed_grid(tmp_path: Path) -> None:
    base = _stage(tmp_path)
    override = [
        {"scales": (1.0,), "angles_deg": (0.0,), "retain_frac": 0.30},
        {"scales": (1.0,), "angles_deg": (0.0, 90.0), "retain_frac": 0.60},
    ]
    report = run_domain_tuning(
        "floorplans-door",
        base,
        methods=("ncc",),
        manifest_root=base,
        grids={"ncc": override},
        out=None,
    )
    (entry,) = report["methods"]
    # Exactly the supplied grid was swept -- the committed 20-entry grid was NOT used.
    assert [t["overrides"] for t in entry["trials"]] == [dict(o) for o in override]
    assert entry["tuned_overrides"] in [dict(o) for o in override]


def _without_latency(value: object) -> object:
    """Strip the wall-clock ``latency_ms`` blocks so two reports can be compared for equality.

    Everything else in the report is deterministic (fixed grids, seeded sampler, committed
    manifest); latency is the one field that legitimately differs between two identical runs.
    """
    if isinstance(value, dict):
        return {k: _without_latency(v) for k, v in value.items() if k != "latency_ms"}
    if isinstance(value, list):
        return [_without_latency(v) for v in value]
    return value


def test_grids_omitted_or_missing_a_method_falls_back_to_the_committed_grid(
    tmp_path: Path,
) -> None:
    """Regression pin: the default path is unchanged, with or without an unrelated ``grids`` key."""
    base = _stage(tmp_path)
    without = run_domain_tuning(
        "floorplans-door", base, methods=("ncc",), manifest_root=base, out=None
    )
    # `grids` given but with no entry for ncc -> same fallback to _TUNING_GRIDS["ncc"].
    with_empty = run_domain_tuning(
        "floorplans-door",
        base,
        methods=("ncc",),
        manifest_root=base,
        grids={"mosse": [{"retain_frac": 0.5}]},
        out=None,
    )
    assert _without_latency(without) == _without_latency(with_empty)
    (entry,) = without["methods"]
    assert len(entry["trials"]) == len(_TUNING_GRIDS["ncc"])


# ------------------------------------------------ size-representative exemplar selection


def _gt_with_areas() -> GroundTruth:
    # Areas 4, 100, 900; the median is 100, so the size-representative pick is box index 1.
    return GroundTruth(
        image_id="sizes",
        boxes=(
            BBox(x=0, y=0, w=2, h=2),  # area 4
            BBox(x=0, y=0, w=10, h=10),  # area 100 (the median)
            BBox(x=0, y=0, w=30, h=30),  # area 900
        ),
        exemplar_index=0,
        source="research",
    )


def test_size_representative_selection_picks_the_median_area_box() -> None:
    gt = _gt_with_areas()
    chosen = sample_exemplars(gt, count=1, seed=0, exemplar_selection="size-representative")
    assert chosen[0].box.area == 100  # the median-area box, not the exemplar_index=0 (area 4) box


def test_size_representative_selection_is_deterministic_and_seed_independent() -> None:
    gt = _gt_with_areas()
    a = sample_exemplars(gt, count=3, seed=0, exemplar_selection="size-representative")
    b = sample_exemplars(gt, count=3, seed=999, exemplar_selection="size-representative")
    # No RNG in this mode: the order is byte-identical regardless of seed.
    assert tuple(e.box.area for e in a) == tuple(e.box.area for e in b)
    # Ordered by closeness to the median area: 100 (delta 0), 4 (delta 96), 900 (delta 800).
    assert tuple(e.box.area for e in a) == (100, 4, 900)


def test_seeded_random_default_is_unchanged_by_the_new_option() -> None:
    gt = _gt_with_areas()
    # The default keyword and an explicit "seeded-random" must produce identical draws.
    default = sample_exemplars(gt, count=2, seed=7)
    explicit = sample_exemplars(gt, count=2, seed=7, exemplar_selection="seeded-random")
    assert tuple(e.box.area for e in default) == tuple(e.box.area for e in explicit)


# ------------------------------------------------ exemplar-count operating points


def test_run_domain_tuning_single_count_keeps_the_flat_shape(tmp_path: Path) -> None:
    base = _stage(tmp_path)
    report = run_domain_tuning(
        "floorplans-door", base, methods=("ncc",), manifest_root=base, out=None
    )
    # Default (exemplar_counts=None) -> committed flat shape: top-level exemplar_count + methods.
    assert report["exemplar_count"] == 1
    assert "methods" in report
    assert "per_count" not in report


def test_run_domain_tuning_multiple_counts_nest_one_block_per_count(tmp_path: Path) -> None:
    base = _stage(tmp_path)
    report = run_domain_tuning(
        "floorplans-door",
        base,
        methods=("ncc",),
        manifest_root=base,
        exemplar_counts=(1, 3),
        out=None,
    )
    assert report["exemplar_counts"] == [1, 3]
    per_count = report["per_count"]
    assert [block["exemplar_count"] for block in per_count] == [1, 3]
    for block in per_count:
        (entry,) = block["methods"]
        assert entry["method"] == "ncc"
        assert "tuned_test" in entry and "default_test" in entry
    # The flat single-count keys are absent in the nested shape.
    assert "methods" not in report
