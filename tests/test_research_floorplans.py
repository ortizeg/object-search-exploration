"""The floor-plan research slice, end to end and fully offline (target-domain eval).

Proves the whole vertical slice on a committed tiny COCO fixture with NO network: convert a
class-filtered floor-plan COCO split -> ``*.gt.json`` -> load through the single loader -> fetch via
the registry -> run ``ncc`` through the research benchmark. The real Roboflow floor-plans-500 export
is a manual drop (``datasets/_incoming/floorplans/``); this fixture stands in for it in the same
COCO shape (door / window / stairs, where stairs is the distractor class the single-class converter
must drop).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from object_search.eval import datasets as dataset_registry
from object_search.eval.benchmark import run_research_benchmark
from object_search.eval.converters import convert_floorplans
from object_search.eval.labels import load_research_ground_truth
from object_search.eval.splits import (
    NativeSplits,
    build_manifest,
    load_split_manifest,
    write_split_manifest,
)
from object_search.provenance import repo_root
from object_search.schemas.geometry import BBox

_FIXTURE_ROOT = repo_root() / "tests" / "fixtures" / "research" / "floorplans"

# Per-image instance counts in the fixture, per class (stairs is the drop-me distractor).
_DOOR_COUNTS = {"valid": {"fp-val-1": 3, "fp-val-2": 2}, "test": {"fp-test-1": 2, "fp-test-2": 1}}
_WINDOW_COUNTS = {"valid": {"fp-val-1": 2, "fp-val-2": 1}, "test": {"fp-test-1": 2, "fp-test-2": 3}}


# --------------------------------------------------------------------------- converter


def test_convert_floorplans_filters_to_target_class(tmp_path: Path) -> None:
    out = tmp_path / "floorplans-door" / "val"
    sidecars = convert_floorplans(_FIXTURE_ROOT / "valid", out, target_class="door")
    assert len(sidecars) == len(_DOOR_COUNTS["valid"])
    for image_id, expected in _DOOR_COUNTS["valid"].items():
        gt = load_research_ground_truth(out / f"{image_id}.gt.json")
        assert gt is not None
        # Exactly the door boxes -- the window and stairs boxes on the same plan are dropped.
        assert gt.achieved_count == expected


def test_convert_floorplans_window_is_independent_of_door(tmp_path: Path) -> None:
    out = tmp_path / "floorplans-window" / "test"
    convert_floorplans(_FIXTURE_ROOT / "test", out, target_class="window")
    for image_id, expected in _WINDOW_COUNTS["test"].items():
        gt = load_research_ground_truth(out / f"{image_id}.gt.json")
        assert gt is not None
        assert gt.achieved_count == expected


def test_convert_floorplans_coco_bbox_round_trips_through_bbox(tmp_path: Path) -> None:
    out = tmp_path / "door"
    convert_floorplans(_FIXTURE_ROOT / "valid", out, target_class="door")
    gt = load_research_ground_truth(out / "fp-val-1.gt.json")
    assert gt is not None
    # fp-val-1's first door annotation is COCO bbox [4,4,10,8] (xywh) -> BBox(4,4,10,8) directly.
    assert gt.boxes[0] == BBox(x=4, y=4, w=10, h=8)
    assert gt.source == "research"


def test_convert_floorplans_skips_images_without_the_class(tmp_path: Path) -> None:
    # stairs exists only on fp-val-1, so a stairs conversion of the val split yields ONE sidecar:
    # an image with zero target-class boxes is skipped, never written empty.
    out = tmp_path / "stairs"
    sidecars = convert_floorplans(_FIXTURE_ROOT / "valid", out, target_class="stairs")
    assert [p.name for p in sidecars] == ["fp-val-1.gt.json"]


def test_convert_floorplans_exemplar_indices_are_seeded_and_stable(tmp_path: Path) -> None:
    first = convert_floorplans(_FIXTURE_ROOT / "valid", tmp_path / "a", target_class="door")
    second = convert_floorplans(_FIXTURE_ROOT / "valid", tmp_path / "b", target_class="door")
    gt_a = load_research_ground_truth(first[0])
    gt_b = load_research_ground_truth(second[0])
    assert gt_a is not None and gt_b is not None
    # Deterministic across conversions (seeded np.random.default_rng, D-11); indices in range.
    assert gt_a.exemplar_indices == gt_b.exemplar_indices
    assert gt_a.effective_exemplar_indices == gt_a.exemplar_indices
    assert all(0 <= i < gt_a.achieved_count for i in gt_a.exemplar_indices)


def test_convert_floorplans_unknown_class_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not among categories"):
        convert_floorplans(_FIXTURE_ROOT / "valid", tmp_path / "x", target_class="elevator")


# --------------------------------------------------------------------------- registry fetch


def test_fetch_floorplans_door_from_incoming_drop(tmp_path: Path) -> None:
    # Stage the fixture as a human-supplied drop under <root>/datasets/_incoming/floorplans/.
    incoming = tmp_path / "datasets" / "_incoming" / "floorplans"
    incoming.mkdir(parents=True)
    for split in ("valid", "test"):
        dst = incoming / split
        dst.mkdir()
        for f in (_FIXTURE_ROOT / split).iterdir():
            dst.joinpath(f.name).write_bytes(f.read_bytes())

    spec = dataset_registry.DATASET_REGISTRY["floorplans-door"]
    out = dataset_registry.fetch(spec, root=tmp_path)
    assert out is not None

    door = tmp_path / "datasets" / "floorplans-door"
    assert len(list((door / "val").glob("*.gt.json"))) == len(_DOOR_COUNTS["valid"])
    assert len(list((door / "test").glob("*.gt.json"))) == len(_DOOR_COUNTS["test"])
    # Provenance recorded the single-class images the converter consumed, with source + licence.
    provenance = json.loads((tmp_path / "datasets" / "provenance.json").read_text())
    block = provenance["datasets"]["floorplans-door"]
    assert "roboflow" in block["source_url"]
    assert len(block["files"]) == len(_DOOR_COUNTS["valid"]) + len(_DOOR_COUNTS["test"])


def test_fetch_floorplans_missing_drop_returns_none(tmp_path: Path) -> None:
    # No _incoming tree: fetch degrades to a skipped None (T-11-05), never raises.
    spec = dataset_registry.DATASET_REGISTRY["floorplans-window"]
    assert dataset_registry.fetch(spec, root=tmp_path) is None


# --------------------------------------------------------------------------- committed manifests


def test_committed_floorplans_manifests_are_native() -> None:
    for key in ("floorplans-door", "floorplans-window"):
        manifest = load_split_manifest(key)
        assert manifest.dataset == key
        assert manifest.val_strategy == "native"
        # Train is intentionally empty (exemplar methods do no training); val tunes, test evaluates.
        assert manifest.train == ()
        assert len(manifest.val) == 56
        assert len(manifest.test) == 28


# --------------------------------------------------------------------------- end-to-end tracer


def _fixture_manifest(tmp_path: Path) -> Path:
    """Convert the fixture test split and write a matching tmp manifest, returning the split dir."""
    out_root = tmp_path / "floorplans-door" / "test"
    convert_floorplans(_FIXTURE_ROOT / "test", out_root, target_class="door")
    manifest = build_manifest(
        "floorplans-door",
        NativeSplits(train=(), val=(), test=tuple(sorted(_DOOR_COUNTS["test"]))),
    )
    write_split_manifest(manifest, root=tmp_path)
    return out_root


def test_ncc_floorplans_door_test_produces_full_metric_rows(tmp_path: Path) -> None:
    out_root = _fixture_manifest(tmp_path)
    report = run_research_benchmark(
        "ncc", "floorplans-door", "test", out_root, exemplar_count=1, manifest_root=tmp_path
    )

    assert report["dataset"] == "floorplans-door"
    per_image = report["per_image"]
    assert len(per_image) == len(_DOOR_COUNTS["test"])
    for row in per_image:
        for key in ("precision", "recall", "f1", "ap", "ap50", "ap75"):
            assert key in row, key
        assert row["true_count"] in set(_DOOR_COUNTS["test"].values())

    overall = report["overall"]
    for key in ("precision", "recall", "f1", "ap", "ap50", "ap75", "mae", "rmse", "nae"):
        assert key in overall, key
    assert overall["n_images"] == len(_DOOR_COUNTS["test"])


def test_research_block_carries_the_three_per_slice_groupings(tmp_path: Path) -> None:
    out_root = _fixture_manifest(tmp_path)
    report = run_research_benchmark(
        "ncc", "floorplans-door", "test", out_root, exemplar_count=1, manifest_root=tmp_path
    )

    slices = report["slices"]
    # All three groupings present (EVAL-10 applied to floor plans).
    assert set(slices) == {"by_symbol_size", "by_crowding", "by_plan_resolution"}

    # by_symbol_size is a GT-box-level RECALL per fixed bucket; every value is a recall in [0, 1]
    # or None (an empty bucket abstains, never fabricates 0).
    by_size = slices["by_symbol_size"]
    assert set(by_size) == {"small", "medium", "large"}
    total_gt = 0
    for bucket in by_size.values():
        recall = bucket["recall"]
        assert recall is None or (0.0 <= recall <= 1.0)
        assert bucket["n_matched"] <= bucket["n_gt"]
        total_gt += bucket["n_gt"]
    # Every labelled door box lands in exactly one size bucket (the fixture records the canvas).
    assert total_gt == sum(_DOOR_COUNTS["test"].values())

    # by_crowding / by_plan_resolution are per-image F1 slices (may be empty if all keys are None,
    # but the fixture carries a canvas size and instance counts, so both have at least one bucket).
    assert slices["by_crowding"]
    assert slices["by_plan_resolution"]
    for grouping in (slices["by_crowding"], slices["by_plan_resolution"]):
        for cell in grouping.values():
            assert "f1" in cell and "recall" in cell  # the research literature-metric column set


def test_sweep_cell_carries_the_slices_block(tmp_path: Path) -> None:
    # Drive run_research_sweep for one cell (ncc x floorplans-door x test). The sweep reads the
    # COMMITTED split manifest (it does not take a manifest_root), so its scene ids are the real
    # floor-plan ids -- absent under tmp_path, hence unlabelled -- but the point here is the
    # PLUMBING: every sweep cell must carry the additive `slices` block, always with all three
    # groupings and the three fixed symbol-size buckets, even when no image loaded.
    from object_search.eval.benchmark import BenchmarkConfig, run_research_sweep

    config = BenchmarkConfig(
        methods=("ncc",),
        datasets=("floorplans-door",),
        splits=("test",),
        exemplar_counts=(1,),
        research_root=str(tmp_path),
        research_out=str(tmp_path / "research-results.json"),
    )
    results = run_research_sweep(config)

    (cell,) = results["cells"]
    assert cell["method"] == "ncc"
    # The additive per-slice block is carried on every sweep cell, not just the single-cell block.
    assert set(cell["slices"]) == {"by_symbol_size", "by_crowding", "by_plan_resolution"}
    assert set(cell["slices"]["by_symbol_size"]) == {"small", "medium", "large"}
