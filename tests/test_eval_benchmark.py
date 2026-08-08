"""The Hydra benchmark runner (EVAL-04): the model-free CI subset and per-slice reporting.

These tests drive the pure ``run_benchmark`` core directly with a constructed config, never
through ``@hydra.main`` -- Hydra seizes ``sys.argv`` and cannot run inside pytest. They run the
CI subset (``ncc`` + classical ``sparse-geo``) over a couple of small chipset images, which needs
no ONNX weights, so this file is part of the model-free CI benchmark itself.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from object_search.eval.benchmark import BenchmarkConfig, run_benchmark
from object_search.eval.labels import chipset_image_ids, real_objects_image_ids


def test_ci_subset_is_model_free_methods_over_chipset() -> None:
    config = BenchmarkConfig(ci=True, ci_image_limit=2)
    methods, images = config.resolve_run_set()
    assert methods == ("ncc", "sparse-geo")  # no learned methods -> no weights needed
    assert len(images) == 2
    assert all(image_id.startswith("chipset") for image_id in images)


def test_full_sweep_includes_chipset_and_synthetic() -> None:
    config = BenchmarkConfig()
    methods, images = config.resolve_run_set()
    assert "dino-dense" in methods  # the full set keeps the learned methods
    assert any(i.startswith("chipset") for i in images)  # NCC-favourable side
    assert "scatter-scaled" in images  # learned-favourable (scale-varied) side


def test_ci_benchmark_writes_results_with_per_slice_breakdowns(tmp_path: Path) -> None:
    out = tmp_path / "results.json"
    config = BenchmarkConfig(ci=True, ci_image_limit=2, out=str(out))
    results = run_benchmark(config)

    # The file is written and matches the returned dict.
    assert out.is_file()
    on_disk = json.loads(out.read_text(encoding="utf-8"))
    assert on_disk == results

    assert results["ci_subset"] is True
    assert results["ap_convention"].startswith("all-point interpolation")

    # Both model-free methods present; no learned method ran.
    methods = results["methods"]
    assert set(methods) == {"ncc", "sparse-geo"}

    for method_name, block in methods.items():
        overall = block["overall"]
        # Per-method precision/recall/AP keys exist (values may be None on abstention -- that is
        # the honest outcome, never fabricated as 0).
        assert "precision" in overall
        assert "recall" in overall
        assert "mean_ap" in overall
        assert overall["n_images"] == 2, method_name

        # Per-canvas-size latency breakdown: the chipset ramp is a latency story, so each canvas
        # size gets its own latency summary.
        by_canvas = block["slices"]["by_canvas_size"]
        assert by_canvas, method_name
        for canvas_key, canvas_block in by_canvas.items():
            assert "x" in canvas_key  # e.g. "320x240"
            assert "latency_ms" in canvas_block
            assert "p50" in canvas_block["latency_ms"]

        # Per-instance-count slice exists too (EVAL-10 slices).
        assert block["slices"]["by_instance_count"]

        # Per-box-size (small/medium/large) recall slice, reused from the research path -- the
        # chipset sidecars carry width/height, so this is populated on the main sweep too.
        by_size = block["slices"]["by_symbol_size"]
        assert set(by_size) == {"small", "medium", "large"}
        for bucket in by_size.values():
            assert "n_gt" in bucket
            assert "recall" in bucket


def test_coverage_reports_unlabelled_images_honestly(tmp_path: Path) -> None:
    out = tmp_path / "results.json"
    # An image id with no sidecar must be reported as unlabelled, not silently dropped. Coverage
    # is computed from ground-truth loading alone, so an empty method set exercises it without
    # running any (slow) search over the full chipset ramp.
    config = BenchmarkConfig(
        ci=False,
        methods=(),
        image_ids=("no-such-image",),
        out=str(out),
    )
    results = run_benchmark(config)
    coverage = results["coverage"]
    assert "no-such-image" in coverage["images_unlabelled"]
    # The chipset images (added automatically) are labelled, so labelled > 0.
    assert coverage["images_labelled"] >= len(chipset_image_ids())


def test_benchmark_config_rejects_unknown_key() -> None:
    with pytest.raises(ValidationError):
        BenchmarkConfig.model_validate({"nope": 1})


def test_real_objects_only_subset_is_real_objects_images_all_methods() -> None:
    config = BenchmarkConfig(real_objects_only=True)
    methods, images = config.resolve_run_set()
    assert methods == config.methods  # full method list, unlike the CI subset
    assert set(images) == set(real_objects_image_ids())
    # No chipset/textured/synthetic ids leak in -- this is a real-objects-ONLY sweep.
    assert all(image_id.startswith("real-") for image_id in images)


def test_real_objects_only_ignores_configured_image_ids() -> None:
    config = BenchmarkConfig(real_objects_only=True, image_ids=("scatter-scaled",))
    _methods, images = config.resolve_run_set()
    assert "scatter-scaled" not in images


def test_ci_wins_over_real_objects_only_when_both_set() -> None:
    config = BenchmarkConfig(ci=True, ci_image_limit=2, real_objects_only=True)
    methods, images = config.resolve_run_set()
    assert methods == ("ncc", "sparse-geo")
    assert all(image_id.startswith("chipset") for image_id in images)


def test_real_objects_only_benchmark_writes_results(tmp_path: Path) -> None:
    out = tmp_path / "real-objects-results.json"
    # Model-free methods only, so this cell runs without fetched ONNX weights (CI-runnable).
    config = BenchmarkConfig(
        real_objects_only=True,
        methods=("ncc", "sparse-geo"),
        out=str(out),
    )
    results = run_benchmark(config)

    assert out.is_file()
    assert results["coverage"]["images_requested"] == len(real_objects_image_ids())
    assert set(results["methods"]) == {"ncc", "sparse-geo"}
    for block in results["methods"].values():
        image_ids = [r["image_id"] for r in block["per_image"]]
        assert all(image_id.startswith("real-") for image_id in image_ids)
