"""The full research sweep: method x dataset x {1,3 exemplars} x {val,test}, offline (EVAL-23/24).

Everything here runs on the committed synthetic fixtures with ``ncc`` only -- no network, no
licence gate, no ONNX weights -- exactly the model-free surface CI can run. It proves:

* the k-shot **late-fusion** runner (:func:`run_multi_exemplar`) unions the per-exemplar matches and
  candidates and NMS-dedupes them, so a repeated instance is not counted once per exemplar, and at
  ``k=1`` it is a pass-through of the single call (the Task 2 ratified mechanism);
* the sweep produces cells for ``{1,3} x {val,test}`` for a carve-val dataset and **test-only**
  cells for CARPK (zero val cells -- D-04);
* every cell carries the full literature metric column set (P/R/F1 + AP/AP50/AP75 + MAE/RMSE/NAE);
* the recall denominator (``len(gt.boxes)``) is identical at 1 and 3 exemplars, so the numbers are
  comparable (the sampled exemplars stay scored in gt);
* the CI subset is untouched -- ``ci=true`` still resolves to the two chipset model-free methods,
  the research sweep excluded.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

from object_search.eval.benchmark import (
    BenchmarkConfig,
    run_multi_exemplar,
    run_research_benchmark,
    run_research_sweep,
)
from object_search.eval.converters import convert_carpk, convert_rpine
from object_search.provenance import repo_root
from object_search.schemas.geometry import BBox, ExemplarBox
from object_search.schemas.search import (
    Candidate,
    LatencyBreakdown,
    Match,
    SearchOutcome,
    SearchResult,
)

_FIXTURES = repo_root() / "tests" / "fixtures" / "research"
_CELL_METRIC_KEYS = ("precision", "recall", "f1", "ap", "ap50", "ap75", "mae", "rmse", "nae")


def _build_research_base(tmp_path: Path) -> Path:
    """Convert the CARPK + RPINE fixtures into a ``base/<dataset>/<split>`` research tree.

    Every fixture image is written into each split dir; the committed manifest restricts which ids
    are actually loaded per split, so extra sidecars are harmless.
    """
    base = tmp_path / "datasets"
    for split in ("val", "test"):
        convert_rpine(_FIXTURES / "rpine", base / "rpine" / split)
        convert_carpk(_FIXTURES / "carpk", base / "carpk" / split)
    return base


# --------------------------------------------------------------------------- run_multi_exemplar


def _fake_spec_fn(box: BBox, score: float) -> object:
    """A method stand-in that returns one match at ``box`` and one sub-threshold candidate."""

    def fn(_image, _exemplar, _config) -> SearchResult:  # positional SearchFn shape; args unused
        return SearchResult(
            method="fake",
            method_version="1",
            outcome=SearchOutcome.OK,
            matches=(Match(box=box, score=score),),
            latency=LatencyBreakdown(preprocess_ms=1.0, inference_ms=1.0, postprocess_ms=1.0),
            threshold_applied=0.5,
            candidates=(Candidate(box=BBox(x=200, y=200, w=10, h=10), score=0.1),),
        )

    return fn


def test_run_multi_exemplar_k1_is_passthrough() -> None:
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    fn = _fake_spec_fn(BBox(x=5, y=5, w=10, h=10), 0.9)
    exemplars = (ExemplarBox(box=BBox(x=5, y=5, w=10, h=10)),)
    result = run_multi_exemplar(fn, image, exemplars, config=object())
    assert len(result.matches) == 1
    assert result.matches[0].box == BBox(x=5, y=5, w=10, h=10)


def test_run_multi_exemplar_unions_then_dedupes() -> None:
    # Three exemplars, each producing the SAME box -> the union is NMS-deduped to one match, never
    # counted three times; the identical candidates collapse to one too.
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    fn = _fake_spec_fn(BBox(x=5, y=5, w=10, h=10), 0.9)
    exemplars = tuple(ExemplarBox(box=BBox(x=5, y=5, w=10, h=10)) for _ in range(3))
    result = run_multi_exemplar(fn, image, exemplars, config=object())
    assert len(result.matches) == 1
    assert len(result.candidates) == 1


def test_run_multi_exemplar_is_defined_in_eval_not_search() -> None:
    # The k-shot fusion lives in the eval layer; no multi-exemplar dispatch was added to any method.
    src = repo_root() / "src" / "object_search"
    assert "def run_multi_exemplar" in (src / "eval" / "benchmark.py").read_text(encoding="utf-8")
    for method_file in (src / "search").glob("*.py"):
        assert "run_multi_exemplar" not in method_file.read_text(encoding="utf-8"), method_file


# --------------------------------------------------------------------------- sweep dimensions


def test_sweep_yields_1_and_3_over_val_and_test_for_carve_val_dataset(tmp_path: Path) -> None:
    base = _build_research_base(tmp_path)
    config = BenchmarkConfig(
        methods=("ncc",),
        datasets=("rpine",),
        splits=("val", "test"),
        exemplar_counts=(1, 3),
        research_root=str(base),
        research_out=str(tmp_path / "research-results.json"),
    )
    results = run_research_sweep(config)
    cells = results["cells"]
    got = {(c["dataset"], c["split"], c["exemplar_count"]) for c in cells}
    assert ("rpine", "val", 1) in got
    assert ("rpine", "val", 3) in got
    assert ("rpine", "test", 1) in got
    assert ("rpine", "test", 3) in got


def test_carpk_is_test_only_no_val_cell(tmp_path: Path) -> None:
    base = _build_research_base(tmp_path)
    config = BenchmarkConfig(
        methods=("ncc",),
        datasets=("carpk",),
        splits=("val", "test"),
        exemplar_counts=(1, 3),
        research_root=str(base),
        research_out=str(tmp_path / "research-results.json"),
    )
    results = run_research_sweep(config)
    splits = {(c["dataset"], c["split"]) for c in results["cells"]}
    assert ("carpk", "test") in splits
    assert ("carpk", "val") not in splits  # D-04: CARPK never emits a val cell.


def test_every_cell_carries_the_full_metric_column_set(tmp_path: Path) -> None:
    base = _build_research_base(tmp_path)
    config = BenchmarkConfig(
        methods=("ncc",),
        datasets=("rpine",),
        splits=("test",),
        exemplar_counts=(1, 3),
        research_root=str(base),
        research_out=str(tmp_path / "research-results.json"),
    )
    results = run_research_sweep(config)
    for cell in results["cells"]:
        for key in _CELL_METRIC_KEYS:
            assert key in cell["overall"], key


def test_recall_denominator_identical_across_1_and_3(tmp_path: Path) -> None:
    base = _build_research_base(tmp_path)
    root = base / "rpine" / "test"
    one = run_research_benchmark("ncc", "rpine", "test", root, exemplar_count=1)
    three = run_research_benchmark("ncc", "rpine", "test", root, exemplar_count=3)
    by_id_one = {r["image_id"]: r["true_count"] for r in one["per_image"]}
    by_id_three = {r["image_id"]: r["true_count"] for r in three["per_image"]}
    assert by_id_one == by_id_three  # sampled exemplars stay in gt -> same denominator (EVAL-23)


def test_research_results_file_is_written_and_gitignored(tmp_path: Path) -> None:
    base = _build_research_base(tmp_path)
    out = repo_root() / "docs" / "benchmark" / "research-results.json"
    config = BenchmarkConfig(
        methods=("ncc",),
        datasets=("carpk",),
        splits=("test",),
        exemplar_counts=(1,),
        research_root=str(base),
    )
    run_research_sweep(config)
    assert out.is_file()
    # docs/benchmark/research-results.json must be gitignored (regenerable, environment-dependent).
    proc = subprocess.run(  # noqa: S603  # `out` is a repo-internal Path, not untrusted input
        ["git", "check-ignore", str(out)],  # noqa: S607
        cwd=repo_root(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, "research-results.json must be gitignored"


# --------------------------------------------------------------------------- CI subset unaffected


def test_ci_subset_excludes_research_even_with_datasets_configured() -> None:
    # A config that also names research datasets must still resolve the CI run to the chipset
    # model-free subset -- the research sweep is a separate path, never in CI.
    config = BenchmarkConfig(ci=True, ci_image_limit=2, datasets=("carpk", "rpine"))
    methods, images = config.resolve_run_set()
    assert methods == ("ncc", "sparse-geo")
    assert all(i.startswith("chipset") for i in images)


# ------------------------------------------------------------ CLI entrypoint (regression)


def test_bench_research_cli_entrypoint_dispatches_and_resolves(tmp_path: Path) -> None:
    """`pixi run bench-research` must reach the sweep through the REAL Hydra CLI.

    Guards two defects the direct-``run_research_sweep`` unit tests could not see (both fire during
    arg/config parsing, before any sidecar is read, so their ABSENCE from the output proves the fix
    regardless of the run's own exit status):

    1. the ``--research`` sentinel selects ``main_research`` from WITHIN ``benchmark.py``'s
       ``__main__`` so Hydra resolves its file-relative ``config_path``. A ``python -c`` call left
       the function's module non-``__main__`` -> ``Primary config module 'conf' not found``.
    2. ``datasets=[...]`` is a plain key override, needing the split manifests OUTSIDE ``conf/``.
       Under ``conf/datasets/`` Hydra reads it as a config group -> ``Could not override
       'datasets'`` / ``No match in the defaults list``.
    """
    proc = subprocess.run(  # noqa: S603  # fixed argv (interpreter + module), no untrusted input
        [
            sys.executable,
            "-m",
            "object_search.eval.benchmark",
            "--research",
            f"research_root={tmp_path}",
            "datasets=[carpk]",
            "splits=[test]",
            "exemplar_counts=[1]",
            "methods=[ncc]",
        ],
        capture_output=True,
        text=True,
        cwd=repo_root(),
        check=False,
    )
    combined = proc.stdout + proc.stderr
    assert "Primary config module 'conf' not found" not in combined, combined
    assert "Could not override 'datasets'" not in combined, combined
    assert "No match in the defaults list" not in combined, combined
