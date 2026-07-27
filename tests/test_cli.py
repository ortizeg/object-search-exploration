"""Tests for the Typer CLI. fetch-models --list must not touch the network."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from object_search.cli import app

runner = CliRunner()


def test_fetch_models_list_exits_zero_and_lists_all_three() -> None:
    result = runner.invoke(app, ["fetch-models", "--list"])
    assert result.exit_code == 0
    for key in ("dinov2-small", "superpoint", "fastsam-s"):
        assert key in result.output


def test_fetch_models_unknown_only_exits_nonzero() -> None:
    result = runner.invoke(app, ["fetch-models", "--only", "nope"])
    assert result.exit_code == 1
    assert "unknown model" in result.output


def test_synth_writes_all_demo_specs(tmp_path: Path) -> None:
    result = runner.invoke(app, ["synth", "--out", str(tmp_path)])
    assert result.exit_code == 0, result.output
    pngs = sorted(tmp_path.glob("*.png"))
    sidecars = sorted(tmp_path.glob("*.gt.json"))
    assert len(pngs) >= 4
    assert len(sidecars) >= 4


def test_synth_unknown_spec_exits_nonzero(tmp_path: Path) -> None:
    result = runner.invoke(app, ["synth", "--out", str(tmp_path), "--spec", "nope"])
    assert result.exit_code == 1
    assert "unknown spec" in result.output


def test_render_samples_exits_zero_and_writes(tmp_path: Path) -> None:
    result = runner.invoke(app, ["render-samples", "--method", "ncc", "--out", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "ncc" / "index.md").is_file()
    assert list((tmp_path / "ncc").glob("*.png"))


def test_render_samples_unknown_method_exits_nonzero(tmp_path: Path) -> None:
    result = runner.invoke(app, ["render-samples", "--method", "nope", "--out", str(tmp_path)])
    assert result.exit_code == 1
    assert "unknown method" in result.output


def test_benchmark_redirects_to_hydra_entrypoint() -> None:
    # The benchmark is a @hydra.main module entry point (Hydra owns sys.argv), so the Typer
    # shim must fail loudly and point at `pixi run bench`, never silently succeed.
    result = runner.invoke(app, ["benchmark"])
    assert result.exit_code == 1
    assert "pixi run bench" in result.output


# ------------------------------------------------------- the synthetic-dataset subcommands


def test_markers_writes_all_specs_with_sidecars(tmp_path: Path) -> None:
    result = runner.invoke(app, ["markers", "--out", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert list(tmp_path.glob("*.png"))
    assert list(tmp_path.glob("*.markers.json"))  # each marker image gets its oracle sidecar


def test_markers_unknown_spec_exits_nonzero(tmp_path: Path) -> None:
    result = runner.invoke(app, ["markers", "--out", str(tmp_path), "--spec", "nope"])
    assert result.exit_code == 1
    assert "unknown marker spec" in result.output


def test_chipset_writes_images(tmp_path: Path) -> None:
    result = runner.invoke(app, ["chipset", "--out", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert list(tmp_path.glob("*.png"))


def test_textured_writes_images(tmp_path: Path) -> None:
    result = runner.invoke(app, ["textured", "--out", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert list(tmp_path.glob("*.png"))


# ------------------------------------------------------- fetch-models: the fetching branches
# (mocked so no network or weight is touched; only the CLI dispatch is under test.)


def test_fetch_models_only_valid_key_fetches_that_spec(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        "object_search.inference.models.fetch",
        lambda spec, force=False: calls.append(spec.key) or Path(spec.dest),
    )
    result = runner.invoke(app, ["fetch-models", "--only", "dinov2-small"])
    assert result.exit_code == 0, result.output
    assert calls == ["dinov2-small"]


def test_fetch_models_default_fetches_all(monkeypatch: pytest.MonkeyPatch) -> None:
    called: list[bool] = []
    monkeypatch.setattr(
        "object_search.inference.models.fetch_all",
        lambda force=False: called.append(True) or {},
    )
    result = runner.invoke(app, ["fetch-models"])
    assert result.exit_code == 0, result.output
    assert called == [True]


# ------------------------------------------------- render-samples: the all-methods marker branch
# (the method loop and the marker gallery are stubbed so the branch runs without weights.)


def test_render_samples_all_skips_marker_gallery_when_fastsam_absent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("object_search.samples.render_samples", lambda names, out_root: [tmp_path])

    def _absent(*_a: object, **_k: object) -> object:
        raise FileNotFoundError("fastsam weight absent")

    monkeypatch.setattr("object_search.search.proposals.default_backend", _absent)
    result = runner.invoke(app, ["render-samples", "--out", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "rendered" in result.output


def test_render_samples_all_renders_marker_gallery_when_backend_available(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("object_search.samples.render_samples", lambda names, out_root: [tmp_path])
    monkeypatch.setattr(
        "object_search.search.proposals.default_backend", lambda providers=None: object()
    )
    monkeypatch.setattr(
        "object_search.samples.render_marker_samples",
        lambda backend, out_root: [tmp_path / "m1", tmp_path / "m2"],
    )
    result = runner.invoke(app, ["render-samples", "--out", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "marker sample artifact(s)" in result.output
