"""Tests for the Typer CLI. fetch-models --list must not touch the network."""

from __future__ import annotations

from pathlib import Path

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


def test_benchmark_fails_loudly() -> None:
    result = runner.invoke(app, ["benchmark"])
    assert result.exit_code == 1
    assert "Phase 8" in result.output
