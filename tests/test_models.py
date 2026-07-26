"""Tests for the model registry and fetch framework (INFRA-11). No network is touched."""

from __future__ import annotations

from pathlib import Path

import pytest

from object_search.inference import models


def test_registry_has_the_expected_models() -> None:
    assert set(models.MODEL_REGISTRY) == {
        "dinov2-small",
        "superpoint",
        "fastsam-s",
        "owlv2-base-patch16",
    }


def test_licence_constraints_are_recorded() -> None:
    fastsam = models.MODEL_REGISTRY["fastsam-s"]
    assert "AGPL-3.0" in fastsam.license
    assert "AGPL" in fastsam.license_note

    superpoint = models.MODEL_REGISTRY["superpoint"]
    assert "non-commercial" in superpoint.license_note.lower()

    dinov2 = models.MODEL_REGISTRY["dinov2-small"]
    assert dinov2.license == "Apache-2.0"
    assert dinov2.revision is not None  # revision must be pinned

    owlv2 = models.MODEL_REGISTRY["owlv2-base-patch16"]
    assert owlv2.license == "Apache-2.0"  # permissive -- the reason it was chosen over T-Rex2
    assert "non-commercial" in owlv2.license_note.lower()  # records why the alternatives were not


def test_each_spec_names_the_phase_that_adds_it() -> None:
    phases = {key: spec.added_in_phase for key, spec in models.MODEL_REGISTRY.items()}
    assert phases == {
        "dinov2-small": 6,
        "superpoint": 5,
        "fastsam-s": 7,
        "owlv2-base-patch16": 8,
    }


def test_github_release_url_is_built_from_repo_and_revision() -> None:
    spec = models.MODEL_REGISTRY["superpoint"]
    url = models._github_release_url(spec)
    assert url == (
        "https://github.com/fabio-sim/LightGlue-ONNX/releases/download/v1.0.0/superpoint.onnx"
    )


def test_models_dir_and_verify_all(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("object_search.provenance.repo_root", lambda: tmp_path)
    directory = models.models_dir()
    assert directory == tmp_path / "models"
    assert directory.is_dir()

    # Nothing downloaded -> every model reports absent.
    assert models.verify_all() == {
        "dinov2-small": False,
        "superpoint": False,
        "fastsam-s": False,
        "owlv2-base-patch16": False,
    }


def test_fetch_skips_when_present_without_network(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("object_search.provenance.repo_root", lambda: tmp_path)
    spec = models.MODEL_REGISTRY["dinov2-small"]  # sha256 is None -> presence alone is enough
    dest = tmp_path / "models" / spec.dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(b"pretend onnx bytes")

    # If this tried to hit the network it would fail; skipping means it returns the path.
    assert models.fetch(spec) == dest
