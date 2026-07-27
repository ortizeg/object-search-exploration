"""Tests for the model registry and fetch framework (INFRA-11). No network is touched.

The download and verify paths are exercised with mocked transports (a fake ``urlopen`` and a
fake ``hf_hub_download``) and with the export dependencies forced absent, so the sha256 gate,
the ``.part`` atomic-rename, the hf-vs-github branch split, the export-env-missing fallback, and
``verify_all``'s hash comparison all run in CI without a byte of real network or any weight.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

from object_search.inference import models
from object_search.inference.models import ModelSpec


class _FakeResponse:
    """A minimal stand-in for the ``urllib.request.urlopen`` context manager."""

    def __init__(self, data: bytes) -> None:
        self._data = data

    def read(self) -> bytes:
        return self._data

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *_exc: object) -> bool:
        return False


def _github_spec(sha256: str | None, *, dest: str = "fake_gh.onnx") -> ModelSpec:
    """A synthetic github-release spec so a test controls the pinned sha256 it asserts against."""
    return ModelSpec(
        key="fake-gh",
        source="github-release",
        repo_id="owner/repo",
        revision="v1.0.0",
        filename="fake.onnx",
        sha256=sha256,
        license="Apache-2.0",
        license_note="synthetic",
        source_note="synthetic",
        dest=dest,
        added_in_phase=99,
    )


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


# --------------------------------------------------------- fetch: the github-release download path


def test_fetch_github_release_downloads_verifies_and_installs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A github-release fetch downloads to ``.part``, verifies the sha256, then renames it in."""
    monkeypatch.setattr("object_search.provenance.repo_root", lambda: tmp_path)
    payload = b"pretend onnx weights"
    spec = _github_spec(hashlib.sha256(payload).hexdigest())
    monkeypatch.setattr(models.urllib.request, "urlopen", lambda _url: _FakeResponse(payload))

    dest = models.fetch(spec)

    assert dest == tmp_path / "models" / "fake_gh.onnx"
    assert dest.read_bytes() == payload
    # The atomic ``.part`` staging file must not survive a successful install.
    assert not dest.with_suffix(".onnx.part").exists()


def test_fetch_raises_and_cleans_up_on_sha256_mismatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A byte-different download refuses to install and leaves no partial file behind (EVAL-09)."""
    monkeypatch.setattr("object_search.provenance.repo_root", lambda: tmp_path)
    spec = _github_spec("0" * 64)  # a sha the payload cannot match
    monkeypatch.setattr(models.urllib.request, "urlopen", lambda _url: _FakeResponse(b"wrong"))

    with pytest.raises(ValueError, match="sha256 mismatch"):
        models.fetch(spec)

    dest = tmp_path / "models" / "fake_gh.onnx"
    assert not dest.exists()  # nothing installed
    assert not dest.with_suffix(".onnx.part").exists()  # the partial was unlinked


def test_fetch_installs_without_verification_when_no_sha256_is_pinned(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A spec with no pinned sha256 installs the download as-is (the verification is skipped)."""
    monkeypatch.setattr("object_search.provenance.repo_root", lambda: tmp_path)
    spec = _github_spec(None)  # no integrity gate declared
    monkeypatch.setattr(models.urllib.request, "urlopen", lambda _url: _FakeResponse(b"any bytes"))

    dest = models.fetch(spec)
    assert dest.read_bytes() == b"any bytes"


def test_fetch_refetches_when_present_file_has_the_wrong_sha256(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A present file whose hash no longer matches is re-downloaded, not trusted."""
    monkeypatch.setattr("object_search.provenance.repo_root", lambda: tmp_path)
    payload = b"the correct bytes"
    spec = _github_spec(hashlib.sha256(payload).hexdigest())
    dest = tmp_path / "models" / "fake_gh.onnx"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(b"stale corrupted bytes")  # present, but wrong hash
    monkeypatch.setattr(models.urllib.request, "urlopen", lambda _url: _FakeResponse(payload))

    assert models.fetch(spec).read_bytes() == payload  # replaced with the correct download


def test_fetch_skips_present_file_when_sha256_matches(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A present file whose hash matches short-circuits before any network call."""
    monkeypatch.setattr("object_search.provenance.repo_root", lambda: tmp_path)
    payload = b"already installed"
    spec = _github_spec(hashlib.sha256(payload).hexdigest())
    dest = tmp_path / "models" / "fake_gh.onnx"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(payload)

    def _boom(_url: str) -> _FakeResponse:  # a network call here would be a bug
        raise AssertionError("fetch must not download when the present file already verifies")

    monkeypatch.setattr(models.urllib.request, "urlopen", _boom)
    assert models.fetch(spec) == dest


# ----------------------------------------------------------- fetch: the hf-hub download path


def test_fetch_hf_hub_downloads_and_installs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An hf-hub fetch copies the hub file into ``models/`` via the ``.part`` staging path."""
    monkeypatch.setattr("object_search.provenance.repo_root", lambda: tmp_path)
    payload = b"hub onnx bytes"
    spec = ModelSpec(
        key="fake-hf",
        source="hf-hub",
        repo_id="org/model",
        revision="abc123",
        filename="model.onnx",
        sha256=hashlib.sha256(payload).hexdigest(),
        license="Apache-2.0",
        license_note="synthetic",
        source_note="synthetic",
        dest="fake_hf.onnx",
        added_in_phase=99,
    )
    hub_file = tmp_path / "hub_cache_model.onnx"
    hub_file.write_bytes(payload)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda **_kw: str(hub_file))

    dest = models.fetch(spec)
    assert dest == tmp_path / "models" / "fake_hf.onnx"
    assert dest.read_bytes() == payload


# ----------------------------------------------------- fetch: the export path with deps absent


def test_fetch_export_fastsam_without_ultralytics_returns_absent_dest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Without the export env, a fastsam export logs the how-to and returns the (absent) dest."""
    monkeypatch.setattr("object_search.provenance.repo_root", lambda: tmp_path)
    monkeypatch.setitem(sys.modules, "ultralytics", None)  # force the ImportError branch

    dest = models.fetch(models.MODEL_REGISTRY["fastsam-s"])
    assert dest == tmp_path / "models" / "fastsam_s.onnx"
    assert not dest.exists()  # export was not attempted; nothing written


def test_fetch_export_owlv2_without_torch_returns_absent_dest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Without the export env, an owlv2 export logs the how-to and returns the (absent) dest."""
    monkeypatch.setattr("object_search.provenance.repo_root", lambda: tmp_path)
    monkeypatch.setitem(sys.modules, "torch", None)  # force the ImportError branch

    dest = models.fetch(models.MODEL_REGISTRY["owlv2-base-patch16"])
    assert dest == tmp_path / "models" / "owlv2_base_patch16.onnx"
    assert not dest.exists()


def test_export_dispatch_rejects_an_unregistered_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A source=export spec with no registered exporter fails loudly, never silently no-ops."""
    monkeypatch.setattr("object_search.provenance.repo_root", lambda: tmp_path)
    spec = ModelSpec(
        key="mystery-export",
        source="export",
        repo_id="x/y",
        revision="v1",
        filename="m.onnx",
        sha256=None,
        license="Apache-2.0",
        license_note="synthetic",
        source_note="synthetic",
        dest="mystery.onnx",
        added_in_phase=99,
    )
    with pytest.raises(ValueError, match="no exporter registered"):
        models.fetch(spec)


# ------------------------------------------------------------------- fetch_all / verify_all


def test_fetch_all_fetches_every_registered_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """fetch_all returns one destination per registry entry, delegating to fetch."""
    monkeypatch.setattr(models, "fetch", lambda spec, force=False: Path(spec.dest))
    result = models.fetch_all()
    assert set(result) == set(models.MODEL_REGISTRY)
    assert result["dinov2-small"] == Path("dinov2_small.onnx")


def test_verify_all_reports_present_none_hash_true_and_mismatch_false(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """verify_all: absent -> False, present+no-pinned-hash -> True, present+wrong-hash -> False."""
    monkeypatch.setattr("object_search.provenance.repo_root", lambda: tmp_path)
    directory = models.models_dir()
    # fastsam-s has sha256=None: mere presence is enough -> True.
    (directory / models.MODEL_REGISTRY["fastsam-s"].dest).write_bytes(b"present")
    # dinov2-small has a pinned sha256; a wrong-byte file must verify False (the integrity gate).
    (directory / models.MODEL_REGISTRY["dinov2-small"].dest).write_bytes(b"corrupted")
    # superpoint is left absent -> False.

    result = models.verify_all()
    assert result["fastsam-s"] is True
    assert result["dinov2-small"] is False
    assert result["superpoint"] is False
