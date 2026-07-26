"""fetch-datasets: provenance, gitignore, graceful absence, and the zip-slip guard (EVAL-21, D-08).

These assert the on-disk data contract the phase bakes in: raw bytes never enter git, every fetched
file's SHA-256 + source URL + licence is recorded, and a licence-gated dataset that has not been
dropped yet degrades gracefully instead of crashing. All run offline against the committed fixture.
"""

from __future__ import annotations

import json
import subprocess
import zipfile
from pathlib import Path

import pytest
from typer.testing import CliRunner

from object_search.cli import app
from object_search.eval import datasets
from object_search.provenance import repo_root

_FIXTURE_ROOT = repo_root() / "tests" / "fixtures" / "research" / "carpk"
_CARPK = datasets.DATASET_REGISTRY["carpk"]


def _build_carpk_zip(dest_zip: Path) -> None:
    """Zip the committed fixture into a CARPK-native archive (Images/ + Annotations/)."""
    dest_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(dest_zip, "w") as zf:
        for sub in ("Images", "Annotations"):
            for path in sorted((_FIXTURE_ROOT / sub).iterdir()):
                zf.write(path, arcname=f"{sub}/{path.name}")


# --------------------------------------------------------------------------- provenance + convert


def test_fetch_from_dropped_zip_writes_provenance(tmp_path: Path) -> None:
    incoming = datasets.incoming_dir(tmp_path) / _CARPK.incoming_subdir
    _build_carpk_zip(incoming / "carpk.zip")

    out = datasets.fetch(_CARPK, root=tmp_path)
    assert out is not None
    assert list(out.glob("*.gt.json"))  # sidecars written
    assert list(out.glob("*.png"))  # scenes co-located beside the labels

    manifest_path = datasets.datasets_dir(tmp_path) / "provenance.json"
    assert manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest["datasets"]["carpk"]["files"]
    assert files, "provenance must record at least one file"
    for entry in files:
        # D-08: every entry carries sha256 + source_url + licence.
        assert entry["sha256"]
        assert entry["source_url"] == _CARPK.source_url
        assert entry["license"] == _CARPK.license


def test_fetch_from_extracted_tree_also_works(tmp_path: Path) -> None:
    # A human may drop the already-extracted Images/ + Annotations/ tree instead of a zip.
    incoming = datasets.incoming_dir(tmp_path) / _CARPK.incoming_subdir
    for sub in ("Images", "Annotations"):
        target = incoming / sub
        target.mkdir(parents=True, exist_ok=True)
        for path in (_FIXTURE_ROOT / sub).iterdir():
            (target / path.name).write_bytes(path.read_bytes())

    out = datasets.fetch(_CARPK, root=tmp_path)
    assert out is not None
    assert len(list(out.glob("*.gt.json"))) == 3


def test_fetch_missing_archive_is_graceful(tmp_path: Path) -> None:
    # No archive dropped: fetch logs the drop instruction and returns None (never crashes the
    # sweep -- T-11-05), and writes no provenance.
    assert datasets.fetch(_CARPK, root=tmp_path) is None
    assert not (datasets.datasets_dir(tmp_path) / "provenance.json").is_file()


def test_verify_all_reflects_conversion(tmp_path: Path) -> None:
    assert datasets.verify_all(tmp_path)["carpk"] is False
    incoming = datasets.incoming_dir(tmp_path) / _CARPK.incoming_subdir
    _build_carpk_zip(incoming / "carpk.zip")
    datasets.fetch(_CARPK, root=tmp_path)
    assert datasets.verify_all(tmp_path)["carpk"] is True


# --------------------------------------------------------------------------- zip-slip guard


def test_zip_slip_member_is_refused(tmp_path: Path) -> None:
    incoming = datasets.incoming_dir(tmp_path) / _CARPK.incoming_subdir
    incoming.mkdir(parents=True, exist_ok=True)
    malicious = incoming / "carpk.zip"
    with zipfile.ZipFile(malicious, "w") as zf:
        zf.writestr("Annotations/ok.txt", "0 0 10 10 1\n")
        zf.writestr("../../escape.txt", "pwned")  # escapes the dataset dir
    with pytest.raises(ValueError, match="zip-slip"):
        datasets.fetch(_CARPK, root=tmp_path)


# --------------------------------------------------------------------------- git hygiene


def test_no_raw_dataset_file_is_git_tracked() -> None:
    # `git ls-files datasets/` must be empty: raw research data never enters git history (D-08).
    result = subprocess.run(
        ["git", "ls-files", "datasets/"],  # noqa: S607
        cwd=repo_root(),
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == ""


def test_gitignore_ignores_datasets() -> None:
    gitignore = (repo_root() / ".gitignore").read_text(encoding="utf-8")
    assert "datasets/" in gitignore


# --------------------------------------------------------------------------- CLI --list


def test_fetch_datasets_list_prints_carpk_and_drop_path() -> None:
    result = CliRunner().invoke(app, ["fetch-datasets", "--list"])
    assert result.exit_code == 0
    assert "carpk" in result.output
    assert _CARPK.license in result.output
    assert "datasets/_incoming/carpk" in result.output


def test_fetch_datasets_unknown_key_errors() -> None:
    result = CliRunner().invoke(app, ["fetch-datasets", "--only", "nope"])
    assert result.exit_code == 1
    assert "unknown dataset" in result.output
