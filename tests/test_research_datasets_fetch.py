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
from object_search.eval.labels import load_research_ground_truth
from object_search.provenance import repo_root

_RESEARCH_FIXTURES = repo_root() / "tests" / "fixtures" / "research"
_FIXTURE_ROOT = _RESEARCH_FIXTURES / "carpk"
_CARPK = datasets.DATASET_REGISTRY["carpk"]


def _build_carpk_zip(dest_zip: Path) -> None:
    """Zip the committed fixture into a CARPK-native archive (Images/ + Annotations/)."""
    dest_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(dest_zip, "w") as zf:
        for sub in ("Images", "Annotations"):
            for path in sorted((_FIXTURE_ROOT / sub).iterdir()):
                zf.write(path, arcname=f"{sub}/{path.name}")


def _fixture_root(key: str) -> Path:
    """The committed native-format fixture tree for dataset ``key``."""
    return _RESEARCH_FIXTURES / key


def _zip_tree(src_root: Path, dest_zip: Path, *, prefix: str = "") -> None:
    """Zip every file under ``src_root`` into ``dest_zip``, preserving its relative structure.

    ``prefix`` wraps the whole tree under one top-level dir, so a single-wrapping-dir archive (the
    common "the whole dataset is inside one folder" drop) can be exercised against the raw-marker
    descent in :func:`datasets._resolve_raw_root`.
    """
    dest_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(dest_zip, "w") as zf:
        for path in sorted(src_root.rglob("*")):
            if path.is_file():
                arcname = str(Path(prefix) / path.relative_to(src_root))
                zf.write(path, arcname=arcname)


def _copy_tree(src_root: Path, dest_root: Path) -> None:
    """Copy every file under ``src_root`` into ``dest_root`` (an already-extracted drop)."""
    for path in sorted(src_root.rglob("*")):
        if path.is_file():
            target = dest_root / path.relative_to(src_root)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())


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


# -------------------------------------------------- every research dataset resolves from _incoming


@pytest.mark.parametrize(
    ("key", "drop_mode"),
    [
        # Both a zipped drop and an already-extracted-tree drop, spread across all four native
        # layouts (FSC's annotations.json marker, RPINE's annotations/ dir marker, CARPK's
        # Annotations/ marker), plus one single-wrapping-dir zip to exercise the marker descent.
        ("carpk", "wrapped-zip"),
        ("fscd147", "zip"),
        ("fscd147", "tree"),
        ("fscd_lvis", "zip"),
        ("fscd_lvis", "tree"),
        ("rpine", "zip"),
        ("rpine", "tree"),
        ("rpine", "wrapped-zip"),
    ],
)
def test_fetch_resolves_each_dataset_from_incoming(
    key: str, drop_mode: str, tmp_path: Path
) -> None:
    # Drop the committed native fixture at _incoming/<subdir>/ as either a zip, a single-wrapping-
    # dir zip, or an extracted tree, then assert fetch converts + records provenance for each.
    spec = datasets.DATASET_REGISTRY[key]
    fixture = _fixture_root(key)
    incoming = datasets.incoming_dir(tmp_path) / spec.incoming_subdir
    if drop_mode == "zip":
        _zip_tree(fixture, incoming / f"{key}.zip")
    elif drop_mode == "wrapped-zip":
        _zip_tree(fixture, incoming / f"{key}.zip", prefix=key)
    else:
        _copy_tree(fixture, incoming)

    out = datasets.fetch(spec, root=tmp_path)
    assert out is not None
    assert out == datasets.datasets_dir(tmp_path) / key / spec.default_split
    sidecars = sorted(out.glob("*.gt.json"))
    assert sidecars, f"{key}: conversion produced at least one sidecar"

    # Every converted sidecar loads back through the single research GT reader (D-10) -- no second
    # ground-truth reader exists for any of the five datasets.
    for sidecar in sidecars:
        gt = load_research_ground_truth(sidecar)
        assert gt is not None
        assert gt.source == "research"
        assert gt.boxes  # min_length=1 already, but assert the loaded truth is non-empty

    # Provenance records sha256 + source_url + license for exactly the converted images (D-08):
    # one entry per produced sidecar, hashing the real source image each converter consumed.
    manifest = json.loads(
        (datasets.datasets_dir(tmp_path) / "provenance.json").read_text(encoding="utf-8")
    )
    files = manifest["datasets"][key]["files"]
    assert len(files) == len(sidecars)
    for entry in files:
        assert entry["sha256"]
        assert entry["source_url"] == spec.source_url
        assert entry["license"] == spec.license


def test_fetch_missing_archive_is_graceful_for_fscd(tmp_path: Path) -> None:
    # A licence-gated dataset with nothing dropped logs the drop instruction and returns None
    # (T-11-05) -- not just CARPK; the graceful-absence path is generalized to every dataset.
    spec = datasets.DATASET_REGISTRY["fscd147"]
    assert datasets.fetch(spec, root=tmp_path) is None
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
