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

import huggingface_hub
import pytest
from PIL import Image
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
        # CARPK is the manual (licence-gated) ``_incoming`` path: a plain zip, a single-wrapping-dir
        # zip (exercising the raw-marker descent), and an already-extracted tree. The HF datasets
        # (rpine/fscd147/fscd_lvis) take the download+normalize path, covered separately below.
        ("carpk", "zip"),
        ("carpk", "wrapped-zip"),
        ("carpk", "tree"),
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


def test_fetch_missing_archive_is_graceful_for_pucpr(tmp_path: Path) -> None:
    # A licence-gated dataset with nothing dropped logs the drop instruction and returns None
    # (T-11-05) -- not just CARPK; the graceful-absence path is generalized to every manual dataset.
    spec = datasets.DATASET_REGISTRY["pucpr_plus"]
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


# ------------------------------------------------------------------ HuggingFace normalize-in-fetch
# The three research datasets (rpine/fscd147/fscd_lvis) download the real HF layout and reshape it
# into the layout each EXISTING converter already expects. These tests build tiny HF-shaped fixtures
# and monkeypatch huggingface_hub's downloaders so NO network is touched, then assert the full
# download -> normalize -> convert -> provenance path.


def _fake_hf_hub_download(zip_path: Path, sha: str = "deadbeef1234"):  # noqa: ANN202
    """A stand-in for ``hf_hub_download`` that returns ``zip_path`` from a snapshots-shaped cache
    dir (so the resolved-revision extraction has a commit sha to find)."""

    def _download(repo_id, filename, repo_type=None, token=None, **_kwargs):  # noqa: ANN202, ANN003
        snapshot = zip_path.parent / "snapshots" / sha
        snapshot.mkdir(parents=True, exist_ok=True)
        target = snapshot / filename
        target.write_bytes(zip_path.read_bytes())
        return str(target)

    return _download


def _fake_snapshot_download(src_tree: Path):  # noqa: ANN202
    """Stand-in for ``snapshot_download`` copying ``src_tree`` into the requested local dir."""

    def _download(  # noqa: ANN202
        repo_id, repo_type=None, local_dir=None, max_workers=None, token=None, **_kw: object
    ):
        _copy_tree(src_tree, Path(local_dir))
        return str(local_dir)

    return _download


def _poly(x: int, y: int, w: int, h: int) -> list[list[int]]:
    """A 4-corner exemplar polygon (the FSC-147 ``box_examples_coordinates`` shape)."""
    return [[x, y], [x, y + h], [x + w, y + h], [x + w, y]]


def _build_fscd147_hf_zip(dest_zip: Path, work: Path) -> None:
    """Build a tiny FSCD-147 HF-shaped archive: FSC147/{annotations,images_384_VarV2}."""
    root = work / "fscd147_src"
    ann = root / "FSC147" / "annotations"
    imgs = root / "FSC147" / "images_384_VarV2"
    ann.mkdir(parents=True)
    imgs.mkdir(parents=True)
    for name in ("v1.jpg", "t1.jpg"):
        Image.new("RGB", (40, 40), (120, 120, 120)).save(imgs / name)

    def _coco(file_name: str, boxes_xywh: list[list[int]]) -> dict:
        return {
            "images": [{"file_name": file_name, "id": 1, "height": 40, "width": 40}],
            "type": "instances",
            "annotations": [
                {"bbox": b, "image_id": 1, "category_id": 1, "id": i + 1, "iscrowd": 0}
                for i, b in enumerate(boxes_xywh)
            ],
            "categories": [{"supercategory": "none", "id": 1, "name": "fg"}],
        }

    (ann / "instances_val.json").write_text(
        json.dumps(_coco("v1.jpg", [[2, 2, 8, 8], [14, 2, 8, 8]]))
    )
    (ann / "instances_test.json").write_text(json.dumps(_coco("t1.jpg", [[2, 2, 10, 10]])))
    a384 = {
        # 4 exemplars for v1 exercises the normalizer's cap-to-3 (the canonical 3-shot protocol).
        "v1.jpg": {
            "box_examples_coordinates": [
                _poly(2, 2, 6, 6),
                _poly(14, 2, 6, 6),
                _poly(2, 2, 6, 6),
                _poly(14, 2, 6, 6),
            ],
            "points": [[3, 3]],
        },
        "t1.jpg": {
            "box_examples_coordinates": [_poly(2, 2, 8, 8), _poly(2, 2, 8, 8), _poly(2, 2, 8, 8)],
            "points": [],
        },
    }
    (ann / "annotation_FSC147_384.json").write_text(json.dumps(a384))
    (ann / "Train_Test_Val_FSC_147.json").write_text(
        json.dumps({"train": [], "val": ["v1.jpg"], "test": ["t1.jpg"]})
    )
    _zip_tree(root, dest_zip)


def _build_rpine_hf_tree(dest: Path) -> None:
    """Build a tiny RPINE HF repo tree: <split>/{images/<id>.jpg,labels/<id>.txt,exemplars.json}."""
    for split, stems in (("train", ("a", "b")), ("val", ("c",))):
        (dest / split / "images").mkdir(parents=True)
        (dest / split / "labels").mkdir(parents=True)
        exemplars: dict[str, list[list[int]]] = {}
        for stem in stems:
            Image.new("RGB", (40, 40)).save(dest / split / "images" / f"{stem}.jpg")
            (dest / split / "labels" / f"{stem}.txt").write_text("2 2 12 12\n14 2 24 12\n")
            exemplars[stem] = [[2, 2, 12, 12]]
        (dest / split / "exemplars.json").write_text(json.dumps(exemplars))


def test_fetch_fscd147_via_huggingface(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    zip_path = tmp_path / "src" / "FSCD_147.zip"
    zip_path.parent.mkdir(parents=True)
    _build_fscd147_hf_zip(zip_path, tmp_path)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _fake_hf_hub_download(zip_path))

    spec = datasets.DATASET_REGISTRY["fscd147"]
    work = tmp_path / "work"
    out = datasets.fetch(spec, root=work)
    assert out is not None

    for split, expected in (("val", 1), ("test", 1)):
        sidecars = sorted((datasets.datasets_dir(work) / "fscd147" / split).glob("*.gt.json"))
        assert len(sidecars) == expected, f"{split} should hold its OWN sidecars"
        for sidecar in sidecars:
            gt = load_research_ground_truth(sidecar)
            assert gt is not None and gt.source == "research" and gt.boxes

    manifest = json.loads((datasets.datasets_dir(work) / "provenance.json").read_text())
    block = manifest["datasets"]["fscd147"]
    assert block["hf_repo"] == spec.hf_repo_id
    assert block["hf_revision"] == "deadbeef1234"
    assert len(block["files"]) == 2  # one per converted image (v1 + t1)
    for entry in block["files"]:
        assert entry["sha256"]
        assert entry["source_url"] == spec.source_url
        assert entry["license"] == spec.license


def test_fetch_rpine_via_huggingface_maps_val_to_test(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src = tmp_path / "rpine_src"
    _build_rpine_hf_tree(src)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", _fake_snapshot_download(src))

    spec = datasets.DATASET_REGISTRY["rpine"]
    work = tmp_path / "work"
    out = datasets.fetch(spec, root=work)
    assert out is not None

    train = datasets.datasets_dir(work) / "rpine" / "train"
    test = datasets.datasets_dir(work) / "rpine" / "test"  # HF val -> our test
    assert len(list(train.glob("*.gt.json"))) == 2  # HF train
    assert len(list(test.glob("*.gt.json"))) == 1  # HF val
    assert list(train.glob("*.png")), "convert_rpine co-locates the scene beside the sidecar"
    # The jpg source was accepted (the one allowed converter change) and loads back as research GT.
    gt = load_research_ground_truth(sorted(train.glob("*.gt.json"))[0])
    assert gt is not None and gt.source == "research" and gt.boxes

    manifest = json.loads((datasets.datasets_dir(work) / "provenance.json").read_text())
    assert manifest["datasets"]["rpine"]["hf_repo"] == spec.hf_repo_id


def test_fetch_fscd_lvis_via_huggingface_happy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # FSCD-LVIS (unseen) is single-class COCO: normalize_fscd_lvis reads unseen_instances_test.json
    # (xywh boxes as GT) and runs convert_rpine (samples exemplars). Exercised on the fixture.
    zip_path = tmp_path / "src" / "FSCD_LVIS.zip"
    zip_path.parent.mkdir(parents=True)
    _zip_tree(_fixture_root("fscd_lvis"), zip_path)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _fake_hf_hub_download(zip_path))

    spec = datasets.DATASET_REGISTRY["fscd_lvis"]
    work = tmp_path / "work"
    out = datasets.fetch(spec, root=work)
    assert out is not None
    sidecars = sorted((datasets.datasets_dir(work) / "fscd_lvis" / "test").glob("*.gt.json"))
    assert sidecars
    for sidecar in sidecars:
        gt = load_research_ground_truth(sidecar)
        assert gt is not None and gt.source == "research" and gt.boxes


def test_fetch_fscd_lvis_unverified_structure_degrades_gracefully(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # If the expected unseen_instances_test.json marker is absent (a changed zip layout), fetch
    # returns None (never crashes the sweep) and writes no provenance -- like a missing archive.
    zip_path = tmp_path / "src" / "FSCD_LVIS.zip"
    zip_path.parent.mkdir(parents=True)
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("some_other_layout/readme.txt", "unknown internal structure")
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _fake_hf_hub_download(zip_path))

    spec = datasets.DATASET_REGISTRY["fscd_lvis"]
    work = tmp_path / "work"
    assert datasets.fetch(spec, root=work) is None
    assert not (datasets.datasets_dir(work) / "provenance.json").is_file()


def test_fetch_huggingface_download_failure_is_graceful(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A rate-limited / failed download degrades to None with an actionable log, no provenance
    # (T-11-05). Backoff sleeps are stubbed so the bounded retry does not slow the suite.
    def _boom(*_args, **_kwargs):  # noqa: ANN002,ANN003,ANN202
        raise RuntimeError("429 Client Error: Too Many Requests")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _boom)
    monkeypatch.setattr(datasets.time, "sleep", lambda _s: None)

    spec = datasets.DATASET_REGISTRY["fscd147"]
    work = tmp_path / "work"
    assert datasets.fetch(spec, root=work) is None
    assert not (datasets.datasets_dir(work) / "provenance.json").is_file()


def test_hf_retry_backs_off_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    slept: list[float] = []
    monkeypatch.setattr(datasets.time, "sleep", lambda seconds: slept.append(seconds))
    calls = {"n": 0}

    def _call() -> str:
        calls["n"] += 1
        if calls["n"] < 3:
            raise ConnectionError("transient")
        return "ok"

    assert datasets._hf_retry(_call) == "ok"
    assert calls["n"] == 3
    assert len(slept) == 2  # two backoffs before the third, successful attempt


def test_hf_retry_non_retriable_raises_immediately(monkeypatch: pytest.MonkeyPatch) -> None:
    def _sleep(_seconds: float) -> None:
        raise AssertionError("non-retriable errors must not back off")

    monkeypatch.setattr(datasets.time, "sleep", _sleep)

    def _call() -> str:
        raise ValueError("permanent")

    with pytest.raises(ValueError, match="permanent"):
        datasets._hf_retry(_call)


def test_revision_from_cache_path() -> None:
    cache = Path("/c/models--x/snapshots/abc123/f.zip")
    assert datasets._revision_from_cache_path(cache) == "abc123"
    assert datasets._revision_from_cache_path(Path("/c/local_dir/f.zip")) is None
