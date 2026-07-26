"""The research-dataset registry and the ``fetch-datasets`` framework (EVAL-21, D-08).

The exact analogue of :mod:`object_search.inference.models` for *data* instead of *weights*. Every
research dataset is one entry in :data:`DATASET_REGISTRY`; the raw bytes are gitignored and arrive
only through ``fetch-datasets``, which records **SHA-256 + source URL + licence** per file in
``datasets/provenance.json`` (mirroring the ``fetch-models`` provenance discipline). No raw dataset
file ever enters git, and each dataset's licence is *recorded*, never re-hosted.

Two threats shape this module (see the plan's threat register):

* **T-11-02 zip-slip.** A malicious archive member named ``../../etc/x`` would escape the dataset
  dir on extraction. :func:`_safe_extract_zip` resolves every member against the destination and
  refuses any that escapes, before writing a single byte.
* **T-11-01 archive tampering.** When a spec pins ``archive_sha256`` the download is verified and a
  mismatch refuses to install (as :func:`object_search.inference.models.fetch` does). CARPK's
  archive is licence-gated and human-supplied, so it is unpinned (``None``): its bytes are
  *recorded* in provenance rather than gated, exactly as the unpinned FastSAM export is.

CARPK is ``requires_manual``: the data sits behind a terms-of-use gate with no unauthenticated
direct-download URL, so ``fetch`` never reaches the network. A human accepts the licence and drops
the archive at ``datasets/_incoming/carpk/``; when it is absent ``fetch`` logs that instruction and
returns ``None`` **without crashing the sweep** (T-11-05, mirroring ``models._export``).
"""

from __future__ import annotations

import json
import zipfile
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from loguru import logger
from pydantic import BaseModel, ConfigDict

from object_search import provenance
from object_search.eval.converters import (
    convert_carpk,
    convert_fscd147,
    convert_fscd_lvis,
    convert_rpine,
)

# Only manual (licence-gated) sources exist this phase. A future direct-download dataset extends
# this Literal in its own plan; keeping it to what is exercised avoids a dead fetch branch.
DatasetSource = Literal["manual-download"]

# Converter dispatch by dataset key -- the one place the dataset->converter mapping lives. Each
# converter has the same public shape ``(raw_root, out_root) -> list[Path]`` (FSCD-LVIS's protocol
# and RPINE's seed take reproducible defaults), so no base class or decorator is needed. PUCPR+
# ships in CARPK's native format, so it reuses the CARPK converter.
_CONVERTERS: Mapping[str, Callable[[Path, Path], list[Path]]] = {
    "carpk": convert_carpk,
    "pucpr_plus": convert_carpk,
    "fscd147": convert_fscd147,
    "fscd_lvis": convert_fscd_lvis,
    "rpine": convert_rpine,
}

_PROVENANCE_FILENAME = "provenance.json"


class DatasetSpec(BaseModel):
    """One research dataset: where it comes from, its licence, and how it lands in ``datasets/``.

    Mirrors :class:`object_search.inference.models.ModelSpec` (frozen, ``extra="forbid"``) plus the
    two fields data needs that weights do not: ``requires_manual`` (the source is licence-gated and
    human-supplied, so ``fetch`` must not try the network) and ``incoming_subdir`` (where the human
    drops the archive under ``datasets/_incoming/``).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    key: str
    source: DatasetSource
    source_url: str
    license: str
    license_note: str
    requires_manual: bool
    incoming_subdir: str
    default_split: str
    # Pinned SHA-256 of the raw archive when one is known; ``None`` for licence-gated,
    # human-supplied archives whose bytes are recorded (not gated) in provenance.
    archive_sha256: str | None
    added_in_phase: int


DATASET_REGISTRY: Mapping[str, DatasetSpec] = {
    # Phase 11 tracer: CARPK -- dense near-identical cars, drone view, single-class ("car"),
    # test-only cross-domain probe (D-04). Native format: Images/<id>.png + Annotations/<id>.txt
    # (`x1 y1 x2 y2 class`). Licence-gated (author terms-of-use), so requires_manual.
    "carpk": DatasetSpec(
        key="carpk",
        source="manual-download",
        source_url="https://lafi.github.io/LPN/",
        license="CARPK terms-of-use (non-commercial research)",
        license_note=(
            "CARPK/PUCPR+ are distributed behind an author terms-of-use gate (request form); "
            "there is no unauthenticated direct-download URL and the data must not be re-hosted. "
            "Recorded here and gitignored; a human accepts the licence and supplies the archive."
        ),
        requires_manual=True,
        incoming_subdir="carpk",
        default_split="test",
        archive_sha256=None,
        added_in_phase=11,
    ),
    # PUCPR+ -- CARPK's sibling (same author release, same native format): ~16k cars, fixed slanted
    # building-camera view. Test-only cross-domain probe (D-04). Licence-gated with CARPK.
    "pucpr_plus": DatasetSpec(
        key="pucpr_plus",
        source="manual-download",
        source_url="https://lafi.github.io/LPN/",
        license="CARPK/PUCPR+ terms-of-use (non-commercial research)",
        license_note=(
            "PUCPR+ ships with CARPK behind the same author terms-of-use gate (request form); no "
            "unauthenticated direct-download URL and the data must not be re-hosted. Recorded and "
            "gitignored; a human accepts the licence and supplies the archive."
        ),
        requires_manual=True,
        incoming_subdir="pucpr_plus",
        default_split="test",
        archive_sha256=None,
        added_in_phase=11,
    ),
    # FSCD-147 -- box-annotated FSC-147 (val/test human boxes, train pseudo). Native train/val/test
    # triple; de-duplicated on load (159 dup images / 11 train<->test leaks, arXiv:2409.15953).
    # Category diversity + leaderboard comparability (D-01). Licence-gated (VinAI/Counting-DETR).
    "fscd147": DatasetSpec(
        key="fscd147",
        source="manual-download",
        source_url="https://research.vinai.io/few-shot-object-counting-and-detection/",
        license="FSC-147 / FSCD-147 research licence (VinAI / Counting-DETR terms)",
        license_note=(
            "FSC-147 images + FSCD-147 boxes are distributed under the VinAI / Counting-DETR "
            "(ECCV'22, https://github.com/VinAIResearch/Counting-DETR) research terms; a human "
            "accepts them and supplies the archive. De-duplicated on load per arXiv:2409.15953."
        ),
        requires_manual=True,
        incoming_subdir="fscd147",
        default_split="test",
        archive_sha256=None,
        added_in_phase=11,
    ),
    # FSCD-LVIS (unseen split) -- multi-class crowded scenes, 377 LVIS classes; the only
    # distractor-rejection stress (D-01). No official val -> seeded carve from train (D-03).
    "fscd_lvis": DatasetSpec(
        key="fscd_lvis",
        source="manual-download",
        source_url="https://github.com/VinAIResearch/Counting-DETR",
        license="FSCD-LVIS research licence (VinAI / Counting-DETR terms)",
        license_note=(
            "FSCD-LVIS is distributed under the VinAI / Counting-DETR research terms; a human "
            "accepts them and supplies the archive. Use the UNSEEN protocol (no official val) for "
            "the headline number; a seeded val is carved from train (D-01/D-03)."
        ),
        requires_manual=True,
        incoming_subdir="fscd_lvis",
        default_split="test",
        archive_sha256=None,
        added_in_phase=11,
    ),
    # RPINE -- the closest match to this project's task: every repetition in a single image is
    # box-annotated, box exemplars (D-01). No official val -> seeded carve from train (D-03).
    "rpine": DatasetSpec(
        key="rpine",
        source="manual-download",
        source_url="https://chipmunk-g4.github.io/TMR/",
        license="RPINE research licence (TMR project terms)",
        license_note=(
            "RPINE ('Repeated Patterns IN Everywhere', https://arxiv.org/html/2508.17636) is "
            "distributed under the TMR project terms; a human accepts them and supplies the "
            "archive. No official val -> a seeded val is carved from train (D-03)."
        ),
        requires_manual=True,
        incoming_subdir="rpine",
        default_split="test",
        archive_sha256=None,
        added_in_phase=11,
    ),
}


def datasets_dir(root: Path | None = None) -> Path:
    """The gitignored ``datasets/`` directory at the repo root, created if missing."""
    directory = (root if root is not None else provenance.repo_root()) / "datasets"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def incoming_dir(root: Path | None = None) -> Path:
    """The ``datasets/_incoming/`` drop directory for licence-gated, human-supplied archives."""
    directory = datasets_dir(root) / "_incoming"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _safe_extract_zip(archive: Path, dest: Path) -> None:
    """Extract ``archive`` into ``dest``, refusing any member that escapes ``dest`` (T-11-02).

    Zip-slip guard: each member's final path is resolved and checked to be inside ``dest`` *before*
    anything is written, so a crafted ``../`` member cannot land outside the dataset directory.
    """
    dest.mkdir(parents=True, exist_ok=True)
    dest_resolved = dest.resolve()
    with zipfile.ZipFile(archive) as zf:
        for member in zf.namelist():
            target = (dest / member).resolve()
            if target != dest_resolved and dest_resolved not in target.parents:
                raise ValueError(
                    f"{archive}: member {member!r} escapes {dest} (zip-slip); refusing to extract"
                )
        zf.extractall(dest)


def _resolve_raw_root(incoming: Path, dataset_dir: Path) -> tuple[Path | None, Path | None]:
    """Locate the raw CARPK tree from an ``_incoming`` drop.

    Accepts either a dropped ``*.zip`` archive (extracted, zip-slip-guarded, into ``dataset_dir/
    _raw``) or an already-extracted tree containing ``Annotations/``. Returns
    ``(raw_root, archive)`` where either element is ``None`` when nothing usable was dropped.
    """
    if not incoming.is_dir():
        return None, None
    archives = sorted(incoming.glob("*.zip"))
    if archives:
        raw_root = dataset_dir / "_raw"
        _safe_extract_zip(archives[0], raw_root)
        # Some archives wrap everything in a single top-level dir; descend to the Annotations tree.
        if not (raw_root / "Annotations").is_dir():
            nested = next((p.parent for p in raw_root.rglob("Annotations") if p.is_dir()), None)
            if nested is not None:
                raw_root = nested
        return raw_root, archives[0]
    if (incoming / "Annotations").is_dir():
        return incoming, None
    return None, None


def fetch(spec: DatasetSpec, *, force: bool = False, root: Path | None = None) -> Path | None:
    """Convert one research dataset from its ``_incoming`` drop into ``datasets/<key>/<split>/``.

    For a ``requires_manual`` dataset this never touches the network: it reads the human-supplied
    archive (or extracted tree) from ``datasets/_incoming/<subdir>/``. When nothing is there it logs
    the exact drop path and returns ``None`` rather than raising, so a missing licence-gated dataset
    degrades that dataset to "skipped" instead of aborting a multi-dataset sweep (T-11-05).

    Args:
        spec: The dataset to fetch.
        force: Reconvert even if the output split already exists.
        root: Optional base dir (defaults to the repo root); tests pass a ``tmp_path``.

    Returns:
        The output split directory, or ``None`` when the archive is absent.

    Raises:
        ValueError: On a zip-slip member (T-11-02) or an ``archive_sha256`` mismatch (T-11-01).
    """
    dataset_dir = datasets_dir(root) / spec.key
    out = dataset_dir / spec.default_split
    if out.is_dir() and any(out.glob("*.gt.json")) and not force:
        logger.info("{}: already converted at {}, skipping", spec.key, out)
        return out

    incoming = incoming_dir(root) / spec.incoming_subdir
    raw_root, archive = _resolve_raw_root(incoming, dataset_dir)
    if raw_root is None:
        incoming.mkdir(parents=True, exist_ok=True)
        logger.warning(
            "{}: no data found. Accept the licence at {} and place the archive (or extracted "
            "Images/ + Annotations/ tree) at {}, then re-run `pixi run fetch-datasets`.",
            spec.key,
            spec.source_url,
            incoming,
        )
        return None

    if archive is not None and spec.archive_sha256 is not None:
        actual = provenance.file_sha256(archive)
        if actual != spec.archive_sha256:
            archive.unlink(missing_ok=True)
            raise ValueError(
                f"{spec.key}: archive sha256 mismatch (expected {spec.archive_sha256}, got "
                f"{actual}); refusing to install data that does not match its pinned hash"
            )

    sidecars = _CONVERTERS[spec.key](raw_root, out)
    write_provenance_manifest(spec, raw_root, sidecars, root=root)
    logger.info("{}: converted {} image(s) into {}", spec.key, len(sidecars), out)
    return out


def _provenance_entries(
    spec: DatasetSpec, raw_root: Path, sidecars: list[Path]
) -> list[dict[str, str]]:
    """One entry per converted image's raw source bytes: sha256 + source_url + licence (D-08)."""
    entries: list[dict[str, str]] = []
    images_dir = raw_root / "Images"
    for sidecar in sidecars:
        image_id = sidecar.name[: -len(".gt.json")]
        source_image = images_dir / f"{image_id}.png"
        if not source_image.is_file():
            continue
        entries.append(
            {
                "image_id": image_id,
                "sha256": provenance.file_sha256(source_image),
                "source_url": spec.source_url,
                "license": spec.license,
            }
        )
    return entries


def write_provenance_manifest(
    spec: DatasetSpec, raw_root: Path, sidecars: list[Path], root: Path | None = None
) -> Path:
    """Record (or refresh) ``datasets/provenance.json`` for ``spec`` -- sha256/source/licence.

    Merges into any existing manifest so multiple datasets accumulate; the entries for ``spec.key``
    are rewritten each fetch so a reconvert cannot leave stale hashes. Keys are sorted on write so
    the file is stable and diffable (the config-hash key-ordering discipline, D-11).
    """
    manifest_path = datasets_dir(root) / _PROVENANCE_FILENAME
    manifest: dict[str, object] = {"datasets": {}}
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if isinstance(existing, dict) and isinstance(existing.get("datasets"), dict):
            manifest = existing

    datasets_block = manifest["datasets"]
    assert isinstance(datasets_block, dict)  # noqa: S101  -- shape guaranteed above
    datasets_block[spec.key] = {
        "source_url": spec.source_url,
        "license": spec.license,
        "fetched_at": datetime.now(UTC).isoformat(),
        "files": _provenance_entries(spec, raw_root, sidecars),
    }

    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    logger.info(
        "{}: recorded provenance for {} file(s) in {}", spec.key, len(sidecars), manifest_path
    )
    return manifest_path


def fetch_all(*, force: bool = False, root: Path | None = None) -> Mapping[str, Path | None]:
    """Fetch every registered dataset. Returns ``{key: output split dir or None}``."""
    return {key: fetch(spec, force=force, root=root) for key, spec in DATASET_REGISTRY.items()}


def verify_all(root: Path | None = None) -> Mapping[str, bool]:
    """For each dataset, whether its converted split exists on disk (with >=1 sidecar)."""
    results: dict[str, bool] = {}
    for key, spec in DATASET_REGISTRY.items():
        out = datasets_dir(root) / spec.key / spec.default_split
        results[key] = out.is_dir() and any(out.glob("*.gt.json"))
    return results
