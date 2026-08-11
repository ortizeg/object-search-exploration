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

Datasets come in two kinds. **Manual** datasets (CARPK/PUCPR+) sit behind a terms-of-use gate with
no unauthenticated direct-download URL, so ``fetch`` never reaches the network: a human accepts the
licence and drops the archive (or an already-extracted tree) at ``datasets/_incoming/<subdir>/``.
**HuggingFace** datasets (RPINE/FSCD-147/FSCD-LVIS) live on an ungated HF mirror; ``fetch``
downloads the real HF layout anonymously (honouring ``HF_TOKEN`` when set -- anonymous downloads are
IP-rate-limited), then **normalizes-in-fetch**: it reshapes that layout into the exact tree each
EXISTING converter expects and runs the converter unchanged (the ``normalize_*`` functions). Either
way, when the data is absent (missing drop, rate-limited download) ``fetch`` logs an actionable
message and returns ``None`` **without crashing the sweep** (T-11-05, mirroring ``models._export``),
so one missing dataset degrades only itself and the rest of a multi-dataset sweep proceeds.

The raw layouts differ per dataset (``Annotations/`` + ``Images/`` for CARPK/PUCPR+;
``annotations.json`` + ``images/`` + ``split.json`` for FSCD-147/FSCD-LVIS; ``annotations/`` +
``images/`` + ``split.json`` for RPINE), so ``fetch`` is layout-agnostic: each :class:`DatasetSpec`
carries a ``raw_marker`` (the relative path that identifies its raw root) and an ``images_subdir``
(where its source scenes live), and :func:`_resolve_raw_root` / :func:`_provenance_entries` are
driven off those rather than any hardcoded CARPK path. For FSCD-147/FSCD-LVIS only the scored
(val/test) images are converted, so provenance is recorded per produced sidecar -- exactly the
images each converter actually consumed.
"""

from __future__ import annotations

import json
import os
import shutil
import time
import zipfile
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, NamedTuple

from loguru import logger
from PIL import Image
from pydantic import BaseModel, ConfigDict

from object_search import provenance
from object_search.eval.converters import (
    convert_carpk,
    convert_floorplans,
    convert_fscd147,
    convert_rpine,
)

# Two source kinds this phase: ``manual-download`` (licence-gated, human drops the archive at
# ``_incoming/``) and ``huggingface`` (an ungated HF repo pulled anonymously, then reshaped into the
# layout each existing converter already expects -- see the ``normalize_*`` functions below).
DatasetSource = Literal["manual-download", "huggingface"]

# Converter dispatch by dataset key -- the one place the dataset->converter mapping lives. Each
# converter has the same public shape ``(raw_root, out_root) -> list[Path]`` (FSCD-LVIS's protocol
# and RPINE's seed take reproducible defaults), so no base class or decorator is needed. PUCPR+
# ships in CARPK's native format, so it reuses the CARPK converter.
_CONVERTERS: Mapping[str, Callable[[Path, Path], list[Path]]] = {
    "carpk": convert_carpk,
    "pucpr_plus": convert_carpk,
    "fscd147": convert_fscd147,
    "rpine": convert_rpine,
}

# Floor-plan dataset keys -> the COCO category kept for that single-class variant. The Roboflow
# floor-plans-500 export is multi-class; converting it once per class (door, window) yields two
# single-class datasets over the same plans, so the harness's single-class GroundTruth is unchanged
# and an exemplar door is scored only against doors. This is the one manual MULTI-split dataset, so
# it takes the HF-style normalize path (below) rather than the flat single-split manual branch.
_FLOORPLANS_CLASS: Mapping[str, str] = {
    "floorplans-door": "door",
    "floorplans-window": "window",
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
    # The relative path *inside the raw tree* that identifies its root -- ``"Annotations"`` (a dir)
    # for carpk/pucpr_plus, ``"annotations.json"`` (a file) for fscd147/fscd_lvis, ``"annotations"``
    # (a dir) for rpine. ``_resolve_raw_root`` locates the raw root by this marker rather than a
    # hardcoded ``Annotations/``, so a dropped archive that wraps everything in one top-level dir is
    # descended into by looking for the marker, per dataset.
    raw_marker: str
    # The subdir holding the source scene images inside the raw tree -- ``"Images"`` for
    # carpk/pucpr_plus, ``"images"`` for fscd147/fscd_lvis/rpine. ``_provenance_entries`` hashes
    # ``<images_subdir>/<image_id>.png`` for each produced sidecar (so FSCD's val/test-only
    # conversion records exactly the images it converted, not every image in the tree).
    images_subdir: str
    # Pinned SHA-256 of the raw archive when one is known; ``None`` for licence-gated,
    # human-supplied archives whose bytes are recorded (not gated) in provenance.
    archive_sha256: str | None
    # HuggingFace source coordinates, set only when ``source == "huggingface"`` (``None`` for the
    # manual datasets so ``extra="forbid"`` stays valid on every spec). ``hf_repo_id`` is the HF
    # repo (a *dataset* repo); ``hf_files`` names the specific files to pull (the one zip) or is
    # ``None`` to snapshot the whole repo.
    hf_repo_id: str | None
    hf_files: tuple[str, ...] | None
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
        raw_marker="Annotations",
        images_subdir="Images",
        archive_sha256=None,
        hf_repo_id=None,
        hf_files=None,
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
        raw_marker="Annotations",
        images_subdir="Images",
        archive_sha256=None,
        hf_repo_id=None,
        hf_files=None,
        added_in_phase=11,
    ),
    # FSCD-147 -- box-annotated FSC-147 (val/test human boxes, train pseudo). Native train/val/test
    # triple; de-duplicated on load (159 dup images / 11 train<->test leaks, arXiv:2409.15953).
    # Category diversity + leaderboard comparability (D-01). Licence-gated (VinAI/Counting-DETR).
    "fscd147": DatasetSpec(
        key="fscd147",
        source="huggingface",
        source_url="https://huggingface.co/datasets/ChipmunkG4/FSCD-147_FSCD-LVIS_temp",
        license="FSC-147 / FSCD-147 research licence (VinAI / Counting-DETR terms)",
        license_note=(
            "FSC-147 images + FSCD-147 boxes are distributed under the VinAI / Counting-DETR "
            "(ECCV'22, https://github.com/VinAIResearch/Counting-DETR) research terms. The "
            "ungated HF mirror ChipmunkG4/FSCD-147_FSCD-LVIS_temp is pulled anonymously, reshaped "
            "into the converter layout, and de-duplicated on load per arXiv:2409.15953."
        ),
        requires_manual=False,
        incoming_subdir="fscd147",
        default_split="test",
        raw_marker="annotations.json",
        images_subdir="images",
        archive_sha256=None,
        hf_repo_id="ChipmunkG4/FSCD-147_FSCD-LVIS_temp",
        hf_files=("FSCD_147.zip",),
        added_in_phase=11,
    ),
    # FSCD-LVIS (unseen split) -- multi-class crowded scenes, 377 LVIS classes; the only
    # distractor-rejection stress (D-01). No official val -> seeded carve from train (D-03).
    "fscd_lvis": DatasetSpec(
        key="fscd_lvis",
        source="huggingface",
        source_url="https://huggingface.co/datasets/ChipmunkG4/FSCD-147_FSCD-LVIS_temp",
        license="FSCD-LVIS research licence (VinAI / Counting-DETR terms)",
        license_note=(
            "FSCD-LVIS is distributed under the VinAI / Counting-DETR research terms. The ungated "
            "HF mirror ChipmunkG4/FSCD-147_FSCD-LVIS_temp ships FSCD_LVIS.zip (6.3GB). UNVERIFIED: "
            "the zip's internal layout has not been confirmed (anonymous download is IP-blocked at "
            "that size); the normalizer wires it BY ANALOGY to FSCD-147 and degrades gracefully "
            "(logs + returns None) if the zip is absent or its structure differs -- see "
            "`normalize_fscd_lvis`. Use the UNSEEN protocol; a seeded val is carved (D-01/D-03)."
        ),
        requires_manual=False,
        incoming_subdir="fscd_lvis",
        default_split="test",
        raw_marker="annotations.json",
        images_subdir="images",
        archive_sha256=None,
        hf_repo_id="ChipmunkG4/FSCD-147_FSCD-LVIS_temp",
        hf_files=("FSCD_LVIS.zip",),
        added_in_phase=11,
    ),
    # RPINE -- the closest match to this project's task: every repetition in a single image is
    # box-annotated, box exemplars (D-01). No official val -> seeded carve from train (D-03).
    "rpine": DatasetSpec(
        key="rpine",
        source="huggingface",
        source_url="https://huggingface.co/datasets/ChipmunkG4/RPINE",
        license="RPINE research licence (TMR project terms)",
        license_note=(
            "RPINE ('Repeated Patterns IN Everywhere', https://arxiv.org/html/2508.17636) is "
            "distributed under the TMR project terms. The ungated HF repo ChipmunkG4/RPINE is "
            "pulled anonymously (whole repo); its HF train -> our train and HF val (the 435-image "
            "eval set) -> our test (see `normalize_rpine`)."
        ),
        requires_manual=False,
        incoming_subdir="rpine",
        default_split="test",
        raw_marker="annotations",
        images_subdir="images",
        archive_sha256=None,
        hf_repo_id="ChipmunkG4/RPINE",
        hf_files=None,
        added_in_phase=11,
    ),
    # Roboflow floor-plans-500 (COCO) -- the target domain: real architectural plans, exemplar
    # search is literal (one door -> all doors). Manual (accept Roboflow terms, drop the COCO export
    # tree), multi-split (native valid+test), converted PER CLASS into two single-class datasets.
    # `raw_marker="test"` locates the export root by its test/ split dir; `images_subdir` is unused
    # (scenes are per-split, so provenance uses the normalizer image_sources map, not a flat walk).
    "floorplans-door": DatasetSpec(
        key="floorplans-door",
        source="manual-download",
        source_url="https://universe.roboflow.com/university-y9nbi/floor-plans-500",
        license="CC BY 4.0 (Roboflow floor-plans-500, university-y9nbi)",
        license_note=(
            "Confirmed 2026-08-10 directly against the creator's (university-y9nbi) own Roboflow "
            "listing -- CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/). A third-party "
            "Kaggle mirror's 'MIT' tag was checked and discounted: a re-uploader cannot "
            "unilaterally relicense someone else's data, so the original creator's stated terms "
            "govern. Recorded here and gitignored; the raw dataset is never re-hosted (only "
            "attributed derivative excerpts are committed elsewhere, e.g. the overlay gallery in "
            "docs/eval/floorplans-findings.md). A human exports it in COCO format and drops the "
            "extracted train/valid/test tree at datasets/_incoming/floorplans."
        ),
        requires_manual=True,
        incoming_subdir="floorplans",
        default_split="test",
        raw_marker="test",
        images_subdir="",
        archive_sha256=None,
        hf_repo_id=None,
        hf_files=None,
        added_in_phase=12,
    ),
    "floorplans-window": DatasetSpec(
        key="floorplans-window",
        source="manual-download",
        source_url="https://universe.roboflow.com/university-y9nbi/floor-plans-500",
        license="CC BY 4.0 (Roboflow floor-plans-500, university-y9nbi)",
        license_note=(
            "Confirmed 2026-08-10 directly against the creator's (university-y9nbi) own Roboflow "
            "listing -- CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/). A third-party "
            "Kaggle mirror's 'MIT' tag was checked and discounted: a re-uploader cannot "
            "unilaterally relicense someone else's data, so the original creator's stated terms "
            "govern. Recorded here and gitignored; the raw dataset is never re-hosted (only "
            "attributed derivative excerpts are committed elsewhere, e.g. the overlay gallery in "
            "docs/eval/floorplans-findings.md). Shares the one dropped train/valid/test tree at "
            "datasets/_incoming/floorplans/ with floorplans-door."
        ),
        requires_manual=True,
        incoming_subdir="floorplans",
        default_split="test",
        raw_marker="test",
        images_subdir="",
        archive_sha256=None,
        hf_repo_id=None,
        hf_files=None,
        added_in_phase=12,
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


def _resolve_raw_root(
    incoming: Path, dataset_dir: Path, marker: str
) -> tuple[Path | None, Path | None]:
    """Locate a dataset's raw tree from an ``_incoming`` drop, driven by its ``marker``.

    ``marker`` is the relative path (a dir like ``Annotations`` / ``annotations``, or a file like
    ``annotations.json``) that identifies the raw root -- ``spec.raw_marker``. Accepts either a
    dropped ``*.zip`` archive (extracted, zip-slip-guarded, into ``dataset_dir/_raw``) or an
    already-extracted tree. When an archive wraps everything in a single top-level dir the marker is
    not at the top, so it is found recursively and the raw root is set to the marker's parent.

    Returns ``(raw_root, archive)`` -- either element is ``None`` when nothing usable was dropped.
    """
    if not incoming.is_dir():
        return None, None
    archives = sorted(incoming.glob("*.zip"))
    if archives:
        raw_root = dataset_dir / "_raw"
        _safe_extract_zip(archives[0], raw_root)
        # Some archives wrap everything in a single top-level dir; descend to the marker's parent.
        if not (raw_root / marker).exists():
            nested = next((p.parent for p in raw_root.rglob(marker)), None)
            if nested is not None:
                raw_root = nested
        return raw_root, archives[0]
    if (incoming / marker).exists():
        return incoming, None
    return None, None


# =============================================================================================
# HuggingFace path: download the real HF layout, reshape it into the layout each EXISTING
# converter already expects ("normalize-in-fetch"), then run that converter UNCHANGED. The
# converters' box-parsing logic is never touched here -- a normalizer only moves/renames files and
# rebuilds the small index files (``annotations.json`` / ``split.json``) the converter reads.
# =============================================================================================

_HF_RETRY_ATTEMPTS = 3
_HF_RETRY_BASE_SECONDS = 2.0

# HF split name -> our split name. RPINE's HF ``val`` (the 435-image eval set) is our frozen test
# surface; its HF ``train`` is our train. FSCD-147 keeps ``val``/``test`` as-is.
_RPINE_SPLIT_MAP: Mapping[str, str] = {"train": "train", "val": "test"}


def _rpine_label_is_convertible(label_path: Path) -> bool:
    """Whether an RPINE label file holds only boxes ``convert_rpine`` will accept.

    The HF RPINE labels contain a small number of malformed/degenerate annotations (e.g. a
    zero-width box). ``convert_rpine`` is strict and raises on the first one, which would abort the
    whole dataset. The normalizer instead SKIPS such an image (honest coverage -- absence, never a
    fabricated box), mirroring how the FSCD converters drop images they cannot represent. This only
    *screens* whole label files; it never edits a box or re-parses the converter's own logic.
    """
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if not parts:
            continue
        if len(parts) < 4:
            return False
        try:
            x1, y1, x2, y2 = (int(parts[i]) for i in range(4))
        except ValueError:
            return False
        if x2 - x1 < 1 or y2 - y1 < 1:
            return False
    return True


class NormalizedDataset(NamedTuple):
    """What a ``normalize_*`` function produced: per-split output dirs, all sidecars, and the source
    image behind each converted sidecar (so provenance hashes exactly the bytes each converter
    consumed, across however many split dirs were written)."""

    splits: dict[str, Path]
    sidecars: list[Path]
    image_sources: dict[str, Path]


def _link_or_copy(src: Path, dst: Path) -> None:
    """Hard-link ``src`` to ``dst`` (cheap, no byte duplication), falling back to a copy across
    filesystems or when a link already exists."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


def _find_path(root: Path, name: str) -> Path | None:
    """First path named ``name`` at/under ``root``, ignoring hidden dirs like ``.cache``."""
    if (root / name).exists():
        return root / name
    return next(
        (
            p
            for p in sorted(root.rglob(name))
            if not any(part.startswith(".") for part in p.relative_to(root).parts)
        ),
        None,
    )


def _image_sources_from_sidecars(sidecars: list[Path], images_dir: Path) -> dict[str, str]:
    """Map each produced sidecar's image id to its normalized source image (png or jpg)."""
    sources: dict[str, str] = {}
    for sidecar in sidecars:
        image_id = sidecar.name[: -len(".gt.json")]
        for ext in (".png", ".jpg", ".jpeg"):
            candidate = images_dir / f"{image_id}{ext}"
            if candidate.is_file():
                sources[image_id] = str(candidate)
                break
    return sources


def normalize_rpine(hf_dir: Path, dataset_dir: Path) -> NormalizedDataset:
    """Reshape the HF ``ChipmunkG4/RPINE`` layout into ``convert_rpine``'s expected raw tree.

    HF layout: ``<split>/images/<id>.jpg`` + ``<split>/labels/<id>.txt`` (GT: one ``x1 y1 x2 y2``
    pixel box per line, which ALREADY matches ``convert_rpine``'s ``annotations/<id>.txt``) +
    ``<split>/exemplars.json``. HF ``train`` -> our ``train``; HF ``val`` -> our ``test``
    (:data:`_RPINE_SPLIT_MAP`). Per split it builds ``annotations/`` (the labels) + ``images/`` (the
    jpgs) and runs ``convert_rpine`` UNCHANGED into ``datasets/rpine/<our_split>/``.

    TODO: RPINE ships real query exemplars in ``exemplars.json``; ``convert_rpine`` currently
    SAMPLES exemplars from the GT boxes (seeded). Wiring the real ones through is a follow-up --
    kept as GT-sampling for now so the converter stays unchanged.
    """
    # Locate the split root: the dir directly holding ``train/`` and/or ``val/`` split subtrees.
    # Check ``hf_dir`` itself first, then its non-hidden children (a single-wrapping-dir repo),
    # ignoring HF's ``.cache`` scaffolding.
    candidates = [
        hf_dir,
        *(p for p in sorted(hf_dir.iterdir()) if p.is_dir() and not p.name.startswith(".")),
    ]
    root = next(
        (
            candidate
            for candidate in candidates
            if any((candidate / hf_split / "labels").is_dir() for hf_split in _RPINE_SPLIT_MAP)
        ),
        hf_dir,
    )
    splits: dict[str, Path] = {}
    sidecars: list[Path] = []
    image_sources: dict[str, Path] = {}
    for hf_split, our_split in _RPINE_SPLIT_MAP.items():
        labels_dir = root / hf_split / "labels"
        images_dir = root / hf_split / "images"
        if not labels_dir.is_dir():
            logger.info("RPINE: HF split {!r} absent under {}, skipping", hf_split, root)
            continue
        raw = dataset_dir / "_raw" / our_split
        (raw / "annotations").mkdir(parents=True, exist_ok=True)
        (raw / "images").mkdir(parents=True, exist_ok=True)
        skipped = 0
        for label in sorted(labels_dir.glob("*.txt")):
            if not _rpine_label_is_convertible(label):
                logger.warning(
                    "RPINE: {} has a malformed/degenerate box, skipping image", label.stem
                )
                skipped += 1
                continue
            _link_or_copy(label, raw / "annotations" / label.name)
            for ext in (".jpg", ".jpeg", ".png"):
                image = images_dir / f"{label.stem}{ext}"
                if image.is_file():
                    _link_or_copy(image, raw / "images" / image.name)
                    break
        if skipped:
            logger.info(
                "RPINE: skipped {} image(s) with malformed labels in HF {!r}", skipped, hf_split
            )
        out = dataset_dir / our_split
        written = convert_rpine(raw, out)
        splits[our_split] = out
        sidecars.extend(written)
        for image_id, src in _image_sources_from_sidecars(written, raw / "images").items():
            image_sources[image_id] = Path(src)
    return NormalizedDataset(splits=splits, sidecars=sorted(sidecars), image_sources=image_sources)


# FSCD-147: the standard 3-shot protocol uses three exemplar boxes; a handful of images in the HF
# annotation file carry 4-5. The unchanged ``convert_fscd147`` requires exactly three, so the
# normalizer caps to the canonical first three (a NORMALIZER choice; the converter is untouched).
_FSCD147_EXEMPLARS = 3


def _build_fscd147_annotations(
    ann_dir: Path,
) -> tuple[dict[str, dict[str, object]], dict[str, list[str]]]:
    """Build ``convert_fscd147``'s merged ``annotations.json`` + ``split.json`` from the FSC-147
    native files (two COCO instance files + the per-image exemplar/point file + the split file).

    COCO ``bbox`` is XYWH; it is converted to the ``[x1, y1, x2, y2]`` (x2/y2 exclusive) the
    converter's ``_xyxy_to_bbox`` expects -- the ONE numeric reshape, and it is a coordinate
    translation, not a re-parse of the converter's own logic.
    """
    a384 = json.loads((ann_dir / "annotation_FSC147_384.json").read_text(encoding="utf-8"))
    merged: dict[str, dict[str, object]] = {}
    for coco_file in ("instances_val.json", "instances_test.json"):
        coco = json.loads((ann_dir / coco_file).read_text(encoding="utf-8"))
        id_to_stem = {img["id"]: Path(img["file_name"]).stem for img in coco["images"]}
        boxes_by_stem: dict[str, list[list[int]]] = {}
        for ann in coco["annotations"]:
            stem = id_to_stem[ann["image_id"]]
            x, y, w, h = (int(v) for v in ann["bbox"])
            # COCO permits out-of-image (negative) box corners; the repo's BBox requires x,y >= 0,
            # so clamp the lower corner to the image origin here (a coordinate reshape at the source
            # boundary, not a change to the converter). x2/y2 keep the box extent.
            x1, y1, x2, y2 = max(0, x), max(0, y), x + w, y + h
            if x2 - x1 < 1 or y2 - y1 < 1:
                # A box that is degenerate even after clamping is annotation noise; drop it (honest
                # coverage) rather than let the strict converter abort the whole split.
                continue
            boxes_by_stem.setdefault(stem, []).append([x1, y1, x2, y2])
        for stem, boxes in boxes_by_stem.items():
            exemplar_entry = a384.get(f"{stem}.jpg")
            if exemplar_entry is None:
                continue
            merged[stem] = {
                "boxes": boxes,
                # Cap 4-5-exemplar images to the canonical three (see _FSCD147_EXEMPLARS).
                "box_examples_coordinates": exemplar_entry["box_examples_coordinates"][
                    :_FSCD147_EXEMPLARS
                ],
                "points": exemplar_entry.get("points", []),
            }
    tt = json.loads((ann_dir / "Train_Test_Val_FSC_147.json").read_text(encoding="utf-8"))
    split = {name: [Path(f).stem for f in tt.get(name, [])] for name in ("train", "val", "test")}
    return merged, split


def normalize_fscd147(hf_dir: Path, dataset_dir: Path) -> NormalizedDataset:
    """Reshape the unzipped FSCD-147 HF layout into ``convert_fscd147``'s expected raw tree.

    HF layout (after unzip): ``FSC147/images_384_VarV2/<id>.jpg`` +
    ``FSC147/annotations/{instances_val,instances_test,annotation_FSC147_384,Train_Test_Val_FSC_147}
    .json``. The normalizer builds the merged ``annotations.json`` + ``split.json`` the converter
    reads (:func:`_build_fscd147_annotations`), re-encodes the val/test jpgs to the ``<id>.png`` the
    converter copies, and runs ``convert_fscd147`` UNCHANGED once per scored split so
    ``datasets/fscd147/val/`` and ``datasets/fscd147/test/`` each hold their own sidecars.

    Only val/test are converted (their boxes are human; train boxes are pseudo, skipped by the
    converter). The converter's own de-dup of the 11 train<->test leaks + pixel-dups is unaffected;
    val/test have zero id overlap, so the leaks (train-side) never reach these two dirs.
    """
    instances = _find_path(hf_dir, "instances_val.json")
    images_src = _find_path(hf_dir, "images_384_VarV2")
    if instances is None or images_src is None:
        logger.warning(
            "fscd147: could not locate FSC-147 annotations/images under {} (looked for "
            "instances_val.json + images_384_VarV2); the HF zip layout may have changed.",
            hf_dir,
        )
        return NormalizedDataset(splits={}, sidecars=[], image_sources={})
    ann_dir = instances.parent
    merged, split = _build_fscd147_annotations(ann_dir)

    raw = dataset_dir / "_raw"
    raw_images = raw / "images"
    raw_images.mkdir(parents=True, exist_ok=True)
    (raw / "annotations.json").write_text(json.dumps(merged), encoding="utf-8")

    # Re-encode each scored jpg to the <id>.png the converter reads (real PNG, dims preserved).
    scored = sorted(set(split["val"]) | set(split["test"]))
    for stem in scored:
        if stem not in merged:
            continue
        source_jpg = images_src / f"{stem}.jpg"
        target_png = raw_images / f"{stem}.png"
        if source_jpg.is_file() and not target_png.is_file():
            with Image.open(source_jpg) as image:
                image.convert("RGB").save(target_png)

    splits: dict[str, Path] = {}
    sidecars: list[Path] = []
    image_sources: dict[str, Path] = {}
    for our_split in ("val", "test"):
        # One split.json per call so the converter emits that split's ids into its OWN dir.
        per_split = {"train": [], "val": [], "test": [], our_split: split[our_split]}
        (raw / "split.json").write_text(json.dumps(per_split), encoding="utf-8")
        out = dataset_dir / our_split
        written = convert_fscd147(raw, out)
        splits[our_split] = out
        sidecars.extend(written)
        for image_id, src in _image_sources_from_sidecars(written, raw_images).items():
            image_sources[image_id] = Path(src)
    return NormalizedDataset(splits=splits, sidecars=sorted(sidecars), image_sources=image_sources)


def normalize_fscd_lvis(hf_dir: Path, dataset_dir: Path) -> NormalizedDataset:
    """Reshape the unzipped FSCD-LVIS (unseen) HF layout into ``convert_rpine``'s raw tree.

    CONFIRMED real layout (verified against the downloaded 6.3GB zip): ``FSCD_LVIS/images/
    <file_name>`` + ``FSCD_LVIS/annotations/unseen_instances_{train,test}.json`` (COCO: ``images[]``
    + ``annotations[]`` with ``bbox`` in xywh). In the **unseen** split every image carries exactly
    ONE category -- all boxes are the target class, with **no distractors** as delivered -- so
    FSCD-LVIS-unseen is single-class-per-image, exactly the shape ``convert_rpine`` handles (all
    boxes are GT; exemplars are sampled from them, seeded). The normalizer writes RPINE-style
    ``annotations/<id>.txt`` (``x1 y1 x2 y2`` px, degenerate boxes dropped) + ``images/`` and runs
    ``convert_rpine`` UNCHANGED. HF unseen ``test`` -> our ``test``; unseen ``train`` -> our
    ``train`` (a val slice is carved from train later, D-03). A future SEEN-split variant would be
    multi-class with distractors (filter COCO boxes by the exemplar category); the unseen release
    ships neither distractors nor explicit exemplar boxes, so single-class GT + sampled exemplars is
    exactly right.
    """
    test_ann = _find_path(hf_dir, "unseen_instances_test.json")
    images_src = _find_path(hf_dir, "images")
    if test_ann is None or images_src is None:
        logger.warning(
            "fscd_lvis: could not locate unseen_instances_test.json + images/ under {} "
            "(the FSCD_LVIS.zip layout may have changed).",
            hf_dir,
        )
        return NormalizedDataset(splits={}, sidecars=[], image_sources={})
    ann_dir = test_ann.parent
    splits: dict[str, Path] = {}
    sidecars: list[Path] = []
    image_sources: dict[str, Path] = {}
    for coco_name, our_split in (
        ("unseen_instances_test.json", "test"),
        ("unseen_instances_train.json", "train"),
    ):
        coco_path = ann_dir / coco_name
        if not coco_path.is_file():
            continue
        coco = json.loads(coco_path.read_text(encoding="utf-8"))
        id_to_name = {img["id"]: img["file_name"] for img in coco["images"]}
        boxes_by_image: dict[int, list[tuple[int, int, int, int]]] = {}
        for annotation in coco["annotations"]:
            x, y, w, h = annotation["bbox"]
            box = (round(x), round(y), round(x + w), round(y + h))
            if box[2] <= box[0] or box[3] <= box[1]:  # drop degenerate (convert_rpine rejects them)
                continue
            boxes_by_image.setdefault(annotation["image_id"], []).append(box)
        raw = dataset_dir / "_raw" / our_split
        (raw / "annotations").mkdir(parents=True, exist_ok=True)
        (raw / "images").mkdir(parents=True, exist_ok=True)
        for image_id, boxes in boxes_by_image.items():
            file_name = id_to_name.get(image_id)
            if not file_name or not boxes:
                continue
            source_image = images_src / str(file_name)
            if not source_image.is_file():
                continue
            lines = "".join(f"{x1} {y1} {x2} {y2}\n" for x1, y1, x2, y2 in boxes)
            stem = Path(str(file_name)).stem
            (raw / "annotations" / f"{stem}.txt").write_text(lines, encoding="utf-8")
            _link_or_copy(source_image, raw / "images" / str(file_name))
        out = dataset_dir / our_split
        written = convert_rpine(raw, out)
        splits[our_split] = out
        sidecars.extend(written)
        for image_id_str, src in _image_sources_from_sidecars(written, raw / "images").items():
            image_sources[image_id_str] = Path(src)
    return NormalizedDataset(splits=splits, sidecars=sorted(sidecars), image_sources=image_sources)


_NORMALIZERS: Mapping[str, Callable[[Path, Path], NormalizedDataset]] = {
    "rpine": normalize_rpine,
    "fscd147": normalize_fscd147,
    "fscd_lvis": normalize_fscd_lvis,
}

# Roboflow floor-plan COCO splits: valid -> our val, test -> our test. Train is intentionally NOT
# converted -- the exemplar-search methods do no training, so a floor-plan "train" split has no role
# and the manifest's train is empty (native strategy, val + test only).
_FLOORPLANS_SPLIT_MAP: Mapping[str, str] = {"valid": "val", "test": "test"}


def normalize_floorplans(raw_root: Path, dataset_dir: Path, target_class: str) -> NormalizedDataset:
    """Convert the dropped Roboflow floor-plan COCO tree into one class's val+test sidecars.

    The floor-plan export is manual (a human accepts Roboflow's terms and drops the export), but
    unlike the flat CARPK tree its scenes live inside each split dir -- so this reuses the HF path's
    :class:`NormalizedDataset` + ``image_sources`` provenance rather than the single-subdir flat
    walk. ``convert_floorplans`` is run once per scored split
    (``valid`` -> ``val``, ``test``), keeping only ``target_class`` boxes so
    ``datasets/floorplans-<class>/{val,test}/`` are single-class. Provenance hashes each converted
    scene at its source path under the raw split dir.
    """
    splits: dict[str, Path] = {}
    sidecars: list[Path] = []
    image_sources: dict[str, Path] = {}
    for roboflow_split, our_split in _FLOORPLANS_SPLIT_MAP.items():
        split_dir = raw_root / roboflow_split
        if not (split_dir / "_annotations.coco.json").is_file():
            logger.info(
                "floorplans: split {!r} absent under {}, skipping", roboflow_split, raw_root
            )
            continue
        out = dataset_dir / our_split
        written = convert_floorplans(split_dir, out, target_class=target_class)
        splits[our_split] = out
        sidecars.extend(written)
        for sidecar in written:
            image_id = sidecar.name[: -len(".gt.json")]
            source = split_dir / f"{image_id}.png"
            if source.is_file():
                image_sources[image_id] = source
    return NormalizedDataset(splits=splits, sidecars=sorted(sidecars), image_sources=image_sources)


def _revision_from_cache_path(path: Path) -> str | None:
    """Extract the resolved commit sha from an HF cache path (``.../snapshots/<sha>/...``)."""
    parts = path.parts
    if "snapshots" in parts:
        index = parts.index("snapshots")
        if index + 1 < len(parts):
            return parts[index + 1]
    return None


def _hf_retry(call: Callable[[], str]) -> str:
    """Call ``call`` with a small bounded exponential backoff on rate-limit (429) / connection
    errors; other errors propagate immediately."""
    last: Exception | None = None
    for attempt in range(_HF_RETRY_ATTEMPTS):
        try:
            return call()
        except Exception as exc:
            last = exc
            message = str(exc).lower()
            retriable = (
                isinstance(exc, ConnectionError)
                or "429" in message
                or "rate limit" in message
                or "too many requests" in message
            )
            if not retriable or attempt == _HF_RETRY_ATTEMPTS - 1:
                raise
            time.sleep(_HF_RETRY_BASE_SECONDS * (2**attempt))
    if last is not None:  # pragma: no cover -- loop returns or raises before here
        raise last
    raise RuntimeError("unreachable")  # pragma: no cover


def _hf_download(spec: DatasetSpec, hf_dir: Path) -> tuple[Path | None, str | None]:
    """Download ``spec``'s HF repo (or specific files) into ``hf_dir``; extract any zip.

    Anonymous by default (no credentials created); honours ``HF_TOKEN`` from the env if present.
    Returns ``(hf_dir, revision)`` on success or ``(None, None)`` on any download failure, after
    logging an actionable message naming the repo and that an ``HF_TOKEN`` may be required
    (anonymous downloads are IP-rate-limited).
    """
    os.environ["HF_HUB_DISABLE_XET"] = "1"  # set BEFORE importing huggingface_hub
    from huggingface_hub import hf_hub_download, snapshot_download

    token = os.environ.get("HF_TOKEN")
    repo_id = spec.hf_repo_id
    if repo_id is None:  # a "huggingface" spec always sets it; guard narrows the type for mypy
        return None, None
    hf_dir.mkdir(parents=True, exist_ok=True)
    revision = "main"
    try:
        if spec.hf_files:
            for filename in spec.hf_files:

                def _download_one(fname: str = filename) -> str:
                    return hf_hub_download(
                        repo_id=repo_id, filename=fname, repo_type="dataset", token=token
                    )

                downloaded = _hf_retry(_download_one)
                revision = _revision_from_cache_path(Path(downloaded)) or revision
                if filename.lower().endswith(".zip"):
                    _safe_extract_zip(Path(downloaded), hf_dir)
                else:
                    _link_or_copy(Path(downloaded), hf_dir / Path(filename).name)
        else:
            local = _hf_retry(
                lambda: snapshot_download(
                    repo_id=repo_id,
                    repo_type="dataset",
                    local_dir=str(hf_dir),
                    max_workers=2,
                    token=token,
                )
            )
            revision = _revision_from_cache_path(Path(local)) or revision
    except Exception as exc:
        logger.warning(
            "{}: HuggingFace download from {!r} failed ({}). Anonymous downloads are "
            "IP-rate-limited; set HF_TOKEN in the environment and re-run fetch-datasets.",
            spec.key,
            spec.hf_repo_id,
            exc,
        )
        return None, None
    return hf_dir, revision


def _fetch_huggingface(spec: DatasetSpec, out: Path, *, root: Path | None = None) -> Path | None:
    """The ``source == "huggingface"`` branch of :func:`fetch`: download -> normalize -> convert."""
    dataset_dir = datasets_dir(root) / spec.key
    hf_dir, revision = _hf_download(spec, dataset_dir / "_hf")
    if hf_dir is None:
        return None
    result = _NORMALIZERS[spec.key](hf_dir, dataset_dir)
    if not result.sidecars:
        logger.warning("{}: normalization produced no sidecars (see warnings above)", spec.key)
        return None
    write_provenance_manifest(
        spec,
        dataset_dir,
        result.sidecars,
        root=root,
        image_sources=result.image_sources,
        hf_revision=revision,
    )
    logger.info(
        "{}: converted {} image(s) across split(s) {} from HF {}@{}",
        spec.key,
        len(result.sidecars),
        sorted(result.splits),
        spec.hf_repo_id,
        revision,
    )
    if out.is_dir() and any(out.glob("*.gt.json")):
        return out
    return next(iter(result.splits.values()))


def _fetch_floorplans(spec: DatasetSpec, out: Path, *, root: Path | None = None) -> Path | None:
    """The manual floor-plan branch of :func:`fetch`: resolve the dropped COCO tree, then normalize.

    Mirrors :func:`_fetch_huggingface` (normalize -> convert -> record image_sources provenance) but
    reads the human-supplied export from ``datasets/_incoming/floorplans/`` with no network, and
    selects the target class from ``spec.key``. Absent data logs an actionable message and returns
    ``None`` rather than raising, so a floor-plan miss degrades to "skipped" like other datasets
    (T-11-05).
    """
    dataset_dir = datasets_dir(root) / spec.key
    incoming = incoming_dir(root) / spec.incoming_subdir
    raw_root, _ = _resolve_raw_root(incoming, dataset_dir, spec.raw_marker)
    if raw_root is None:
        incoming.mkdir(parents=True, exist_ok=True)
        logger.warning(
            "{}: no data found. Export floor-plans-500 in COCO format from {} and place the "
            "extracted train/valid/test tree at {}, then re-run `pixi run fetch-datasets`.",
            spec.key,
            spec.source_url,
            incoming,
        )
        return None
    result = normalize_floorplans(raw_root, dataset_dir, _FLOORPLANS_CLASS[spec.key])
    if not result.sidecars:
        logger.warning("{}: conversion produced no sidecars (see warnings above)", spec.key)
        return None
    write_provenance_manifest(
        spec, raw_root, result.sidecars, root=root, image_sources=result.image_sources
    )
    logger.info(
        "{}: converted {} image(s) across split(s) {} from {}",
        spec.key,
        len(result.sidecars),
        sorted(result.splits),
        raw_root,
    )
    if out.is_dir() and any(out.glob("*.gt.json")):
        return out
    return next(iter(result.splits.values()), None)


def fetch(spec: DatasetSpec, *, force: bool = False, root: Path | None = None) -> Path | None:
    """Convert one research dataset into ``datasets/<key>/<split>/``.

    Two source kinds. A ``manual-download`` dataset never touches the network: it reads the
    human-supplied archive (or extracted tree) from ``datasets/_incoming/<subdir>/``. A
    ``huggingface`` downloads the real HF layout anonymously, reshapes it into the layout the
    existing converter expects, then runs that converter unchanged (:func:`_fetch_huggingface`).
    Either way, when the data is absent ``fetch`` logs an actionable message and returns ``None``
    rather than raising, so one missing dataset degrades to "skipped" instead of aborting a
    multi-dataset sweep (T-11-05).

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

    # Floor-plans is the one manual MULTI-split dataset: it takes the normalize path (COCO -> per
    # class -> val+test) rather than the flat single-split manual branch below.
    if spec.key in _FLOORPLANS_CLASS:
        return _fetch_floorplans(spec, out, root=root)

    if spec.source == "huggingface":
        return _fetch_huggingface(spec, out, root=root)

    incoming = incoming_dir(root) / spec.incoming_subdir
    raw_root, archive = _resolve_raw_root(incoming, dataset_dir, spec.raw_marker)
    if raw_root is None:
        incoming.mkdir(parents=True, exist_ok=True)
        logger.warning(
            "{}: no data found. Accept the licence at {} and place the archive (or extracted "
            "tree containing {!r}) at {}, then re-run `pixi run fetch-datasets`.",
            spec.key,
            spec.source_url,
            spec.raw_marker,
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
    """One entry per converted image's raw source bytes: sha256 + source_url + licence (D-08).

    Iterates the *produced* sidecars, so a dataset that converts only a subset of its images (FSCD's
    val/test-only conversion) records provenance for exactly those images, not every image in the
    raw tree. The source image lives at ``<images_subdir>/<image_id>.png`` (``Images`` for CARPK's
    capitalized layout, ``images`` for the FSC/RPINE layouts) -- ``spec.images_subdir``.
    """
    entries: list[dict[str, str]] = []
    images_dir = raw_root / spec.images_subdir
    for sidecar in sidecars:
        image_id = sidecar.name[: -len(".gt.json")]
        # Native CARPK ships PNG; the HF mirrors ship JPG. Hash whichever the converter consumed.
        source_image = next(
            (
                candidate
                for ext in (".png", ".jpg", ".jpeg")
                if (candidate := images_dir / f"{image_id}{ext}").is_file()
            ),
            None,
        )
        if source_image is None:
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


def _hf_provenance_entries(
    spec: DatasetSpec, image_sources: Mapping[str, Path]
) -> list[dict[str, str]]:
    """Provenance entries for the HF path, hashing the normalized image behind each sidecar.

    The HF converters write across several split dirs from several normalized image dirs, so the
    flat ``image_id -> normalized image`` map the normalizer returned drives this rather than
    the single-``images_subdir`` walk :func:`_provenance_entries` uses for the manual path.
    """
    entries: list[dict[str, str]] = []
    for image_id, source_image in sorted(image_sources.items()):
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
    spec: DatasetSpec,
    raw_root: Path,
    sidecars: list[Path],
    root: Path | None = None,
    *,
    image_sources: Mapping[str, Path] | None = None,
    hf_revision: str | None = None,
) -> Path:
    """Record (or refresh) ``datasets/provenance.json`` for ``spec`` -- sha256/source/licence.

    Merges into any existing manifest so multiple datasets accumulate; the entries for ``spec.key``
    are rewritten each fetch so a reconvert cannot leave stale hashes. Keys are sorted on write so
    the file is stable and diffable (the config-hash key-ordering discipline, D-11).

    For the HuggingFace path, pass ``image_sources`` (the normalizer's ``image_id -> normalized
    image`` map) and ``hf_revision``; the block then also records the HF repo id and the resolved
    commit revision alongside the per-file hashes.
    """
    manifest_path = datasets_dir(root) / _PROVENANCE_FILENAME
    manifest: dict[str, object] = {"datasets": {}}
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if isinstance(existing, dict) and isinstance(existing.get("datasets"), dict):
            manifest = existing

    datasets_block = manifest["datasets"]
    assert isinstance(datasets_block, dict)  # noqa: S101  -- shape guaranteed above
    block: dict[str, object] = {
        "source_url": spec.source_url,
        "license": spec.license,
        "fetched_at": datetime.now(UTC).isoformat(),
        "files": (
            _hf_provenance_entries(spec, image_sources)
            if image_sources is not None
            else _provenance_entries(spec, raw_root, sidecars)
        ),
    }
    if spec.hf_repo_id is not None:
        block["hf_repo"] = spec.hf_repo_id
    if hf_revision is not None:
        block["hf_revision"] = hf_revision
    datasets_block[spec.key] = block

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
