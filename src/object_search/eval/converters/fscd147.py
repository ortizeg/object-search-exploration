"""FSCD-147 native annotations -> the repo's ``*.gt.json`` sidecar schema, with de-duplication.

FSCD-147 is box-annotated FSC-147 (D-06, EVAL-21/22). Its native form is FSC-147's COCO-style
annotation JSON keyed by image id::

    {
      "<image_id>": {
        "box_examples_coordinates": [[[x, y], ...4 corners...], ...3 exemplars...],
        "points": [[x, y], ...],          # dot annotations -- ignored here (D-06: no dots)
        "boxes":  [[x1, y1, x2, y2], ...]  # one per object; HUMAN in val/test, pseudo in train
      }, ...
    }

plus a ``split.json`` naming the native ``train`` / ``val`` / ``test`` triple. **Only val/test
images carry human boxes**, so only those are converted for scoring; train boxes are pseudo-labels
and are skipped (D-06). The three native exemplar boxes per image become ``exemplar_indices`` (the
first is ``exemplar_index``): the 3-exemplar run honours all three, the 1-exemplar run takes the
first.

Boundary conversion (the one place two conventions meet)
--------------------------------------------------------
Each ``[x1, y1, x2, y2]`` (``x2``/``y2`` exclusive) converts to the repo's half-open
:class:`~object_search.schemas.geometry.BBox` **at this boundary and nowhere else**::

    BBox(x=x1, y=y1, w=x2 - x1, h=y2 - y1)

so no downstream code ever sees the source convention. The converter never parses a sidecar back;
:func:`object_search.eval.labels.load_research_ground_truth` reads the output through the single
``_parse_sidecar`` (D-10) -- there is no second ground-truth reader.

De-duplication (D-07, threat T-11-07)
-------------------------------------
FSC-147 is documented to be contaminated ("Mind the Prompt", **arXiv:2409.15953**): 159 images
recur as 334 **pixel-identical duplicates**, and **11 images leak across the train<->test
boundary**. Both inflate precision/recall/counts if scored, so :func:`dedup_fscd147` drops them
**before any split manifest is built**, so a leaked or duplicated id can never reach the scorer:

* **Train<->test leaks.** The 11 documented leaks are, by their defining property, ids that appear
  in **more than one split** -- so they are caught *structurally* (no fragile id list to transcribe
  or let drift). Any leak documented outside the split files can be added to
  :data:`_DOCUMENTED_TRAIN_TEST_LEAK_IDS`; the structural check already covers the 11.
* **Pixel-identical duplicates.** Detected by content hash (``provenance.file_sha256``); the
  lexicographically-first id is kept as canonical and the rest are dropped.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

from loguru import logger
from PIL import Image
from pydantic import BaseModel, ConfigDict

from object_search.schemas.geometry import BBox

# FSC-147 ships exactly three exemplar boxes per image; the harness runs at 1 and 3 exemplars.
_FSCD147_EXEMPLAR_COUNT = 3

# The count documented in arXiv:2409.15953. Recorded so the source magnitude is visible in-code;
# the actual removal is structural (an id in >1 split) plus content-hash de-duplication, so no
# transcribed filename list can drift out of date. See the module docstring.
DOCUMENTED_FSC147_TRAIN_TEST_LEAK_COUNT = 11

# Additional documented train<->test leaked ids, if any are ever published outside the split files.
# Empty by default: the 11 known leaks appear in both train and test and are caught structurally.
_DOCUMENTED_TRAIN_TEST_LEAK_IDS: frozenset[str] = frozenset()


class Fscd147Splits(BaseModel):
    """The native FSC-147 train/val/test id triple, frozen so a cleaned split cannot be mutated."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    train: tuple[str, ...] = ()
    val: tuple[str, ...] = ()
    test: tuple[str, ...] = ()


class Fscd147DedupResult(BaseModel):
    """The outcome of :func:`dedup_fscd147`: the cleaned splits plus what was dropped and why."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    splits: Fscd147Splits
    dropped_leaks: tuple[str, ...] = ()
    dropped_duplicates: tuple[str, ...] = ()

    @property
    def removed_count(self) -> int:
        """Total ids removed: leaks + (duplicate copies - canonical copies)."""
        return len(self.dropped_leaks) + len(self.dropped_duplicates)


def dedup_fscd147(
    splits: Fscd147Splits,
    image_hashes: Mapping[str, str],
    *,
    documented_leak_ids: frozenset[str] = _DOCUMENTED_TRAIN_TEST_LEAK_IDS,
) -> Fscd147DedupResult:
    """Drop the documented train<->test leaks and the pixel-identical duplicates (D-07, T-11-07).

    Args:
        splits: The native FSC-147 train/val/test id triple.
        image_hashes: ``image_id -> file_sha256`` for the images. Ids sharing a hash are
            pixel-identical duplicates; only the lexicographically-first is kept.
        documented_leak_ids: Extra leaked ids to drop unconditionally. Empty by default -- the 11
            documented FSC-147 leaks are caught structurally (they appear in more than one split).

    Returns:
        A :class:`Fscd147DedupResult` whose ``splits`` contain no id in more than one split and
        none of the dropped leaks or duplicates. ``removed_count`` equals
        ``leaks + (duplicate_copies - canonical_copies)``.
    """
    # 1. Record which split(s) each id appears in.
    membership: dict[str, list[str]] = {}
    for split_name, ids in (("train", splits.train), ("val", splits.val), ("test", splits.test)):
        for image_id in ids:
            membership.setdefault(image_id, []).append(split_name)

    # 2. Leaks: any id in >1 split (the 11 documented train<->test leaks by definition), plus any
    #    explicitly documented id. Dropped from every split so they cannot reach the scorer.
    leaks = {image_id for image_id, splits_in in membership.items() if len(splits_in) > 1}
    leaks |= documented_leak_ids & membership.keys()

    # 3. Pixel-identical duplicates by content hash, among ids not already dropped as leaks. Keep
    #    the lexicographically-first id as canonical; drop the rest.
    by_hash: dict[str, list[str]] = {}
    for image_id in membership:
        if image_id in leaks:
            continue
        digest = image_hashes.get(image_id)
        if digest is not None:
            by_hash.setdefault(digest, []).append(image_id)
    duplicates: set[str] = set()
    for ids_with_hash in by_hash.values():
        if len(ids_with_hash) > 1:
            duplicates |= set(ids_with_hash) - {min(ids_with_hash)}

    removed = leaks | duplicates

    def _clean(ids: Sequence[str]) -> tuple[str, ...]:
        return tuple(sorted(image_id for image_id in ids if image_id not in removed))

    logger.info("FSCD-147 dedup: dropped {} leak(s) + {} duplicate(s)", len(leaks), len(duplicates))
    return Fscd147DedupResult(
        splits=Fscd147Splits(
            train=_clean(splits.train), val=_clean(splits.val), test=_clean(splits.test)
        ),
        dropped_leaks=tuple(sorted(leaks)),
        dropped_duplicates=tuple(sorted(duplicates)),
    )


def load_native_splits(raw_root: Path) -> Fscd147Splits:
    """Read ``split.json`` under ``raw_root`` into a :class:`Fscd147Splits`."""
    payload = json.loads((raw_root / "split.json").read_text(encoding="utf-8"))
    return Fscd147Splits(
        train=tuple(payload.get("train", ())),
        val=tuple(payload.get("val", ())),
        test=tuple(payload.get("test", ())),
    )


def _xyxy_to_bbox(box: Sequence[int]) -> BBox:
    """Convert one ``[x1, y1, x2, y2]`` (x2/y2 exclusive) to the repo's half-open :class:`BBox`."""
    x1, y1, x2, y2 = (int(v) for v in box)
    w, h = x2 - x1, y2 - y1
    if w < 1 or h < 1:
        raise ValueError(f"FSCD-147 box {list(box)} is degenerate (w={w}, h={h})")
    # The one boundary conversion: source xyxy -> repo half-open BBox.
    return BBox(x=x1, y=y1, w=w, h=h)


def _polygon_to_bbox(polygon: Sequence[Sequence[float]]) -> BBox:
    """Convert an exemplar polygon (4 corners ``[[x, y], ...]``) to its bounding :class:`BBox`."""
    xs = [float(pt[0]) for pt in polygon]
    ys = [float(pt[1]) for pt in polygon]
    x1, y1 = round(min(xs)), round(min(ys))
    x2, y2 = round(max(xs)), round(max(ys))
    return _xyxy_to_bbox([x1, y1, x2, y2])


def _exemplar_indices(
    boxes: Sequence[BBox], exemplar_polygons: Sequence[object]
) -> tuple[int, ...]:
    """Map each native exemplar polygon to the index of the object box it best overlaps.

    FSC-147's three exemplar boxes are three of the annotated instances, so each is matched to the
    object box with the highest IoU. Returns their indices in native order (first == the designated
    single exemplar). Raises if the source does not carry exactly three exemplars.
    """
    if len(exemplar_polygons) != _FSCD147_EXEMPLAR_COUNT:
        raise ValueError(
            f"FSCD-147 image must carry {_FSCD147_EXEMPLAR_COUNT} exemplar boxes, "
            f"got {len(exemplar_polygons)}"
        )
    indices: list[int] = []
    for polygon in exemplar_polygons:
        exemplar_box = _polygon_to_bbox(cast(Sequence[Sequence[float]], polygon))
        best = max(range(len(boxes)), key=lambda i: boxes[i].iou(exemplar_box))
        indices.append(best)
    return tuple(indices)


def convert_fscd147(raw_root: Path, out_root: Path) -> list[Path]:
    """Convert an FSCD-147 raw tree into co-located ``*.gt.json`` sidecars under ``out_root``.

    Only **val/test** images are converted (their boxes are human; train boxes are pseudo and
    skipped per D-06). An image with no ``boxes`` produces no sidecar (honest coverage -- absence,
    never an empty sidecar). De-duplication is a **separate** step (:func:`dedup_fscd147`) applied
    at manifest-build time, so it can see all splits and every image hash at once.

    Args:
        raw_root: An FSCD-147 tree with ``images/<id>.png``, ``annotations.json`` + ``split.json``.
        out_root: Destination directory (created if missing) for the sidecars and copied scenes.

    Returns:
        The written sidecar paths, sorted by image id.

    Raises:
        FileNotFoundError: If ``annotations.json`` or ``split.json`` is missing.
        ValueError: If an image has fewer than three exemplars or a degenerate box.
    """
    annotations_path = raw_root / "annotations.json"
    if not annotations_path.is_file():
        raise FileNotFoundError(f"FSCD-147 annotations not found at {annotations_path}")
    annotations = json.loads(annotations_path.read_text(encoding="utf-8"))
    splits = load_native_splits(raw_root)
    images_dir = raw_root / "images"
    out_root.mkdir(parents=True, exist_ok=True)

    # Only val/test carry human boxes and are scored; a duplicated id across val/test is written
    # once (dedup governs which ids reach a manifest, not the on-disk sidecar set).
    scored_ids = sorted(set(splits.val) | set(splits.test))
    written: list[Path] = []
    for image_id in scored_ids:
        entry = annotations.get(image_id)
        if not isinstance(entry, dict):
            continue
        raw_boxes = entry.get("boxes")
        if not raw_boxes:
            # No box annotation -> no sidecar (D-06: honest coverage, never a synthesized box).
            continue

        boxes = [_xyxy_to_bbox(box) for box in raw_boxes]
        exemplar_indices = _exemplar_indices(boxes, entry.get("box_examples_coordinates", []))

        image_path = images_dir / f"{image_id}.png"
        if not image_path.is_file():
            logger.warning("FSCD-147: {} has no image at {}, skipping", image_id, image_path)
            continue
        with Image.open(image_path) as image:
            width, height = image.size

        sidecar = {
            "image": f"{image_id}.png",
            "width": width,
            "height": height,
            "achieved_n": len(boxes),
            "exemplar_index": exemplar_indices[0],
            "exemplar_indices": list(exemplar_indices),
            "boxes": [box.model_dump() for box in boxes],
        }
        sidecar_path = out_root / f"{image_id}.gt.json"
        sidecar_path.write_text(
            json.dumps(sidecar, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        shutil.copyfile(image_path, out_root / f"{image_id}.png")
        written.append(sidecar_path)

    logger.info("FSCD-147: converted {} image(s) into {}", len(written), out_root)
    return sorted(written)
