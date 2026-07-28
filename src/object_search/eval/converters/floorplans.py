"""Roboflow floor-plan COCO exports -> the repo's ``*.gt.json`` sidecar schema (D-10).

The Roboflow *floor-plans-500* export is the first **target-domain** research dataset: real
architectural floor plans where the product's exemplar-search framing is literal -- draw one
``door`` (or ``window``), find every other instance in the same plan. It ships in COCO format with
native ``train`` / ``valid`` / ``test`` splits, each a directory holding ``_annotations.coco.json``
plus the scene PNGs beside it::

    <split>/_annotations.coco.json   # COCO: images[], annotations[] (bbox xywh), categories[]
    <split>/<file_name>.png

Two facts make this converter its own file rather than a branch of an existing one:

* **Multi-class, converted per class.** The export carries ``bathroom``/``door``/``perimeter``/
  ``stairs``/``window`` (plus an unused ``floorplans`` supercategory). The harness's
  :class:`~object_search.eval.labels.GroundTruth` is single-class (all boxes are GT), so this
  converter is called **once per target class**: it keeps only that class's boxes, so
  ``floorplans-door`` and ``floorplans-window`` are two single-class datasets over the same images.
  An exemplar door is then scored against exactly the doors -- the recall denominator is the door
  count, never doors+windows.
* **Per-split images, so it converts one split at a time.** Unlike CARPK's flat ``Images/`` tree,
  floor-plan scenes live inside each split dir. ``normalize_floorplans`` (in
  :mod:`object_search.eval.datasets`) calls this once per scored split (``valid`` -> ``val``,
  ``test``) and records provenance from the
  returned ``image_id -> source png`` map.

Boundary conversion (the one place two conventions meet)
--------------------------------------------------------
COCO ``bbox`` is ``[x, y, w, h]`` in float pixels -- already the repo's :class:`BBox` shape. It
converts **here and nowhere else**: corners are rounded to int, the top-left is clamped to the image
origin (COCO permits negative corners; ``BBox`` requires ``x, y >= 0``), and a box degenerate after
rounding (``w < 1`` or ``h < 1``) is dropped as annotation noise rather than aborting the split::

    BBox(x=max(0, round(x)), y=max(0, round(y)), w=round(w), h=round(h))

No native exemplar box ships, so -- exactly like :mod:`object_search.eval.converters.rpine` -- the
exemplar indices are **sampled from the GT boxes, seeded** (``np.random.default_rng(seed)``, never
``cv2.setRNGSeed``, D-11), stable per image via an FNV-1a offset over the image id. The output is
read back through the single ``_parse_sidecar`` (D-10); no second reader is created.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
from loguru import logger

from object_search.schemas.geometry import BBox

# Number of exemplar indices sampled per image (clamped to the box count). The harness runs at 1 and
# 3 exemplars, so up to three are recorded and the run selects among them (mirrors convert_rpine).
_FLOORPLANS_EXEMPLAR_COUNT = 3

_ANNOTATION_FILE = "_annotations.coco.json"


def _coco_bbox_to_bbox(bbox: list[float]) -> BBox | None:
    """Convert one COCO ``[x, y, w, h]`` (float px) to a half-open :class:`BBox`, or ``None``.

    Returns ``None`` for a box that is degenerate (``w < 1`` or ``h < 1``) after rounding -- COCO
    annotation noise that the strict ``BBox`` would reject; dropping it keeps coverage honest rather
    than aborting the whole split on one bad annotation.
    """
    x, y, w, h = bbox
    x0, y0 = max(0, round(x)), max(0, round(y))
    w0, h0 = round(w), round(h)
    if w0 < 1 or h0 < 1:
        return None
    # The one boundary conversion: COCO xywh -> repo half-open BBox.
    return BBox(x=x0, y=y0, w=w0, h=h0)


def _image_seed_offset(image_id: str) -> int:
    """A small, stable per-image offset so each image's exemplar sampler is independent (D-11).

    Deliberately **not** Python's builtin ``hash`` (salted per process, so not reproducible across
    runs). A fixed FNV-1a over the id bytes gives the same offset every run -- identical to
    :func:`object_search.eval.converters.rpine._image_seed_offset` by design (both sample exemplars
    from GT boxes), inlined here so this converter reads top to bottom on its own.
    """
    digest = 0x811C9DC5
    for byte in image_id.encode("utf-8"):
        digest = ((digest ^ byte) * 0x01000193) & 0xFFFFFFFF
    return digest


def _sample_exemplar_indices(n_boxes: int, seed: int) -> tuple[int, ...]:
    """Deterministically sample up to three exemplar indices from ``[0, n_boxes)`` (D-11).

    Seeded with ``np.random.default_rng(seed)`` so the same seed yields the byte-identical choice.
    Returns the indices sorted; the first is the designated single exemplar.
    """
    rng = np.random.default_rng(seed)
    count = min(_FLOORPLANS_EXEMPLAR_COUNT, n_boxes)
    chosen = rng.choice(n_boxes, size=count, replace=False)
    return tuple(int(i) for i in sorted(chosen))


def _target_category_ids(coco: dict[str, object], target_class: str, path: Path) -> set[int]:
    """The COCO category id(s) whose name is ``target_class`` (matched by name, not a fixed id)."""
    categories = coco.get("categories")
    if not isinstance(categories, list):
        raise ValueError(f"{path}: COCO file has no 'categories' list")
    ids = {c["id"] for c in categories if isinstance(c, dict) and c.get("name") == target_class}
    if not ids:
        names = sorted(str(c.get("name")) for c in categories if isinstance(c, dict))
        raise ValueError(f"{path}: class {target_class!r} not among categories {names}")
    return ids


def convert_floorplans(
    split_dir: Path, out_root: Path, *, target_class: str, seed: int = 0
) -> list[Path]:
    """Convert one COCO split, filtered to ``target_class``, into ``*.gt.json`` sidecars.

    Args:
        split_dir: A Roboflow split directory holding ``_annotations.coco.json`` and the scene PNGs.
        out_root: Destination directory (created if missing) for the sidecars and copied scenes --
            e.g. ``datasets/floorplans-door/val``.
        target_class: The category name to keep (``"door"`` / ``"window"``); every other class's
            boxes are dropped so the sidecar is single-class.
        seed: Seed for the exemplar-index sampler (D-11); the byte-stable output records it
            implicitly. The same seed reproduces the same exemplar choice exactly.

    Returns:
        The written sidecar paths, sorted by image id. An image with **zero** boxes of
        ``target_class`` is skipped (no exemplar search is possible with no instances) rather than
        written as an empty label.

    Raises:
        FileNotFoundError: If ``split_dir/_annotations.coco.json`` does not exist.
        ValueError: If the COCO file lacks ``target_class`` among its categories.
    """
    annotation_path = split_dir / _ANNOTATION_FILE
    if not annotation_path.is_file():
        raise FileNotFoundError(f"floor-plan COCO annotations not found at {annotation_path}")
    out_root.mkdir(parents=True, exist_ok=True)

    coco = json.loads(annotation_path.read_text(encoding="utf-8"))
    target_ids = _target_category_ids(coco, target_class, annotation_path)
    images = {img["id"]: img for img in coco["images"]}

    # Collect the target class's boxes per image, in stable COCO annotation-id order.
    boxes_by_image: dict[int, list[BBox]] = {}
    for ann in coco["annotations"]:
        if ann["category_id"] not in target_ids:
            continue
        box = _coco_bbox_to_bbox(ann["bbox"])
        if box is not None:
            boxes_by_image.setdefault(ann["image_id"], []).append(box)

    written: list[Path] = []
    # Iterate images in image-id order for a deterministic, diffable produced set.
    for coco_image_id, image in sorted(images.items(), key=lambda kv: str(kv[1]["file_name"])):
        boxes = boxes_by_image.get(coco_image_id)
        if not boxes:
            # No instances of the target class in this plan; nothing to search for -- skip honestly.
            continue
        file_name = str(image["file_name"])
        image_id = Path(file_name).stem
        source_image = split_dir / file_name
        if not source_image.is_file():
            logger.warning("floorplans: {} has no image at {}, skipping", image_id, source_image)
            continue

        exemplar_indices = _sample_exemplar_indices(len(boxes), seed + _image_seed_offset(image_id))
        sidecar = {
            "image": f"{image_id}.png",
            "width": int(image["width"]),
            "height": int(image["height"]),
            "achieved_n": len(boxes),
            "exemplar_index": exemplar_indices[0],
            "exemplar_indices": list(exemplar_indices),
            "boxes": [box.model_dump() for box in boxes],
        }
        sidecar_path = out_root / f"{image_id}.gt.json"
        sidecar_path.write_text(
            json.dumps(sidecar, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        shutil.copyfile(source_image, out_root / f"{image_id}.png")
        written.append(sidecar_path)

    logger.info(
        "floorplans[{}]: converted {} image(s) into {}", target_class, len(written), out_root
    )
    return sorted(written)
