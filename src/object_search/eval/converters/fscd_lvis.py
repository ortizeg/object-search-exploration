"""FSCD-LVIS (unseen split) native annotations -> the repo's ``*.gt.json`` sidecar schema (D-06).

FSCD-LVIS is the **distractor-rejection** dataset: multi-class crowded scenes where several
repeated objects share an image, so it measures whether a method finds the *right* object rather
than any repeat (the gap FSC-147's single-class-per-image labelling leaves). Its native form is a
COCO-style annotation JSON keyed by image id::

    {
      "<image_id>": {
        "exemplar_category": <category_id>,           # the class the exemplars belong to
        "box_examples_coordinates": [[[x, y], ...4...], ...3 exemplars...],
        "annotations": [ {"box": [x1, y1, x2, y2], "category": <id>}, ... ]  # ALL classes present
      }, ...
    }

plus a ``split.json`` naming the ``train`` / ``test`` ids (the **unseen** protocol has **no
official val** -- one is carved from train, seeded, by :func:`object_search.eval.splits.carve_val`).

Only the boxes of the **exemplar category** become ground truth -- those are the instances a
correct search must find. Boxes of *other* categories are the distractors: they exist in the image
pixels but are deliberately **not** ground truth, so a method that returns them is scored as a false
positive (exactly the distractor-rejection signal). No dot-only annotation is turned into a box
(D-06): an image with no exemplar-category box produces no sidecar.

Boundary conversion (the one place two conventions meet)
--------------------------------------------------------
Each ``[x1, y1, x2, y2]`` (x2/y2 exclusive) converts to the repo's half-open
:class:`~object_search.schemas.geometry.BBox` **at this boundary and nowhere else**::

    BBox(x=x1, y=y1, w=x2 - x1, h=y2 - y1)

The output is read back through the single ``_parse_sidecar`` (D-10); no second reader is created.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, cast

from loguru import logger
from PIL import Image

from object_search.schemas.geometry import BBox

_FSCD_LVIS_EXEMPLAR_COUNT = 3
Protocol = Literal["unseen", "seen"]


def _xyxy_to_bbox(box: Sequence[int]) -> BBox:
    """Convert one ``[x1, y1, x2, y2]`` (x2/y2 exclusive) to the repo's half-open :class:`BBox`."""
    x1, y1, x2, y2 = (int(v) for v in box)
    w, h = x2 - x1, y2 - y1
    if w < 1 or h < 1:
        raise ValueError(f"FSCD-LVIS box {list(box)} is degenerate (w={w}, h={h})")
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
    """Map each native exemplar polygon to the target-category box index it best overlaps."""
    if len(exemplar_polygons) != _FSCD_LVIS_EXEMPLAR_COUNT:
        raise ValueError(
            f"FSCD-LVIS image must carry {_FSCD_LVIS_EXEMPLAR_COUNT} exemplar boxes, "
            f"got {len(exemplar_polygons)}"
        )
    indices: list[int] = []
    for polygon in exemplar_polygons:
        exemplar_box = _polygon_to_bbox(cast(Sequence[Sequence[float]], polygon))
        best = max(range(len(boxes)), key=lambda i: boxes[i].iou(exemplar_box))
        indices.append(best)
    return tuple(indices)


def convert_fscd_lvis(
    raw_root: Path, out_root: Path, *, protocol: Protocol = "unseen"
) -> list[Path]:
    """Convert an FSCD-LVIS raw tree into co-located ``*.gt.json`` sidecars under ``out_root``.

    Only the **exemplar-category** boxes are emitted as ground truth; other-category boxes are the
    distractors and are intentionally excluded (D-06). An image with no exemplar-category box
    produces no sidecar (honest coverage).

    Args:
        raw_root: An FSCD-LVIS tree with ``images/<id>.png``, ``annotations.json`` + ``split.json``.
        out_root: Destination directory (created if missing) for sidecars and copied scenes.
        protocol: ``"unseen"`` (the standard Counting-DETR generalization eval; the headline number)
            or ``"seen"``. Only ``"unseen"`` is exercised this phase.

    Returns:
        The written sidecar paths, sorted by image id.

    Raises:
        FileNotFoundError: If ``annotations.json`` is missing.
        NotImplementedError: If ``protocol`` is not ``"unseen"``.
        ValueError: If an image has fewer than three exemplars or a degenerate box.
    """
    if protocol != "unseen":
        raise NotImplementedError(
            f"FSCD-LVIS protocol {protocol!r} not wired this phase; use 'unseen' (D-01)"
        )
    annotations_path = raw_root / "annotations.json"
    if not annotations_path.is_file():
        raise FileNotFoundError(f"FSCD-LVIS annotations not found at {annotations_path}")
    annotations = json.loads(annotations_path.read_text(encoding="utf-8"))
    images_dir = raw_root / "images"
    out_root.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for image_id in sorted(annotations):
        entry = annotations[image_id]
        if not isinstance(entry, dict):
            continue
        target_category = entry.get("exemplar_category")
        # Ground truth is the exemplar category only; other categories are distractors (excluded).
        target_boxes = [
            _xyxy_to_bbox(ann["box"])
            for ann in entry.get("annotations", [])
            if ann.get("category") == target_category
        ]
        if not target_boxes:
            # No box for the queried class -> no sidecar (D-06: honest coverage, never fabricated).
            continue

        exemplar_indices = _exemplar_indices(
            target_boxes, entry.get("box_examples_coordinates", [])
        )

        image_path = images_dir / f"{image_id}.png"
        if not image_path.is_file():
            logger.warning("FSCD-LVIS: {} has no image at {}, skipping", image_id, image_path)
            continue
        with Image.open(image_path) as image:
            width, height = image.size

        sidecar = {
            "image": f"{image_id}.png",
            "width": width,
            "height": height,
            "achieved_n": len(target_boxes),
            "exemplar_index": exemplar_indices[0],
            "exemplar_indices": list(exemplar_indices),
            "boxes": [box.model_dump() for box in target_boxes],
        }
        sidecar_path = out_root / f"{image_id}.gt.json"
        sidecar_path.write_text(
            json.dumps(sidecar, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        shutil.copyfile(image_path, out_root / f"{image_id}.png")
        written.append(sidecar_path)

    logger.info("FSCD-LVIS: converted {} image(s) into {}", len(written), out_root)
    return sorted(written)
