"""CARPK native annotations -> the repo's ``*.gt.json`` sidecar schema (D-10, EVAL-21).

CARPK is the tracer dataset: single-class ("car"), drone view, and the simplest of the four
native formats -- one plain-text box file per image. Each ``Annotations/<image_id>.txt`` line is::

    x1 y1 x2 y2 class

with **pixel corner-inclusive** ``(x1, y1)`` top-left and ``(x2, y2)`` bottom-right, and ``class``
always ``1``. Images live beside the annotations in ``Images/<image_id>.png``.

Boundary conversion (the one place two conventions meet)
--------------------------------------------------------
The repo's :class:`~object_search.schemas.geometry.BBox` is half-open ``[x, x2) x [y, y2)`` with
``w == x2 - x``. CARPK's corners convert **at this boundary and nowhere else**::

    BBox(x=x1, y=y1, w=x2 - x1, h=y2 - y1)

so the converted sidecar carries the repo convention inwards and no downstream code ever sees the
CARPK convention (the half-open-vs-closed rule in ``geometry.py``).

Output
------
For each image this writes ``out_root/<image_id>.gt.json`` in the existing schema (``boxes``,
``exemplar_index=0``, ``exemplar_indices=[0]``, top-level ``width``/``height``, ``achieved_n``) and
**copies the scene image** to ``out_root/<image_id>.png`` so the sidecar and its pixels are
co-located under the (gitignored) ``datasets/carpk/test/`` tree the benchmark loads from. No second
ground-truth reader is created: :func:`object_search.eval.labels.load_research_ground_truth` reads
these back through the same ``_parse_sidecar``.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from loguru import logger
from PIL import Image

from object_search.schemas.geometry import BBox

# CARPK is single-class; the class column is present in the native format but carries no
# information for an exemplar-search harness, so it is parsed and discarded rather than stored.
_CARPK_CLASS_COLUMN = 4


def _parse_annotation_line(line: str, path: Path) -> BBox | None:
    """Parse one ``x1 y1 x2 y2 class`` line into a half-open :class:`BBox`, or ``None`` if blank.

    Raises:
        ValueError: If a non-blank line is malformed or would produce a degenerate (<1px) box.
    """
    parts = line.split()
    if not parts:
        return None
    if len(parts) < _CARPK_CLASS_COLUMN:
        raise ValueError(f"{path}: malformed CARPK line {line!r} (need at least x1 y1 x2 y2)")
    x1, y1, x2, y2 = (int(parts[i]) for i in range(4))
    w, h = x2 - x1, y2 - y1
    if w < 1 or h < 1:
        raise ValueError(
            f"{path}: CARPK box ({x1},{y1},{x2},{y2}) is degenerate (w={w}, h={h}); "
            f"expected corner-inclusive x1<x2 and y1<y2"
        )
    # The one boundary conversion: CARPK corner-inclusive -> repo half-open BBox.
    return BBox(x=x1, y=y1, w=w, h=h)


def convert_carpk(raw_root: Path, out_root: Path) -> list[Path]:
    """Convert a CARPK raw tree into co-located ``*.gt.json`` sidecars under ``out_root``.

    Args:
        raw_root: A CARPK tree with ``Images/<id>.png`` and ``Annotations/<id>.txt``. The
            ``ImageSets/`` split lists are not required here -- every annotated image found is
            converted; membership in a split is decided by the committed split manifest.
        out_root: Destination directory (created if missing) for the converted sidecars and copied
            scene images -- e.g. ``datasets/carpk/test``.

    Returns:
        The written sidecar paths, sorted by image id, so a caller can record exactly what was
        produced (provenance) deterministically.

    Raises:
        FileNotFoundError: If ``raw_root/Annotations`` does not exist.
        ValueError: If an annotation file is malformed or an image has zero boxes.
    """
    annotations_dir = raw_root / "Annotations"
    images_dir = raw_root / "Images"
    if not annotations_dir.is_dir():
        raise FileNotFoundError(f"CARPK annotations dir not found at {annotations_dir}")
    out_root.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for annotation_path in sorted(annotations_dir.glob("*.txt")):
        image_id = annotation_path.stem
        image_path = images_dir / f"{image_id}.png"
        if not image_path.is_file():
            logger.warning(
                "CARPK: annotation {} has no image at {}, skipping", image_id, image_path
            )
            continue

        boxes = [
            box
            for line in annotation_path.read_text(encoding="utf-8").splitlines()
            if (box := _parse_annotation_line(line, annotation_path)) is not None
        ]
        if not boxes:
            raise ValueError(f"{annotation_path}: no boxes; a labelled CARPK image has >= 1 car")

        with Image.open(image_path) as image:
            width, height = image.size

        sidecar = {
            "image": f"{image_id}.png",
            "width": width,
            "height": height,
            "achieved_n": len(boxes),
            "exemplar_index": 0,
            # Additive multi-exemplar field (Task 1): CARPK is single-exemplar, so the list is just
            # the first box; the field exists uniformly so FSCD-* three-exemplar sidecars fit later.
            "exemplar_indices": [0],
            "boxes": [box.model_dump() for box in boxes],
        }
        sidecar_path = out_root / f"{image_id}.gt.json"
        sidecar_path.write_text(
            json.dumps(sidecar, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        # Co-locate the pixels so the benchmark's research scene loader finds them beside the label.
        shutil.copyfile(image_path, out_root / f"{image_id}.png")
        written.append(sidecar_path)

    logger.info("CARPK: converted {} image(s) into {}", len(written), out_root)
    return sorted(written)
