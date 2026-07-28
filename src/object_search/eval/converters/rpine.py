"""RPINE native annotations -> the repo's ``*.gt.json`` sidecar schema (D-06, EVAL-22).

RPINE ("Repeated Patterns IN Everywhere") is the closest match to this project's task: **every
repetition in a single image is box-annotated**, and queries are box exemplars. Its native form is
one plain-text box file per image::

    annotations/<image_id>.txt   # one `x1 y1 x2 y2` box per line (x2/y2 exclusive)
    images/<image_id>.png

There is no per-image class column (RPINE annotates *repeats*, not object categories) and no native
exemplar box: the harness needs one designated exemplar per image, so this converter **samples the
exemplar indices from the ground-truth boxes, seeded** (``np.random.default_rng(seed)`` -- never
``cv2.setRNGSeed``, D-11), so the choice is byte-stable across runs. Up to three indices are drawn
(the 1- and 3-exemplar runs both read them); the first is the designated ``exemplar_index``.

Boundary conversion (the one place two conventions meet)
--------------------------------------------------------
Each ``x1 y1 x2 y2`` (x2/y2 exclusive) converts to the repo's half-open
:class:`~object_search.schemas.geometry.BBox` **at this boundary and nowhere else**::

    BBox(x=x1, y=y1, w=x2 - x1, h=y2 - y1)

The output is read back through the single ``_parse_sidecar`` (D-10); no second reader is created.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
from loguru import logger
from PIL import Image

from object_search.schemas.geometry import BBox

# Number of exemplar indices sampled per image (clamped to the box count). The harness runs at 1
# and 3 exemplars, so up to three are recorded and the run selects among them.
_RPINE_EXEMPLAR_COUNT = 3


def _parse_box_line(line: str, path: Path) -> BBox | None:
    """Parse one ``x1 y1 x2 y2`` line into a half-open :class:`BBox`, or ``None`` if blank."""
    parts = line.split()
    if not parts:
        return None
    if len(parts) < 4:
        raise ValueError(f"{path}: malformed RPINE line {line!r} (need x1 y1 x2 y2)")
    x1, y1, x2, y2 = (int(parts[i]) for i in range(4))
    w, h = x2 - x1, y2 - y1
    if w < 1 or h < 1:
        raise ValueError(f"{path}: RPINE box ({x1},{y1},{x2},{y2}) is degenerate (w={w}, h={h})")
    # The one boundary conversion: source xyxy -> repo half-open BBox.
    return BBox(x=x1, y=y1, w=w, h=h)


def _image_seed_offset(image_id: str) -> int:
    """A small, stable per-image offset so each image's exemplar sampler is independent (D-11).

    Deliberately **not** Python's builtin ``hash`` (salted per process, so not reproducible across
    runs). A fixed FNV-1a over the id bytes gives the same offset every run.
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
    count = min(_RPINE_EXEMPLAR_COUNT, n_boxes)
    chosen = rng.choice(n_boxes, size=count, replace=False)
    return tuple(int(i) for i in sorted(chosen))


def convert_rpine(raw_root: Path, out_root: Path, *, seed: int = 0) -> list[Path]:
    """Convert an RPINE raw tree into co-located ``*.gt.json`` sidecars under ``out_root``.

    Args:
        raw_root: An RPINE tree with ``images/<id>.png`` and ``annotations/<id>.txt``.
        out_root: Destination directory (created if missing) for sidecars and copied scenes.
        seed: Seed for the exemplar-index sampler (D-11); recorded implicitly by the byte-stable
            output. The same seed reproduces the same exemplar choice exactly.

    Returns:
        The written sidecar paths, sorted by image id.

    Raises:
        FileNotFoundError: If ``raw_root/annotations`` does not exist.
        ValueError: If an annotation file is malformed or has zero boxes.
    """
    annotations_dir = raw_root / "annotations"
    images_dir = raw_root / "images"
    if not annotations_dir.is_dir():
        raise FileNotFoundError(f"RPINE annotations dir not found at {annotations_dir}")
    out_root.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for annotation_path in sorted(annotations_dir.glob("*.txt")):
        image_id = annotation_path.stem
        # RPINE's native release ships PNG scenes; the HuggingFace mirror (ChipmunkG4/RPINE) ships
        # JPG. Accept either -- this is the only difference between the two sources at this layer.
        image_path = next(
            (
                candidate
                for ext in (".png", ".jpg", ".jpeg")
                if (candidate := images_dir / f"{image_id}{ext}").is_file()
            ),
            None,
        )
        if image_path is None:
            logger.warning("RPINE: {} has no image in {}, skipping", image_id, images_dir)
            continue

        boxes = [
            box
            for line in annotation_path.read_text(encoding="utf-8").splitlines()
            if (box := _parse_box_line(line, annotation_path)) is not None
        ]
        if not boxes:
            raise ValueError(f"{annotation_path}: no boxes; a labelled RPINE image has >= 1 repeat")

        # Seed the exemplar choice on the image id so each image is independent yet reproducible.
        exemplar_indices = _sample_exemplar_indices(len(boxes), seed + _image_seed_offset(image_id))

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

    logger.info("RPINE: converted {} image(s) into {}", len(written), out_root)
    return sorted(written)
