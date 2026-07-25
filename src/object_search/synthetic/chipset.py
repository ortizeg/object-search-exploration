"""The chip-insertion benchmark set (EVAL-19).

Ten images, ten canvas sizes ramping from 320x240 to 6000x4000. Each image gets one distinct,
randomly-generated *textured* chip pasted ``N in {5, 10, 15}`` times at strictly
non-overlapping positions on a white background. Because every instance is pasted by us at a
known rectangle, the ground truth is exact **by construction** -- precision, recall and AP are
computable with no human rating and no hand-labelling, which makes this the objective harness
for tuning method parameters and comparing methods after every change.

Two invariants are load-bearing and tested:

* **Strict non-overlap.** Placement rejection-samples against already-placed boxes with a hard
  attempt cap; every pair of ground-truth boxes has IoU exactly 0. That is what makes
  precision/recall unambiguous, with no duplicate-versus-fragment judgement anywhere.
* **The recorded count is the ACHIEVED count, never the requested N.** If the attempt cap is
  hit the image simply holds fewer chips, and the sidecar records how many were actually
  placed. A ground-truth file that overstated the count would silently depress every method's
  recall and make the benchmark lie.

The chip is textured (several saturated primitives) on purpose: a flat chip would trip Method
1's low-variance template guard, so a flat benchmark could never exercise NCC honestly.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from object_search.schemas.geometry import BBox
from object_search.schemas.records import SliceMetadata
from object_search.synthetic.generator import SyntheticImage

# Master seed for the whole set: every per-image and per-chip seed derives from this, so the
# ten specs are fixed and regenerate identically.
_MASTER_SEED = 20260724
_PLACEMENT_MAX_ATTEMPTS = 2000
_INSET = 2  # keep every instance fully inside the frame; a clipped chip is neither clearly
# present nor clearly absent, and that ambiguity is exactly what this set avoids.
_PLACEMENT_GAP = 2  # a small guaranteed gap so no two chips even touch.

# Canvas sizes ramp from small to very large, in order.
CHIP_CANVAS_SIZES: tuple[tuple[int, int], ...] = (
    (320, 240),
    (512, 384),
    (800, 600),
    (1024, 768),
    (1600, 1200),
    (2048, 1536),
    (2560, 1920),
    (3200, 2400),
    (4096, 3072),
    (6000, 4000),
)
_INSTANCE_CHOICES = (5, 10, 15)


class ChipSpec(BaseModel):
    """One randomly-generated chip -- the single "object class" for one benchmark image."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    seed: int
    size: int = Field(ge=8)


class ChipImageSpec(BaseModel):
    """One benchmark image: a canvas, a chip, and how many times to paste it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    image_id: str
    width: int = Field(ge=64)
    height: int = Field(ge=64)
    n_instances: int = Field(ge=1)
    chip: ChipSpec
    seed: int


def _chip_size_for(width: int, height: int) -> int:
    """Scale the chip with the canvas so instances stay findable at every resolution."""
    return int(np.clip(round(min(width, height) * 0.06), 24, 160))


def _build_specs() -> tuple[ChipImageSpec, ...]:
    rng = np.random.default_rng(_MASTER_SEED)
    specs: list[ChipImageSpec] = []
    for index, (width, height) in enumerate(CHIP_CANVAS_SIZES, start=1):
        n_instances = int(rng.choice(_INSTANCE_CHOICES))
        chip_seed = int(rng.integers(0, 2**31))
        image_seed = int(rng.integers(0, 2**31))
        specs.append(
            ChipImageSpec(
                image_id=f"chipset-{index:02d}",
                width=width,
                height=height,
                n_instances=n_instances,
                chip=ChipSpec(seed=chip_seed, size=_chip_size_for(width, height)),
                seed=image_seed,
            )
        )
    return tuple(specs)


CHIPSET_SPECS: tuple[ChipImageSpec, ...] = _build_specs()


def render_chip(spec: ChipSpec) -> npt.NDArray[np.uint8]:
    """Render the textured chip for ``spec`` deterministically from its seed."""
    rng = np.random.default_rng(spec.seed)
    size = spec.size
    chip = np.empty((size, size, 3), dtype=np.uint8)
    chip[:, :] = rng.integers(30, 220, size=3)

    def _color() -> tuple[int, int, int]:
        vals = rng.integers(0, 256, size=3)
        return (int(vals[0]), int(vals[1]), int(vals[2]))

    n_primitives = int(rng.integers(4, 8))
    for _ in range(n_primitives):
        kind = int(rng.integers(0, 4))
        if kind == 0:  # filled rectangle
            x0, x1 = sorted(int(v) for v in rng.integers(0, size, size=2))
            y0, y1 = sorted(int(v) for v in rng.integers(0, size, size=2))
            cv2.rectangle(chip, (x0, y0), (x1, y1), _color(), thickness=-1, lineType=cv2.LINE_8)
        elif kind == 1:  # filled circle
            cx, cy = int(rng.integers(0, size)), int(rng.integers(0, size))
            radius = int(rng.integers(2, max(3, size // 3)))
            cv2.circle(chip, (cx, cy), radius, _color(), thickness=-1, lineType=cv2.LINE_8)
        elif kind == 2:  # filled triangle
            pts = rng.integers(0, size, size=(3, 2)).astype(np.int32)
            cv2.fillPoly(chip, [pts], _color(), lineType=cv2.LINE_8)
        else:  # line segment
            p0 = (int(rng.integers(0, size)), int(rng.integers(0, size)))
            p1 = (int(rng.integers(0, size)), int(rng.integers(0, size)))
            cv2.line(chip, p0, p1, _color(), thickness=int(rng.integers(1, 4)), lineType=cv2.LINE_8)
    return chip


def _overlaps(x: int, y: int, size: int, placed: list[BBox]) -> bool:
    """True if a ``size`` box at ``(x, y)`` comes within ``_PLACEMENT_GAP`` of any placed box."""
    gap = _PLACEMENT_GAP
    for box in placed:
        if not (
            x + size + gap <= box.x
            or box.x + box.w + gap <= x
            or y + size + gap <= box.y
            or box.y + box.h + gap <= y
        ):
            return True
    return False


def generate_chipset_image(spec: ChipImageSpec) -> SyntheticImage:
    """Generate one benchmark image with exact, non-overlapping ground-truth boxes."""
    rng = np.random.default_rng(spec.seed)
    canvas = np.full((spec.height, spec.width, 3), 255, dtype=np.uint8)
    chip = render_chip(spec.chip)
    size = spec.chip.size

    x_hi = spec.width - size - _INSET
    y_hi = spec.height - size - _INSET
    placed: list[BBox] = []
    attempts = 0
    while len(placed) < spec.n_instances and attempts < _PLACEMENT_MAX_ATTEMPTS:
        attempts += 1
        x = int(rng.integers(_INSET, x_hi + 1))
        y = int(rng.integers(_INSET, y_hi + 1))
        if _overlaps(x, y, size, placed):
            continue
        canvas[y : y + size, x : x + size] = chip
        placed.append(BBox(x=x, y=y, w=size, h=size))

    if len(placed) < spec.n_instances:
        logger.warning(
            f"{spec.image_id}: placed {len(placed)}/{spec.n_instances} chips after {attempts} "
            f"attempts; recording the achieved count as ground truth"
        )

    placed.sort(key=lambda box: (box.y, box.x))
    slice_metadata = SliceMetadata(true_instance_count=len(placed))
    # Reuse the synthetic dataclass so downstream code treats both sets identically. The spec
    # is not a SyntheticSpec, so it is not attached; the sidecar carries the chip provenance.
    return SyntheticImage(
        image=canvas,
        boxes=tuple(placed),
        spec=None,
        slice_metadata=slice_metadata,
    )


def write_chipset(out_dir: Path, *, force: bool = False, exemplar_index: int = 0) -> list[Path]:
    """Generate all ten images plus ``<image_id>.gt.json`` sidecars into ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for spec in CHIPSET_SPECS:
        image_path = out_dir / f"{spec.image_id}.png"
        if image_path.is_file() and not force:
            logger.info(f"{spec.image_id}: exists, skipping (use --force to overwrite)")
            written.append(image_path)
            continue
        result = generate_chipset_image(spec)
        if not cv2.imwrite(str(image_path), result.image):
            raise OSError(f"failed to write chip image to {image_path}")
        sidecar = out_dir / f"{spec.image_id}.gt.json"
        payload = {
            "image": image_path.name,
            "width": spec.width,
            "height": spec.height,
            "seed": spec.seed,
            "chip": spec.chip.model_dump(mode="json"),
            "requested_n": spec.n_instances,
            "achieved_n": len(result.boxes),
            "exemplar_index": exemplar_index,
            "boxes": [box.model_dump(mode="json") for box in result.boxes],
        }
        sidecar.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        logger.info(
            f"{spec.image_id}: {spec.width}x{spec.height}, {len(result.boxes)} chips "
            f"(requested {spec.n_instances})"
        )
        written.append(image_path)
    return written
