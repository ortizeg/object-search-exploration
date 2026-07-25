"""Synthetic image generator with **exact** ground truth (EVAL-03).

Why this exists
---------------
Every other image the project uses needs a human to draw ground-truth boxes. These do not:
the generator *places* each instance at a known pose, so the true boxes are known by
construction and precision/recall are computable with zero labelling. That makes this the
cheapest possible signal for tuning a method and for the determinism and correctness tests.

Two properties are load-bearing and are both covered by tests:

1. **Byte-identical output for a given seed.** A single ``np.random.default_rng(spec.seed)``
   drives every random draw, in a fixed order, and shapes are drawn with ``cv2.LINE_8`` (no
   anti-aliasing) so the pixels do not depend on the OpenCV build. No ``random`` module, no
   unseeded numpy call, no iteration over an unordered set appears anywhere below.
2. **The ground-truth box is the axis-aligned bounding box of the *actually drawn* rotated
   shape**, computed from the transformed vertices -- not the nominal ``instance_size``. A
   rotated square has a strictly larger AABB than its side length, and using the nominal box
   would make every downstream IoU quietly wrong.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import numpy.typing as npt
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from object_search.schemas.geometry import BBox
from object_search.schemas.records import SliceMetadata

# Rejection-sampling attempt cap for scatter mode. Hitting it means the canvas is too full;
# we then place fewer instances and record the real count rather than looping forever.
_SCATTER_MAX_ATTEMPTS = 2000

ShapeName = Literal["rect", "triangle", "circle", "plus", "chevron"]


class SyntheticSpec(BaseModel):
    """Everything needed to generate one synthetic scene, deterministically from ``seed``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    seed: int = 0
    width: int = Field(default=960, ge=16)
    height: int = Field(default=640, ge=16)
    mode: Literal["lattice", "scatter"] = "lattice"
    shape: ShapeName = "plus"
    n_instances: int = Field(default=12, ge=1)
    instance_size: int = Field(default=56, ge=6)
    scale_jitter: float = Field(default=0.0, ge=0.0, le=0.9)
    rotation_jitter_deg: float = Field(default=0.0, ge=0.0, le=180.0)
    position_jitter: float = Field(default=0.0, ge=0.0, le=1.0)
    n_distractors: int = Field(default=0, ge=0)
    clutter: float = Field(default=0.0, ge=0.0, le=1.0)
    fg_color: tuple[int, int, int] = (40, 90, 220)
    bg_color: tuple[int, int, int] = (235, 235, 235)


@dataclass(frozen=True)
class SyntheticImage:
    """A generated scene and its exact ground truth.

    ``image`` is a NumPy array and deliberately lives on a dataclass rather than the frozen
    Pydantic schema -- ndarrays do not belong in an inter-layer contract.
    """

    image: npt.NDArray[np.uint8]
    boxes: tuple[BBox, ...]
    # ``None`` for the chip-insertion set, which reuses this dataclass but is not driven by a
    # SyntheticSpec (its provenance lives in the chipset sidecar instead).
    spec: SyntheticSpec | None
    slice_metadata: SliceMetadata


# -- shape geometry -----------------------------------------------------------------------


def _unit_polygon(shape: ShapeName) -> npt.NDArray[np.float64]:
    """Vertices of ``shape`` centred at the origin, half-extent 1.0. ``circle`` returns empty."""
    if shape == "rect":
        return np.array([(-1, -1), (1, -1), (1, 1), (-1, 1)], dtype=np.float64)
    if shape == "triangle":
        return np.array([(0, -1), (1, 1), (-1, 1)], dtype=np.float64)
    if shape == "plus":
        t = 1.0 / 3.0
        return np.array(
            [
                (-t, -1),
                (t, -1),
                (t, -t),
                (1, -t),
                (1, t),
                (t, t),
                (t, 1),
                (-t, 1),
                (-t, t),
                (-1, t),
                (-1, -t),
                (-t, -t),
            ],
            dtype=np.float64,
        )
    if shape == "chevron":
        return np.array(
            [(-1, -1), (0, -0.5), (1, -1), (1, -0.2), (0, 0.7), (-1, -0.2)],
            dtype=np.float64,
        )
    # circle
    return np.empty((0, 2), dtype=np.float64)


def _rotate(points: npt.NDArray[np.float64], angle_deg: float) -> npt.NDArray[np.float64]:
    theta = np.deg2rad(angle_deg)
    cos_t, sin_t = float(np.cos(theta)), float(np.sin(theta))
    rotation = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float64)
    return points @ rotation.T


def _draw_instance(
    canvas: npt.NDArray[np.uint8],
    shape: ShapeName,
    cx: float,
    cy: float,
    half: float,
    angle_deg: float,
    color: tuple[int, int, int],
) -> BBox:
    """Draw one instance and return the AABB of the pixels actually drawn."""
    if shape == "circle":
        radius = round(half)
        centre = (round(cx), round(cy))
        cv2.circle(canvas, centre, radius, color, thickness=-1, lineType=cv2.LINE_8)
        x0, y0 = centre[0] - radius, centre[1] - radius
        x1, y1 = centre[0] + radius, centre[1] + radius
    else:
        verts = _rotate(_unit_polygon(shape) * half, angle_deg)
        verts[:, 0] += cx
        verts[:, 1] += cy
        pts = np.round(verts).astype(np.int32)
        cv2.fillPoly(canvas, [pts], color, lineType=cv2.LINE_8)
        x0, y0 = int(pts[:, 0].min()), int(pts[:, 1].min())
        x1, y1 = int(pts[:, 0].max()), int(pts[:, 1].max())

    height, width = canvas.shape[:2]
    x0 = max(0, min(x0, width - 1))
    y0 = max(0, min(y0, height - 1))
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    # Drawn pixels are inclusive of the max coordinate; a half-open box adds one.
    return BBox(x=x0, y=y0, w=x1 - x0 + 1, h=y1 - y0 + 1)


# -- background ---------------------------------------------------------------------------


def _paint_background(spec: SyntheticSpec, rng: np.random.Generator) -> npt.NDArray[np.uint8]:
    canvas = np.empty((spec.height, spec.width, 3), dtype=np.uint8)
    canvas[:, :] = spec.bg_color
    if spec.clutter <= 0.0:
        return canvas

    # Seeded Gaussian noise plus a few low-contrast blobs, all scaled by clutter strength.
    noise = rng.normal(0.0, 25.0 * spec.clutter, size=canvas.shape)
    blended = canvas.astype(np.float64) + noise
    n_blobs = round(6 * spec.clutter)
    for _ in range(n_blobs):
        bx = int(rng.integers(0, spec.width))
        by = int(rng.integers(0, spec.height))
        br = int(rng.integers(spec.width // 12, spec.width // 5 + 1))
        shade = float(rng.integers(-30, 31)) * spec.clutter
        overlay = blended.copy()
        cv2.circle(overlay, (bx, by), br, (shade, shade, shade), thickness=-1, lineType=cv2.LINE_8)
        blended = 0.7 * blended + 0.3 * overlay
    return np.clip(blended, 0, 255).astype(np.uint8)


# -- placement ----------------------------------------------------------------------------


def _lattice_centres(
    spec: SyntheticSpec, margin: float, rng: np.random.Generator
) -> list[tuple[float, float]]:
    cols = int(np.ceil(np.sqrt(spec.n_instances)))
    rows = int(np.ceil(spec.n_instances / cols))
    usable_w = spec.width - 2 * margin
    usable_h = spec.height - 2 * margin
    cell_w = usable_w / max(cols, 1)
    cell_h = usable_h / max(rows, 1)
    centres: list[tuple[float, float]] = []
    for index in range(spec.n_instances):
        row, col = divmod(index, cols)
        cx = margin + (col + 0.5) * cell_w
        cy = margin + (row + 0.5) * cell_h
        if spec.position_jitter > 0.0:
            cx += float(rng.uniform(-0.5, 0.5)) * spec.position_jitter * cell_w
            cy += float(rng.uniform(-0.5, 0.5)) * spec.position_jitter * cell_h
        centres.append((cx, cy))
    return centres


def _scatter_centres(
    spec: SyntheticSpec, margin: float, rng: np.random.Generator
) -> list[tuple[float, float]]:
    min_gap = 2.0 * margin
    placed: list[tuple[float, float]] = []
    attempts = 0
    while len(placed) < spec.n_instances and attempts < _SCATTER_MAX_ATTEMPTS:
        attempts += 1
        cx = float(rng.uniform(margin, spec.width - margin))
        cy = float(rng.uniform(margin, spec.height - margin))
        if all((cx - px) ** 2 + (cy - py) ** 2 >= min_gap**2 for px, py in placed):
            placed.append((cx, cy))
    if len(placed) < spec.n_instances:
        logger.warning(
            f"scatter mode placed {len(placed)}/{spec.n_instances} instances after "
            f"{attempts} attempts (canvas too full); recording the achieved count"
        )
    return placed


# -- top-level generation -----------------------------------------------------------------


def synthesize(spec: SyntheticSpec) -> SyntheticImage:
    """Generate the scene described by ``spec`` with exact ground-truth boxes."""
    rng = np.random.default_rng(spec.seed)
    canvas = _paint_background(spec, rng)

    # A margin large enough that a max-scale, worst-case-rotated instance stays in frame.
    max_half = spec.instance_size / 2.0 * (1.0 + spec.scale_jitter) * np.sqrt(2.0)
    margin = float(max_half) + 2.0

    centres = (
        _lattice_centres(spec, margin, rng)
        if spec.mode == "lattice"
        else _scatter_centres(spec, margin, rng)
    )

    boxes: list[BBox] = []
    scales: list[float] = []
    angles: list[float] = []
    for cx, cy in centres:
        scale = 1.0 + float(rng.uniform(-1.0, 1.0)) * spec.scale_jitter
        angle = float(rng.uniform(-1.0, 1.0)) * spec.rotation_jitter_deg
        scales.append(scale)
        angles.append(angle)
        half = spec.instance_size / 2.0 * scale
        boxes.append(_draw_instance(canvas, spec.shape, cx, cy, half, angle, spec.fg_color))

    # Distractors are drawn but NOT recorded as ground truth -- they are false-positive bait,
    # a different shape and a shifted colour so a method can genuinely be fooled by them.
    _draw_distractors(canvas, spec, margin, rng)

    boxes.sort(key=lambda box: (box.y, box.x))

    slice_metadata = SliceMetadata(
        true_instance_count=len(boxes),
        instance_scale_min=min(scales) if scales else None,
        instance_scale_max=max(scales) if scales else None,
        rotation_min_deg=min(angles) if angles else None,
        rotation_max_deg=max(angles) if angles else None,
        clutter_level=spec.clutter,
    )
    return SyntheticImage(
        image=canvas, boxes=tuple(boxes), spec=spec, slice_metadata=slice_metadata
    )


def _draw_distractors(
    canvas: npt.NDArray[np.uint8],
    spec: SyntheticSpec,
    margin: float,
    rng: np.random.Generator,
) -> None:
    if spec.n_distractors <= 0:
        return
    distractor_shape: ShapeName = "circle" if spec.shape != "circle" else "rect"
    # A hue-ish shift by rotating the BGR channels, so distractors read as "similar but not".
    b, g, r = spec.fg_color
    distractor_color = (g, r, b)
    for _ in range(spec.n_distractors):
        cx = float(rng.uniform(margin, spec.width - margin))
        cy = float(rng.uniform(margin, spec.height - margin))
        half = spec.instance_size / 2.0 * float(rng.uniform(0.7, 1.1))
        angle = float(rng.uniform(-1.0, 1.0)) * spec.rotation_jitter_deg
        _draw_instance(canvas, distractor_shape, cx, cy, half, angle, distractor_color)


def save(out: SyntheticImage, image_path: Path) -> Path:
    """Write the PNG and a ``<stem>.gt.json`` sidecar so ground truth travels with the image."""
    image_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(image_path), out.image):
        raise OSError(f"failed to write synthetic image to {image_path}")
    sidecar = image_path.with_suffix(".gt.json")
    payload = {
        "image": image_path.name,
        "spec": out.spec.model_dump(mode="json") if out.spec is not None else None,
        "slice_metadata": out.slice_metadata.model_dump(mode="json"),
        "boxes": [box.model_dump(mode="json") for box in out.boxes],
    }
    sidecar.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    logger.info(f"wrote {image_path.name} with {len(out.boxes)} ground-truth boxes")
    return image_path


# -- the committed demo set ---------------------------------------------------------------

DEMO_SPECS: Mapping[str, SyntheticSpec] = {
    # Identical instances on a grid -- the NCC-favourable base case.
    "lattice-plain": SyntheticSpec(seed=1, mode="lattice", shape="plus", n_instances=12),
    # Instances spaced so they nearly touch -- the case that proves local-max beats NMS (Phase 2).
    "lattice-touching": SyntheticSpec(
        seed=2, mode="lattice", shape="rect", n_instances=16, instance_size=72, position_jitter=0.0
    ),
    # Scale + rotation variation -- should favour dino-dense over template matching.
    "scatter-scaled": SyntheticSpec(
        seed=3,
        mode="scatter",
        shape="triangle",
        n_instances=10,
        scale_jitter=0.5,
        rotation_jitter_deg=30.0,
    ),
    # Heavy clutter and distractors -- false-positive stress.
    "cluttered-distractors": SyntheticSpec(
        seed=4, mode="scatter", shape="plus", n_instances=8, clutter=0.6, n_distractors=8
    ),
}
