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

from object_search.schemas.geometry import BBox, Point
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


# ======================================================================================
# Marker mode (Milestone 2) -- markers with an *exact* tip, direction and centroid.
#
# The Milestone 1 generator draws objects to be *found*. This mode draws the query gesture
# itself -- an arrow, a dot, a caret -- whose tip and pointing direction are known by
# construction and so serve as exact oracles for the orientation estimator (M2-02). The
# same two load-bearing properties hold: one ``np.random.default_rng(seed)`` drives every
# draw in a fixed order, and shapes use ``cv2.LINE_8`` so pixels do not depend on the
# OpenCV build. An arrow has a shaft plus a filled triangular head, so its heavier,
# narrowing-to-a-point end is unambiguous -- exactly the signal the arrowhead-mass heuristic
# keys off. A dot is rotationally symmetric and therefore carries **no** direction, which is
# the spec-required "return the centroid and no guessed direction" case.
# ======================================================================================

MarkerName = Literal["arrow", "dot", "caret"]


class MarkerSpec(BaseModel):
    """Everything needed to draw one marker scene deterministically from ``seed``.

    Attributes:
        seed: The single seed the one RNG is built from.
        width: Canvas width in pixels.
        height: Canvas height in pixels.
        marker: Which gesture to draw. ``arrow`` and ``caret`` point; ``dot`` does not.
        n_markers: How many markers to place, non-overlapping.
        arrow_len: Full tail-to-tip length of an arrow/caret in pixels; a dot's diameter is
            derived from it so all markers occupy a comparable footprint.
        rotation_jitter_deg: Unused placeholder kept symmetric with :class:`SyntheticSpec`;
            each marker already draws a *fully* random orientation, which is the point.
        with_targets: When true, draw a target object a known distance past each pointing
            marker's tip, along its direction, and record its box in the ground truth.
        target_gap: Pixels from the tip to the near edge of the target object.
        thickness: Shaft/stroke thickness in pixels.
        fg_color: Marker colour (BGR).
        bg_color: Background colour (BGR).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    seed: int = 0
    width: int = Field(default=640, ge=16)
    height: int = Field(default=480, ge=16)
    marker: MarkerName = "arrow"
    n_markers: int = Field(default=3, ge=1)
    arrow_len: int = Field(default=64, ge=8)
    rotation_jitter_deg: float = Field(default=0.0, ge=0.0, le=180.0)
    with_targets: bool = False
    target_gap: int = Field(default=28, ge=0)
    thickness: int = Field(default=3, ge=1)
    fg_color: tuple[int, int, int] = (40, 90, 220)
    bg_color: tuple[int, int, int] = (235, 235, 235)


@dataclass(frozen=True)
class MarkerGT:
    """Exact ground truth for one drawn marker.

    ``tip`` and ``centroid`` are always present; ``direction`` is a **unit** vector for a
    pointing marker and ``None`` for a symmetric one (a dot). ``target`` is the box of the
    pointed-at object when ``with_targets`` was set, else ``None``.
    """

    box: BBox
    tip: Point
    direction: tuple[float, float] | None
    centroid: Point
    target: BBox | None


@dataclass(frozen=True)
class MarkerImage:
    """A generated marker scene and its exact per-marker ground truth."""

    image: npt.NDArray[np.uint8]
    markers: tuple[MarkerGT, ...]
    spec: MarkerSpec


def _aabb_of(points: npt.NDArray[np.float64], width: int, height: int) -> BBox:
    """Clipped, half-open AABB of ``points`` (rows of ``(x, y)``)."""
    pts = np.round(points).astype(np.int64)
    x0 = max(0, min(int(pts[:, 0].min()), width - 1))
    y0 = max(0, min(int(pts[:, 1].min()), height - 1))
    x1 = max(0, min(int(pts[:, 0].max()), width - 1))
    y1 = max(0, min(int(pts[:, 1].max()), height - 1))
    return BBox(x=x0, y=y0, w=x1 - x0 + 1, h=y1 - y0 + 1)


def _draw_arrow(
    canvas: npt.NDArray[np.uint8],
    center: tuple[float, float],
    direction: tuple[float, float],
    arrow_len: float,
    thickness: int,
    color: tuple[int, int, int],
    *,
    open_head: bool,
) -> tuple[Point, npt.NDArray[np.float64]]:
    """Draw an arrow (or caret) pointing along ``direction``; return its tip and hull points.

    The head carries deliberately more mass than the tail: a *filled* triangle for an arrow,
    an open V for a caret. That asymmetry -- heavy, narrowing head versus thin, flat tail --
    is exactly what the estimator's arrowhead-mass heuristic recovers.
    """
    dx, dy = direction
    nx, ny = -dy, dx  # unit perpendicular
    cx, cy = center
    half = arrow_len / 2.0
    tip = (cx + dx * half, cy + dy * half)
    tail = (cx - dx * half, cy - dy * half)

    head_len = arrow_len * 0.35
    head_half_w = arrow_len * 0.20
    base = (tip[0] - dx * head_len, tip[1] - dy * head_len)
    left = (base[0] + nx * head_half_w, base[1] + ny * head_half_w)
    right = (base[0] - nx * head_half_w, base[1] - ny * head_half_w)

    if open_head:
        # Caret: shaft plus two open barbs, no fill -- still head-heavy but hollow.
        cv2.line(
            canvas,
            (round(tail[0]), round(tail[1])),
            (round(tip[0]), round(tip[1])),
            color,
            thickness,
            lineType=cv2.LINE_8,
        )
        for barb in (left, right):
            cv2.line(
                canvas,
                (round(tip[0]), round(tip[1])),
                (round(barb[0]), round(barb[1])),
                color,
                thickness,
                lineType=cv2.LINE_8,
            )
    else:
        cv2.line(
            canvas,
            (round(tail[0]), round(tail[1])),
            (round(tip[0]), round(tip[1])),
            color,
            thickness,
            lineType=cv2.LINE_8,
        )
        head = np.array([tip, left, right], dtype=np.float64)
        cv2.fillPoly(canvas, [np.round(head).astype(np.int32)], color, lineType=cv2.LINE_8)

    hull = np.array([tail, tip, left, right], dtype=np.float64)
    return Point(x=float(tip[0]), y=float(tip[1])), hull


def _draw_target(
    canvas: npt.NDArray[np.uint8],
    tip: Point,
    direction: tuple[float, float],
    gap: float,
    size: float,
    color: tuple[int, int, int],
    width: int,
    height: int,
) -> BBox:
    """Draw a filled square target ``gap`` px past ``tip`` along ``direction``; return its box."""
    dx, dy = direction
    half = size / 2.0
    tcx = tip.x + dx * (gap + half)
    tcy = tip.y + dy * (gap + half)
    corners = np.array(
        [
            (tcx - half, tcy - half),
            (tcx + half, tcy - half),
            (tcx + half, tcy + half),
            (tcx - half, tcy + half),
        ],
        dtype=np.float64,
    )
    cv2.fillPoly(canvas, [np.round(corners).astype(np.int32)], color, lineType=cv2.LINE_8)
    return _aabb_of(corners, width, height)


def synthesize_markers(spec: MarkerSpec) -> MarkerImage:
    """Draw ``spec.n_markers`` markers with exact tip/direction/centroid ground truth.

    Placement is rejection-sampled to be non-overlapping, drawing from one
    ``np.random.default_rng(spec.seed)`` in a fixed order (position, then orientation, per
    marker) so the output is byte-identical for a given seed. A ``dot`` is drawn as a filled
    circle and reports ``direction=None``; an ``arrow``/``caret`` reports a unit direction.
    """
    rng = np.random.default_rng(spec.seed)
    canvas = np.empty((spec.height, spec.width, 3), dtype=np.uint8)
    canvas[:, :] = spec.bg_color

    # Footprint radius large enough that a target (if any) also clears its neighbours.
    reach = spec.arrow_len * (1.0 if not spec.with_targets else 1.9)
    radius = reach * 0.6
    margin = radius + 2.0
    min_gap = 2.0 * radius

    centres: list[tuple[float, float]] = []
    attempts = 0
    while len(centres) < spec.n_markers and attempts < _SCATTER_MAX_ATTEMPTS:
        attempts += 1
        cx = float(rng.uniform(margin, spec.width - margin))
        cy = float(rng.uniform(margin, spec.height - margin))
        if all((cx - px) ** 2 + (cy - py) ** 2 >= min_gap**2 for px, py in centres):
            centres.append((cx, cy))
    if len(centres) < spec.n_markers:
        logger.warning(
            f"marker mode placed {len(centres)}/{spec.n_markers} markers after {attempts} "
            f"attempts (canvas too full); recording the achieved count"
        )

    target_color = (spec.fg_color[1], spec.fg_color[2], spec.fg_color[0])
    target_size = spec.arrow_len * 0.5

    markers: list[MarkerGT] = []
    for cx, cy in centres:
        angle = float(rng.uniform(-np.pi, np.pi))
        direction = (float(np.cos(angle)), float(np.sin(angle)))

        if spec.marker == "dot":
            radius_px = max(1, round(spec.arrow_len * 0.25))
            cv2.circle(
                canvas, (round(cx), round(cy)), radius_px, spec.fg_color, -1, lineType=cv2.LINE_8
            )
            box = BBox(
                x=max(0, round(cx) - radius_px),
                y=max(0, round(cy) - radius_px),
                w=2 * radius_px + 1,
                h=2 * radius_px + 1,
            ).clipped_to(spec.width, spec.height)
            markers.append(
                MarkerGT(
                    box=box,
                    tip=Point(x=cx, y=cy),
                    direction=None,
                    centroid=Point(x=cx, y=cy),
                    target=None,
                )
            )
            continue

        tip, hull = _draw_arrow(
            canvas,
            (cx, cy),
            direction,
            float(spec.arrow_len),
            spec.thickness,
            spec.fg_color,
            open_head=(spec.marker == "caret"),
        )
        target = None
        if spec.with_targets:
            target = _draw_target(
                canvas,
                tip,
                direction,
                float(spec.target_gap),
                target_size,
                target_color,
                spec.width,
                spec.height,
            )
        markers.append(
            MarkerGT(
                box=_aabb_of(hull, spec.width, spec.height),
                tip=tip,
                direction=direction,
                centroid=Point(x=cx, y=cy),
                target=target,
            )
        )

    return MarkerImage(image=canvas, markers=tuple(markers), spec=spec)


def save_marker_image(out: MarkerImage, image_path: Path) -> Path:
    """Write the PNG and a ``<stem>.markers.json`` sidecar carrying every marker's exact GT."""
    image_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(image_path), out.image):
        raise OSError(f"failed to write marker image to {image_path}")
    sidecar = image_path.with_suffix(".markers.json")
    payload = {
        "image": image_path.name,
        "spec": out.spec.model_dump(mode="json"),
        "markers": [
            {
                "box": marker.box.model_dump(mode="json"),
                "tip": marker.tip.model_dump(mode="json"),
                "direction": list(marker.direction) if marker.direction is not None else None,
                "centroid": marker.centroid.model_dump(mode="json"),
                "target": marker.target.model_dump(mode="json") if marker.target else None,
            }
            for marker in out.markers
        ],
    }
    sidecar.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    logger.info(f"wrote {image_path.name} with {len(out.markers)} marker(s)")
    return image_path


MARKER_DEMO_SPECS: Mapping[str, MarkerSpec] = {
    # Bare arrows -- the clean orientation-estimation case.
    "arrows": MarkerSpec(seed=11, marker="arrow", n_markers=4),
    # Arrows each pointing at a known target object a fixed gap away.
    "arrows-with-targets": MarkerSpec(
        seed=12, marker="arrow", n_markers=3, with_targets=True, target_gap=30
    ),
    # Symmetric dots -- the "no direction" case.
    "dots": MarkerSpec(seed=13, marker="dot", n_markers=5),
}
