"""Marker reference-point and orientation estimation -- the one genuinely new CV piece (M2-02).

A marker (an arrow, a caret, a dot) is a *gesture*: the exploration needs to know **where** it
points from and **which way** it points. This module answers both, by two paths that prefer the
free one:

Transform path (preferred)
    When the marker-finding method fitted a per-instance 2x3 affine (sparse-geo fills
    ``Match.transform``, a flattened row-major ``[a, b, tx, c, d, ty]``), the rotation falls out
    directly as ``theta = atan2(c, a)``. The 180-degree flip a bare rotation cannot resolve is
    settled by mapping the *exemplar* marker's own tip through the same transform: the pointing
    direction runs from the instance centroid toward that mapped tip, which is also taken as the
    reference point.

PCA path (fallback)
    When no transform is available (ncc, dino-dense supply none), the principal axis of the
    marker's foreground mask is recovered by PCA. The axis is a line, not a ray, so the tip is
    disambiguated by an **arrowhead-mass heuristic**: an arrow is head-heavy (a filled triangle
    plus shaft), so the side of the centroid carrying more foreground mass is the head; the tip is
    the farthest foreground pixel on that side and the direction runs centroid -> tip.

Symmetric / low-confidence
    A dot, a plus or a blob has near-balanced mass either side of the centroid. Guessing a
    direction there would be a fabricated signal, so this returns the **centroid and
    ``direction=None``** with a low confidence -- never a guessed direction (spec-required).

Pre-processing (explicit)
    The foreground mask is the Otsu threshold of each pixel's Euclidean colour distance from the
    background, where the background colour is the **median of the crop's one-pixel border ring**.
    The border median (not the four corners) is used because a marker's axis-aligned bounding box
    is tangent to the marker at its extreme points -- an arrow's tip and tail routinely land on box
    corners -- so a corner sample would be marker-coloured and poison the estimate, whereas a
    single tip pixel is negligible against a whole ring of background. Distance-from-background
    (rather than a fixed grey threshold) is in turn robust to the marker's fill ratio: a thin
    arrow occupies a minority of its box while a dot occupies a majority.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np
import numpy.typing as npt

from object_search.schemas.geometry import BBox, Point

# Below this normalised mass-asymmetry the two ends are indistinguishable: return no direction.
_SYMMETRY_TOL = 0.12
# Fewer foreground pixels than this is too little to fit an axis to; treat as symmetric.
_MIN_FOREGROUND = 12


@dataclass(frozen=True)
class MarkerGeometry:
    """Where a marker points from and which way.

    Attributes:
        reference_point: The point object proposals are scored *from*, in scene pixels -- the
            arrow tip for a pointing marker, the centroid for a symmetric one.
        direction: A **unit** pointing vector in scene pixels, or ``None`` when the marker is
            symmetric / the tip could not be disambiguated. ``None`` is a real answer, not a
            failure: it means "no direction to guess".
        confidence: ``[0, 1]`` mass-asymmetry for the PCA path (``1.0`` for the transform path),
            so a caller can down-weight a shaky estimate rather than trust it blindly.
    """

    reference_point: Point
    direction: tuple[float, float] | None
    confidence: float


def theta_from_transform(transform: tuple[float, ...]) -> float:
    """Recover the rotation angle (radians) of a flattened 2x3 affine ``[a, b, tx, c, d, ty]``.

    ``a = s*cos(theta)`` and ``c = s*sin(theta)`` for a similarity, so ``atan2(c, a)`` cancels the
    scale and returns the rotation directly.
    """
    a, _b, _tx, c, _d, _ty = transform
    return math.atan2(c, a)


def _apply_affine(transform: tuple[float, ...], point: Point) -> Point:
    """Map ``point`` through a flattened 2x3 affine ``[a, b, tx, c, d, ty]``."""
    a, b, tx, c, d, ty = transform
    return Point(x=a * point.x + b * point.y + tx, y=c * point.x + d * point.y + ty)


def foreground_mask(crop: npt.NDArray[np.uint8]) -> npt.NDArray[np.bool_]:
    """Boolean foreground mask: Otsu on each pixel's colour distance from the corner-median bg."""
    h, w = crop.shape[:2]
    border = np.concatenate(
        [
            crop[0, :].reshape(-1, 3),
            crop[h - 1, :].reshape(-1, 3),
            crop[:, 0].reshape(-1, 3),
            crop[:, w - 1].reshape(-1, 3),
        ]
    ).astype(np.float64)
    background = np.median(border, axis=0)
    dist = np.linalg.norm(crop.astype(np.float64) - background, axis=2)
    peak = float(dist.max())
    if peak <= 1e-6:
        return np.zeros((h, w), dtype=np.bool_)
    dist_u8 = np.clip(dist / peak * 255.0, 0, 255).astype(np.uint8)
    _, binary = cv2.threshold(dist_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary > 0


def _pca_geometry(crop: npt.NDArray[np.uint8], box: BBox) -> MarkerGeometry:
    """Estimate geometry from the foreground mask alone (no transform available)."""
    mask = foreground_mask(crop)
    ys, xs = np.nonzero(mask)
    # Scene-pixel centroid is always defined; it is the fallback reference point.
    if xs.size < _MIN_FOREGROUND:
        return MarkerGeometry(
            reference_point=Point(x=box.cx, y=box.cy), direction=None, confidence=0.0
        )

    coords = np.stack([xs, ys], axis=1).astype(np.float64)  # (N, 2) as (x, y)
    centroid = coords.mean(axis=0)
    centroid_pt = Point(x=float(centroid[0] + box.x), y=float(centroid[1] + box.y))

    # 1. Principal axis by PCA on the centred coordinates.
    centered = coords - centroid
    cov = centered.T @ centered / float(xs.size)
    _eigvals, eigvecs = np.linalg.eigh(cov)
    axis = eigvecs[:, -1]  # eigenvector of the largest eigenvalue

    # 2. Arrowhead-mass heuristic: the head-heavy side of the centroid is the pointing side.
    projection = centered @ axis
    n_pos = int(np.count_nonzero(projection > 0.0))
    n_neg = int(xs.size - n_pos)
    asymmetry = abs(n_pos - n_neg) / float(xs.size)

    # 3. Symmetric marker -> centroid and NO direction; never a guessed one.
    if asymmetry < _SYMMETRY_TOL:
        return MarkerGeometry(reference_point=centroid_pt, direction=None, confidence=asymmetry)

    head_sign = 1.0 if n_pos > n_neg else -1.0
    # 4. Tip = the farthest foreground pixel along the axis on the head side.
    signed = projection * head_sign
    tip_idx = int(np.argmax(signed))
    tip_pt = Point(x=float(coords[tip_idx, 0] + box.x), y=float(coords[tip_idx, 1] + box.y))

    vx = tip_pt.x - centroid_pt.x
    vy = tip_pt.y - centroid_pt.y
    norm = math.hypot(vx, vy)
    if norm <= 1e-9:
        return MarkerGeometry(reference_point=centroid_pt, direction=None, confidence=asymmetry)
    return MarkerGeometry(
        reference_point=tip_pt, direction=(vx / norm, vy / norm), confidence=asymmetry
    )


def estimate_geometry(
    crop: npt.NDArray[np.uint8],
    box: BBox,
    transform: tuple[float, ...] | None = None,
    exemplar_tip: Point | None = None,
) -> MarkerGeometry:
    """Estimate one marker instance's reference point and pointing direction.

    Args:
        crop: The BGR marker crop, ``image[box.y:box.y2, box.x:box.x2]``.
        box: The marker's box in scene pixels (used to lift crop coordinates back to the scene).
        transform: The per-instance 2x3 affine if the marker method fitted one, else ``None``.
        exemplar_tip: The exemplar marker's own tip in scene pixels; only used on the transform
            path to resolve the pointing sign. ``None`` falls back to the raw rotation angle.

    Returns:
        A :class:`MarkerGeometry`. ``direction`` is ``None`` for a symmetric marker.
    """
    if transform is not None:
        theta = theta_from_transform(transform)
        centre = Point(x=box.cx, y=box.cy)
        if exemplar_tip is not None:
            mapped_tip = _apply_affine(transform, exemplar_tip)
            vx = mapped_tip.x - centre.x
            vy = mapped_tip.y - centre.y
            norm = math.hypot(vx, vy)
            if norm > 1e-9:
                return MarkerGeometry(
                    reference_point=mapped_tip,
                    direction=(vx / norm, vy / norm),
                    confidence=1.0,
                )
        # No exemplar tip (or a degenerate mapping): fall back to the raw rotation ray.
        return MarkerGeometry(
            reference_point=centre,
            direction=(math.cos(theta), math.sin(theta)),
            confidence=1.0,
        )

    return _pca_geometry(crop, box)
