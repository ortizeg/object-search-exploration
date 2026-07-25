"""Rendering helpers: match overlays, similarity heatmaps, keypoint diagnostics -- an offering.

Everything here must work **headless**: CI and the batch renderer have no display. Two rules
follow. Box and point overlays use OpenCV drawing directly -- no backend, no figure, nothing
to misconfigure. The one place a real colormap is wanted, the similarity heatmap, uses
matplotlib's colormap tables; and because *any* matplotlib import can otherwise try to open a
display, the ``Agg`` backend is forced at import time, before anything else touches the
library. That is the whole reason the two ``matplotlib`` lines sit above the other imports.

The heatmap encoder returns the Phase 1 :class:`HeatmapPayload` -- a base64 PNG plus the
**true** ``vmin``/``vmax`` of the underlying float map -- so the UI can label the colour scale
with real numbers instead of implying every map spans 0..1 (schema docstring; PITFALLS.md 6.8
on why dense arrays travel as PNG, not JSON).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless backend; MUST precede any other matplotlib import (no display)

import base64
from collections.abc import Sequence

import cv2
import numpy as np
import numpy.typing as npt
from matplotlib import colormaps

from object_search.schemas import ExemplarBox, HeatmapPayload, Match, Point

# BGR colours (OpenCV order). The exemplar is deliberately a different hue and a thicker line
# from an ordinary match: Method 2 labels its own self-match as the exemplar, and the render
# must not let that read as just another detection.
_MATCH_COLOR = (0, 200, 0)  # green
_EXEMPLAR_COLOR = (255, 0, 255)  # magenta
_KEYPOINT_COLOR = (0, 165, 255)  # orange
_CORR_COLOR = (0, 255, 255)  # yellow
_LABEL_BAR_H = 22
_HEATMAP_CMAP = "viridis"


def _to_bgr(image: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    """Return a 3-channel BGR uint8 copy, accepting either a grayscale or BGR input."""
    if image.ndim == 2:
        return np.asarray(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), dtype=np.uint8)
    return image.copy()


def draw_matches(
    image: npt.NDArray[np.uint8],
    matches: Sequence[Match],
    *,
    exemplar: ExemplarBox | None = None,
    show_scores: bool = True,
) -> npt.NDArray[np.uint8]:
    """Draw each match box (with its score) over the scene, exemplar rendered distinctly.

    Args:
        image: BGR or grayscale scene, uint8.
        matches: The matches to draw. A match flagged ``is_exemplar`` is drawn in the
            distinct exemplar style, not the ordinary-match style.
        exemplar: Optional explicit exemplar box to draw distinctly and label, for callers
            that carry the query separately from the match list.
        show_scores: When True, annotate each ordinary match with its score.

    Returns:
        A new BGR uint8 image; the input is not mutated.
    """
    canvas = _to_bgr(image)
    for match in matches:
        x, y, x2, y2 = match.box.xyxy
        if match.is_exemplar:
            _draw_box(canvas, (x, y, x2, y2), _EXEMPLAR_COLOR, thickness=3, label="EXEMPLAR")
        else:
            label = f"{match.score:.2f}" if show_scores else None
            _draw_box(canvas, (x, y, x2, y2), _MATCH_COLOR, thickness=2, label=label)

    if exemplar is not None:
        ex = exemplar.box
        _draw_box(canvas, ex.xyxy, _EXEMPLAR_COLOR, thickness=3, label="EXEMPLAR")

    return canvas


def _draw_box(
    canvas: npt.NDArray[np.uint8],
    xyxy: tuple[int, int, int, int],
    color: tuple[int, int, int],
    *,
    thickness: int,
    label: str | None,
) -> None:
    """Draw one rectangle (and optional label) in place. x2/y2 are exclusive -> draw to -1."""
    x, y, x2, y2 = xyxy
    cv2.rectangle(canvas, (x, y), (x2 - 1, y2 - 1), color, thickness)
    if label:
        cv2.putText(
            canvas, label, (x, max(0, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA
        )


def heatmap_png_b64(
    response: npt.NDArray[np.floating],
    *,
    vmin: float | None = None,
    vmax: float | None = None,
) -> HeatmapPayload:
    """Colour-map a 2-D response and return it as a base64 PNG plus its true value range.

    Args:
        response: The float similarity/correlation map, shape ``(H, W)``.
        vmin: Low end of the colour scale. When None, the map's finite minimum -- so the
            payload reports the *honest* range, not an assumed 0..1.
        vmax: High end. When None, the map's finite maximum.

    Returns:
        A :class:`HeatmapPayload` whose ``vmin``/``vmax`` are the actual range the colormap
        spanned, so the UI can label the scale truthfully.

    Raises:
        ValueError: If ``response`` is not 2-D.
    """
    if response.ndim != 2:
        raise ValueError(f"response must be a 2-D map, got shape {response.shape}")

    arr = np.asarray(response, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    lo = float(vmin) if vmin is not None else (float(finite.min()) if finite.size else 0.0)
    hi = float(vmax) if vmax is not None else (float(finite.max()) if finite.size else 1.0)
    span = hi - lo if hi > lo else 1e-9  # a constant map colours flat rather than dividing by 0

    normalized = np.clip((np.nan_to_num(arr, nan=lo) - lo) / span, 0.0, 1.0)
    rgba = colormaps[_HEATMAP_CMAP](normalized)  # (H, W, 4) float in 0..1
    rgb = (rgba[:, :, :3] * 255.0).astype(np.uint8)
    bgr = rgb[:, :, ::-1]  # cv2 encodes BGR; keep the round-trip consistent

    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise ValueError("failed to PNG-encode the heatmap")
    png_b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return HeatmapPayload(
        png_b64=png_b64,
        width=arr.shape[1],
        height=arr.shape[0],
        vmin=lo,
        vmax=hi,
    )


def draw_keypoints(
    image: npt.NDArray[np.uint8],
    points: Sequence[Point],
) -> npt.NDArray[np.uint8]:
    """Draw a small circle at each keypoint. Returns a new BGR uint8 image."""
    canvas = _to_bgr(image)
    for point in points:
        cv2.circle(canvas, (round(point.x), round(point.y)), 3, _KEYPOINT_COLOR, -1)
    return canvas


def draw_correspondences(
    crop: npt.NDArray[np.uint8],
    scene: npt.NDArray[np.uint8],
    correspondences: Sequence[tuple[Point, Point]],
) -> npt.NDArray[np.uint8]:
    """Side-by-side ``[crop | scene]`` canvas with a line per correspondence.

    Each correspondence is a ``(src, dst)`` pair: ``src`` is drawn on the left crop panel and
    ``dst`` on the right scene panel, joined by a line. Panels are top-aligned and padded to a
    common height. Returns a new BGR uint8 image of width ``crop_w + scene_w``.
    """
    left = _to_bgr(crop)
    right = _to_bgr(scene)
    h = max(left.shape[0], right.shape[0])
    canvas = np.zeros((h, left.shape[1] + right.shape[1], 3), dtype=np.uint8)
    canvas[: left.shape[0], : left.shape[1]] = left
    canvas[: right.shape[0], left.shape[1] :] = right

    offset = left.shape[1]
    for src, dst in correspondences:
        p_src = (round(src.x), round(src.y))
        p_dst = (round(dst.x) + offset, round(dst.y))
        cv2.circle(canvas, p_src, 3, _CORR_COLOR, -1)
        cv2.circle(canvas, p_dst, 3, _CORR_COLOR, -1)
        cv2.line(canvas, p_src, p_dst, _CORR_COLOR, 1, cv2.LINE_AA)
    return canvas


def compose_panel(tiles: Sequence[tuple[str, npt.NDArray[np.uint8]]]) -> npt.NDArray[np.uint8]:
    """Compose labelled tiles side by side into one image, for sample runs and the README.

    Args:
        tiles: ``(label, image)`` pairs. Images may be grayscale or BGR and of differing
            sizes; each gets a labelled bar above it and all are padded to a common height.

    Returns:
        A single BGR uint8 image, tiles concatenated left to right.

    Raises:
        ValueError: If ``tiles`` is empty.
    """
    if not tiles:
        raise ValueError("compose_panel requires at least one tile")

    labelled: list[npt.NDArray[np.uint8]] = [_labelled_tile(label, img) for label, img in tiles]
    height = max(tile.shape[0] for tile in labelled)
    padded: list[npt.NDArray[np.uint8]] = []
    for tile in labelled:
        canvas = np.zeros((height, tile.shape[1], 3), dtype=np.uint8)
        canvas[: tile.shape[0], : tile.shape[1]] = tile
        padded.append(canvas)
    return np.asarray(cv2.hconcat(padded), dtype=np.uint8)


def _labelled_tile(label: str, image: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    """Return the image with a labelled bar stacked on top, as BGR uint8."""
    body = _to_bgr(image)
    bar = np.zeros((_LABEL_BAR_H, body.shape[1], 3), dtype=np.uint8)
    cv2.putText(
        bar,
        label,
        (4, _LABEL_BAR_H - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return np.asarray(cv2.vconcat([bar, body]), dtype=np.uint8)
