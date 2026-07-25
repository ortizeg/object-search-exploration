"""Peak extraction from a dense response map -- three strategies, one offering.

Peak extraction is where template-style methods actually fail: a similarity map with two
nearby instances is easy to turn into *one* detection by accident, and that undercounts
recall on exactly the repeated-instance images this project exists to search. So this module
ships three *selectable* strategies rather than a single buried heuristic, and the difference
between them is measurable:

- ``"nms"`` -- every above-floor location becomes a template-sized box, then greedy IoU NMS.
  This is the baseline that **merges touching instances**: two peaks whose boxes overlap by
  more than the IoU threshold collapse into one. It exists precisely so that failure is
  demonstrable and comparable, not hidden.
- ``"local-max"`` (default) -- a ``maximum_filter`` with a footprint **derived from the
  exemplar crop size**, keeping points that equal their local maximum and clear the floor.
  Because the footprint is size-aware (not a magic pixel count), two instances a crop-width
  apart survive as two peaks where ``"nms"`` merges them. Demonstrating this separation is a
  Phase 2 success criterion.
- ``"watershed"`` -- distance-transform the thresholded map, take the distance maxima as
  markers, and emit one peak per marked region. The distance transform of two touching blobs
  has two maxima, so this also separates them; it is the recommended strategy for the dense
  similarity maps of the DINOv2 method (PITFALLS.md 4.6). Secondary to ``local-max`` here.

The strategy selector is a small ``if / elif`` at the top of :func:`extract_peaks` -- **not**
a class hierarchy or a registry. This is a leaf utility and must stay readable top to bottom;
hidden dispatch would fight the repo's single most important convention.

Every strategy sorts its output by ``(-score, y, x)`` -- the same total order as
:mod:`object_search.search.common.nms` -- so a symmetric map returns the same peaks in the
same order between runs (the reproducibility constraint; PITFALLS.md 6.3).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import numpy.typing as npt
from loguru import logger
from pydantic import BaseModel, ConfigDict
from scipy import ndimage

from object_search.schemas import BBox
from object_search.search.common.nms import nms

PeakStrategy = Literal["nms", "local-max", "watershed"]


class Peak(BaseModel):
    """A single extracted peak: an integer pixel location and its response value.

    Frozen, like every other cross-boundary value in the codebase, so a peak that has been
    handed to a caller cannot be mutated behind its back.

    Attributes:
        x: Column index into the response map.
        y: Row index into the response map.
        score: The response value at ``(y, x)``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    x: int
    y: int
    score: float


def _odd_at_least_three(n: int) -> int:
    """Round a footprint side up to the nearest odd number, floored at 3.

    ``maximum_filter`` behaves surprisingly on even-sized footprints (the window is not
    centred), and a 1- or 2-pixel footprint makes every pixel its own maximum. An odd side
    ``>= 3`` gives a genuinely centred neighbourhood.
    """
    n = max(3, n)
    return n if n % 2 == 1 else n + 1


def _sanitize(response: npt.NDArray[np.float32]) -> npt.NDArray[np.float64]:
    """Replace non-finite values so they can never be selected as a maximum.

    A NaN in the response map is returned as the maximum by ``argmax`` on some platforms
    (PITFALLS.md 1.1 / §1.8 note on non-finite maps). Map NaN and -inf to ``-inf`` (never a
    peak) and +inf down to the finite maximum, in float64 so the downstream comparisons are
    layout-independent (PITFALLS.md 6.7).
    """
    arr = np.asarray(response, dtype=np.float64)
    finite_max = float(arr[np.isfinite(arr)].max()) if np.isfinite(arr).any() else 0.0
    return np.nan_to_num(arr, nan=-np.inf, posinf=finite_max, neginf=-np.inf)


def _finalize(peaks: list[Peak], max_peaks: int) -> list[Peak]:
    """Sort by the canonical ``(-score, y, x)`` total order and truncate to ``max_peaks``."""
    ordered = sorted(peaks, key=lambda p: (-p.score, p.y, p.x))
    return ordered[:max_peaks]


def extract_peaks(
    response: npt.NDArray[np.float32],
    *,
    strategy: PeakStrategy = "local-max",
    template_w: int,
    template_h: int,
    floor: float,
    max_peaks: int = 50,
    suppression_radius_frac: float = 0.5,
    nms_iou: float = 0.3,
) -> list[Peak]:
    """Extract peaks from a response map under one of three strategies.

    Args:
        response: The float correlation/similarity map, shape ``(H, W)``. Non-finite values
            are sanitised so they cannot masquerade as maxima.
        strategy: ``"nms"`` (merges touching instances), ``"local-max"`` (default, size-aware
            separation), or ``"watershed"`` (distance-transform separation).
        template_w: Exemplar crop width **at the response map's scale**. The suppression
            footprint is derived from this, which is what makes ``local-max`` size-aware
            instead of using a fixed pixel count.
        template_h: Exemplar crop height at the response map's scale.
        floor: Minimum response value for a location to be considered. Strict ``>``.
        max_peaks: Cap on the number of returned peaks after sorting.
        suppression_radius_frac: Footprint side as a fraction of the template size. ``0.5``
            (the default) means a peak suppresses others within half a crop.
        nms_iou: IoU threshold for the ``"nms"`` strategy only.

    Returns:
        Peaks sorted by ``(-score, y, x)``, truncated to ``max_peaks``. Empty when nothing
        clears ``floor``.

    Raises:
        ValueError: On an unknown ``strategy`` (the ``Literal`` type already guards callers;
            this catches a bad runtime string) or a non-2-D ``response``.
    """
    if response.ndim != 2:
        raise ValueError(f"response must be a 2-D map, got shape {response.shape}")

    resp = _sanitize(response)
    fh = _odd_at_least_three(round(template_h * suppression_radius_frac))
    fw = _odd_at_least_three(round(template_w * suppression_radius_frac))

    # Small dispatch -- three leaf functions, no hierarchy, no registry.
    if strategy == "nms":
        return _finalize(_peaks_nms(resp, template_w, template_h, floor, nms_iou), max_peaks)
    if strategy == "local-max":
        return _finalize(_peaks_local_max(resp, fh, fw, floor), max_peaks)
    if strategy == "watershed":
        return _finalize(_peaks_watershed(resp, fh, fw, floor), max_peaks)

    raise ValueError(f"unknown peak strategy {strategy!r}; expected nms | local-max | watershed")


def _peaks_nms(
    resp: npt.NDArray[np.float64],
    template_w: int,
    template_h: int,
    floor: float,
    nms_iou: float,
) -> list[Peak]:
    """Baseline: every above-floor location becomes a template box, then greedy IoU NMS.

    This is the strategy that *merges* touching instances -- two peaks whose template-sized
    boxes overlap by more than ``nms_iou`` collapse to one. That is the point: it is the
    control the size-aware strategies are measured against.
    """
    ys, xs = np.where(resp > floor)
    if ys.size == 0:
        return []

    # A box of template size centred on each above-floor location (top-left clamped to >= 0,
    # since BBox forbids negative origins). The exact anchoring is irrelevant to the merge
    # behaviour, only the mutual overlap is.
    boxes: list[BBox] = []
    scores: list[float] = []
    for y, x in zip(ys.tolist(), xs.tolist(), strict=True):
        x0 = max(0, x - template_w // 2)
        y0 = max(0, y - template_h // 2)
        boxes.append(BBox(x=x0, y=y0, w=template_w, h=template_h))
        scores.append(float(resp[y, x]))

    kept = nms(boxes, scores, iou_threshold=nms_iou)
    # Report each survivor at its ORIGINAL peak pixel, not the clamped box corner.
    return [Peak(x=int(xs[i]), y=int(ys[i]), score=float(resp[ys[i], xs[i]])) for i in kept]


def _peaks_local_max(
    resp: npt.NDArray[np.float64],
    fh: int,
    fw: int,
    floor: float,
) -> list[Peak]:
    """Default: size-aware local maxima via ``maximum_filter``, then radius de-duplication.

    A location is a candidate when it equals the maximum of its ``(fh, fw)`` neighbourhood
    and clears the floor. Because the footprint is derived from the crop size, two instances a
    crop-width apart each remain a local maximum -- which is exactly the separation that plain
    NMS destroys.
    """
    filtered = ndimage.maximum_filter(resp, size=(fh, fw), mode="nearest")
    ys, xs = np.where((resp == filtered) & (resp > floor))
    if ys.size == 0:
        return []

    candidates = [
        Peak(x=int(x), y=int(y), score=float(resp[y, x]))
        for y, x in zip(ys.tolist(), xs.tolist(), strict=True)
    ]
    # Drop near-duplicates within the footprint radius, keeping the strongest. A flat-topped
    # peak equals its own maximum at several adjacent pixels; without this it would report a
    # cluster. Greedy in (-score, y, x) order so the survivor is deterministic.
    radius = max(fh, fw) // 2
    ordered = sorted(candidates, key=lambda p: (-p.score, p.y, p.x))
    kept: list[Peak] = []
    for cand in ordered:
        if all((cand.x - k.x) ** 2 + (cand.y - k.y) ** 2 >= radius * radius for k in kept):
            kept.append(cand)
    return kept


def _peaks_watershed(
    resp: npt.NDArray[np.float64],
    fh: int,
    fw: int,
    floor: float,
) -> list[Peak]:
    """Distance-transform separation: markers at distance maxima, one peak per region.

    Threshold the map, take the Euclidean distance transform of the mask, and treat the
    distance maxima as region markers. Two touching blobs produce two distance maxima (the
    transform dips at their shared waist), so this separates them where connected-components
    labelling alone would fuse them (PITFALLS.md 4.6).
    """
    mask = resp > floor
    if not mask.any():
        return []

    distance = ndimage.distance_transform_edt(mask)
    dist_max = ndimage.maximum_filter(distance, size=(fh, fw), mode="nearest")
    marker_mask = mask & (distance == dist_max) & (distance > 0.0)

    labelled, n_labels = ndimage.label(marker_mask)
    peaks: list[Peak] = []
    # Skip label 0: ndimage.label (like connectedComponentsWithStats) reserves 0 for the
    # BACKGROUND. Emitting it would produce a full-map false peak (PITFALLS.md 4.6).
    for label in range(1, n_labels + 1):
        ys, xs = np.where(labelled == label)
        # Within this marker region pick the single highest-response pixel.
        best = int(np.argmax(resp[ys, xs]))
        y, x = int(ys[best]), int(xs[best])
        peaks.append(Peak(x=x, y=y, score=float(resp[y, x])))

    if not peaks:
        logger.debug("watershed found a non-empty mask but no interior distance maxima")
    return peaks
