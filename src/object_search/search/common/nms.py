"""Deterministic greedy IoU non-maximum suppression -- an offering, not a requirement.

Why this lives in a shared module at all: **deterministic tie-breaking**. A plain greedy NMS
keeps whichever of two equally-scoring boxes it happens to visit first, and "first" is
whatever order the caller built its candidate list -- which on a symmetric synthetic lattice
(EVAL-03), where repeated instances are pixel-identical and every score is exactly ``1.0``,
is not stable between runs. That silently breaks the "same image + box + method + config =>
identical results" constraint. Research measured it directly (PITFALLS.md 6.3): iterating the
six permutations of three boxes (two tied) through a greedy NMS yields two different kept
sets.

The fix is to impose a **total order** on the boxes before suppression, so ties break on
geometry rather than arrival: sort by ``(-score, y, x)``. Score descending is the usual NMS
priority; ``y`` then ``x`` is a deterministic, caller-order-independent tie-break. Never sort
by score alone.

This returns the **kept indices** into the input sequences, not the boxes, so a caller can
carry its own per-box payload (a ``Match``, a ``Candidate``, a pyramid level) through
suppression without this module needing to know the payload type.

Deliberately a plain Python loop with no vectorisation: this runs on tens of boxes, and the
readability of the one file the practitioner reads matters more than microseconds here.
"""

from __future__ import annotations

from collections.abc import Sequence

from object_search.schemas import BBox

# Boxes overlapping MORE than this are suppressed. The boundary is STRICT (`>`): a pair whose
# IoU is exactly `iou_threshold` is KEPT. Documented here because 6.3-style reproducibility
# depends on the boundary being pinned, and a test asserts this exact convention.
_SUPPRESS_IF_IOU_STRICTLY_GREATER_THAN = True  # informational; see the `>` in the loop below


def nms(
    boxes: Sequence[BBox],
    scores: Sequence[float],
    iou_threshold: float,
) -> list[int]:
    """Greedy IoU non-maximum suppression with deterministic tie-breaking.

    Args:
        boxes: The candidate boxes. Indexed in parallel with ``scores``.
        scores: One score per box; higher is stronger. May contain exact ties -- that is the
            whole reason this function pins a total order.
        iou_threshold: A surviving box suppresses any later box whose IoU with it is
            **strictly greater** than this value. At exactly the threshold the box is kept.
            ``0.3`` is the project default (:class:`NCCConfig.nms_iou`).

    Returns:
        The indices of the kept boxes, in the canonical ``(-score, y, x)`` order (not input
        order). Returning indices rather than boxes lets the caller carry its own per-box
        payload through suppression.

    Raises:
        ValueError: If ``boxes`` and ``scores`` differ in length -- a parallel-index API
            where the indices don't line up is a silent-corruption bug, not a warning.
    """
    if len(boxes) != len(scores):
        raise ValueError(
            f"boxes and scores must be parallel; got {len(boxes)} boxes and {len(scores)} scores"
        )
    if not boxes:
        return []

    # 1. Impose a TOTAL order: score DESC, then y ASC, then x ASC. This is the load-bearing
    #    line -- ties break on geometry, never on the caller's enumeration order, so the
    #    output is byte-identical across runs (PITFALLS.md 6.3).
    order = sorted(
        range(len(boxes)),
        key=lambda i: (-scores[i], boxes[i].y, boxes[i].x),
    )

    # 2. Greedy sweep in that order: take the strongest remaining box, then drop every later
    #    box that overlaps it by MORE than the threshold. `>` (not `>=`) is the pinned
    #    boundary convention: a box at exactly `iou_threshold` survives.
    kept: list[int] = []
    suppressed: set[int] = set()
    for idx in order:
        if idx in suppressed:
            continue
        kept.append(idx)
        for other in order:
            if other in suppressed or other == idx:
                continue
            if boxes[idx].iou(boxes[other]) > iou_threshold:
                suppressed.add(other)

    return kept
