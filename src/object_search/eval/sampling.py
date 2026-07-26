"""Seeded exemplar sampling: query every method at 1 and 3 exemplars, reproducibly (EVAL-23, D-11).

The harness scores each method at **two operating points** and reports both, because they answer
different questions (D-05):

* **1 exemplar** -- the product's real operating point (the UI draws one box). The headline UX
  number.
* **3 exemplars** -- the published-benchmark convention, so our numbers sit next to the few-shot
  counting/detection leaderboards.

They are the same *selection* looked at two ways, not two independent draws: the 1-exemplar set is
the **first element of the 3-exemplar set**. That is the whole reason this is one function returning
an ordered tuple rather than two samplers -- a re-draw at ``count=1`` would make the 1-vs-3
comparison confounded by which boxes happened to be chosen.

How the selection is built (one canonical ordering, sliced by ``count``):

1. **Native exemplars first.** Datasets that ship their own exemplar boxes (FSCD-* provide three
   per image; the converter records them in :attr:`GroundTruth.exemplar_indices`) have those boxes
   placed at the front, in their native order. So a 3-exemplar run on FSCD-* returns exactly the
   dataset's own three exemplars -- comparable to the published protocol -- and the 1-exemplar run
   takes their first.
2. **Then a seeded draw of the remainder.** Any positions still needed (an image with fewer than
   ``count`` native exemplars, e.g. CARPK, whose converter records a single one) are filled by
   drawing distinct *other* GT boxes with :func:`numpy.random.default_rng` seeded from config --
   **never** ``cv2.setRNGSeed``, which controls nothing here (D-11). The seed therefore moves only
   this sampled tail; the native prefix is seed-independent.

Reproducibility (a project constraint): the same ``(gt, count, seed)`` yields byte-identical
:class:`ExemplarBox` tuples, because the only stochastic step is a single seeded permutation over a
sorted index list.
"""

from __future__ import annotations

import numpy as np

from object_search.eval.labels import GroundTruth
from object_search.schemas.geometry import ExemplarBox


def _selection_order(gt: GroundTruth, seed: int) -> list[int]:
    """The canonical ordering of GT box indices: native exemplars, then a seeded draw of the rest.

    Native exemplar indices (:attr:`GroundTruth.exemplar_indices`) come first in their recorded
    order, de-duplicated; the remaining indices follow in one ``default_rng(seed)`` permutation.
    Slicing this by ``count`` gives the prefix property for free, and the seed reaches only the
    non-native tail.
    """
    seen: set[int] = set()
    native: list[int] = []
    for index in gt.exemplar_indices:
        if index not in seen:
            seen.add(index)
            native.append(index)

    remaining = [i for i in range(len(gt.boxes)) if i not in seen]
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(len(remaining))
    sampled = [remaining[int(p)] for p in permutation]
    return native + sampled


def sample_exemplars(gt: GroundTruth, *, count: int, seed: int) -> tuple[ExemplarBox, ...]:
    """Draw ``count`` exemplar boxes from ``gt``, native-first then seeded (EVAL-23, D-11).

    The returned tuple is a prefix-stable selection: ``sample_exemplars(gt, count=1, seed=s)`` is
    always ``sample_exemplars(gt, count=3, seed=s)[:1]``. Native exemplar boxes (FSCD-* ship three)
    are honoured at the front in their recorded order; any further boxes needed are drawn distinctly
    from the rest of ``gt.boxes`` with :func:`numpy.random.default_rng` seeded from ``seed``.

    Args:
        gt: The image's ground truth. Its :attr:`GroundTruth.exemplar_indices` supplies the native
            exemplars (empty for a dataset that ships none, e.g. sampled draws for CARPK's tail).
        count: How many exemplars to return (``1`` = product operating point, ``3`` = literature
            convention). Must be ``>= 1``.
        seed: Config seed for the permutation of the non-native remainder. Reproducible: the same
            seed yields the same tail; it never perturbs the native prefix.

    Returns:
        A tuple of ``count`` :class:`ExemplarBox` (fewer when ``gt`` has fewer than ``count`` boxes
        -- see below), sampled without replacement so the exemplars are distinct.

    Raises:
        ValueError: If ``count < 1``. A zero-exemplar query is meaningless -- every method needs at
            least one positive example to search from.

    Note:
        **Out-of-range is explicit, not an error.** When ``count`` exceeds the number of GT boxes,
        every box is returned (the selection is simply as long as the image allows) rather than
        raising or indexing past the end -- a 3-exemplar request on a 2-instance image is a real,
        expected case, not a bug.
    """
    if count < 1:
        raise ValueError(f"count={count} must be >= 1; a search needs at least one exemplar")
    order = _selection_order(gt, seed)
    chosen = order[: min(count, len(gt.boxes))]
    return tuple(ExemplarBox(box=gt.boxes[index]) for index in chosen)
