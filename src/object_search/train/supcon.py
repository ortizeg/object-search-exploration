"""Supervised-contrastive (SupCon) loss over OWLv2 ``class_embeds`` -- the torch-free specification.

Torch-free on purpose (numpy + loguru only), mirroring the split
:mod:`object_search.train.owlv2_targets` already established: the part that is easy to get
*silently* wrong lives in ``src/`` where ``pixi run test`` gates every line of it with no torch, no
weights, and no GPU. ``scripts/finetune_owlv2.py`` carries a torch mirror of :func:`supcon_loss`
that is checked against this module numerically by ``finetune-owlv2 --self-check``, because torch
lives only in the ``export`` pixi environment and ``pytest`` is not installed there.

Why a contrastive objective at all
----------------------------------
Quick task 260801-8zy fine-tuned OWLv2 on this domain with a 5-way text-conditioned classification
loss and measured a negative result. Its diagnosed reason: the training objective was a *proxy*.
``owlv2-oneshot`` does not classify at inference -- it ranks scene patches by **L2-normalized cosine
similarity** to one image-derived query embedding, in the ``class_embeds`` space
(``search/owlv2_oneshot.py``: ``_l2_normalize(target.class_embeds, axis=1) @ query_embedding``).
SupCon trains exactly that property: same-class instances cosine-close, different-class and
background patches cosine-far. It is a directly-matched objective rather than a correlate.

The decisions, with their reasoning
-----------------------------------
**D-hg1-01 -- the formulation is L_out.** With ``z`` L2-normalized and ``A(i)`` every other pooled
row except ``i`` itself, ``P(i) subset A(i)`` the same-class anchors::

    L = sum_i (-1/|P(i)|) sum_{p in P(i)} log( exp(z_i.z_p/tau) / sum_{a in A(i)} exp(z_i.z_a/tau) )

Khosla et al. (2020) §3 show ``L_out <= L_in`` and that ``L_out`` is the empirically superior of the
two. The ``L_in`` variant -- summing the positives *inside* the log -- is the common
misimplementation, and a quieter bug than it looks: it still trains, and the curve still falls.

**D-hg1-02 -- temperature tau = 0.07.** The value SupCon reports its headline results with, and the
SimCLR/MoCo convention. It is exposed as ``supcon_temperature`` on ``FinetuneConfig`` so it is
written into ``train_log.json`` beside the numbers it produced, never hard-coded here.

**D-hg1-03 -- background patches are denominator-only negatives, and this is load-bearing.**
``owlv2-oneshot``'s measured floor-plan failure is precision 0.01-0.11 at high recall: *background
scores too high*. A SupCon over matched anchors alone only separates door-from-window; it cannot
touch door-from-background, and shipping it would repeat 260801-8zy's mistake in a new costume. So
patches whose grid-cell centre lies in no ground-truth box (:func:`background_patch_mask`,
:func:`sample_background_indices`) are pooled in as rows that appear **only in the denominator**:
never anchors, never positives. The ``negative_only`` argument of :func:`supcon_loss` is that rule,
and :func:`supcon_loss` enforces it rather than trusting the caller.

**D-hg1-04 -- the contrastive pool is the EFFECTIVE batch, not the micro-batch.** On the 197-image
train split the class balance is door 1822, window 1413, bathroom 283, perimeter 267, stairs 177
boxes, so at ``--batch-size 2`` a micro-batch holds ~1.8 stairs boxes on average. An anchor with no
same-class positive contributes exactly zero (see below), so a per-micro-batch pool would give the
rare classes no gradient almost every step and the loss would quietly be a door/window loss wearing
a five-class label. Pooling across ``--grad-accum 4`` gives an effective batch of 8 images and ~7
stairs boxes. **Stated cost, not hidden:** the backward is deferred to the accumulation boundary, so
``grad_accum`` micro-batch graphs are retained. In the primary ``headonly`` arm this is cheap -- the
ViT runs under ``no_grad``, so what is retained is ``class_head.dense0`` plus the box head, not the
backbone. With an unfrozen backbone it retains ``grad_accum`` ViT graphs and needs a lower
``--grad-accum``; that is one more reason the primary contrastive run is ``headonly``.
Rejected alternatives: raising ``--batch-size`` (memory at 960x960 / 3600 patches, and it changes
the optimizer schedule so the arm stops being comparable to the three already-measured ones), and
accepting sparse rare-class positives (it silently converts a 5-class objective into a 2-class one).

An anchor with no positive contributes zero -- never an error, never a NaN
--------------------------------------------------------------------------
This is a correctness requirement rather than defensive padding. Pooled batches routinely contain a
singleton class, and the natural implementations of ``L_out`` divide by ``|P(i)|``. Such an anchor
is dropped from the sum **and from the mean divisor**: it is not evidence of "no separation", it is
an absence of evidence, and averaging a fabricated ``0.0`` into the loss would dilute the real
anchors by an amount that depends on the batch composition. The same distinction the repo's
nullable human-count rule makes.

The patch grid, and the one silent way to get it wrong
------------------------------------------------------
Patch index ``i`` maps to ``(row, col) = divmod(i, grid)`` in **row-major** order, and that cell's
centre in normalized coordinates is ``((col + 0.5) / grid, (row + 0.5) / grid)`` -- in the
**padded-square** frame, the same frame ``ImageTargets.boxes`` are normalized in. Both halves were
confirmed against the installed ``transformers`` rather than assumed:
``Owlv2ForObjectDetection.normalize_grid_corner_coordinates(4, 4)`` emits its 16 corners in
x-fastest order (index 1 is corner ``(0.50, 0.25)``, row 0 / col 1), and ``compute_box_bias`` builds
per-patch box prior from exactly those coordinates. If either the raster order or the frame were
wrong, background sampling would draw "negatives" from *inside* the objects, and the loss would fall
while training the precise opposite of what it claims (threat T-hg1-01).

The diagnostic, and why its components are nullable
---------------------------------------------------
:func:`cosine_gap_report` measures the property the loss is *supposed* to move -- mean same-class
cosine against mean different-class and mean anchor-to-background cosine -- so a run carries its own
answer to "did the objective move the thing ``owlv2-oneshot`` actually scores with?", independently
of whether F1 followed (D-hg1-06). Every component is ``float | None`` and a component with no
contributing pair is ``None``, **never** ``0.0``: a pool with one anchor per class has no same-class
pair at all, and reporting ``0.0`` there would read as "measured, no separation" when the truth is
"not measurable". The same distinction the repo's nullable human-count rule makes, for the same
reason -- a fabricated zero is indistinguishable from a real one once it is in a table.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt
from loguru import logger

# Guards the L2 normalization of a zero vector, which OWLv2 can emit for a fully-padded patch. The
# same convention (and the same value) as ``search/owlv2_oneshot._l2_normalize``: a zero vector
# stays zero rather than becoming NaN.
_EPS = 1e-12


def _l2_normalize(matrix: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Row-wise L2 normalization. A zero row stays zero rather than becoming NaN (see :data:`_EPS`).

    Shared by :func:`supcon_loss` and :func:`cosine_gap_report` so the loss and the diagnostic that
    reports on it can never disagree about what "cosine" means.
    """
    normalized: npt.NDArray[np.float64] = matrix / np.maximum(
        np.linalg.norm(matrix, axis=1, keepdims=True), _EPS
    )
    return normalized


def patch_grid_size(num_patches: int) -> int:
    """Return the side of the square patch grid holding ``num_patches`` patches.

    OWLv2 at the repo's pinned 960x960 / patch-16 operating point gives ``patch_grid_size(3600) ==
    60``. Derived from the tensor the model actually returned rather than restated as a constant, so
    a re-export at a different resolution fails loudly here instead of silently mis-indexing every
    background patch.

    Args:
        num_patches: The patch-axis length of a ``class_embeds`` tensor.

    Returns:
        The integer grid side.

    Raises:
        ValueError: If ``num_patches`` is not a positive perfect square.
    """
    if num_patches <= 0:
        raise ValueError(f"num_patches must be >= 1, got {num_patches}")
    side = math.isqrt(num_patches)
    if side * side != num_patches:
        raise ValueError(f"num_patches must be a perfect square, got {num_patches}")
    return side


def background_patch_mask(
    boxes: npt.NDArray[np.floating],
    grid: int,
) -> npt.NDArray[np.bool_]:
    """Mark the patch-grid cells whose centre lies in NO ground-truth box.

    Cell ``i`` is at ``(row, col) = divmod(i, grid)`` with centre
    ``((col + 0.5) / grid, (row + 0.5) / grid)`` -- see the module docstring for why that raster
    order and that frame, and for what breaks silently if either is wrong.

    Args:
        boxes: ``(n, 4)`` float ``(cx, cy, w, h)`` normalized over the padded-square side, i.e.
            exactly :attr:`object_search.train.owlv2_targets.ImageTargets.boxes`. May be empty.
        grid: The patch-grid side, from :func:`patch_grid_size`.

    Returns:
        ``(grid * grid,)`` bool array, ``True`` where the cell is background.

    Raises:
        ValueError: If ``grid`` is not positive or ``boxes`` is not ``(n, 4)``.
    """
    if grid <= 0:
        raise ValueError(f"grid must be >= 1, got {grid}")
    box_array = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    if box_array.shape[1] != 4:
        raise ValueError(f"boxes must be (n, 4) cxcywh, got {np.shape(boxes)}")

    cell_index = np.arange(grid * grid)
    row, col = np.divmod(cell_index, grid)
    centre_x = (col + 0.5) / grid
    centre_y = (row + 0.5) / grid

    if box_array.shape[0] == 0:
        return np.ones(grid * grid, dtype=bool)

    # Inclusive bounds: a centre exactly on a box edge counts as inside, so an edge cell is never
    # sampled as a negative on a floating-point tie.
    box_cx, box_cy, box_w, box_h = (box_array[:, i] for i in range(4))
    inside_x = np.abs(centre_x[:, None] - box_cx[None, :]) <= box_w[None, :] / 2.0
    inside_y = np.abs(centre_y[:, None] - box_cy[None, :]) <= box_h[None, :] / 2.0
    inside_any: npt.NDArray[np.bool_] = np.any(inside_x & inside_y, axis=1)
    return ~inside_any


def sample_background_indices(
    boxes: npt.NDArray[np.floating],
    grid: int,
    count: int,
    rng: np.random.Generator,
) -> npt.NDArray[np.int64]:
    """Draw up to ``count`` background patch indices, without replacement, deterministically.

    The denominator-only negatives of D-hg1-03. Every returned index is a cell
    :func:`background_patch_mask` marked ``True``, so a sampled row can never be a patch the model
    is simultaneously being told is a door.

    ``rng`` is an explicitly-seeded :func:`numpy.random.default_rng`, passed in by the caller --
    never a bare ``random`` call and never a fresh generator built here, so the repo's
    reproducibility rule holds: the same seed draws the same patches, and a training run's
    background sample is a function of its config rather than of the interpreter's global state.

    Fewer than ``count`` indices is a normal return, not an error: a floor plan whose annotations
    tile most of the page genuinely has fewer background cells than asked for, and an exception
    there would abort a legitimate training run over an image that simply has little background.

    Args:
        boxes: ``(n, 4)`` ``(cx, cy, w, h)`` normalized over the padded-square side. May be empty.
        grid: The patch-grid side, from :func:`patch_grid_size`.
        count: How many background patches to draw. ``0`` is a supported disable and returns an
            empty array (it is what ``supcon_background_negatives=0`` means).
        rng: The seeded generator to draw from.

    Returns:
        An ``int64`` array of at most ``count`` patch indices, **ascending** -- so the pooled row
        order is a function of the sampled set rather than of the draw order.

    Raises:
        ValueError: If ``count`` is negative, or ``grid``/``boxes`` are rejected by
            :func:`background_patch_mask`.
    """
    if count < 0:
        raise ValueError(f"count must be >= 0, got {count}")
    if count == 0:
        return np.zeros(0, dtype=np.int64)

    candidates = np.flatnonzero(background_patch_mask(boxes, grid)).astype(np.int64)
    if candidates.size == 0:
        logger.debug(
            "sample_background_indices: every one of the {} cell(s) is covered by a box; "
            "no background negatives for this image",
            grid * grid,
        )
        return np.zeros(0, dtype=np.int64)
    if candidates.size <= count:
        return candidates

    drawn: npt.NDArray[np.int64] = np.sort(rng.choice(candidates, size=count, replace=False))
    return drawn


def cosine_gap_report(
    anchor_embeddings: npt.NDArray[np.floating],
    anchor_labels: npt.NDArray[np.integer],
    background_embeddings: npt.NDArray[np.floating],
) -> dict[str, float | None]:
    """Mean same-class / different-class / anchor-to-background cosine, and the two gaps.

    The diagnostic of D-hg1-06: it measures the property the contrastive objective claims to train,
    in the same L2-normalized space ``owlv2-oneshot`` scores in, so a run can distinguish "the loss
    moved the cosine property but F1 did not follow" from "the loss never moved it". Recorded at
    epoch 0 as well as after every epoch, because without the pre-training measurement the numbers
    have no reference point inside the run.

    Every component is nullable and a component with no contributing pair is ``None``, never
    ``0.0`` -- see the module docstring for why that distinction is load-bearing rather than
    fastidious.

    Args:
        anchor_embeddings: ``(n, d)`` matched-anchor rows. May be empty.
        anchor_labels: ``(n,)`` class labels aligned with ``anchor_embeddings``.
        background_embeddings: ``(m, d)`` background rows. May be empty.

    Returns:
        ``{"same_class_mean", "diff_class_mean", "background_mean", "gap_class",
        "gap_background"}``, each ``float`` or ``None``. ``gap_class = same_class_mean -
        diff_class_mean`` and ``gap_background = same_class_mean - background_mean``, both ``None``
        if either operand is.

    Raises:
        ValueError: If either array is not 2-D, the labels do not align with the anchors, or the
            two embedding sets disagree about ``d``.
    """
    anchors = np.asarray(anchor_embeddings, dtype=np.float64)
    background = np.asarray(background_embeddings, dtype=np.float64)
    if anchors.ndim != 2:
        raise ValueError(f"anchor_embeddings must be (n, d), got shape {anchors.shape}")
    if background.ndim != 2:
        raise ValueError(f"background_embeddings must be (m, d), got shape {background.shape}")
    labels = np.asarray(anchor_labels)
    if labels.shape != (anchors.shape[0],):
        raise ValueError(f"anchor_labels must be ({anchors.shape[0]},), got {labels.shape}")
    if anchors.size and background.size and anchors.shape[1] != background.shape[1]:
        raise ValueError(
            f"anchors are {anchors.shape[1]}-D but background is {background.shape[1]}-D"
        )

    same_class_mean: float | None = None
    diff_class_mean: float | None = None
    background_mean: float | None = None

    if anchors.shape[0] >= 2:
        normalized = _l2_normalize(anchors)
        # Upper triangle only: each unordered pair counted once, and never a row against itself
        # (whose cosine is 1 by construction and would inflate the same-class mean).
        rows, cols = np.triu_indices(anchors.shape[0], k=1)
        pair_cosine = (normalized @ normalized.T)[rows, cols]
        same_pair = labels[rows] == labels[cols]
        if bool(np.any(same_pair)):
            same_class_mean = float(pair_cosine[same_pair].mean())
        if bool(np.any(~same_pair)):
            diff_class_mean = float(pair_cosine[~same_pair].mean())

    if anchors.shape[0] and background.shape[0]:
        background_mean = float((_l2_normalize(anchors) @ _l2_normalize(background).T).mean())

    return {
        "same_class_mean": same_class_mean,
        "diff_class_mean": diff_class_mean,
        "background_mean": background_mean,
        "gap_class": (
            None
            if same_class_mean is None or diff_class_mean is None
            else same_class_mean - diff_class_mean
        ),
        "gap_background": (
            None
            if same_class_mean is None or background_mean is None
            else same_class_mean - background_mean
        ),
    }


def supcon_loss(
    embeddings: npt.NDArray[np.floating],
    labels: npt.NDArray[np.integer],
    temperature: float,
    *,
    negative_only: npt.NDArray[np.bool_] | None = None,
) -> float:
    """Supervised-contrastive loss (Khosla et al. 2020, the ``L_out`` form) over a pooled batch.

    This is the **specification**; ``scripts/finetune_owlv2.py``'s ``supcon_loss_torch`` mirrors it
    line for line and ``finetune-owlv2 --self-check`` asserts the two agree to ``<1e-6``.

    Every embedding is L2-normalized inside, so the returned value depends only on the *directions*
    of the rows -- which is what the cosine scoring at inference sees, and what makes the loss
    invariant to a positive rescaling of any row.

    Args:
        embeddings: ``(n, d)`` float array of pooled embeddings (matched anchors first, then any
            background rows).
        labels: ``(n,)`` integer class labels. Entries at ``negative_only`` positions are ignored.
        temperature: The SupCon temperature ``tau`` (D-hg1-02 pins the default at 0.07). Must be
            positive; it divides the similarities, so a lower value sharpens the weighting and
            raises the loss of an imperfect configuration.
        negative_only: ``(n,)`` bool array, ``True`` for rows that appear **only** in the
            denominator: never anchors, never positives (D-hg1-03, the background patches). ``None``
            means every row is a full participant.

    Returns:
        The mean loss over the anchors that have at least one same-class positive. Exactly ``0.0``
        when no anchor does -- never NaN, and never an exception.

    Raises:
        ValueError: If the shapes disagree or ``temperature`` is not positive.
    """
    z = np.asarray(embeddings, dtype=np.float64)
    if z.ndim != 2:
        raise ValueError(f"embeddings must be (n, d), got shape {z.shape}")
    y = np.asarray(labels)
    if y.shape != (z.shape[0],):
        raise ValueError(f"labels must be ({z.shape[0]},), got {y.shape}")
    if temperature <= 0.0:
        raise ValueError(f"temperature must be > 0, got {temperature}")

    count = z.shape[0]
    if negative_only is None:
        denominator_only = np.zeros(count, dtype=bool)
    else:
        denominator_only = np.asarray(negative_only, dtype=bool)
        if denominator_only.shape != (count,):
            raise ValueError(f"negative_only must be ({count},), got {denominator_only.shape}")

    if count < 2:  # a pool of one has no other row to contrast against
        return 0.0

    # 1. L2-normalize, so the dot product below IS cosine -- the space owlv2-oneshot scores in.
    z = _l2_normalize(z)
    similarity = (z @ z.T) / temperature

    # 2. A(i) is every other row INCLUDING the denominator-only ones; P(i) is the same-class rows,
    #    excluding self and excluding anything denominator-only on either side.
    self_pair = np.eye(count, dtype=bool)
    in_denominator = ~self_pair
    positives = (
        (y[:, None] == y[None, :])
        & ~self_pair
        & ~denominator_only[None, :]
        & ~denominator_only[:, None]
    )

    # 3. log sum_{a in A(i)} exp(sim), with the row max subtracted. A naive exp over similarities
    #    scaled by 1/0.07 overflows float32 into a plausible-looking `inf` (threat T-hg1-07).
    row_max = np.max(np.where(in_denominator, similarity, -np.inf), axis=1, keepdims=True)
    shifted = np.where(in_denominator, np.exp(similarity - row_max), 0.0)
    log_denominator = row_max[:, 0] + np.log(np.sum(shifted, axis=1))
    log_prob = similarity - log_denominator[:, None]

    # 4. Mean over the CONTRIBUTING anchors only. An anchor with no positive is dropped from the
    #    sum and from the divisor -- absence of evidence, not evidence of no separation.
    positive_counts = positives.sum(axis=1)
    contributing = positive_counts > 0
    if not bool(np.any(contributing)):
        logger.debug(
            "supcon_loss: no anchor in the pool of {} has a same-class positive; loss is 0.0", count
        )
        return 0.0

    positive_log_prob = np.where(positives, log_prob, 0.0).sum(axis=1)
    per_anchor = -positive_log_prob[contributing] / positive_counts[contributing]
    return float(per_anchor.mean())
