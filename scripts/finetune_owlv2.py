"""Fine-tune OWLv2 on the Roboflow floor-plans-500 train split (quick task 260801-8zy).

Run this ONLY in the ``export`` pixi environment, which carries ``torch`` and ``transformers``::

    # arm A (primary): heads only, frozen vision tower
    pixi run -e export finetune-owlv2 --out models/finetune/owlv2-floorplans-headonly

    # arm B (stretch): the whole exported vision path unfrozen
    pixi run -e export finetune-owlv2 --unfreeze-all --out models/finetune/owlv2-floorplans-full

Why this exists
---------------
The floor-plan domain-shift investigation established that ``owlv2-oneshot``'s ~0.01-0.11 precision
on architectural floor plans is a genuine domain-fit failure of OWLv2's **pretrained embedding
space**, and the two cheap mitigations (tiling, query augmentation) were tried and reverted.
Fine-tuning is the one lever that changes the embedding space itself. This script settles the
question with a measured number rather than an argument -- and a negative result is a valid answer.

Text-conditioned training improves an image-guided method, with zero method changes
------------------------------------------------------------------------------------
``owlv2-oneshot`` runs OWLv2 in its *image-guided* mode (encode the exemplar crop, cosine-score
every scene patch). This script trains the *text-conditioned* path. That is not a mismatch: in
``transformers``, ``Owlv2ForObjectDetection.forward`` and ``.image_guided_detection`` call the
**same** ``class_predictor`` / ``box_predictor`` over the **same** ``vision_model`` -- exactly the
three modules ``scripts/export_owlv2.py``'s ``_VisionGraph`` exports. So training the text path
improves the exported graph, and the method module is not touched at all.

``Owlv2ForObjectDetection.forward`` has **no built-in detection loss** (no ``labels`` argument, no
matcher -- the open upstream gap huggingface/transformers#33664), so the matching and the losses are
supplied here, reusing ``transformers.loss.loss_for_object_detection`` rather than hand-writing
Hungarian matching or GIoU.

Exactly which parameters reach the exported graph
-------------------------------------------------
This decides the freeze strategy, so it is written down rather than assumed. ``_VisionGraph``
(``inference/models.py``) is ``image_embedder -> box_predictor + class_predictor(query_embeds=None)``,
so the exported weights are precisely:

* ``owlv2.vision_model``      -- the ViT-B/16 tower and its ``post_layernorm``;
* ``layer_norm``              -- the class-token merge norm inside ``image_embedder``;
* ``box_head``                -- all of it;
* ``class_head.dense0``       -- the projection that produces ``class_embeds``.

Everything else is trained-but-not-exported, or not trained at all:

* ``class_head.logit_shift`` / ``logit_scale`` are trained (they sit in the ``pred_logits`` path the
  loss reads) but are **not** exported -- the method does its own cosine scoring in NumPy. They are
  kept trainable so the classification loss can absorb calibration drift there instead of distorting
  ``dense0``, which is the part that does transfer.
* ``objectness_head`` is **not** exported *and* receives no gradient (this loss has no objectness
  term), so it is left frozen. Unfreezing it would advertise a trainable head that is inert --
  worse than no knob at all.
* the text tower (``owlv2.text_model``, ``owlv2.text_projection``) is frozen in **every** arm. It is
  not exported, and leaving it trainable would let the loss fall by moving the five text queries
  toward the current image features rather than by improving the image features themselves -- a
  reduction in training loss that provably cannot transfer to the image-guided path.

The split against ``src/object_search/train/``
----------------------------------------------
All torch lives here, in one top-to-bottom readable file, because this is the artifact an ML
practitioner reads and edits. The torch-free glue that decides *what the model is trained on* --
the config schema, the class-index mapping, the COCO -> OWLv2 target conversion, the deterministic
batch order, the learning-rate schedule -- lives in :mod:`object_search.train.owlv2_targets`, where
``pixi run test`` gates it with no torch, no weights, and no GPU. ``pixi run lint`` / ``typecheck``
cover ``src/`` and ``tests/`` only, which is the existing precedent set by
``scripts/export_owlv2.py``.

Pre-processing (exact, and shared with inference)
-------------------------------------------------
Images go through the repo's own :func:`object_search.inference.owlv2.owlv2_preprocess_tensor`, not
through ``Owlv2Processor``'s image path -- so training and inference are provably one code path:
BGR->RGB, rescale ``1/255``, **pad bottom-right** to a square of side ``max(H, W)`` with grey
``0.5``, resize to ``960x960`` bilinear, CLIP mean/std. Targets are normalized over that **same
padded-square side** (see the ``owlv2_targets`` module docstring for why per-axis ``(W, H)``
normalization would be a silent, plausible-looking bug).

The second objective: ``--loss-mode contrastive`` (quick task 260805-hg1)
-------------------------------------------------------------------------
The recipe above trains a **5-way text-conditioned classification** and measured a negative result
on this domain. Its diagnosed reason is that the objective is a *proxy*: ``owlv2-oneshot`` does not
classify at inference, it ranks scene patches by **L2-normalized cosine similarity** to one
image-derived query embedding, in the ``class_embeds`` space. Nothing in a focal loss over
``pred_logits`` asks for that space to be well-separated -- ``class_head.logit_shift`` and
``logit_scale`` can absorb a great deal of the improvement, and neither is exported.

``--loss-mode contrastive`` replaces the focal term with a **supervised-contrastive loss over
``class_embeds`` itself** (Khosla et al. 2020's ``L_out``; the torch-free specification, and every
decision behind it, is in :mod:`object_search.train.supcon`). It asks for exactly the inference-time
property: same-class instances cosine-close, different-class instances and **background patches**
cosine-far. Background patches are denominator-only negatives and are load-bearing rather than a
refinement -- the measured failure is precision 0.01-0.11 at high recall, i.e. background scoring
too high, which an anchor-only contrastive term cannot touch (D-hg1-03).

Two things are deliberately unchanged in every mode: ``loss_bbox``/``loss_giou`` stay in the total
(the box head still has to be trained for the exported graph to produce usable boxes, and a
contrastive objective supervises no geometry at all), and ``focal`` remains the **default**, with
its per-epoch numbers asserted identical to a fixture captured before any of this existed.

The contrastive pool is the EFFECTIVE batch, and the pooled anchors are diagnosed
---------------------------------------------------------------------------------
Per D-hg1-04 the SupCon term is computed **once per optimizer step** over the anchors and background
rows of all ``--grad-accum`` micro-batches, not once per micro-batch: at ``--batch-size 2`` a single
micro-batch holds ~1.8 stairs boxes, and an anchor with no same-class positive contributes nothing,
so a per-micro-batch pool would quietly reduce a five-class objective to a door/window one. The cost
is stated rather than hidden -- in contrastive/both mode the backward is deferred to the
accumulation boundary, so ``grad_accum`` micro-batch graphs are retained (cheap on the primary
``headonly`` arm, where the ViT runs under ``no_grad``; expensive with an unfrozen backbone, which
is one more reason the primary contrastive run is ``headonly``).

Per D-hg1-06 every contrastive run also records ``val_cos_gap`` -- mean same-class, different-class
and anchor-to-background cosine over the val anchors -- **including an epoch-0 entry measured before
the first optimizer step**. Without that pre-training reference the gap has no meaning inside the
run, and the run could not distinguish "the loss moved the property the method uses but F1 did not
follow" from "the loss never moved it", which is precisely the question 260801-8zy could not answer.
The epoch-0 record is a no-gradient pass over both splits, so its ``train_*`` columns are the
pretrained model's losses rather than an epoch of training.

Reproducibility
---------------
Same seed => identical per-epoch losses, which the task's verify step asserts by diffing two runs'
``train_log.json``. One ``--seed`` drives ``torch.manual_seed``, the ``np.random.default_rng`` epoch
shuffler, and the background-negative sampler; the epoch record holds numbers only (never a duration
or a timestamp), so the log files of two same-seed runs compare equal. Validation draws its
background patches from a generator re-seeded at the start of **every** evaluation, so an
epoch-to-epoch change in ``val_cos_gap`` is the model moving and never the sample moving.

The third objective: ``--supcon-crop-context`` (quick task 260808-dla)
------------------------------------------------------------------------
``--loss-mode contrastive`` (above) trains SupCon over **scene-context** forward passes only. A
follow-up diagnostic (a second vast.ai instance, 260805-hg1's disposition) found the mechanism
behind its sharp negative result: ``owlv2-oneshot``'s inference-time self-similarity calibration
depends on a **crop**-context query embedding (the exemplar crop encoded alone) agreeing, in
cosine, with the **scene**-context embedding of that same region (``self_score``) -- and plain
scene-to-scene SupCon never touches the crop-context forward pass at all. For the contrastive
checkpoint, ``self_score`` went from +0.71 (pretrained) to -0.297, flipping the calibration
threshold negative and retaining ~86% of all scene patches instead of ~25-30%.

``--supcon-crop-context`` adds a crop-encoded anchor to the SupCon pool, explicitly teaching the
model that crop-context and scene-context embeddings of the same object should be cosine-close --
the exact property calibration depends on.

**D-dla-01 -- crop-context anchors are ORDINARY same-class positives. Zero changes to
``supcon_loss``/``supcon_loss_torch``'s math.** The crop-context embedding for a training image's
picked ground-truth box is appended to ``ContrastiveRows.anchors``/``.labels`` alongside the
existing scene-context matched anchors. The existing label-based positive/negative machinery
already pulls same-labeled rows together and pushes different-labeled/background rows apart --
exactly "pulled toward its own scene-context ground-truth-box patch and other same-class scene
patches, pushed away from different-class and background scene patches." No special-casing.

**D-dla-02 -- ONE crop-context anchor per training image per micro-batch pass, not one per
ground-truth box.** The box is chosen by ``rng.integers(0, n_boxes)`` from the SAME seeded
generator already threaded through ``_contrastive_rows``, consumed AFTER background sampling so a
``--supcon-crop-context``-disabled run's rng stream (and therefore its background samples) is
byte-identical to before this task. Floor plans average ~20 boxes/image; a crop-context forward is
a full independent 960x960 ViT pass (unlike scene anchors, which piggyback for free on the one
scene forward already computed), so one crop per box would roughly 10-20x the model's forward-pass
compute per step. One crop per image roughly DOUBLES it instead, and pooling across
``--grad-accum`` micro-batches (D-hg1-04) plus multiple epochs of re-sampling gives broad box/class
coverage across the run without the 10-20x cost.

**D-dla-03 -- crop-context supervision is OPT-IN, layered on ``--loss-mode contrastive``/``both``,
never changing their default meaning.** ``supcon_crop_context: bool = False`` and
``--supcon-crop-context``. The already-committed ``contrastive`` numbers and artifacts must remain
reproducible from the current script state -- the same behavior-preservation discipline
``--loss-mode focal`` already gets (D-hg1-05).

**D-dla-04 -- the crop query-patch selection reuses ``owlv2_oneshot.py``'s OWN selection logic via
a shared, refactored function, not a training-side reimplementation.**
``select_query_patch_index(class_embeds, boxes_cxcywh, iou_frac) -> int`` is extracted from
``select_query_embedding`` (which becomes a thin wrapper over it); training imports and calls the
SAME function. This is the strongest available guarantee against the crop-preprocessing/selection-
fidelity risk this task exists to close: literal code reuse, not "written to be equivalent."
Training also reuses ``owlv2_preprocess_tensor`` and ``boxes_to_pixels`` for the same reason.

**D-dla-05 -- ``supcon_query_iou_frac`` (default 0.8) is a separate ``FinetuneConfig`` field
mirroring ``Owlv2OneshotConfig.query_iou_frac``'s default, not a hard-coded constant.** So the
training-time selection threshold is visible in ``train_log.json``'s logged config and
independently sweepable -- but its default is pinned EQUAL to the inference config's default,
because training must select the query patch the way inference does.

**D-dla-06 -- a new torch-free diagnostic, ``crop_scene_agreement``
(:mod:`object_search.train.supcon`), measures the property this fix targets DIRECTLY and
INSTANCE-LEVEL.** For each instance where a crop-context anchor was built, its cosine similarity
against the SPECIFIC scene-context patch the Hungarian matcher assigned to that SAME ground-truth
box (not a class average) -- the same instance-level pairing ``self_score`` measures at inference.
Logged per-epoch on val ONLY (mirroring ``val_cos_gap``/D-hg1-06), including an epoch-0
pre-training reference, as ``val_crop_scene_agreement`` in ``train_log.json``'s ``epochs`` array,
entirely ABSENT (never ``null``) when ``supcon_crop_context`` is ``False`` or the mode is
``focal`` -- preserving both existing modes' ``epochs`` array shape exactly (verified by the
preflight/postflight fixture diff in ``.planning/quick/260808-dla-.../preflight-*-train-log.json``).
The pooling granularity mirrors ``val_cos_gap``'s: measured over the WHOLE val split rather than
per ``grad_accum`` pool, because it is a summary statistic reported once per epoch, not a training
signal computed at the accumulation boundary.

A degenerate ground-truth box never crashes training: ``_crop_context_rows`` retries a different
box (up to 3 attempts) when ``boxes_to_pixels`` rejects one after pixel rounding, then skips that
image's crop-context anchor for the step and logs at DEBUG -- never an exception, never a NaN.

Crop-margin + rotation/mirror augmentation (quick task 260808-w8c)
-------------------------------------------------------------------
Two further, independently-motivated levers on the ``contrastive-crop`` recipe above, sequenced so
the cheap one (a local, no-GPU inference sweep against the already-exported checkpoint -- see
``docs/reports/owlv2-floorplans-finetune.md``'s fourth experiment) is measured before any GPU money
funds the second.

**D-w8c-01 -- crop context-margin is ONE shared, model-free function,
``expand_box_with_margin`` (``search/owlv2_oneshot.py``), used by BOTH the inference-time query
crop and (opt-in, via ``supcon_crop_margin_frac``) this training-time crop-context anchor.** Not
two independent implementations -- the same fidelity guarantee D-dla-04 gave query-patch
selection.

**D-w8c-02 -- the margin is applied to the crop-context anchor's ALREADY-VALIDATED pixel box, in
``_crop_context_rows``, independent of the augmentation flag.** Default 0.0 leaves the anchor
exactly as tight as 260808-dla measured it.

**D-w8c-06 -- ``--supcon-crop-augment`` adds exactly ONE additional same-class SupCon positive per
training image per micro-batch pass -- not a rotation bank.** A rotated (90/180/270) or mirrored
(horizontal/vertical) view of the SAME crop, SAME ground-truth box, SAME label as the existing
crop-context anchor (D-dla-01's "ordinary positive" rule, zero changes to ``supcon_loss``'s math).
Ignored (a no-op) when ``supcon_crop_context`` is off.

**D-w8c-07 -- the augmentation choice is drawn from the SAME seeded ``rng`` used for crop-context
box selection, but ONLY AFTER the per-image box-selection loop has consumed its draws for the
whole micro-batch.** A ``--supcon-crop-augment``-off run's ``rng`` stream -- and therefore its
crop-context box selection and background sampling -- is unaffected by this task's presence in the
file, so ``--supcon-crop-context`` alone continues to reproduce 260808-dla's already-committed
numbers byte-for-byte (verified by the preflight/postflight fixture diff in
``.planning/quick/260808-w8c-.../preflight-contrastive-crop-train-log.json``).

**D-w8c-08 -- augmentation is applied to the RAW crop pixels (before ``owlv2_preprocess_tensor``),
on the SAME already-validated pixel box the base crop-context anchor uses, and is batched into the
SAME single ``_forward_batch`` call as the base crop anchors (base rows first, augmented rows
appended after) rather than a second forward pass.** No new degenerate-box exposure; the extra
compute is one crop-sized forward per image, not two.

**D-w8c-09 -- the augmented anchor does NOT feed ``crop_scene_agreement``'s diagnostic pool.**
D-dla-06's diagnostic stays exactly the property it was defined to measure -- the CANONICAL crop
vs. its matched scene patch -- so 260808-dla's already-reported numbers and this task's own
crop-context-alone regression check remain comparable. The augmented view is purely an additional
SupCon training signal.

This is a DIFFERENT mechanism from an earlier, already-reverted mitigation described in
``docs/reports/owlv2-floorplans-finetune.md``'s "Why this was tried" section: "rotation/mirror
query-embedding augmentation," which mutated the INFERENCE-time query embedding itself and zeroed
a near-symmetric window's only true positive. D-w8c-06 augments TRAINING-time SupCon positives,
never the shipped inference query -- the two must not be conflated.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt
import torch
from loguru import logger
from transformers import Owlv2ForObjectDetection, Owlv2Processor
from transformers.loss.loss_for_object_detection import (
    HungarianMatcher,
    ImageLoss,
    sigmoid_focal_loss,
)

# Make the repo's `src/` importable when this script is run directly (the same bootstrap
# scripts/export_owlv2.py uses).
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from object_search.inference.owlv2 import owlv2_preprocess_tensor  # noqa: E402
from object_search.schemas import BBox  # noqa: E402
from object_search.search.owlv2_oneshot import (  # noqa: E402
    boxes_to_pixels,
    expand_box_with_margin,
    select_query_patch_index,
)
from object_search.train.owlv2_targets import (  # noqa: E402
    FLOORPLAN_CLASSES,
    OWLV2_NUM_PATCHES,
    FinetuneConfig,
    ImageTargets,
    coco_to_owlv2_targets,
    cosine_warmup_factor,
    deterministic_batches,
    warmup_steps_for,
)
from object_search.train.supcon import (  # noqa: E402
    cosine_gap_report,
    crop_scene_agreement,
    patch_grid_size,
    sample_background_indices,
    supcon_loss,
)

_BASE_MODEL = "google/owlv2-base-patch16-ensemble"
_DEFAULT_TRAIN_COCO = "datasets/_incoming/floorplans/train/_annotations.coco.json"
_DEFAULT_VAL_COCO = "datasets/_incoming/floorplans/valid/_annotations.coco.json"
_DEFAULT_OUT = "models/finetune/owlv2-floorplans-headonly"

# The loss terms a `focal` run logs -- EXACTLY the three it has always logged. `contrastive` and
# `both` append `loss_supcon` (see `_loss_keys_for`), so a focal run's `epochs` array keeps its
# current shape and stays byte-comparable with the three already-measured arms.
_LOSS_KEYS = ("loss_ce", "loss_bbox", "loss_giou")
_SUPCON_KEY = "loss_supcon"

# Guards the L2 normalization of a zero embedding, matching `object_search.train.supcon._EPS` and
# `search/owlv2_oneshot._l2_normalize`: a zero vector stays zero rather than becoming NaN.
_SUPCON_EPS = 1e-12


def _loss_keys_for(config: FinetuneConfig) -> tuple[str, ...]:
    """The loss terms this run logs. Mode-dependent, so `focal` keeps exactly its historic keys."""
    if config.loss_mode == "focal":
        return _LOSS_KEYS
    return (*_LOSS_KEYS, _SUPCON_KEY)


# ---------------------------------------------------------------- 1. matching and the losses


class Owlv2HungarianMatcher(HungarianMatcher):
    """Hungarian matching whose classification cost is a **sigmoid**, matching the focal objective.

    The stock matcher builds its class cost from ``logits.softmax(-1)`` because DETR's head is a
    softmax over ``num_classes + 1`` with a no-object column. OWLv2's ``logits`` are
    ``[batch, num_patches, num_text_queries]`` with **no** background column and are trained here
    with sigmoid focal loss, so a softmax cost would make the matching disagree with the loss it
    feeds -- the matcher would rank a patch by its probability *relative to the other four classes*
    while the loss scores each class independently.

    The alternative considered and rejected: keep the stock softmax cost (zero code, and the
    ranking is often similar). It was not chosen because a matcher that optimizes a different
    quantity from the loss is precisely the kind of silent inconsistency that produces a
    plausible-looking training curve and a worse model. ``-sigmoid`` is the Deformable-DETR
    convention.
    """

    @torch.no_grad()
    def forward(self, outputs, targets):
        batch_size, num_queries = outputs["logits"].shape[:2]

        # [batch * num_patches, num_text_queries] -- sigmoid, NOT softmax (see the docstring).
        out_prob = outputs["logits"].flatten(0, 1).sigmoid()
        out_bbox = outputs["pred_boxes"].flatten(0, 1)

        target_ids = torch.cat([t["class_labels"] for t in targets])
        target_bbox = torch.cat([t["boxes"] for t in targets])

        class_cost = -out_prob[:, target_ids]
        bbox_cost = torch.cdist(out_bbox, target_bbox, p=1)
        giou_cost = -_generalized_box_iou(_cxcywh_to_xyxy(out_bbox), _cxcywh_to_xyxy(target_bbox))

        cost_matrix = self.bbox_cost * bbox_cost + self.class_cost * class_cost
        cost_matrix = cost_matrix + self.giou_cost * giou_cost
        cost_matrix = cost_matrix.view(batch_size, num_queries, -1).cpu()

        from scipy.optimize import linear_sum_assignment

        sizes = [len(t["boxes"]) for t in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(cost_matrix.split(sizes, -1))]
        return [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices
        ]


class Owlv2ImageLoss(ImageLoss):
    """``ImageLoss`` with ``loss_labels`` replaced by sigmoid focal loss over the per-query logits.

    The stock ``loss_labels`` (and ``loss_cardinality``) are DETR-shaped and **cannot** be used
    here: both assume a softmax head with an extra no-object logit column -- ``target_classes`` is
    filled with ``self.num_classes`` as the background index, and ``empty_weight`` carries
    ``num_classes + 1`` entries. OWLv2's ``logits`` have no background column, so the stock loss
    would index a class that does not exist. This override is the one place the stock module must
    be replaced; ``loss_boxes`` (L1 + GIoU) and ``_get_source_permutation_idx`` are inherited
    unchanged, and ``"cardinality"`` is simply never requested.
    """

    def __init__(self, matcher, num_classes, losses, *, focal_alpha, focal_gamma):
        # eos_coef only sizes the unused `empty_weight` buffer; nothing here reads it.
        super().__init__(matcher=matcher, num_classes=num_classes, eos_coef=0.1, losses=losses)
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

    def loss_labels(self, outputs, targets, indices, num_boxes):
        source_logits = outputs["logits"]  # [batch, num_patches, num_text_queries]

        # One-hot over the MATCHED (patch, class) pairs only; every unmatched patch is all-zero,
        # i.e. "no class fires here" -- which is how a sigmoid head expresses background.
        batch_idx, source_idx = self._get_source_permutation_idx(indices)
        target_classes = torch.cat(
            [t["class_labels"][j] for t, (_, j) in zip(targets, indices, strict=True)]
        )
        target = torch.zeros_like(source_logits)
        target[batch_idx, source_idx, target_classes] = 1.0

        # sigmoid_focal_loss reduces as mean-over-patches then sum; multiplying back by the patch
        # count restores a per-patch sum (the Deformable-DETR scaling), so the classification term
        # keeps a magnitude comparable to the box terms instead of vanishing against 3600 patches.
        loss_ce = (
            sigmoid_focal_loss(
                source_logits,
                target,
                num_boxes,
                alpha=self.focal_alpha,
                gamma=self.focal_gamma,
            )
            * source_logits.shape[1]
        )
        return {"loss_ce": loss_ce}


def _cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """``(cx, cy, w, h)`` -> ``(x1, y1, x2, y2)``, the corner form the GIoU helper expects."""
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


def _generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Pairwise GIoU, delegated to transformers' implementation (never hand-written here)."""
    from transformers.loss.loss_for_object_detection import generalized_box_iou

    return generalized_box_iou(boxes1, boxes2)


def supcon_loss_torch(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
    negative_only_count: int = 0,
) -> torch.Tensor:
    """Differentiable mirror of :func:`object_search.train.supcon.supcon_loss` (the ``L_out`` form).

    The NumPy function in ``src/`` is the **specification** -- it is the one ``pixi run test`` can
    gate, because torch lives only in the ``export`` pixi environment and ``pytest`` is not
    installed there. This mirror exists so the loss is differentiable; ``finetune-owlv2
    --self-check`` asserts the two agree to ``<1e-6`` on the same inputs and that this one leaves a
    finite, non-zero gradient. Keep the two in step: the numbered steps below match that module's.

    ``negative_only_count`` is expressed as a **trailing row count** rather than a bool mask because
    the caller builds the pool by concatenation -- matched anchors first, then background patches --
    so "the last k rows are background" is the shape the data already has.

    Args:
        embeddings: ``(n, d)`` pooled embeddings, anchors first then background.
        labels: ``(n,)`` int64 class labels. The trailing background entries are ignored.
        temperature: SupCon ``tau``.
        negative_only_count: How many trailing rows are denominator-only (D-hg1-03): they appear in
            every anchor's denominator but are never anchors and never positives.

    Returns:
        A scalar tensor. Exactly zero (and still attached to the graph, so ``.backward()`` is
        always safe) when no anchor has a same-class positive.
    """
    count = int(embeddings.shape[0])
    if count < 2:
        return embeddings.sum() * 0.0

    # 1. L2-normalize, so the matmul IS cosine -- the space owlv2-oneshot scores in.
    normalized = embeddings / embeddings.norm(dim=1, keepdim=True).clamp_min(_SUPCON_EPS)
    similarity = (normalized @ normalized.t()) / temperature

    # 2. A(i) is every other row INCLUDING the background ones; P(i) is the same-class rows,
    #    excluding self and excluding anything denominator-only on either side.
    self_pair = torch.eye(count, dtype=torch.bool, device=embeddings.device)
    in_denominator = ~self_pair
    denominator_only = torch.zeros(count, dtype=torch.bool, device=embeddings.device)
    if negative_only_count > 0:
        denominator_only[count - negative_only_count :] = True
    positives = (
        (labels[:, None] == labels[None, :])
        & ~self_pair
        & ~denominator_only[None, :]
        & ~denominator_only[:, None]
    )

    # 3. log sum_{a in A(i)} exp(sim), row max subtracted. A naive exp over similarities scaled by
    #    1/0.07 overflows float32 into a plausible-looking `inf` (threat T-hg1-07).
    row_max = similarity.masked_fill(~in_denominator, float("-inf")).max(dim=1, keepdim=True).values
    shifted = torch.where(
        in_denominator, (similarity - row_max).exp(), torch.zeros_like(similarity)
    )
    log_denominator = row_max[:, 0] + shifted.sum(dim=1).log()
    log_prob = similarity - log_denominator[:, None]

    # 4. Mean over the CONTRIBUTING anchors only -- an anchor with no positive is dropped from the
    #    sum AND from the divisor, never averaged in as a fabricated zero.
    positive_counts = positives.sum(dim=1)
    contributing = positive_counts > 0
    if not bool(contributing.any()):
        return embeddings.sum() * 0.0

    positive_log_prob = torch.where(positives, log_prob, torch.zeros_like(log_prob)).sum(dim=1)
    per_anchor = -positive_log_prob[contributing] / positive_counts[contributing]
    return per_anchor.mean()


# ------------------------------------------------------------------- 2. data -> model tensors


def _load_pixel_values(image_path: Path) -> torch.Tensor:
    """Read one scene PNG and run the repo's OWLv2 preprocessing -> ``[1, 3, 960, 960]`` float32.

    Deliberately the SAME function the inferencer uses, so a preprocessing change can never drift
    between training and inference (the most common source of silently-wrong ONNX pipelines).
    """
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"could not read {image_path}")
    tensor, _side = owlv2_preprocess_tensor(image)
    return torch.from_numpy(tensor)


def _read_crop_pixels(image_path: Path, box: BBox) -> npt.NDArray[np.uint8]:
    """Read one scene PNG and slice it to ``box`` in RAW pixel space -- BGR ``uint8``, un-preprocessed.

    Mirrors ``owlv2-oneshot``'s OWN inference-time exemplar-crop encode (``search/owlv2_oneshot.py``:
    ``image[crop_box.y:y2, crop_box.x:x2]``) line for line (D-dla-04, verified-fact 2). Split out
    from the old ``_load_crop_pixel_values`` so the raw-pixel step is reusable by both the base and
    the (opt-in) augmented crop-context anchor (D-w8c-08).
    """
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"could not read {image_path}")
    return np.ascontiguousarray(image[box.y : box.y2, box.x : box.x2], dtype=np.uint8)


def _load_crop_pixel_values(image_path: Path, box: BBox) -> torch.Tensor:
    """Read one scene PNG, crop it to ``box`` in RAW pixel space, and preprocess -> ``[1,3,960,960]``.

    Mirrors :func:`_load_pixel_values` exactly, but on the cropped sub-array rather than the whole
    image: the crop gets its own independent pad-to-square-then-resize based on the CROP's own
    aspect ratio, never the scene's.
    """
    tensor, _side = owlv2_preprocess_tensor(_read_crop_pixels(image_path, box))
    return torch.from_numpy(tensor)


_AUGMENT_CHOICES = 5


def _augment_crop_pixels(crop: npt.NDArray[np.uint8], choice: int) -> npt.NDArray[np.uint8]:
    """Rotate or mirror a raw BGR crop (D-w8c-08): 3 rotations + 2 mirrors, chosen by ``choice``.

    ``np.rot90``/reversed-slice mirrors return NON-contiguous views, and ``owlv2_preprocess_tensor``
    calls ``cv2.cvtColor`` first, which requires a contiguous array (verified_fact 5) -- always
    ``np.ascontiguousarray`` the result before preprocessing.
    """
    if choice == 0:
        transformed = np.rot90(crop, k=1)
    elif choice == 1:
        transformed = np.rot90(crop, k=2)
    elif choice == 2:
        transformed = np.rot90(crop, k=3)
    elif choice == 3:
        transformed = crop[:, ::-1, :]
    elif choice == 4:
        transformed = crop[::-1, :, :]
    else:
        raise ValueError(f"choice must be in [0, {_AUGMENT_CHOICES}), got {choice}")
    return np.ascontiguousarray(transformed, dtype=np.uint8)


def _load_augmented_crop_pixel_values(image_path: Path, box: BBox, choice: int) -> torch.Tensor:
    """The SAME crop as :func:`_load_crop_pixel_values`, rotated/mirrored per ``choice``, preprocessed."""
    crop = _augment_crop_pixels(_read_crop_pixels(image_path, box), choice)
    tensor, _side = owlv2_preprocess_tensor(crop)
    return torch.from_numpy(tensor)


def _pick_crop_box_index(target: ImageTargets, rng: np.random.Generator) -> int:
    """Pick ONE ground-truth box index at random for this image's crop-context anchor (D-dla-02).

    One per image per micro-batch pass, not one per box -- see the module docstring's
    "crop-context extension" section for the cost tradeoff this pins.
    """
    return int(rng.integers(0, target.boxes.shape[0]))


def _batch_targets(
    batch: list[ImageTargets], device: torch.device
) -> list[dict[str, torch.Tensor]]:
    """The per-image ``{class_labels, boxes}`` dicts ``ImageLoss``/the matcher expect."""
    return [
        {
            "class_labels": torch.from_numpy(target.class_labels).to(device),
            "boxes": torch.from_numpy(target.boxes).to(device),
        }
        for target in batch
    ]


def _tokenize_class_names(processor: Owlv2Processor, device: torch.device):
    """Tokenize the five class names ONCE; the same ids are reused for every batch.

    All five classes are trained even though only ``door`` and ``window`` are evaluated: more
    discriminative classes should sharpen door-vs-not-door separation rather than dilute it. That
    is a hypothesis, and the report confirms or refutes it against the measured numbers.
    """
    encoded = processor(
        text=[list(FLOORPLAN_CLASSES)], return_tensors="pt", padding=True, truncation=True
    )
    return encoded["input_ids"].to(device), encoded["attention_mask"].to(device)


def _load_split(
    coco_path: Path, limit_images: int | None, label: str
) -> tuple[list[ImageTargets], Path]:
    """Read one COCO split into targets (optionally truncated), plus its image directory."""
    if not coco_path.is_file():
        raise SystemExit(
            f"no {label} COCO annotations at {coco_path}. The floor plans are a MANUAL dataset: "
            f"drop the Roboflow export at datasets/_incoming/floorplans/{{train,valid,test}} first."
        )
    targets = coco_to_owlv2_targets(json.loads(coco_path.read_text()))
    if limit_images is not None:
        # The list is sorted by file_name, so the truncation picks the SAME images every run.
        targets = targets[:limit_images]
    if not targets:
        raise SystemExit(f"{coco_path} produced no usable targets")
    logger.info(f"{label}: {len(targets)} image(s) from {coco_path}")
    return targets, coco_path.parent


# ------------------------------------------------------- 3. the freeze arms and the param groups


@dataclass(frozen=True)
class FreezePlan:
    """Which parameters train, at which learning rate, and under which arm name."""

    arm: str
    head_params: list[torch.nn.Parameter]
    backbone_params: list[torch.nn.Parameter]
    backbone_frozen: bool
    trainable_params: int
    total_params: int

    @property
    def all_trainable(self) -> list[torch.nn.Parameter]:
        return [*self.head_params, *self.backbone_params]


def _apply_freeze_strategy(model: Owlv2ForObjectDetection, config: FinetuneConfig) -> FreezePlan:
    """Freeze everything, then unfreeze exactly the arm's parameters. Returns the two LR groups.

    Three arms, all of which keep the text tower frozen (see the module docstring for why that is
    a correctness requirement and not a thrift measure):

    * ``headonly``  (default)          -- ``box_head`` + ``class_head`` only;
    * ``last{N}``   (--unfreeze-last-n)-- plus the last N vision-encoder blocks at ``lr_backbone``;
    * ``full``      (--unfreeze-all)   -- plus the whole vision tower and the class-token
      ``layer_norm``, still at ``lr_backbone``.

    ``--unfreeze-all`` is deliberately "the whole **exported** vision path" rather than a literal
    ``requires_grad_(True)`` over every parameter: the text tower and ``objectness_head`` are
    excluded because neither reaches the ONNX graph (the head additionally receives no gradient
    from this loss). Optimizing weights that cannot transfer only buys a prettier training curve.
    """
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    # Always trainable: the two heads that ARE in the exported graph. `class_head.dense0` is the
    # projection the method's cosine scoring runs on; `class_head.logit_shift`/`logit_scale` also
    # train (they are in the pred_logits path) but are not exported.
    head_params: list[torch.nn.Parameter] = []
    for head in (model.box_head, model.class_head):
        for parameter in head.parameters():
            parameter.requires_grad_(True)
            head_params.append(parameter)

    backbone_params: list[torch.nn.Parameter] = []
    encoder_layers = model.owlv2.vision_model.encoder.layers
    if config.unfreeze_all:
        arm = "full"
        backbone_modules = [model.owlv2.vision_model, model.layer_norm]
    elif config.unfreeze_last_n > 0:
        last_n = min(config.unfreeze_last_n, len(encoder_layers))
        arm = f"last{last_n}"
        backbone_modules = list(encoder_layers[-last_n:])
    else:
        arm = "headonly"
        backbone_modules = []

    for module in backbone_modules:
        for parameter in module.parameters():
            if parameter.requires_grad:  # already claimed by the head group; never list it twice
                continue
            parameter.requires_grad_(True)
            backbone_params.append(parameter)

    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in head_params + backbone_params)
    return FreezePlan(
        arm=arm,
        head_params=head_params,
        backbone_params=backbone_params,
        backbone_frozen=not backbone_params,
        trainable_params=trainable,
        total_params=total,
    )


# --------------------------------------------------------- 4. the forward pass, an epoch, and val


@dataclass(frozen=True)
class Runtime:
    """Everything a forward pass needs, bundled so the epoch functions stay readable."""

    model: Owlv2ForObjectDetection
    criterion: Owlv2ImageLoss
    config: FinetuneConfig
    device: torch.device
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    plan: FreezePlan
    use_bf16: bool


def _forward_batch(runtime: Runtime, pixel_values: torch.Tensor):
    """One text-conditioned forward pass -> ``(logits, pred_boxes, class_embeds)``.

    ``class_embeds`` is ``[batch, num_patches, 512]`` and is **already computed** by
    ``class_predictor`` -- it used to be bound to ``_class_embeds`` and thrown away. Keeping it is
    the entire wiring the contrastive objective needs; no extra forward computation is added. It is
    the space ``owlv2-oneshot`` scores in at inference (``search/owlv2_oneshot.py`` L2-normalizes
    it and takes a cosine against the query embedding), which is why a loss over it trains the
    property the method actually uses rather than a correlate of it.

    Two paths, and the difference is memory, not maths:

    * **backbone unfrozen** -- plain ``model(...)``, gradients flow through the ViT.
    * **backbone frozen** (the primary arm) -- ``image_text_embedder`` runs under
      ``torch.no_grad()`` and only the heads are re-implemented here, mirroring
      ``Owlv2ForObjectDetection.forward`` line for line. The ViT's per-layer activations are then
      never retained, which is most of the memory and a good part of the wall-clock of a 3600-patch
      backward pass. The next available speedup -- caching each image's frozen ``feature_map`` once
      and reusing it every epoch -- was deliberately NOT taken: it turns a readable loop into a
      cache-invalidation problem, and the GPU arms that matter (``last{N}``, ``full``) cannot use it
      anyway.

    ``forward`` expects ``input_ids`` shaped ``[batch * num_text_queries, seq]`` (it reshapes them
    internally), so the single tokenized query block is repeated once per image in the batch.
    """
    batch_size = pixel_values.shape[0]
    input_ids = runtime.input_ids.repeat(batch_size, 1)
    attention_mask = runtime.attention_mask.repeat(batch_size, 1)

    if not runtime.plan.backbone_frozen:
        outputs = runtime.model(
            input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_mask
        )
        return outputs.logits, outputs.pred_boxes, outputs.class_embeds

    with torch.no_grad():
        query_embeds, feature_map, _outputs = runtime.model.image_text_embedder(
            input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_mask
        )

    batch, grid_h, grid_w, hidden = feature_map.shape
    image_feats = feature_map.reshape(batch, grid_h * grid_w, hidden)
    num_queries = input_ids.shape[0] // batch
    query_embeds = query_embeds.reshape(batch, num_queries, query_embeds.shape[-1])
    # A padded (all-zero-first-token) query is masked out, exactly as the stock forward does.
    query_mask = input_ids.reshape(batch, num_queries, input_ids.shape[-1])[..., 0] > 0

    logits, class_embeds = runtime.model.class_predictor(image_feats, query_embeds, query_mask)
    pred_boxes = runtime.model.box_predictor(image_feats, feature_map)
    return logits, pred_boxes, class_embeds


@dataclass(frozen=True, eq=False)
class ContrastiveRows:
    """The rows one micro-batch contributes to the pooled SupCon set.

    ``eq=False`` because the fields are tensors (element-wise ``==`` has no single truth value).

    Attributes:
        anchors: ``(n_boxes, 512)`` ``class_embeds`` rows at the patch indices the Hungarian
            matcher assigned to this micro-batch's ground-truth boxes. In
            ``--supcon-crop-context`` mode this ALSO includes the crop-context anchors (D-dla-01):
            ordinary same-class positives, appended alongside the scene-matched rows.
        labels: ``(n_boxes,)`` int64 class labels, aligned with ``anchors``.
        background: ``(n_background, 512)`` rows sampled from patches whose grid-cell centre lies
            in no ground-truth box -- denominator-only negatives (D-hg1-03).
        crop_diag_crop: ``(n_crop, 512)`` crop-context embeddings, DETACHED -- diagnostic only,
            never fed to the loss (D-dla-06). ``(0, 512)`` when crop-context is off or an image's
            crop anchor was skipped this step.
        crop_diag_scene: ``(n_crop, 512)`` the SAME instances' matched scene-context embeddings,
            DETACHED and aligned row-for-row with ``crop_diag_crop`` -- the pairing
            :func:`object_search.train.supcon.crop_scene_agreement` measures (Task 2).
    """

    anchors: torch.Tensor
    labels: torch.Tensor
    background: torch.Tensor
    crop_diag_crop: torch.Tensor
    crop_diag_scene: torch.Tensor


def _crop_context_rows(
    runtime: Runtime,
    class_embeds: torch.Tensor,
    indices: list[tuple[torch.Tensor, torch.Tensor]],
    batch: list[ImageTargets],
    image_dir: Path,
    rng: np.random.Generator,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    """Build ONE crop-context SupCon anchor per image in this micro-batch (D-dla-01/02/04).

    For each image: pick one ground-truth box (:func:`_pick_crop_box_index`, consuming ``rng``
    AFTER the caller's background sampling -- see the module docstring's "crop-context extension"
    section for why that ordering keeps a flag-off run's rng stream byte-identical), convert it to
    scene PIXEL coordinates (:func:`~object_search.search.owlv2_oneshot.boxes_to_pixels`, verified
    fact 3), retrying a different box up to 3 times if it degenerates after pixel rounding, then
    skip that image's crop anchor for this step (logged at DEBUG, never an exception -- the
    must-have degenerate-box guarantee).

    All of this micro-batch's valid crops are batched into ONE extra ``_forward_batch`` call. Each
    crop's query patch is then picked by :func:`~object_search.search.owlv2_oneshot.
    select_query_patch_index` -- the SAME function ``owlv2-oneshot`` calls at inference (D-dla-04)
    -- on a DETACHED NumPy view, and the TORCH row at that index is gathered (keeping the gradient
    path into ``class_head.dense0``) and returned as an ordinary same-class positive.

    When ``config.supcon_crop_margin_frac > 0``, the validated pixel box is grown by
    :func:`~object_search.search.owlv2_oneshot.expand_box_with_margin` -- the SAME function
    ``owlv2-oneshot``'s inference crop uses (D-w8c-01) -- BEFORE cropping (D-w8c-02).

    When ``config.supcon_crop_augment`` is set, AFTER this per-image box-selection loop has
    consumed its ``rng`` draws for the whole micro-batch (D-w8c-07, so a flag-off run's rng stream
    is unaffected), one additional rotated/mirrored view of each valid crop is built
    (:func:`_augment_crop_pixels`) and batched into the SAME ``_forward_batch`` call, base rows
    first, augmented rows appended after (D-w8c-08). Each augmented row is an ordinary same-class
    positive carrying the SAME label as its base sibling (D-w8c-06) and is excluded from the
    ``crop_scene_agreement`` diagnostic pool (D-w8c-09).

    Also returns the DETACHED ``(crop, scene)`` diagnostic pair for each successfully-built BASE
    anchor, by looking up the box's matched scene patch in the ALREADY-COMPUTED Hungarian
    ``indices`` (no second matcher call, D-dla-06) -- silently skipping the diagnostic pairing
    (never the anchor itself) if the Hungarian solve did not happen to match that particular box
    this step.

    Returns:
        ``(anchor_rows, label_rows, diag_crop_rows, diag_scene_rows)`` -- four lists of tensors for
        the caller to ``torch.cat``. All empty when no image in the micro-batch produced a valid
        crop-context row.
    """
    max_retries = 3
    crop_targets: list[tuple[int, int, Path, BBox]] = []
    for image_index, target in enumerate(batch):
        if target.boxes.shape[0] == 0:  # guaranteed non-empty by coco_to_owlv2_targets; defensive
            continue
        image_path = image_dir / target.file_name
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            logger.debug("crop-context: could not read {}, skipping its crop anchor", image_path)
            continue
        orig_h, orig_w = int(image.shape[0]), int(image.shape[1])

        pixel_box: BBox | None = None
        chosen_box_idx = -1
        for _attempt in range(max_retries):
            box_idx = _pick_crop_box_index(target, rng)
            candidate = boxes_to_pixels(target.boxes[box_idx : box_idx + 1], orig_w, orig_h)[0]
            if candidate is not None:
                if runtime.config.supcon_crop_margin_frac > 0.0:
                    candidate = expand_box_with_margin(
                        candidate, runtime.config.supcon_crop_margin_frac, orig_w, orig_h
                    )
                pixel_box, chosen_box_idx = candidate, box_idx
                break
            logger.debug(
                "crop-context: box {} of {} degenerated after pixel rounding, retrying",
                box_idx,
                target.file_name,
            )
        if pixel_box is None:
            logger.debug(
                "crop-context: no valid GT box for {} after {} attempt(s); skipping its crop "
                "anchor for this step",
                target.file_name,
                max_retries,
            )
            continue
        crop_targets.append((image_index, chosen_box_idx, image_path, pixel_box))

    anchor_rows: list[torch.Tensor] = []
    label_rows: list[torch.Tensor] = []
    diag_crop_rows: list[torch.Tensor] = []
    diag_scene_rows: list[torch.Tensor] = []
    if not crop_targets:
        return anchor_rows, label_rows, diag_crop_rows, diag_scene_rows

    base_tensors = [_load_crop_pixel_values(path, box) for _, _, path, box in crop_targets]
    augment_choices: list[int] = []
    if runtime.config.supcon_crop_augment:
        augment_choices = [int(rng.integers(0, _AUGMENT_CHOICES)) for _ in crop_targets]
        augmented_tensors = [
            _load_augmented_crop_pixel_values(path, box, choice)
            for (_, _, path, box), choice in zip(crop_targets, augment_choices, strict=True)
        ]
        crop_pixel_values = torch.cat(base_tensors + augmented_tensors).to(runtime.device)
    else:
        crop_pixel_values = torch.cat(base_tensors).to(runtime.device)

    with torch.autocast(
        device_type=runtime.device.type, dtype=torch.bfloat16, enabled=runtime.use_bf16
    ):
        _crop_logits, crop_pred_boxes, crop_class_embeds = _forward_batch(
            runtime, crop_pixel_values
        )
    crop_class_embeds = crop_class_embeds.float()
    crop_pred_boxes_np = crop_pred_boxes.float().detach().cpu().numpy()

    def _select_row(row: int) -> torch.Tensor:
        selected = select_query_patch_index(
            crop_class_embeds[row].detach().cpu().numpy(),
            crop_pred_boxes_np[row],
            runtime.config.supcon_query_iou_frac,
        )
        return crop_class_embeds[row, selected]  # torch, differentiable

    n_base = len(crop_targets)
    for row, (image_index, box_idx, _path, _box) in enumerate(crop_targets):
        crop_row = _select_row(row)

        target = batch[image_index]
        anchor_rows.append(crop_row.unsqueeze(0))
        label_rows.append(
            torch.tensor(
                [int(target.class_labels[box_idx])],
                dtype=torch.int64,
                device=class_embeds.device,
            )
        )

        source_idx, target_idx = indices[image_index]
        matched = target_idx == box_idx
        if bool(matched.any()):
            scene_patch_idx = int(source_idx[matched][0].item())
            diag_crop_rows.append(crop_row.detach().unsqueeze(0))
            diag_scene_rows.append(class_embeds[image_index, scene_patch_idx].detach().unsqueeze(0))

    if augment_choices:
        for offset, (image_index, box_idx, _path, _box) in enumerate(crop_targets):
            augmented_row = _select_row(n_base + offset)
            target = batch[image_index]
            anchor_rows.append(augmented_row.unsqueeze(0))
            label_rows.append(
                torch.tensor(
                    [int(target.class_labels[box_idx])],
                    dtype=torch.int64,
                    device=class_embeds.device,
                )
            )
            # D-w8c-09: the augmented view is excluded from crop_scene_agreement's diagnostic pool.

    return anchor_rows, label_rows, diag_crop_rows, diag_scene_rows


def _contrastive_rows(
    runtime: Runtime,
    outputs: dict[str, torch.Tensor],
    class_embeds: torch.Tensor,
    targets: list[dict[str, torch.Tensor]],
    batch: list[ImageTargets],
    image_dir: Path,
    rng: np.random.Generator,
) -> ContrastiveRows:
    """Gather this micro-batch's SupCon rows: the matched anchors, plus background negatives.

    The anchor selection is the **existing** ``Owlv2HungarianMatcher`` assignment, not new matching
    logic: the anchor for a ground-truth box is by construction the same patch the box loss
    supervises, which is what makes the contrastive term and the box term agree about what a "door"
    is. The gather mirrors ``ImageLoss._get_source_permutation_idx``.

    The matcher is run once more here rather than having the criterion hand its indices back. That
    is a deliberate trade: a duplicated, deterministic, ``no_grad`` Hungarian solve (microseconds
    against a 3600-patch ViT forward) in exchange for not threading cached mutable state out of a
    ``transformers`` base class. It runs in contrastive/both mode only.

    The background rows (D-hg1-03) are drawn from the **numpy** ``ImageTargets.boxes`` rather than
    the batched tensors, because ``sample_background_indices`` is the tested torch-free
    specification and its coordinate frame is the frame those boxes are already in. The grid side is
    derived from the tensor the model actually returned, so a re-export at another resolution is
    caught here instead of silently mis-indexing every negative.

    AFTER the scene-anchor and background-negative gather (D-dla-02) -- so a
    ``supcon_crop_context=False`` run's ``rng`` stream and background samples stay byte-identical
    to before this task -- optionally gathers one crop-context anchor per image
    (:func:`_crop_context_rows`), when ``config.supcon_crop_context`` is set and the mode is not
    ``focal``.
    """
    indices = runtime.criterion.matcher(outputs, targets)
    num_patches = int(class_embeds.shape[1])
    grid = patch_grid_size(num_patches)
    if num_patches != OWLV2_NUM_PATCHES:
        logger.warning(
            "class_embeds carries {} patches ({}x{}), not the pinned {} -- background sampling "
            "follows the tensor, but check that the export operating point is what you intended",
            num_patches,
            grid,
            grid,
            OWLV2_NUM_PATCHES,
        )

    anchors: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    background: list[torch.Tensor] = []
    for image_index, (source_idx, target_idx) in enumerate(indices):
        source = source_idx.to(class_embeds.device)
        anchors.append(class_embeds[image_index, source])
        labels.append(targets[image_index]["class_labels"][target_idx.to(class_embeds.device)])

        sampled = sample_background_indices(
            batch[image_index].boxes, grid, runtime.config.supcon_background_negatives, rng
        )
        if sampled.size:
            background.append(
                class_embeds[
                    image_index,
                    torch.from_numpy(sampled).to(device=class_embeds.device, dtype=torch.int64),
                ]
            )

    empty = class_embeds.new_zeros((0, class_embeds.shape[-1]))
    diag_crop: list[torch.Tensor] = []
    diag_scene: list[torch.Tensor] = []
    if runtime.config.supcon_crop_context and runtime.config.loss_mode != "focal":
        crop_anchors, crop_labels, diag_crop, diag_scene = _crop_context_rows(
            runtime, class_embeds, indices, batch, image_dir, rng
        )
        anchors.extend(crop_anchors)
        labels.extend(crop_labels)

    return ContrastiveRows(
        anchors=torch.cat(anchors) if anchors else empty,
        labels=(
            torch.cat(labels)
            if labels
            else torch.zeros(0, dtype=torch.int64, device=class_embeds.device)
        ),
        background=torch.cat(background) if background else empty,
        crop_diag_crop=torch.cat(diag_crop) if diag_crop else empty,
        crop_diag_scene=torch.cat(diag_scene) if diag_scene else empty,
    )


def _pooled_supcon(rows: Sequence[ContrastiveRows], config: FinetuneConfig) -> torch.Tensor | None:
    """SupCon over a pooled set of micro-batches -> a scalar tensor, or ``None`` if there is no pool.

    Taking a sequence rather than one ``ContrastiveRows`` is what makes D-hg1-04 a one-line caller
    change: pass one element for a per-micro-batch pool, or ``grad_accum`` elements for the
    effective batch. Background rows are concatenated AFTER every anchor, which is the trailing-row
    convention :func:`supcon_loss_torch` expects.
    """
    if not rows:
        return None
    anchors = torch.cat([row.anchors for row in rows])
    if anchors.shape[0] == 0:
        return None

    background = torch.cat([row.background for row in rows])
    labels = torch.cat([row.labels for row in rows])
    pooled = torch.cat([anchors, background])
    # A background row's label is never read (the denominator-only mask excludes it on both sides);
    # -1 is used so an accidental read would be loud rather than silently valid.
    pooled_labels = torch.cat([labels, labels.new_full((background.shape[0],), -1)])

    return supcon_loss_torch(
        pooled, pooled_labels, config.supcon_temperature, int(background.shape[0])
    )


def _batch_loss(
    runtime: Runtime, batch: list[ImageTargets], image_dir: Path, rng: np.random.Generator
) -> tuple[dict[str, torch.Tensor], ContrastiveRows | None]:
    """Forward one micro-batch (bf16 autocast on CUDA) -> the loss terms in fp32, plus SupCon rows.

    The autocast region covers the forward only. Hungarian matching and the loss reductions run in
    fp32 on purpose: bf16 has ~3 decimal digits of mantissa, which is enough for a ViT's activations
    and nowhere near enough for a cost matrix whose argmin decides which patch supervises which box.

    The returned ``ContrastiveRows`` is ``None`` in ``focal`` mode, so that arm does no extra work
    at all and stays byte-comparable with the three already-measured arms -- including its
    consumption of ``rng``, which focal mode never touches here.
    """
    pixel_values = torch.cat(
        [_load_pixel_values(image_dir / target.file_name) for target in batch]
    ).to(runtime.device)

    with torch.autocast(
        device_type=runtime.device.type, dtype=torch.bfloat16, enabled=runtime.use_bf16
    ):
        logits, pred_boxes, class_embeds = _forward_batch(runtime, pixel_values)

    outputs = {"logits": logits.float(), "pred_boxes": pred_boxes.float()}
    targets = _batch_targets(batch, runtime.device)
    loss_dict = runtime.criterion(outputs, targets)

    rows = None
    if runtime.config.loss_mode != "focal":
        rows = _contrastive_rows(
            runtime, outputs, class_embeds.float(), targets, batch, image_dir, rng
        )
    return loss_dict, rows


def _total_loss(loss_dict: dict[str, torch.Tensor], config: FinetuneConfig) -> torch.Tensor:
    """The per-micro-batch terms: ``w_class * loss_ce`` (focal/both) ``+ w_bbox``/``w_giou`` boxes.

    ``loss_bbox`` and ``loss_giou`` are in EVERY mode: ``box_head`` still has to be trained for the
    exported graph to produce usable boxes, and a contrastive objective supervises no geometry at
    all. Only the classification-side term is swapped (D-hg1-05).

    The contrastive term is deliberately **not** added here. It is computed once per optimizer step
    over the pooled effective batch (D-hg1-04), which is a different granularity from this
    function's, so folding it in would misrepresent where it is applied. The caller adds
    ``w_contrast * loss_supcon`` at the accumulation boundary.
    """
    total = config.w_bbox * loss_dict["loss_bbox"] + config.w_giou * loss_dict["loss_giou"]
    if config.loss_mode in ("focal", "both"):
        total = total + config.w_class * loss_dict["loss_ce"]
    return total


def _mean_losses(sums: dict[str, float], batches: int) -> dict[str, float]:
    """Per-micro-batch means of the total and of each term -- the numbers the log records."""
    divisor = float(max(batches, 1))
    return {key: value / divisor for key, value in sums.items()}


def _epoch_means(
    sums: dict[str, float],
    batches: int,
    supcon_sum: float,
    supcon_pools: int,
    config: FinetuneConfig,
) -> dict[str, float]:
    """Fold the two different granularities into one set of epoch numbers.

    ``loss_ce``/``loss_bbox``/``loss_giou`` are per-**micro-batch** quantities; the contrastive term
    is a per-**pool** one (D-hg1-04: once per optimizer step over the effective batch), so the two
    cannot share a divisor. Each is averaged over its own count and the reported total is
    ``mean(box terms) + w_contrast * mean(supcon)`` -- which is exactly what the accumulated
    gradient represents, since averaging a per-micro-batch ``w_contrast * supcon_i`` gives the same
    thing. In ``focal`` mode this returns ``_mean_losses`` untouched.
    """
    means = _mean_losses(sums, batches)
    if config.loss_mode == "focal":
        return means
    supcon = supcon_sum / float(max(supcon_pools, 1))
    means[_SUPCON_KEY] = supcon
    means["loss"] = means["loss"] + config.w_contrast * supcon
    return means


def _backward_accumulated(
    runtime: Runtime,
    micro_totals: list[torch.Tensor],
    rows: Sequence[ContrastiveRows],
) -> float | None:
    """One optimizer step's deferred backward, in contrastive/both mode. Returns the pooled SupCon.

    The whole point of D-hg1-04 lives here: the micro-batch box/focal totals were kept as live
    graphs rather than backwarded on the spot, so the contrastive term can be computed **once**
    over the anchors and background rows of the entire effective batch and the sum backwarded in
    one pass. Division by ``grad_accum`` happens in the same place the focal path does it, so
    raising ``--grad-accum`` still does not silently raise the effective learning rate; the
    contrastive term is added undivided because it is already a single per-step quantity.

    Returns ``None`` when the pool had no anchor with a same-class positive -- the loss then
    genuinely has no contrastive component to report for that step, and averaging in a fabricated
    ``0.0`` would make the reported curve depend on batch composition.
    """
    total = torch.stack(micro_totals).sum() / runtime.config.grad_accum
    supcon = _pooled_supcon(rows, runtime.config)
    if supcon is not None:
        total = total + runtime.config.w_contrast * supcon
    total.backward()
    return None if supcon is None else float(supcon.item())


def _train_one_epoch(
    runtime: Runtime,
    targets: list[ImageTargets],
    image_dir: Path,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    rng: np.random.Generator,
    steps_taken: int,
) -> tuple[dict[str, float], int]:
    """One shuffled pass over the training split. Returns ``(mean losses, cumulative step count)``.

    Gradient accumulation divides each micro-batch loss by ``grad_accum`` before ``backward()``, so
    the accumulated gradient is the mean over the effective batch rather than its sum -- otherwise
    raising ``--grad-accum`` would silently also raise the effective learning rate.

    Two backward strategies, and which one runs is decided by ``--loss-mode``:

    * ``focal`` -- backward each micro-batch immediately, exactly as before. Nothing about this path
      changed when the contrastive objective was added, which is what the preflight-fixture
      assertion in the task's verify step checks.
    * ``contrastive``/``both`` -- hold each micro-batch's box/focal total as a live graph, pool the
      SupCon rows, and backward once at the accumulation boundary (D-hg1-04 and
      :func:`_backward_accumulated`). ``rng`` also feeds the background sampler in these modes, so
      the same seed draws the same negatives.
    """
    config = runtime.config
    deferred = config.loss_mode != "focal"
    runtime.model.train()
    if runtime.plan.backbone_frozen:
        # A frozen module in train() mode is a no-op for OWLv2 (dropout is 0.0 and LayerNorm keeps
        # no running stats), but eval() states the intent and is robust to a config that changes.
        runtime.model.owlv2.eval()
    runtime.model.owlv2.text_model.eval()  # frozen in every arm

    sums = dict.fromkeys(("loss", *_LOSS_KEYS), 0.0)
    batches = 0
    pending = 0
    pending_totals: list[torch.Tensor] = []
    pending_rows: list[ContrastiveRows] = []
    supcon_sum = 0.0
    supcon_pools = 0
    optimizer.zero_grad(set_to_none=True)

    for indices in deterministic_batches(len(targets), config.batch_size, rng):
        batch = [targets[index] for index in indices]
        loss_dict, rows = _batch_loss(runtime, batch, image_dir, rng)
        loss = _total_loss(loss_dict, config)

        if deferred:
            pending_totals.append(loss)  # backwarded at the accumulation boundary, not here
            if rows is not None:
                pending_rows.append(rows)
        else:
            (loss / config.grad_accum).backward()

        pending += 1
        batches += 1
        sums["loss"] += float(loss.item())
        for key in _LOSS_KEYS:
            sums[key] += float(loss_dict[key].item())

        if pending == config.grad_accum:
            steps_taken, pooled = _apply_step(
                runtime, optimizer, scheduler, steps_taken, pending_totals, pending_rows
            )
            if pooled is not None:
                supcon_sum += pooled
                supcon_pools += 1
            pending, pending_totals, pending_rows = 0, [], []
            if config.max_steps is not None and steps_taken >= config.max_steps:
                logger.info(f"stopping early at --max-steps {config.max_steps}")
                return (
                    _epoch_means(sums, batches, supcon_sum, supcon_pools, config),
                    steps_taken,
                )

    if pending:  # flush a short tail rather than discarding its gradient
        steps_taken, pooled = _apply_step(
            runtime, optimizer, scheduler, steps_taken, pending_totals, pending_rows
        )
        if pooled is not None:
            supcon_sum += pooled
            supcon_pools += 1

    return _epoch_means(sums, batches, supcon_sum, supcon_pools, config), steps_taken


def _apply_step(
    runtime: Runtime,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    steps_taken: int,
    micro_totals: list[torch.Tensor],
    rows: Sequence[ContrastiveRows],
) -> tuple[int, float | None]:
    """Backward any deferred micro-batches, then take one optimizer step.

    ``micro_totals`` is always empty in ``focal`` mode (that path backwards as it goes), so this
    reduces to the original :func:`_optimizer_step` call there -- one code path, two behaviours,
    and no branch duplicated across the loop body and its tail flush.
    """
    pooled = _backward_accumulated(runtime, micro_totals, rows) if micro_totals else None
    return _optimizer_step(runtime, optimizer, scheduler, steps_taken), pooled


def _optimizer_step(
    runtime: Runtime,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    steps_taken: int,
) -> int:
    grad_norm = torch.nn.utils.clip_grad_norm_(runtime.plan.all_trainable, runtime.config.grad_clip)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    steps_taken += 1
    if steps_taken % 10 == 1:
        rates = " / ".join(f"{group['lr']:.2e}" for group in optimizer.param_groups)
        logger.debug(f"step {steps_taken}: grad-norm {float(grad_norm):.3f}, lr {rates}")
    return steps_taken


@torch.no_grad()
def _evaluate(
    runtime: Runtime, targets: list[ImageTargets], image_dir: Path
) -> tuple[dict[str, float], dict[str, float | None] | None, dict[str, float | None] | None]:
    """The identical loss over a held-out split, plus the cosine diagnostics. No optimizer step.

    No shuffle: the val number must depend only on the weights, so that an epoch-to-epoch change is
    the model moving and not the batch composition moving. For the same reason the background
    sampler runs from a generator re-seeded **here**, at every call: the val negatives are then the
    same patches at epoch 0 and at epoch 8, and a move in ``val_cos_gap`` cannot be the sample
    moving. ``test`` is never touched by this script.

    The contrastive term is pooled over ``grad_accum`` micro-batches, matching training's
    granularity so the val and train numbers mean the same thing -- and bounding the pairwise
    similarity matrix, which pooling the whole split would not.

    Returns:
        ``(mean losses, cosine gap report, crop/scene agreement report)``. The cosine gap report is
        ``None`` in ``focal`` mode, exactly as before this task. The crop/scene agreement report
        (D-dla-06, quick task 260808-dla) is ``None`` whenever ``config.supcon_crop_context`` is
        ``False`` -- including in ``focal`` mode -- which is what keeps a flag-off run's ``epochs``
        array exactly the shape it had before this task existed.
    """
    config = runtime.config
    contrastive = config.loss_mode != "focal"
    runtime.model.eval()
    sums = dict.fromkeys(("loss", *_LOSS_KEYS), 0.0)
    batches = 0
    rng = np.random.default_rng(config.seed)

    pending_rows: list[ContrastiveRows] = []
    supcon_sum = 0.0
    supcon_pools = 0
    anchor_chunks: list[np.ndarray] = []
    label_chunks: list[np.ndarray] = []
    background_chunks: list[np.ndarray] = []
    diag_crop_chunks: list[np.ndarray] = []
    diag_scene_chunks: list[np.ndarray] = []

    def pool_pending() -> None:
        """Score one pool's worth of held rows -- the val counterpart of the accumulation boundary.

        The contrastive term MUST be in the val total: it is what selects the checkpoint, and a val
        loss made only of box terms would pick the epoch with the best boxes rather than the best
        embedding space, silently defeating the whole experiment.
        """
        nonlocal supcon_sum, supcon_pools
        pooled = _pooled_supcon(pending_rows, config)
        if pooled is not None:
            supcon_sum += float(pooled.item())
            supcon_pools += 1
        pending_rows.clear()

    for start in range(0, len(targets), config.batch_size):
        batch = targets[start : start + config.batch_size]
        loss_dict, rows = _batch_loss(runtime, batch, image_dir, rng)
        total = _total_loss(loss_dict, config)

        sums["loss"] += float(total.item())
        for key in _LOSS_KEYS:
            sums[key] += float(loss_dict[key].item())
        batches += 1

        if rows is not None:
            pending_rows.append(rows)
            anchor_chunks.append(rows.anchors.cpu().numpy())
            label_chunks.append(rows.labels.cpu().numpy())
            background_chunks.append(rows.background.cpu().numpy())
            if config.supcon_crop_context:
                diag_crop_chunks.append(rows.crop_diag_crop.cpu().numpy())
                diag_scene_chunks.append(rows.crop_diag_scene.cpu().numpy())
            if len(pending_rows) == config.grad_accum:
                pool_pending()

    if pending_rows:
        pool_pending()

    losses = _epoch_means(sums, batches, supcon_sum, supcon_pools, config)
    if not contrastive:
        return losses, None, None

    # The gap IS measured over the whole split, unlike the loss: it is a summary statistic rather
    # than a training signal, and pooling it is what makes it comparable epoch to epoch.
    width = 1 if not anchor_chunks else int(anchor_chunks[0].shape[1])
    gap = cosine_gap_report(
        np.concatenate(anchor_chunks) if anchor_chunks else np.zeros((0, width)),
        np.concatenate(label_chunks) if label_chunks else np.zeros(0, dtype=np.int64),
        np.concatenate(background_chunks) if background_chunks else np.zeros((0, width)),
    )

    # Same pooling discipline as the gap above: measured over the WHOLE split (never fabricated,
    # None components thread through crop_scene_agreement untouched -- D-dla-06).
    crop_scene: dict[str, float | None] | None = None
    if config.supcon_crop_context:
        crop_scene = crop_scene_agreement(
            np.concatenate(diag_crop_chunks) if diag_crop_chunks else np.zeros((0, width)),
            np.concatenate(diag_scene_chunks) if diag_scene_chunks else np.zeros((0, width)),
        )
    return losses, gap, crop_scene


# ---------------------------------------------------------------------------------- 5. the CLI


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune OWLv2 on the floor-plans train split.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    defaults = FinetuneConfig()
    parser.add_argument("--train-coco", type=Path, default=Path(_DEFAULT_TRAIN_COCO))
    parser.add_argument(
        "--val-coco",
        type=Path,
        default=Path(_DEFAULT_VAL_COCO),
        help="Held-out split for per-epoch val loss and checkpoint selection. NEVER the test split.",
    )
    parser.add_argument("--out", type=Path, default=Path(_DEFAULT_OUT))
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=defaults.grad_accum,
        help="Micro-batches per optimizer step. In contrastive/both mode this ALSO sets the "
        "contrastive pool: SupCon is computed once per step over all of them (D-hg1-04), so the "
        "rare classes get same-class positives -- at the cost of retaining that many micro-batch "
        "graphs until the accumulation boundary. Cheap with a frozen backbone (only the heads are "
        "retained); with --unfreeze-all it retains that many ViT graphs, so lower it there.",
    )
    parser.add_argument("--lr-head", type=float, default=defaults.lr_head)
    parser.add_argument("--lr-backbone", type=float, default=defaults.lr_backbone)
    parser.add_argument("--weight-decay", type=float, default=defaults.weight_decay)
    parser.add_argument("--grad-clip", type=float, default=defaults.grad_clip)
    parser.add_argument("--warmup-frac", type=float, default=defaults.warmup_frac)
    parser.add_argument(
        "--unfreeze-last-n",
        type=int,
        default=defaults.unfreeze_last_n,
        help="Also train the last N vision-encoder blocks at --lr-backbone (0 = heads only).",
    )
    parser.add_argument(
        "--unfreeze-all",
        action="store_true",
        help="Arm B: also train the whole vision tower at --lr-backbone. The text tower stays "
        "frozen in every arm.",
    )
    parser.add_argument(
        "--loss-mode",
        choices=("focal", "contrastive", "both"),
        default=defaults.loss_mode,
        help="focal (DEFAULT, the already-measured recipe) = sigmoid focal loss over the "
        "text-conditioned logits. contrastive = a supervised-contrastive loss over class_embeds -- "
        "the space owlv2-oneshot actually scores in -- replacing the focal term. both = the sum. "
        "The L1/GIoU box terms are in every mode.",
    )
    parser.add_argument(
        "--supcon-temperature",
        type=float,
        default=defaults.supcon_temperature,
        help="SupCon temperature tau (contrastive/both mode).",
    )
    parser.add_argument(
        "--supcon-background-negatives",
        type=int,
        default=defaults.supcon_background_negatives,
        help="Background patches sampled per image as DENOMINATOR-ONLY negatives: grid cells whose "
        "normalized centre lies in no ground-truth box. They are never anchors and never "
        "positives. Load-bearing rather than a refinement -- the measured failure is background "
        "scoring too high, which an anchor-only SupCon cannot address. 0 disables them.",
    )
    parser.add_argument(
        "--w-contrast",
        type=float,
        default=defaults.w_contrast,
        help="Weight of the supervised-contrastive term in the total loss.",
    )
    parser.add_argument(
        "--supcon-crop-context",
        action="store_true",
        help="Quick task 260808-dla: ALSO build one crop-context SupCon anchor per training "
        "image, encoded via the SAME preprocessing/query-selection path owlv2-oneshot's inference "
        "runs for its exemplar crop (D-dla-01/02/04). Opt-in, layered on --loss-mode "
        "contrastive/both; ignored in focal mode. Default off leaves focal and flag-off "
        "contrastive byte-identical to before this flag existed (D-dla-03).",
    )
    parser.add_argument(
        "--supcon-query-iou-frac",
        type=float,
        default=defaults.supcon_query_iou_frac,
        help="Crop-context anchor's query-patch selection threshold, pinned equal to "
        "Owlv2OneshotConfig.query_iou_frac's default so training selects the query patch the way "
        "inference does (D-dla-05). Only read when --supcon-crop-context is set.",
    )
    parser.add_argument(
        "--supcon-crop-margin-frac",
        type=float,
        default=defaults.supcon_crop_margin_frac,
        help="Quick task 260808-w8c: grow the crop-context anchor's ground-truth box by this "
        "fraction of its own size before cropping, via the SAME expand_box_with_margin "
        "owlv2-oneshot's inference crop uses (D-w8c-01/02). Default 0.0 leaves the crop-context "
        "anchor exactly as tight as 260808-dla measured it. Only read when --supcon-crop-context "
        "is set.",
    )
    parser.add_argument(
        "--supcon-crop-augment",
        action="store_true",
        help="Quick task 260808-w8c: add ONE additional rotated/mirrored view of the SAME "
        "crop-context anchor as a second same-class SupCon positive per image (D-w8c-06/07/08). "
        "Opt-in, layered on --supcon-crop-context; ignored when that is off. Default off leaves "
        "the already-measured contrastive-crop arm byte-identical to before this flag existed.",
    )
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="Assert the torch SupCon mirror agrees with the NumPy specification in "
        "object_search.train.supcon and that it has a finite non-zero gradient, then exit. This is "
        "how the torch half is gated: torch lives only in the export pixi environment, where "
        "pytest is not installed.",
    )
    parser.add_argument("--limit-images", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument(
        "--no-bf16",
        action="store_true",
        help="Disable bf16 autocast on CUDA (it is off on CPU/MPS regardless).",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
        help="auto = cuda when available, else cpu (mps is opt-in: it is a local convenience).",
    )
    return parser.parse_args(argv)


def _config_from_args(args: argparse.Namespace) -> FinetuneConfig:
    """The one place CLI flags become the frozen, logged, self-describing config."""
    return FinetuneConfig(
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        lr_head=args.lr_head,
        lr_backbone=args.lr_backbone,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        warmup_frac=args.warmup_frac,
        unfreeze_last_n=args.unfreeze_last_n,
        unfreeze_all=args.unfreeze_all,
        loss_mode=args.loss_mode,
        supcon_temperature=args.supcon_temperature,
        w_contrast=args.w_contrast,
        supcon_background_negatives=args.supcon_background_negatives,
        supcon_crop_context=args.supcon_crop_context,
        supcon_query_iou_frac=args.supcon_query_iou_frac,
        supcon_crop_margin_frac=args.supcon_crop_margin_frac,
        supcon_crop_augment=args.supcon_crop_augment,
        max_steps=args.max_steps,
        limit_images=args.limit_images,
    )


def _run_self_check() -> None:
    """Assert ``supcon_loss_torch`` mirrors the NumPy specification, and that it differentiates.

    Verified fact 4 of this task: ``pixi run test`` cannot import torch (it lives only in the
    ``export`` environment) and ``pytest`` is not installed in ``export``. So the NumPy function is
    the tested specification and this flag is how the torch mirror is gated. Both halves matter:

    * **numeric parity** catches a transcription slip (an ``L_in`` fold, a sign, a mask polarity);
    * **a finite non-zero gradient** catches the failure parity alone cannot -- a loss that computes
      the right number through ``no_grad``/``detach``/an integer cast trains nothing at all, and
      would produce a perfectly plausible flat curve.

    The comparison runs in float64 so any difference is the *maths*, not float32 rounding.

    Raises:
        SystemExit: If the two disagree, or the gradient is not finite and non-zero.
    """
    rng = np.random.default_rng(0)
    embeddings = rng.normal(size=(24, 16))
    labels = rng.integers(0, 4, size=24)
    background_rows = 6
    temperature = FinetuneConfig().supcon_temperature

    # The trailing rows carry REAL labels that collide with the anchors', so this also checks that
    # the denominator-only rule is enforced inside both implementations rather than by the caller.
    negative_only = np.zeros(len(labels), dtype=bool)
    negative_only[-background_rows:] = True
    expected = supcon_loss(embeddings, labels, temperature, negative_only=negative_only)

    tensor = torch.tensor(embeddings, dtype=torch.float64, requires_grad=True)
    actual = supcon_loss_torch(
        tensor, torch.tensor(labels, dtype=torch.int64), temperature, background_rows
    )
    delta = abs(float(actual.item()) - expected)

    actual.backward()
    gradient = tensor.grad
    if gradient is None:
        raise SystemExit("self-check FAILED: supcon_loss_torch produced no gradient at all")
    finite = bool(torch.isfinite(gradient).all())
    magnitude = float(gradient.abs().sum().item())

    logger.info(
        f"self-check: numpy {expected:.12f} vs torch {float(actual.item()):.12f} "
        f"(|delta| {delta:.3e}); gradient finite={finite}, sum|grad|={magnitude:.6f}"
    )
    if delta >= 1e-6:
        raise SystemExit(f"self-check FAILED: torch and numpy SupCon disagree by {delta:.3e}")
    if not finite or magnitude == 0.0:
        raise SystemExit(
            f"self-check FAILED: gradient finite={finite}, sum|grad|={magnitude} -- a loss with no "
            "gradient trains nothing while still printing a plausible curve"
        )
    logger.info("self-check PASSED: torch mirrors the NumPy specification and differentiates")


def _format_gap(gap: dict[str, float | None] | None) -> str:
    """One-line rendering of a cosine-gap report, with ``None`` shown as ``n/a`` and never as 0."""
    if gap is None:
        return "n/a"
    return " ".join(
        f"{key}={'n/a' if value is None else f'{value:+.4f}'}" for key, value in gap.items()
    )


def _resolve_device(choice: str) -> torch.device:
    if choice != "auto":
        return torch.device(choice)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_optimizer(
    plan: FreezePlan, config: FinetuneConfig, total_steps: int
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    """Two-group AdamW (heads at ``lr_head``, backbone at ``lr_backbone``) + cosine warmup.

    The backbone group is omitted entirely rather than added empty when the arm is heads-only, so
    the logged LR columns match the parameters that actually move.
    """
    groups = [{"params": plan.head_params, "lr": config.lr_head}]
    if plan.backbone_params:
        groups.append({"params": plan.backbone_params, "lr": config.lr_backbone})
    optimizer = torch.optim.AdamW(groups, weight_decay=config.weight_decay)

    warmup = warmup_steps_for(total_steps, config.warmup_frac)
    logger.info(f"schedule: {total_steps} optimizer step(s), {warmup} of them warmup")
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: cosine_warmup_factor(step, total_steps, warmup)
    )
    return optimizer, scheduler


def _planned_steps(image_count: int, config: FinetuneConfig) -> int:
    """Total optimizer steps the schedule is stretched over (the short tail batch counts)."""
    micro_batches = -(-image_count // config.batch_size)  # ceil
    per_epoch = -(-micro_batches // config.grad_accum)
    total = max(1, per_epoch * config.epochs)
    return min(total, config.max_steps) if config.max_steps is not None else total


def _epoch_record(
    epoch: int,
    steps: int,
    train_losses: dict[str, float],
    val_losses: dict[str, float],
    loss_keys: Sequence[str],
    *,
    saved: bool,
    cos_gap: dict[str, float | None] | None,
    crop_scene: dict[str, float | None] | None = None,
) -> dict[str, object]:
    """One row of the ``epochs`` array. Numbers only -- no timestamp, no duration.

    Built in one place so the epoch-0 reference row (D-hg1-06) and the per-epoch rows cannot drift
    into different shapes, and so a ``focal`` run provably emits exactly the keys it always has:
    ``loss_keys`` excludes ``loss_supcon`` there and both ``cos_gap`` and ``crop_scene`` are
    ``None``. ``crop_scene`` (D-dla-06, quick task 260808-dla) mirrors ``cos_gap``'s optional-key
    handling exactly: present only when ``config.supcon_crop_context`` is ``True``, absent (not
    ``null``) otherwise -- so a ``--supcon-crop-context``-disabled run's ``epochs`` array shape is
    unchanged by this task.
    """
    record: dict[str, object] = {
        "epoch": epoch,
        "steps": steps,
        "train_loss": train_losses["loss"],
        **{f"train_{key}": train_losses[key] for key in loss_keys},
        "val_loss": val_losses["loss"],
        **{f"val_{key}": val_losses[key] for key in loss_keys},
        "saved": saved,
    }
    if cos_gap is not None:
        record["val_cos_gap"] = cos_gap
    if crop_scene is not None:
        record["val_crop_scene_agreement"] = crop_scene
    return record


def _write_train_log(
    log_path: Path,
    config: FinetuneConfig,
    plan: FreezePlan,
    epochs: list[dict[str, object]],
    best_epoch: int,
    best_val: float,
    args: argparse.Namespace,
    use_bf16: bool,
) -> None:
    """The record the report reads: the resolved config, the arm, the seed, and the curve."""
    log_path.write_text(
        json.dumps(
            {
                "arm": plan.arm,
                "seed": config.seed,
                "config": config.model_dump(mode="json"),
                "classes": list(FLOORPLAN_CLASSES),
                "base_model": _BASE_MODEL,
                "train_coco": str(args.train_coco),
                "val_coco": str(args.val_coco),
                "trainable_params": plan.trainable_params,
                "total_params": plan.total_params,
                "bf16": use_bf16,
                "best_epoch": best_epoch,
                "best_val_loss": best_val,
                "epochs": epochs,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def main(argv: list[str] | None = None) -> None:
    """Fine-tune OWLv2 on the floor-plan train split and save the best-val HuggingFace checkpoint."""
    args = _parse_args(argv)
    if args.self_check:
        # Before any data or weights are touched: this gate is about the loss function, and it must
        # be runnable on a laptop with no dataset and no network.
        _run_self_check()
        return
    config = _config_from_args(args)
    loss_keys = _loss_keys_for(config)

    # Reproducibility: one explicit seed drives torch and the numpy epoch shuffler. Never a bare
    # random.shuffle, and never cv2.setRNGSeed (a no-op for anything that matters here, D-11).
    torch.manual_seed(config.seed)
    device = _resolve_device(args.device)
    use_bf16 = device.type == "cuda" and not args.no_bf16 and torch.cuda.is_bf16_supported()

    # 1. Targets: COCO xywh px -> cxcywh normalized by the PADDED-SQUARE side (torch-free, tested).
    train_targets, train_dir = _load_split(args.train_coco, config.limit_images, "train")
    val_targets, val_dir = _load_split(args.val_coco, config.limit_images, "val")

    # 2. Model + processor. The text tower is frozen in every arm (it is not in the exported graph).
    logger.info(f"loading {_BASE_MODEL} onto {device} (bf16 autocast: {use_bf16})")
    model = Owlv2ForObjectDetection.from_pretrained(_BASE_MODEL).to(device)
    processor = Owlv2Processor.from_pretrained(_BASE_MODEL)
    input_ids, attention_mask = _tokenize_class_names(processor, device)

    # 3. Freeze strategy -> the two learning-rate groups.
    plan = _apply_freeze_strategy(model, config)
    logger.info(
        f"arm {plan.arm}: trainable {plan.trainable_params:,} / {plan.total_params:,} params "
        f"({100.0 * plan.trainable_params / plan.total_params:.2f}%) -- "
        f"{len(plan.head_params)} head tensor(s), {len(plan.backbone_params)} backbone tensor(s)"
    )

    # 4. Loss: sigmoid-consistent matching + focal labels + inherited L1/GIoU boxes. "cardinality"
    #    is EXCLUDED -- it assumes DETR's no-object logit column, which OWLv2 does not have.
    matcher = Owlv2HungarianMatcher(
        class_cost=config.class_cost, bbox_cost=config.bbox_cost, giou_cost=config.giou_cost
    )
    criterion = Owlv2ImageLoss(
        matcher=matcher,
        num_classes=len(FLOORPLAN_CLASSES),
        losses=["labels", "boxes"],
        focal_alpha=config.focal_alpha,
        focal_gamma=config.focal_gamma,
    ).to(device)
    runtime = Runtime(
        model=model,
        criterion=criterion,
        config=config,
        device=device,
        input_ids=input_ids,
        attention_mask=attention_mask,
        plan=plan,
        use_bf16=use_bf16,
    )

    # 5. Train, validating after every epoch and keeping only the best-val checkpoint.
    total_steps = _planned_steps(len(train_targets), config)
    optimizer, scheduler = _build_optimizer(plan, config, total_steps)
    rng = np.random.default_rng(config.seed)
    log_path = args.out / "train_log.json"
    epochs: list[dict[str, object]] = []
    best_val = float("inf")
    best_epoch = 0
    steps_taken = 0
    args.out.mkdir(parents=True, exist_ok=True)

    # 5a. The epoch-0 reference point (D-hg1-06), contrastive/both only. One no-gradient pass over
    #     each split before a single weight has moved, so the run carries its own pretrained
    #     baseline for the cosine property owlv2-oneshot actually scores with. `focal` runs skip it
    #     entirely and stay comparable, row for row, with the three already-measured arms.
    if config.loss_mode != "focal":
        logger.info("epoch 0: measuring the pre-training reference point (no optimizer step)")
        zero_train, _, _ = _evaluate(runtime, train_targets, train_dir)
        zero_val, zero_gap, zero_crop_scene = _evaluate(runtime, val_targets, val_dir)
        logger.info(
            f"epoch 0: train {zero_train['loss']:.4f} val {zero_val['loss']:.4f} "
            f"(supcon {zero_val[_SUPCON_KEY]:.4f}) cos-gap {_format_gap(zero_gap)}"
        )
        if zero_crop_scene is not None:
            logger.info(f"epoch 0: val crop/scene self-score {_format_gap(zero_crop_scene)}")
        epochs.append(
            _epoch_record(
                0,
                0,
                zero_train,
                zero_val,
                loss_keys,
                saved=False,
                cos_gap=zero_gap,
                crop_scene=zero_crop_scene,
            )
        )

    for epoch in range(1, config.epochs + 1):
        train_losses, steps_taken = _train_one_epoch(
            runtime, train_targets, train_dir, optimizer, scheduler, rng, steps_taken
        )
        val_losses, cos_gap, crop_scene = _evaluate(runtime, val_targets, val_dir)

        improved = val_losses["loss"] < best_val
        if improved:
            best_val, best_epoch = val_losses["loss"], epoch
            model.save_pretrained(args.out)
            processor.save_pretrained(args.out)

        logger.info(
            f"epoch {epoch}/{config.epochs}: train {train_losses['loss']:.4f} "
            f"val {val_losses['loss']:.4f} "
            f"(ce {val_losses['loss_ce']:.4f} bbox {val_losses['loss_bbox']:.4f} "
            f"giou {val_losses['loss_giou']:.4f}) "
            f"{'-> SAVED (best val)' if improved else '-- not saved'}"
        )
        if cos_gap is not None:
            logger.info(f"epoch {epoch}: val cos-gap {_format_gap(cos_gap)}")
        if crop_scene is not None:
            logger.info(f"epoch {epoch}: val crop/scene self-score {_format_gap(crop_scene)}")

        epochs.append(
            _epoch_record(
                epoch,
                steps_taken,
                train_losses,
                val_losses,
                loss_keys,
                saved=improved,
                cos_gap=cos_gap,
                crop_scene=crop_scene,
            )
        )
        # Rewritten every epoch so a long GPU run that dies at hour three still leaves its curve.
        _write_train_log(log_path, config, plan, epochs, best_epoch, best_val, args, use_bf16)

        if config.max_steps is not None and steps_taken >= config.max_steps:
            break

    if best_epoch == 0:
        raise SystemExit("no epoch improved on an infinite val loss -- nothing was saved")
    logger.info(
        f"saved the epoch-{best_epoch} checkpoint (val loss {best_val:.4f}) to {args.out}; "
        f"curve in {log_path}"
    )


if __name__ == "__main__":
    main()
