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

Reproducibility
---------------
Same seed => identical per-epoch losses, which the task's verify step asserts by diffing two runs'
``train_log.json``. One ``--seed`` drives ``torch.manual_seed`` and the ``np.random.default_rng``
epoch shuffler; the epoch record holds numbers only (never a duration or a timestamp), so the log
files of two same-seed runs compare equal.
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
from object_search.train.owlv2_targets import (  # noqa: E402
    FLOORPLAN_CLASSES,
    FinetuneConfig,
    ImageTargets,
    coco_to_owlv2_targets,
    cosine_warmup_factor,
    deterministic_batches,
    warmup_steps_for,
)
from object_search.train.supcon import supcon_loss  # noqa: E402

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
            matcher assigned to this micro-batch's ground-truth boxes.
        labels: ``(n_boxes,)`` int64 class labels, aligned with ``anchors``.
        background: ``(n_background, 512)`` rows sampled from patches whose grid-cell centre lies
            in no ground-truth box -- denominator-only negatives (D-hg1-03).
    """

    anchors: torch.Tensor
    labels: torch.Tensor
    background: torch.Tensor


def _contrastive_rows(
    runtime: Runtime,
    outputs: dict[str, torch.Tensor],
    class_embeds: torch.Tensor,
    targets: list[dict[str, torch.Tensor]],
) -> ContrastiveRows:
    """Gather this micro-batch's SupCon anchors: the matched patch for each ground-truth box.

    The anchor selection is the **existing** ``Owlv2HungarianMatcher`` assignment, not new matching
    logic: the anchor for a ground-truth box is by construction the same patch the box loss
    supervises, which is what makes the contrastive term and the box term agree about what a "door"
    is. The gather mirrors ``ImageLoss._get_source_permutation_idx``.

    The matcher is run once more here rather than having the criterion hand its indices back. That
    is a deliberate trade: a duplicated, deterministic, ``no_grad`` Hungarian solve (microseconds
    against a 3600-patch ViT forward) in exchange for not threading cached mutable state out of a
    ``transformers`` base class. It runs in contrastive/both mode only.
    """
    indices = runtime.criterion.matcher(outputs, targets)

    anchors: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    for image_index, (source_idx, target_idx) in enumerate(indices):
        source = source_idx.to(class_embeds.device)
        anchors.append(class_embeds[image_index, source])
        labels.append(targets[image_index]["class_labels"][target_idx.to(class_embeds.device)])

    empty_background = class_embeds.new_zeros((0, class_embeds.shape[-1]))
    return ContrastiveRows(
        anchors=torch.cat(anchors) if anchors else empty_background,
        labels=(
            torch.cat(labels)
            if labels
            else torch.zeros(0, dtype=torch.int64, device=class_embeds.device)
        ),
        background=empty_background,
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
    runtime: Runtime, batch: list[ImageTargets], image_dir: Path
) -> tuple[dict[str, torch.Tensor], ContrastiveRows | None]:
    """Forward one micro-batch (bf16 autocast on CUDA) -> the loss terms in fp32, plus SupCon rows.

    The autocast region covers the forward only. Hungarian matching and the loss reductions run in
    fp32 on purpose: bf16 has ~3 decimal digits of mantissa, which is enough for a ViT's activations
    and nowhere near enough for a cost matrix whose argmin decides which patch supervises which box.

    The returned ``ContrastiveRows`` is ``None`` in ``focal`` mode, so that arm does no extra work
    at all and stays byte-comparable with the three already-measured arms.
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
        rows = _contrastive_rows(runtime, outputs, class_embeds.float(), targets)
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
    """
    config = runtime.config
    loss_keys = _loss_keys_for(config)
    runtime.model.train()
    if runtime.plan.backbone_frozen:
        # A frozen module in train() mode is a no-op for OWLv2 (dropout is 0.0 and LayerNorm keeps
        # no running stats), but eval() states the intent and is robust to a config that changes.
        runtime.model.owlv2.eval()
    runtime.model.owlv2.text_model.eval()  # frozen in every arm

    sums = dict.fromkeys(("loss", *loss_keys), 0.0)
    batches = 0
    pending = 0
    optimizer.zero_grad(set_to_none=True)

    for indices in deterministic_batches(len(targets), config.batch_size, rng):
        batch = [targets[index] for index in indices]
        loss_dict, rows = _batch_loss(runtime, batch, image_dir)
        loss = _total_loss(loss_dict, config)
        if config.loss_mode != "focal":
            supcon = _pooled_supcon([] if rows is None else [rows], config)
            loss_dict[_SUPCON_KEY] = (
                torch.zeros((), device=runtime.device) if supcon is None else supcon
            )
            if supcon is not None:
                loss = loss + config.w_contrast * supcon

        (loss / config.grad_accum).backward()
        pending += 1
        batches += 1
        sums["loss"] += float(loss.item())
        for key in loss_keys:
            sums[key] += float(loss_dict[key].item())

        if pending == config.grad_accum:
            steps_taken = _optimizer_step(runtime, optimizer, scheduler, steps_taken)
            pending = 0
            if config.max_steps is not None and steps_taken >= config.max_steps:
                logger.info(f"stopping early at --max-steps {config.max_steps}")
                return _mean_losses(sums, batches), steps_taken

    if pending:  # flush a short tail rather than discarding its gradient
        steps_taken = _optimizer_step(runtime, optimizer, scheduler, steps_taken)

    return _mean_losses(sums, batches), steps_taken


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
def _evaluate(runtime: Runtime, targets: list[ImageTargets], image_dir: Path) -> dict[str, float]:
    """The identical loss over the held-out ``valid`` split, in deterministic file_name order.

    No shuffle: the val number must depend only on the weights, so that an epoch-to-epoch change is
    the model moving and not the batch composition moving. ``test`` is never touched by this script.
    """
    config = runtime.config
    loss_keys = _loss_keys_for(config)
    runtime.model.eval()
    sums = dict.fromkeys(("loss", *loss_keys), 0.0)
    batches = 0

    for start in range(0, len(targets), config.batch_size):
        batch = targets[start : start + config.batch_size]
        loss_dict, rows = _batch_loss(runtime, batch, image_dir)
        total = _total_loss(loss_dict, config)
        if config.loss_mode != "focal":
            # The contrastive term MUST be in the val total: it is what selects the checkpoint, and
            # a val loss made only of box terms would pick the epoch with the best boxes rather
            # than the best embedding space -- silently defeating the whole experiment.
            supcon = _pooled_supcon([] if rows is None else [rows], config)
            loss_dict[_SUPCON_KEY] = (
                torch.zeros((), device=runtime.device) if supcon is None else supcon
            )
            if supcon is not None:
                total = total + config.w_contrast * supcon

        sums["loss"] += float(total.item())
        for key in loss_keys:
            sums[key] += float(loss_dict[key].item())
        batches += 1

    return _mean_losses(sums, batches)


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
    parser.add_argument("--grad-accum", type=int, default=defaults.grad_accum)
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
        "--w-contrast",
        type=float,
        default=defaults.w_contrast,
        help="Weight of the supervised-contrastive term in the total loss.",
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

    for epoch in range(1, config.epochs + 1):
        train_losses, steps_taken = _train_one_epoch(
            runtime, train_targets, train_dir, optimizer, scheduler, rng, steps_taken
        )
        val_losses = _evaluate(runtime, val_targets, val_dir)

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

        # Numbers only: no timestamps and no durations, so two same-seed runs' logs compare equal.
        epochs.append(
            {
                "epoch": epoch,
                "steps": steps_taken,
                "train_loss": train_losses["loss"],
                **{f"train_{key}": train_losses[key] for key in loss_keys},
                "val_loss": val_losses["loss"],
                **{f"val_{key}": val_losses[key] for key in loss_keys},
                "saved": improved,
            }
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
