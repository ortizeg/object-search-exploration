"""Fine-tune OWLv2 on the Roboflow floor-plans-500 train split (quick task 260801-8zy).

Run this ONLY in the ``export`` pixi environment, which carries ``torch`` and ``transformers``::

    pixi run -e export finetune-owlv2 --out models/finetune/owlv2-floorplans-headonly

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

The split against ``src/object_search/train/``
----------------------------------------------
All torch lives here, in one top-to-bottom readable file, because this is the artifact an ML
practitioner reads and edits. The torch-free glue that decides *what the model is trained on* --
the config schema, the class-index mapping, the COCO -> OWLv2 target conversion, the deterministic
batch order -- lives in :mod:`object_search.train.owlv2_targets`, where ``pixi run test`` gates it
with no torch, no weights, and no GPU. ``pixi run lint`` / ``typecheck`` cover ``src/`` and
``tests/`` only, which is the existing precedent set by ``scripts/export_owlv2.py``.

Pre-processing (exact, and shared with inference)
-------------------------------------------------
Images go through the repo's own :func:`object_search.inference.owlv2.owlv2_preprocess_tensor`, not
through ``Owlv2Processor``'s image path -- so training and inference are provably one code path:
BGR->RGB, rescale ``1/255``, **pad bottom-right** to a square of side ``max(H, W)`` with grey
``0.5``, resize to ``960x960`` bilinear, CLIP mean/std. Targets are normalized over that **same
padded-square side** (see the ``owlv2_targets`` module docstring for why per-axis ``(W, H)``
normalization would be a silent, plausible-looking bug).
"""

from __future__ import annotations

import argparse
import json
import sys
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
)

_BASE_MODEL = "google/owlv2-base-patch16-ensemble"
_DEFAULT_TRAIN_COCO = "datasets/_incoming/floorplans/train/_annotations.coco.json"
_DEFAULT_OUT = "models/finetune/owlv2-floorplans-headonly"


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


# ------------------------------------------------------------------------ 3. the training step


def _forward_batch(
    model: Owlv2ForObjectDetection,
    pixel_values: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
):
    """One text-conditioned forward pass over a batch of images.

    ``forward`` expects ``input_ids`` shaped ``[batch * num_text_queries, seq]`` (it reshapes them
    internally), so the single tokenized query block is repeated once per image in the batch.
    """
    batch_size = pixel_values.shape[0]
    return model(
        input_ids=input_ids.repeat(batch_size, 1),
        pixel_values=pixel_values,
        attention_mask=attention_mask.repeat(batch_size, 1),
    )


def _total_loss(loss_dict: dict[str, torch.Tensor], config: FinetuneConfig) -> torch.Tensor:
    """``w_class * loss_ce + w_bbox * loss_bbox + w_giou * loss_giou`` (defaults 1 / 5 / 2)."""
    return (
        config.w_class * loss_dict["loss_ce"]
        + config.w_bbox * loss_dict["loss_bbox"]
        + config.w_giou * loss_dict["loss_giou"]
    )


def _freeze_to_heads(model: Owlv2ForObjectDetection) -> list[torch.nn.Parameter]:
    """Freeze everything, then unfreeze the three prediction heads. Returns the trainable params.

    ``class_head`` carries OWLv2's learned ``logit_scale`` / ``logit_shift``, so training it also
    trains the model's own calibration -- which is exactly what the method's compressed-cosine
    thresholding problem needs.
    """
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    trainable: list[torch.nn.Parameter] = []
    for head in (model.box_head, model.class_head, model.objectness_head):
        for parameter in head.parameters():
            parameter.requires_grad_(True)
            trainable.append(parameter)
    return trainable


# ---------------------------------------------------------------------------------- 4. the CLI


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune OWLv2 on the floor-plans train split.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--train-coco", type=Path, default=Path(_DEFAULT_TRAIN_COCO))
    parser.add_argument("--out", type=Path, default=Path(_DEFAULT_OUT))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr-head", type=float, default=1e-4)
    parser.add_argument("--limit-images", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
        help="auto = cuda when available, else cpu (mps is opt-in: it is a local convenience).",
    )
    return parser.parse_args(argv)


def _resolve_device(choice: str) -> torch.device:
    if choice != "auto":
        return torch.device(choice)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(argv: list[str] | None = None) -> None:
    """Fine-tune OWLv2's heads on the floor-plan train split and save a HuggingFace checkpoint."""
    args = _parse_args(argv)
    config = FinetuneConfig(
        seed=args.seed,
        batch_size=args.batch_size,
        lr_head=args.lr_head,
        max_steps=args.max_steps,
        limit_images=args.limit_images,
    )

    # Reproducibility: one explicit seed drives torch and the numpy epoch shuffler. Never a bare
    # random.shuffle, and never cv2.setRNGSeed (a no-op for anything that matters here, D-11).
    torch.manual_seed(config.seed)
    device = _resolve_device(args.device)

    # 1. Targets: COCO xywh px -> cxcywh normalized by the PADDED-SQUARE side (torch-free, tested).
    if not args.train_coco.is_file():
        raise SystemExit(
            f"no COCO annotations at {args.train_coco}. The floor plans are a MANUAL dataset: "
            f"drop the Roboflow export at datasets/_incoming/floorplans/{{train,valid,test}} first."
        )
    coco = json.loads(args.train_coco.read_text())
    targets = coco_to_owlv2_targets(coco)
    if config.limit_images is not None:
        targets = targets[: config.limit_images]
    if not targets:
        raise SystemExit(f"{args.train_coco} produced no usable targets")
    image_dir = args.train_coco.parent

    # 2. Model + processor. The text tower is frozen in every arm (it is not in the exported graph).
    logger.info(f"loading {_BASE_MODEL} onto {device}")
    model = Owlv2ForObjectDetection.from_pretrained(_BASE_MODEL).to(device)
    processor = Owlv2Processor.from_pretrained(_BASE_MODEL)
    input_ids, attention_mask = _tokenize_class_names(processor, device)

    # 3. Freeze strategy: heads only (arm A). Task 2 adds the --unfreeze-* arms.
    trainable = _freeze_to_heads(model)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in trainable)
    logger.info(
        f"trainable {trainable_params:,} / {total_params:,} params "
        f"({100.0 * trainable_params / total_params:.2f}%) -- heads only"
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

    optimizer = torch.optim.AdamW(trainable, lr=config.lr_head, weight_decay=config.weight_decay)

    # 5. The thin slice: iterate batches in the deterministic file_name order and take real steps.
    model.train()
    rng = np.random.default_rng(config.seed)
    order = [int(i) for i in rng.permutation(len(targets))]
    steps = 0
    for start in range(0, len(order), config.batch_size):
        batch = [targets[i] for i in order[start : start + config.batch_size]]
        pixel_values = torch.cat(
            [_load_pixel_values(image_dir / target.file_name) for target in batch]
        ).to(device)

        outputs = _forward_batch(model, pixel_values, input_ids, attention_mask)
        loss_dict = criterion(
            {"logits": outputs.logits, "pred_boxes": outputs.pred_boxes},
            _batch_targets(batch, device),
        )
        loss = _total_loss(loss_dict, config)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, config.grad_clip)
        optimizer.step()
        steps += 1
        logger.info(
            f"step {steps}: loss {loss.item():.4f} "
            f"(ce {loss_dict['loss_ce'].item():.4f} bbox {loss_dict['loss_bbox'].item():.4f} "
            f"giou {loss_dict['loss_giou'].item():.4f})"
        )
        if config.max_steps is not None and steps >= config.max_steps:
            break

    # 6. Save a plain HuggingFace checkpoint -- exactly what `export_owlv2.py --checkpoint` reads.
    args.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out)
    processor.save_pretrained(args.out)
    logger.info(f"saved checkpoint to {args.out} after {steps} step(s)")


if __name__ == "__main__":
    main()
