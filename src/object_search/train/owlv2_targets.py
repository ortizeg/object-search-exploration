"""COCO floor-plan annotations -> OWLv2 detection targets, plus the fine-tuning config schema.

Torch-free on purpose (numpy + pydantic + loguru only), so ``pixi run test`` gates every line of
it in the default env with no weights and no GPU. The torch half of the recipe lives in
``scripts/finetune_owlv2.py``; this module owns the part that is easy to get silently wrong.

The normalization denominator is the PADDED-SQUARE side, not the width or the height
--------------------------------------------------------------------------------------
This is the load-bearing fact of the whole file. OWLv2's preprocessing
(:func:`object_search.inference.owlv2.owlv2_preprocess_tensor`, and the HuggingFace processor it
mirrors) **pads the image bottom-right to a square of side ``max(H, W)``** and only then resizes to
:data:`~object_search.inference.owlv2.OWLV2_IMAGE_SIZE` (960). Because the pad is bottom-right and
not centred, the content origin stays at the top-left, and OWLv2's own ``pred_boxes`` are
``(cx, cy, w, h)`` normalized to ``[0, 1]`` **over that padded square** -- which the method module
already relies on when it maps a prediction back to scene pixels by a plain multiply by
``max(H, W)`` (see ``search/owlv2_oneshot.boxes_to_pixels``).

Training targets must therefore use the **same** denominator. Normalizing by ``(W, H)``
per-axis -- the reflex, and what almost every DETR-style example does -- would stretch every target
box along the short axis by ``max(H, W) / min(H, W)``. On this dataset that is a factor of up to
~1.4, so the model would be trained to predict systematically skewed boxes and would then be
evaluated against un-skewed ground truth. It would still train, the loss would still go down, and
the result would be quietly wrong. Hence the exact-float assertions in
``tests/test_train_owlv2_targets.py``.

What is dropped, and why it is dropped rather than clamped
----------------------------------------------------------
Two annotation-noise guards, both drops:

* a box that is **degenerate** after rounding (``w < 1`` or ``h < 1`` px) -- the same rule
  ``eval/converters/floorplans.py`` already applies when it builds the ground-truth sidecars, so the
  training targets and the evaluation targets agree on what counts as a box;
* a box that falls **outside the image** (any corner beyond ``[0, W] x [0, H]``).

Neither is clamped into a valid-looking box: a clamp would silently turn a bad annotation into a
plausible training target, which is worse than losing it (threat T-8zy-04). Both are counted and
logged. On the Roboflow floor-plans-500 export as committed, **neither guard fires** -- all 3962
train and 1180 valid boxes are in-range and non-degenerate -- so these are defensive, not silent
data surgery.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from object_search.inference.owlv2 import OWLV2_IMAGE_SIZE, OWLV2_PATCH

# The OWLv2 patch-grid size, DERIVED from the inferencer's pinned operating point rather than
# restated: 960 / 16 = 60 per side => 3600 patches, which is the query dimension of the ``logits``
# tensor the classification loss builds its one-hot target over. Deriving it here (torch-free, so
# ``pixi run test`` gates it) keeps the magic number out of the training script entirely.
OWLV2_NUM_PATCHES: int = (OWLV2_IMAGE_SIZE // OWLV2_PATCH) ** 2

# The five real floor-plan categories, as a TUPLE so the class-index mapping is order-stable across
# runs and machines (a set's iteration order is not a contract). Index i here is the i-th OWLv2 text
# query at training time. The Roboflow export also carries a "floorplans" supercategory row, which
# is not an object class and is dropped by name.
FLOORPLAN_CLASSES: tuple[str, ...] = ("door", "window", "bathroom", "perimeter", "stairs")

# A box whose side rounds below this is annotation noise, not an object (mirrors
# eval/converters/floorplans._coco_bbox_to_bbox, so training and ground truth agree).
_MIN_SIDE_PX = 1


class FinetuneConfig(BaseModel):
    """Every knob of the OWLv2 fine-tuning recipe, frozen and self-describing.

    Frozen + ``extra="forbid"`` for the same reason every method config in this repo is: the
    resolved config is written verbatim into ``train_log.json`` beside the checkpoint, so the run
    that produced a reported number can be reconstructed exactly. Each ``description`` is the text
    ``--help`` and the log line show, so it is written here once.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    seed: int = Field(
        default=0,
        ge=0,
        description=(
            "Seeds torch and the numpy epoch shuffler (np.random.default_rng). The repo's "
            "reproducibility rule: same seed => identical per-epoch losses."
        ),
    )
    epochs: int = Field(default=8, ge=1, description="Passes over the training split.")
    batch_size: int = Field(
        default=2,
        ge=1,
        description=(
            "Images per forward pass. OWLv2 runs at a fixed 960x960 with 3600 patches, so memory "
            "grows fast; raise --grad-accum rather than this on a small card."
        ),
    )
    lr_head: float = Field(
        default=1e-4,
        gt=0.0,
        description="AdamW learning rate for the always-trained heads (box/class/objectness).",
    )
    lr_backbone: float = Field(
        default=1e-5,
        gt=0.0,
        description=(
            "AdamW learning rate for any unfrozen vision-encoder blocks. An order of magnitude "
            "below lr_head, the standard detection-fine-tuning ratio."
        ),
    )
    unfreeze_last_n: int = Field(
        default=0,
        ge=0,
        description=(
            "Additionally unfreeze the last N vision-encoder blocks at lr_backbone. 0 = the "
            "heads-only arm (arm A)."
        ),
    )
    unfreeze_all: bool = Field(
        default=False,
        description=(
            "Unfreeze the whole vision tower at lr_backbone (arm B, the stretch comparison). The "
            "text tower stays frozen in EVERY arm -- it is not part of the exported graph."
        ),
    )
    warmup_frac: float = Field(
        default=0.1,
        ge=0.0,
        lt=1.0,
        description=(
            "Fraction of the total optimizer steps spent linearly warming the learning rate up "
            "before the cosine decay begins. 0 = no warmup (straight into the cosine)."
        ),
    )
    weight_decay: float = Field(default=1e-4, ge=0.0, description="AdamW weight decay.")
    grad_clip: float = Field(
        default=1.0,
        gt=0.0,
        description="Global grad-norm clip applied before every optimizer step.",
    )
    grad_accum: int = Field(
        default=1,
        ge=1,
        description="Micro-batches accumulated per optimizer step (raises the effective batch).",
    )
    class_cost: float = Field(
        default=1.0,
        ge=0.0,
        description="Hungarian matching cost weight on the classification term (OWL-ViT: 1).",
    )
    bbox_cost: float = Field(
        default=5.0,
        ge=0.0,
        description="Hungarian matching cost weight on the L1 box term (OWL-ViT: 5).",
    )
    giou_cost: float = Field(
        default=2.0,
        ge=0.0,
        description="Hungarian matching cost weight on the GIoU term (OWL-ViT: 2).",
    )
    focal_alpha: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        description="Sigmoid-focal alpha for the classification loss (Deformable-DETR default).",
    )
    focal_gamma: float = Field(
        default=2.0, ge=0.0, description="Sigmoid-focal gamma for the classification loss."
    )
    w_class: float = Field(
        default=1.0, ge=0.0, description="Weight of the classification term in the total loss."
    )
    w_bbox: float = Field(
        default=5.0, ge=0.0, description="Weight of the L1 box term in the total loss."
    )
    w_giou: float = Field(
        default=2.0, ge=0.0, description="Weight of the GIoU term in the total loss."
    )
    loss_mode: Literal["focal", "contrastive", "both"] = Field(
        default="focal",
        description=(
            "Which classification-side objective trains the embedding. focal (the DEFAULT, and "
            "the already-measured recipe) = sigmoid focal loss over the text-conditioned logits. "
            "contrastive = a supervised-contrastive loss over class_embeds, which is the space "
            "owlv2-oneshot actually scores in, replacing the focal term. both = the sum of the "
            "two. loss_bbox and loss_giou stay in the total loss in EVERY mode -- the box head "
            "still has to be trained for the exported graph to produce usable boxes."
        ),
    )
    supcon_temperature: float = Field(
        default=0.07,
        gt=0.0,
        description=(
            "SupCon temperature tau (Khosla et al. 2020's headline value, and the SimCLR/MoCo "
            "convention). It divides the cosine similarities, so lower = sharper weighting of the "
            "hardest negatives. Only read in contrastive/both mode."
        ),
    )
    w_contrast: float = Field(
        default=1.0,
        ge=0.0,
        description=(
            "Weight of the supervised-contrastive term in the total loss (contrastive/both mode)."
        ),
    )
    supcon_background_negatives: int = Field(
        default=64,
        ge=0,
        description=(
            "Background patches sampled per image as DENOMINATOR-ONLY negatives: grid cells whose "
            "normalized centre lies in no ground-truth box. They are never anchors and never "
            "positives. This is load-bearing rather than a refinement -- owlv2-oneshot's measured "
            "floor-plan failure is low precision at high recall, i.e. background scoring too high, "
            "which an anchor-only SupCon cannot address. 0 disables them."
        ),
    )
    supcon_crop_context: bool = Field(
        default=False,
        description=(
            "Quick task 260808-dla: also build ONE crop-context SupCon anchor per training image "
            "(D-dla-02), by cropping a ground-truth box's RAW scene pixels and running it through "
            "the crop-context forward pass -- the same query-encoding path owlv2-oneshot's "
            "inference runs for its exemplar crop. Opt-in and layered on contrastive/both "
            "(D-dla-03): default False leaves the already-measured `focal` and flag-off "
            "`contrastive` arms byte-identical to before this flag existed. Ignored in focal mode."
        ),
    )
    supcon_query_iou_frac: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description=(
            "The crop-context anchor's query-patch selection threshold (D-dla-05), passed to "
            "select_query_patch_index exactly as Owlv2OneshotConfig.query_iou_frac is at "
            "inference. Pinned equal to that field's default so training selects the query patch "
            "the way inference does; only read when supcon_crop_context is True."
        ),
    )
    max_steps: int | None = Field(
        default=None,
        ge=1,
        description="Stop after this many optimizer steps (smoke tests / the tracer). None = full.",
    )
    limit_images: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Use only the first N images of each split, in the deterministic file_name order. "
            "Applies to BOTH train and val so a smoke run stays a smoke run. None = all."
        ),
    )


@dataclass(frozen=True, eq=False)
class ImageTargets:
    """One image's detection targets in the shape ``ImageLoss``/``HungarianMatcher`` expect.

    ``eq=False`` because the fields are NumPy arrays (element-wise ``==`` has no single truth
    value); these are never compared for equality.

    Attributes:
        image_id: The COCO ``images[].id``, kept so a target can be traced back to its annotation.
        file_name: The scene PNG's name, relative to the split directory.
        boxes: ``(n, 4)`` float32 ``(cx, cy, w, h)`` normalized to ``[0, 1]`` over the
            **padded-square** side ``max(H, W)`` -- see the module docstring.
        class_labels: ``(n,)`` int64 indices into :data:`FLOORPLAN_CLASSES`.
    """

    image_id: int
    file_name: str
    boxes: npt.NDArray[np.float32]
    class_labels: npt.NDArray[np.int64]


def _class_index_by_category_id(
    categories: Sequence[Mapping[str, Any]],
    classes: Sequence[str],
) -> dict[int, int]:
    """Map COCO ``category_id`` -> index into ``classes``, matched by NAME.

    Matching by name, not by id, is deliberate and mirrors ``eval/converters/floorplans.py``: the
    Roboflow export's numeric ids are an artefact of the export (``door`` is id 2 there, and a
    re-export could renumber them), while the names are the stable contract. Categories whose name
    is not in ``classes`` -- notably the "floorplans" supercategory row -- are simply absent from
    the returned map, so their annotations are skipped.
    """
    by_name = {name: index for index, name in enumerate(classes)}
    return {
        int(category["id"]): by_name[str(category["name"])]
        for category in categories
        if str(category["name"]) in by_name
    }


def coco_to_owlv2_targets(
    coco: Mapping[str, Any],
    *,
    classes: Sequence[str] = FLOORPLAN_CLASSES,
) -> list[ImageTargets]:
    """Convert one COCO split dict to OWLv2 training targets, sorted by ``file_name``.

    The one boundary conversion in the training path: COCO ``bbox`` ``[x, y, w, h]`` in float
    pixels -> ``(cx, cy, w, h)`` normalized by the **padded-square side** ``max(H, W)`` (see the
    module docstring for why that denominator and not ``(W, H)``).

    Images with no surviving annotation are dropped from the result rather than emitted with an
    empty box array: a zero-box target contributes nothing to a Hungarian match and would only make
    ``num_boxes`` normalization noisier.

    The result is sorted by ``file_name`` so the batching order is a property of the data, not of
    the JSON's insertion order -- one of the four things the repo's reproducibility rule actually
    pins (D-11).
    """
    index_by_category = _class_index_by_category_id(coco.get("categories", []), classes)
    images = {int(image["id"]): image for image in coco.get("images", [])}

    boxes_by_image: dict[int, list[list[float]]] = {}
    labels_by_image: dict[int, list[int]] = {}
    dropped_degenerate = 0
    dropped_out_of_range = 0
    dropped_unknown_class = 0

    for annotation in coco.get("annotations", []):
        category_id = int(annotation["category_id"])
        if category_id not in index_by_category:
            dropped_unknown_class += 1
            continue
        image = images.get(int(annotation["image_id"]))
        if image is None:  # an annotation pointing at no image is corrupt input, not a target
            dropped_out_of_range += 1
            continue

        width, height = float(image["width"]), float(image["height"])
        x, y, w, h = (float(v) for v in annotation["bbox"])

        # Guard 1: degenerate after rounding -- the same rule the GT converter applies.
        if round(w) < _MIN_SIDE_PX or round(h) < _MIN_SIDE_PX:
            dropped_degenerate += 1
            continue
        # Guard 2: outside the image. DROPPED, never clamped -- a clamp would turn a bad
        # annotation into a plausible-looking training target (threat T-8zy-04).
        if x < 0.0 or y < 0.0 or x + w > width or y + h > height:
            dropped_out_of_range += 1
            continue

        # The normalization denominator: the PADDED-SQUARE side, because OWLv2 pads bottom-right.
        side = max(width, height)
        boxes_by_image.setdefault(int(image["id"]), []).append(
            [(x + w / 2.0) / side, (y + h / 2.0) / side, w / side, h / side]
        )
        labels_by_image.setdefault(int(image["id"]), []).append(index_by_category[category_id])

    targets = [
        ImageTargets(
            image_id=image_id,
            file_name=str(images[image_id]["file_name"]),
            boxes=np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
            class_labels=np.asarray(labels_by_image[image_id], dtype=np.int64),
        )
        for image_id, boxes in boxes_by_image.items()
    ]
    targets.sort(key=lambda target: target.file_name)

    logger.info(
        "coco_to_owlv2_targets: {} image(s), {} box(es) over classes {}; dropped "
        "{} degenerate, {} out-of-range, {} unknown-class annotation(s)",
        len(targets),
        sum(int(target.boxes.shape[0]) for target in targets),
        tuple(classes),
        dropped_degenerate,
        dropped_out_of_range,
        dropped_unknown_class,
    )
    return targets


def deterministic_batches(
    count: int,
    batch_size: int,
    rng: np.random.Generator,
) -> Iterator[list[int]]:
    """Yield shuffled index batches for one epoch, from an explicitly seeded numpy Generator.

    Split out of the training loop so the ordering -- the part that decides whether two same-seed
    runs produce the same losses -- is covered by ``pixi run test`` without torch. The shuffle is
    ``rng.permutation``: never ``random.shuffle`` (unseeded global state) and never
    ``cv2.setRNGSeed`` (a no-op for anything that matters here, D-11).

    The final batch is short rather than dropped -- with 197 training images, dropping it would
    silently discard up to ``batch_size - 1`` of them every epoch.
    """
    order = rng.permutation(count)
    for start in range(0, count, batch_size):
        yield [int(index) for index in order[start : start + batch_size]]


def cosine_warmup_factor(
    step: int,
    total_steps: int,
    warmup_steps: int,
    *,
    min_factor: float = 0.0,
) -> float:
    """Learning-rate multiplier at 0-based optimizer ``step``: linear warmup, then cosine decay.

    Torch-free and pure, so the shape of the schedule is gated by ``pixi run test`` rather than
    only observable as a column in a training log. ``scripts/finetune_owlv2.py`` hands this straight
    to ``torch.optim.lr_scheduler.LambdaLR``, which multiplies each param group's base ``lr`` by the
    returned factor -- so the two-group (head / backbone) split keeps its ratio at every step.

    Warmup counts from 1, not 0: at ``step == 0`` with ``warmup_steps == 4`` the factor is ``0.25``,
    not ``0.0``. Starting at exactly zero would make the first optimizer step a guaranteed no-op,
    which on a short fine-tune (a few hundred steps) is a measurable waste rather than a rounding
    detail. The two branches meet continuously at ``step == warmup_steps`` (both give ``1.0``).

    Args:
        step: 0-based optimizer step.
        total_steps: Total optimizer steps in the run; the cosine reaches ``min_factor`` at the end.
        warmup_steps: Steps of linear warmup. 0 disables warmup.
        min_factor: Floor the cosine decays to, as a fraction of the base learning rate.

    Returns:
        A multiplier in ``[min_factor, 1.0]``.

    Raises:
        ValueError: If ``step`` is negative, ``total_steps`` is not positive, ``warmup_steps`` is
            negative or not below ``total_steps``, or ``min_factor`` is outside ``[0, 1]``.
    """
    if step < 0:
        raise ValueError(f"step must be >= 0, got {step}")
    if total_steps <= 0:
        raise ValueError(f"total_steps must be >= 1, got {total_steps}")
    if warmup_steps < 0 or warmup_steps >= total_steps:
        raise ValueError(f"warmup_steps must be in [0, {total_steps}), got {warmup_steps}")
    if not 0.0 <= min_factor <= 1.0:
        raise ValueError(f"min_factor must be in [0, 1], got {min_factor}")

    if warmup_steps > 0 and step < warmup_steps:
        return (step + 1) / warmup_steps

    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    return min_factor + (1.0 - min_factor) * 0.5 * (1.0 + math.cos(math.pi * progress))


def warmup_steps_for(total_steps: int, warmup_frac: float) -> int:
    """Resolve ``warmup_frac`` to a step count :func:`cosine_warmup_factor` will accept.

    Rounds to the nearest step and clamps into ``[0, total_steps - 1]``, so a smoke run with a
    handful of steps cannot end up spending its whole budget in warmup (which would make the
    schedule silently constant) and cannot produce an out-of-range value the schedule rejects.
    """
    if total_steps <= 0:
        raise ValueError(f"total_steps must be >= 1, got {total_steps}")
    return max(0, min(total_steps - 1, round(warmup_frac * total_steps)))
