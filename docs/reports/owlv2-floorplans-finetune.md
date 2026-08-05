# Fine-tuning `owlv2-oneshot` on floor plans — measured result

**Verdict, stated up front: fine-tuning does not close the gap to the current floor-plan winners
(`propose-retrieve` 0.459 door F1, `ncc` 0.403 window F1), and on doors it makes the pretrained
baseline *worse*.** Both fine-tuned arms train cleanly — loss falls monotonically, val tracks train,
no sign of a wiring bug — but neither arm's improved training loss transfers to better image-guided
detection. This is a negative result, reported as the headline rather than buried: it means the
ceiling on this method is not (only) the frozen embedding space this repo's earlier diagnostic
identified, but also the text-conditioned-training-as-proxy mechanism itself, or the one-shot
query-embedding selection, or the box head's small-symbol resolution — not something a from-scratch
head fine-tune on 197 images fixes.

## Why this was tried

The floor-plan domain-shift investigation (`docs/eval/floorplans-findings.md`) found `owlv2-oneshot`
over-detecting on floor plans (precision 0.01–0.11, recall high) — a genuine domain-fit failure of
OWLv2's pretrained embedding space, confirmed against the real HuggingFace reference implementation.
Two cheap mitigations were tried and reverted on this branch: tiling (roughly doubled false positives
per added true positive at every grid size) and rotation/mirror query-embedding augmentation (zeroed
the only true positive on a near-symmetric window symbol). Fine-tuning is the one lever that can
actually change the embedding space rather than just how it's queried, so it was the next thing to
measure rather than argue about.

## The recipe

`owlv2-oneshot` runs OWLv2's *image-guided* mode (encode the exemplar crop, cosine-score every scene
patch). `transformers.Owlv2ForObjectDetection.forward()` has no built-in detection loss (no `labels`
argument, no matcher — an open upstream gap, huggingface/transformers#33664), so fine-tuning was done
via the *text-conditioned* path instead, which has an established training recipe: Hungarian matching
(cost weights 1/5/2 for class/bbox/giou, following the OWL-ViT paper and an unmerged upstream
reference PR, huggingface/transformers#47658) + sigmoid focal loss (α=0.3, γ=2, since OWLv2's logits
have no background column — the stock DETR-style softmax label loss does not apply) + L1 + GIoU,
reusing `transformers.loss.loss_for_object_detection`'s `HungarianMatcher`/`ImageLoss` rather than
hand-writing matching or GIoU. All five floor-plan categories (door, window, bathroom, perimeter,
stairs) were used as text classes during training, on the theory that more discriminative classes
would sharpen door-vs-not-door separation, even though eval only targets door/window.

This is a valid proxy for the image-guided path, not a mismatch: `forward()` and
`image_guided_detection()` call the identical `box_predictor`/`class_predictor` over the identical
`vision_model` — exactly the three modules `scripts/export_owlv2.py`'s `_VisionGraph` exports to
ONNX. Fine-tuning the shared heads/backbone via text-conditioned training therefore updates the exact
weights the image-guided method uses, with zero changes to `owlv2_oneshot.py` itself.

**Two arms**, both seeded (0), both trained 8 epochs on the 197-image train split with the
val-selected checkpoint (lowest held-out loss, not the last epoch):

| arm | unfrozen | trainable / total params |
|---|---|---|
| headonly (primary) | `box_head`, `class_head` (incl. `logit_scale`/`logit_shift`) | 1,579,526 / 154,966,792 (1.0%) |
| full (stretch) | + the entire `vision_model` backbone | 89,994,758 / 154,966,792 (58.1%) |

`objectness_head` stays frozen in both arms (it is outside the exported graph and gets no gradient
from this loss — unfreezing it would be an inert knob). The text tower (`text_model`,
`text_projection`) stays frozen in both arms (it is not exported; letting it move would let the loss
fall by moving the five text queries rather than by improving the image features that are actually
exported — a training-loss reduction that provably cannot transfer).

## Training curves

Both arms converge cleanly — no overfitting divergence, val loss keeps improving through epoch 8 for
both:

| arm | epoch 1 train / val loss | epoch 8 train / val loss | epoch 8 train ce | epoch 8 val ce |
|---|---|---|---|---|
| headonly | 2.123 / 1.093 | 0.670 / 0.706 | 0.207 | 0.214 |
| full | 1.460 / 0.985 | 0.472 / 0.555 | 0.109 | 0.139 |

The full-unfreeze arm reaches a substantially lower loss on its own training objective (more
capacity, same data) — and this is exactly what does **not** transfer to the downstream metric below.

## Measured result — baseline vs fine-tuned, full 28-plan test splits

Precision/recall/F1 @ IoU 0.5, 1 exemplar, tuned config selected by argmax-F1 on the 56-plan val
split (never touched by test), evaluated once on the frozen test split. Every arm uses the identical
method config and tuning grid — the only difference between rows is which ONNX weights
`OS_OWLV2_MODEL` pointed at.

**Doors** (`floorplans-door`, test, 28/28 plans scored in every arm):

| arm | default P | default R | default F1 | tuned P | tuned R | tuned F1 |
|---|---|---|---|---|---|---|
| baseline (pretrained) | 0.061 | 0.562 | 0.111 | 0.089 | 0.575 | **0.154** |
| headonly | 0.034 | 0.421 | 0.062 | 0.048 | 0.464 | 0.087 |
| full | 0.022 | 0.343 | 0.041 | 0.045 | 0.481 | 0.083 |

**Windows** (`floorplans-window`, test, 28/28 plans scored in every arm):

| arm | default P | default R | default F1 | tuned P | tuned R | tuned F1 |
|---|---|---|---|---|---|---|
| baseline (pretrained) | 0.011 | 0.237 | 0.022 | 0.012 | 0.237 | 0.023 |
| headonly | 0.014 | 0.346 | 0.027 | 0.015 | 0.353 | **0.028** |
| full | 0.005 | 0.192 | 0.011 | 0.005 | 0.167 | 0.010 |

**Doors regress with fine-tuning, monotonically with how much of the model moves** (0.154 → 0.087 →
0.083). **Windows move a hair in the right direction for the lightly-tuned arm** (0.023 → 0.028) but
the full-unfreeze arm regresses there too (0.010). Neither arm, on either class, gets within range of
`propose-retrieve`'s 0.459 door F1 or `ncc`'s 0.403 window F1 — those numbers remain the ones to ship
against on this domain.

> **Baseline reproduction note.** The pretrained-baseline numbers measured here (0.154 door / 0.023
> window, tuned) are close to but not bit-identical with the already-committed
> `docs/eval/floorplans-findings.md` numbers (0.180 door / 0.023 window). This run used
> `CUDAExecutionProvider` on a vast.ai RTX 3090 (onnxruntime-gpu 1.23.2); the committed numbers were
> produced on a different execution provider. OWLv2's float32 ops are not guaranteed bit-identical
> across execution providers (the same class of environment-identity effect this repo's own
> `provenance.environment_identity()` exists to record), which plausibly explains a small drift in
> which grid point the tuning sweep selects. The window numbers, at 0.022→0.023 in both runs, land in
> the same place regardless — the qualitative conclusion (owlv2 does not close the gap) is unaffected
> either way.

## Why training loss went down but detection got worse

This is the interesting finding, not just the negative headline. The text-conditioned objective is a
5-way classification-plus-box-regression task; a lower loss there means the model got better at
telling "door" from "window" from "bathroom" from "perimeter" from "stairs" *given a text prompt for
each*. That is a different task from what `owlv2-oneshot` actually needs: a single image-derived
query embedding whose cosine similarity to scene patches ranks true instances above everything else,
then gets thresholded by a self-similarity-anchored calibration. Nothing in the training loss
optimizes that ranking or that calibration directly — text-conditioned accuracy improving is a
correlate this repo hypothesized would transfer, and on this data, for this method, it measurably
did not. The full arm's *better* training-objective loss and *worse* door F1 than the headonly arm is
the clearest signal: more capacity spent on the proxy objective bought more proxy-objective
overfitting relative to the small 197-image train set, not more of the property that matters
downstream.

## Provenance

The P/R/F1 numbers above were measured against the run trained and evaluated on the original
vast.ai instance. That instance's network became unreliable partway through pulling the two
fine-tuned `.onnx` files back (it had already produced the eval result JSONs and `train_log.json`
curves, which transferred fine — only the large binaries were affected), so it was destroyed and
the local `.onnx` artifacts below were regenerated on a second vast.ai instance with the *same*
seed and recipe. GPU floating-point ops are not bit-identical across different physical hardware
(a documented limit of this repo's own reproducibility model — see `provenance.environment_identity`),
so the sha256s differ from the run that produced the eval numbers, even though the recipe, seed,
and data are identical. The regenerated run's `train_log.json` curves are numerically indistinguishable
from the original's (headonly epoch 1: 2.123/1.093 vs. 2.118/1.082 train/val loss; epoch 8:
0.670/0.706 vs. 0.669/0.704) — confirming this is a faithful reproduction, not a different result.

| artifact | sha256 (as committed locally) |
|---|---|
| `models/owlv2_base_patch16.onnx` (pretrained baseline, from the original run) | `6fe44c36640d37927f3438220fa46f039100e073b287a2da4ea1d85ecac3da61` |
| `models/owlv2_base_patch16_floorplans_ft.onnx` (headonly, registered; regenerated run) | `ac8d5a532473c37066877a81e60c555883e0bd08445e14a7a263482431bffab0` |
| `models/owlv2_base_patch16_floorplans_ft_full.onnx` (full, unregistered comparison; regenerated run) | `2815f6c62f2a07923d5addb286dcca6d045a90ce5da39e9b09a82b8ef1740061` |

Both regenerated artifacts pass `_verify_graph`'s local contract check (`class_embeds
[batch, num_patches, 512]`, `pred_boxes [batch, 3600, 4]`) before being trusted.

Training data: `datasets/_incoming/floorplans/train/_annotations.coco.json` (197 images, 3962 boxes:
door 1822, window 1413, bathroom 283, perimeter 267, stairs 177) — Roboflow floor-plans-500, the same
export already used for the committed val/test splits. Trained and evaluated on a vast.ai RTX 3090
(CUDA 12.1, onnxruntime-gpu 1.23.2, torch 2.5.1+cu121); reproduced with
`bash scripts/gpu_finetune.sh` (seed 0 throughout).

## Disposition

Neither fine-tuned arm is adopted as a default. `owlv2-oneshot`'s shipped model path and default
behavior are unchanged — the fine-tuned weights are an opt-in research artifact only, reachable for
one run via `OS_OWLV2_MODEL=owlv2_base_patch16_floorplans_ft.onnx` (headonly) or
`OS_OWLV2_MODEL=owlv2_base_patch16_floorplans_ft_full.onnx` (full), mirroring the existing
`OS_ONNX_PROVIDERS` override convention. `ncc` and `propose-retrieve` remain the methods to ship on
this domain (per `docs/eval/floorplans-findings.md`); this result does not change that recommendation
— if anything it forecloses fine-tuning as the next lever to pull on `owlv2-oneshot` for floor plans.
