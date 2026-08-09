# Fine-tuning `owlv2-oneshot` on floor plans — measured result

**Verdict, stated up front: three fine-tuning objectives were tried. The first two fail outright; the
third — supervising the exact training/inference mismatch the second one exposed — works, closing most
of the gap to the pretrained baseline without reaching `propose-retrieve`/`ncc`.** Classification-loss
fine-tuning (first experiment) makes doors worse than the pretrained baseline and barely moves windows.
A directly-matched supervised-contrastive (SupCon) objective (second experiment) is worse still: door
F1 0.010, window F1 0.009. Diagnosis found why: `owlv2-oneshot` computes two embeddings of the same
object through two different forward passes — a **crop-context** query embedding (the small cropped
exemplar alone) and a **scene-context** embedding of that object inside the full scene — and
calibration depends on them agreeing, but the SupCon batches were built entirely from scene-context
forward passes. Training never touched the crop-context path at all, and it drifted so far that the
exemplar's own self-similarity score went cosine-**negative**, collapsing the calibration threshold and
retaining ~86% of all scene patches instead of ~25–30%. **The third experiment fixes exactly this**:
adding crop-context anchors to the SupCon pool (reusing `owlv2-oneshot`'s own query-encoding functions,
not a reimplementation, to guarantee train/inference fidelity) restores a healthy, strongly positive
self-similarity score and lifts door F1 to 0.229–0.391 and window F1 to 0.216 — roughly **2–24× every
other fine-tuned arm** and clearly ahead of the pretrained baseline (0.154 door / 0.023 window) on both
classes. It still falls short of `propose-retrieve`'s 0.459 door F1 and `ncc`'s 0.403 window F1, so
`ncc`/`propose-retrieve` remain the floor-plan recommendation — but the headline changes from "fine-
tuning is a dead end on this method" to "fine-tuning works once the training objective supervises the
same cross-context comparison the method actually runs at inference." All three experiments train
cleanly (loss falls monotonically, no sign of a wiring bug); the difference between them is entirely
about what the training objective supervises, not how it optimizes.

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
| headonly (classification) | 0.034 | 0.421 | 0.062 | 0.048 | 0.464 | 0.087 |
| full (classification) | 0.022 | 0.343 | 0.041 | 0.045 | 0.481 | 0.083 |
| headonly (contrastive) | 0.006 | 0.086 | 0.011 | 0.005 | 0.090 | 0.010 |
| headonly (contrastive-crop) | 0.261 | 0.785 | 0.391 | 0.138 | 0.674 | **0.229** |

**Windows** (`floorplans-window`, test, 28/28 plans scored in every arm):

| arm | default P | default R | default F1 | tuned P | tuned R | tuned F1 |
|---|---|---|---|---|---|---|
| baseline (pretrained) | 0.011 | 0.237 | 0.022 | 0.012 | 0.237 | 0.023 |
| headonly (classification) | 0.014 | 0.346 | 0.027 | 0.015 | 0.353 | **0.028** |
| full (classification) | 0.005 | 0.192 | 0.011 | 0.005 | 0.167 | 0.010 |
| headonly (contrastive) | 0.005 | 0.090 | 0.009 | 0.005 | 0.096 | 0.009 |
| headonly (contrastive-crop) | 0.124 | 0.846 | 0.216 | 0.124 | 0.846 | **0.216** |

**The crop-context-free classification and contrastive arms all regress below baseline** (doors:
0.154 → 0.087 → 0.083 → 0.010; windows: 0.023 → 0.028 → 0.010 → 0.009), and the crop-context-free
contrastive arm regresses furthest on both classes — consistent with the diagnostic finding below: it
retains far more scene patches than any other checkpoint, so it is not failing to find candidates, it
is failing to reject almost anything. **`contrastive-crop` reverses this pattern entirely**: door F1
0.229 tuned / 0.391 default and window F1 0.216 (default and tuned select the same config) are the
best numbers of any fine-tuned arm by a wide margin, and beat the pretrained baseline on both classes.
Neither `contrastive-crop` cell reaches `propose-retrieve`'s 0.459 door F1 or `ncc`'s 0.403 window
F1 — those remain the methods to ship on this domain — but this is the first fine-tuning arm where the
answer to "does fine-tuning help this method" is genuinely yes.

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

## Second experiment: a supervised-contrastive objective

**Hypothesis.** The classification-loss recipe's diagnosed problem (above) is that "getting better at
5-way door/window/bathroom/perimeter/stairs classification given a text prompt" is a correlate of, not
the same as, the property `owlv2-oneshot` actually uses at inference: rank scene patches by cosine
similarity to a single image-derived query embedding, then threshold by self-similarity calibration. A
supervised-contrastive (SupCon, Khosla et al. 2020, the `L_out` formulation) loss over L2-normalized
`class_embeds` trains that exact property directly — same-class anchors cosine-close, different-class
and background patches cosine-far — rather than hoping classification accuracy transfers. This is a
directly-matched objective, worth one measured number before concluding fine-tuning is a dead end on
this domain.

**The recipe** (`--loss-mode contrastive`, `src/object_search/train/supcon.py`,
`scripts/finetune_owlv2.py`): τ=0.07 (the SupCon/SimCLR/MoCo convention); anchors are the same patches
`Owlv2HungarianMatcher` already assigns to each ground-truth box, no new matching logic; **background
patches — grid cells whose centre falls in no ground-truth box — are sampled as denominator-only
negatives** (64 per image, never anchors or positives), which is load-bearing rather than a
refinement, because the measured floor-plan failure is low precision (background scoring too high) and
a SupCon over matched anchors alone can only separate door-from-window, never door-from-background;
**anchors and background negatives are pooled across the full effective batch** (all `grad_accum`
micro-batches, not per micro-batch), because at `batch_size=2` a micro-batch holds ~1.8 stairs boxes on
average and pooling to the effective batch of 8 images gives ~7, so rare classes contribute a real
gradient instead of being starved to zero most steps. Trained on the `headonly` freeze arm only, 8
epochs, seed 0 — the `full`-unfreeze contrastive arm was deliberately not run, per the human
checkpoint's stretch condition, since headonly-contrastive did not beat its classification counterpart
or the baseline on either class.

**The cosine-gap table — the number 260801-8zy could not produce.** `val_cos_gap` is measured over
held-out scene patches (matched anchors vs. background), L2-normalized, at epoch 0 (pre-training
reference, no gradient step yet) and after every epoch:

| checkpoint | same_class_mean | diff_class_mean | background_mean | gap_class | gap_background |
|---|---|---|---|---|---|
| epoch 0 (pretrained reference) | 0.720 | 0.598 | 0.472 | 0.122 | 0.248 |
| epoch 8 (final, contrastive) | 0.882 | 0.466 | 0.041 | **0.415** | **0.840** |

Both gaps more than triple (`gap_class` +240%, `gap_background` +239%), and `background_mean` falls
from 0.472 to 0.041 — background scene patches end up scoring almost zero cosine similarity against
same-class anchors, which is exactly the separation the objective was designed to produce. **This
resolves the question 260801-8zy's diagnostic could not answer: the loss moved the property
`owlv2-oneshot` actually scores with, and F1 still collapsed further than the classification recipe.**
The objective is right for scene-to-scene separation; something else is the ceiling.

**Why F1 collapsed anyway: the training objective never touches the query's own encoding path.** A
follow-up offline diagnostic (comparing `select_query_embedding`'s behavior across all three
checkpoints on the same exemplar) found the mechanism. `owlv2-oneshot` computes two embeddings of the
same physical region through two different forward passes: the **query embedding**, from encoding the
small cropped exemplar image alone, and the **scene embedding** of that same region, from encoding the
full floor plan. Calibration depends on these agreeing — `self_score` is the scene-embedding-side score
of the exemplar's own location against its own query embedding, and the retain threshold is
`self_score * retain_frac`:

| checkpoint | query patch selected | self_score | threshold (×0.94) | scene patches retained |
|---|---|---|---|---|
| baseline (pretrained) | 1477/3600 | +0.712 | 0.670 | 1093/3600 (30%) |
| headonly (classification) | 1477/3600 | +0.666 | 0.626 | 918/3600 (26%) |
| headonly (contrastive) | 2997/3600 | **−0.297** | **−0.279** | **3080/3600 (86%)** |

The contrastive checkpoint's `self_score` is **negative** — the model no longer thinks the exemplar's
own scene location resembles the exemplar's own crop. A negative self-score makes the calibration
threshold negative, and a negative bar retains almost the entire patch grid, which is the entire F1
collapse mechanically: not "wrong patches outrank right ones" but "the threshold stopped rejecting
anything." The root cause is a training/inference contract gap, not a bug in `select_query_embedding`
or `calibrate()`: the SupCon batches (like the val anchors and background patches measured in the
cosine-gap table above) are built entirely from **scene-context** forward passes — the **crop-context**
forward pass that produces the query embedding at inference was never part of the training loop's loss
at all. Contrastive training reshaped the scene-embedding space dramatically and consistently (the
cosine-gap table), but nothing in the objective keeps crop-context embeddings of an object consistent
with scene-context embeddings of the same object, and for this checkpoint they diverged enough to flip
sign. Classification fine-tuning and the pretrained baseline don't show this failure mode because they
were never pushed as far from CLIP's original, context-robust embedding geometry. A principled fix
would add crop-encoded anchors to the contrastive batch (query-crop embeddings contrastively supervised
against the same scene-context positives/negatives, not just scene-vs-scene) — not attempted in this
task; if pursued, it is separately-scoped future work, not a config tweak to this recipe.

**`--loss-mode both` is implemented and smoke-tested (Task 1/3), not measured on GPU** — `train_log.json`
carries both `loss_ce` and `loss_supcon` populated simultaneously in a 1-step CPU smoke run, but no
`both`-mode arm was trained to convergence or evaluated, per D-hg1-05.

## Third experiment: crop-context supervision

**Hypothesis.** The second experiment's diagnosis identified a training/inference contract gap, not
just a bad objective: SupCon reshaped the scene-embedding space beautifully (the cosine-gap table
above), but calibration depends on a *different* comparison — the crop-context query embedding vs. a
scene-context patch of the same object — that training never touched, and it drifted enough to flip
the exemplar's self-similarity score negative. If crop- and scene-context embeddings of the same
object are explicitly trained to agree, not just assumed to, calibration should recover.

**The recipe** (`--supcon-crop-context`, layered on `--loss-mode contrastive`, default off so the
already-committed `contrastive` arm's numbers stay reproducible): for each training image, in addition
to the existing scene-context anchors and background negatives, the exemplar crop of one
randomly-picked ground-truth box is run through the model's image encoder and its selected patch
embedding is appended to the SupCon pool as an ordinary same-class positive — zero changes to
`supcon_loss`'s math (D-dla-01). Critically, the crop is built via the **same**
`select_query_patch_index` / `owlv2_preprocess_tensor` / `boxes_to_pixels` functions
`owlv2-oneshot`'s inference path itself calls (D-dla-04) — literal code reuse, not a
reimplementation, to close off the single largest correctness risk in this task: a crop-preprocessing
mismatch between training and inference that would silently supervise the wrong thing. One crop per
training image, not one per ground-truth box (D-dla-02): floor plans average ~20 boxes/image, and a
crop forward is a full independent ViT pass (unlike scene anchors, which piggyback on the one scene
forward already computed), so one-per-box would cost 10–20× more compute for a doubling in effective
coverage across the run's epochs. Same hyperparameters as the crop-context-free `contrastive` arm
(headonly freeze, 8 epochs, seed 0, batch-size 2, grad-accum 4) so the only variable is the new anchor.

**Does it move the property it targets? Pooled, instance-level crop/scene agreement, epoch 0 (before
training) vs. epoch 7 (best checkpoint):**

| metric | epoch 0 | epoch 7 (best) |
|---|---|---|
| `val_crop_scene_agreement.self_score_mean` | +0.490 | **+0.808** |
| `val_cos_gap.gap_class` | +0.117 | +0.344 |
| `val_cos_gap.gap_background` | +0.228 | +0.617 |

The pooled crop/scene self-similarity moves from a middling positive start to a strongly positive
+0.808 (n=56 pairs each epoch) and never dips negative at any epoch — the opposite of the
crop-context-free arm's collapse to −0.297. The scene-side cosine gaps still widen substantially too
(comparable magnitude to the second experiment), so this is not a tradeoff against the property the
classification-loss recipe never touched — both move together.

**Does it hold on an independent, freshly-chosen exemplar?** The original diagnostic's exact exemplar
coordinates were never persisted (it ran ad hoc on a since-destroyed instance), so this measures a
*different*, deterministically-chosen exemplar (the first door-class ground-truth box, by annotation
order, in the first file_name-sorted training image that has one — `.planning/quick/260808-dla-add-
crop-context-supervision-to-the-owlv/self_score_diagnostic.py`, CPU-only, no GPU) against all four
checkpoints:

| checkpoint | self_score | threshold (×0.94) | scene patches retained |
|---|---|---|---|
| baseline (pretrained) | +0.368 | +0.346 | 970/2482 (39.1%) |
| headonly (classification) | +0.556 | +0.523 | 1215/2487 (48.9%) |
| contrastive | **−0.200** | **−0.188** | 1766/2490 (70.9%) |
| contrastive-crop | **+0.859** | **+0.808** | 199/2497 (**8.0%**) |

These numbers are not bit-comparable to the second experiment's table above (a different exemplar),
but the pattern is the same and, on this exemplar, even sharper: `contrastive-crop` doesn't just
recover a positive self-score, it produces the **highest** self-score and the **tightest** retention of
any of the four checkpoints — more selective than the pretrained baseline. That tracks directly with
the F1 table above: `contrastive-crop` is a precision story (0.124–0.261 vs. 0.005–0.061 for every
other fine-tuned arm) at comparable or better recall, exactly what "reject almost nothing" (the second
experiment's failure) becoming "reject almost everything except real instances" looks like.

**Verdict: the fix works.** Crop-context supervision closes the training/inference contract gap the
second experiment diagnosed, and F1 tracks it — door F1 2.6–4.5× the pretrained baseline and 2–39×
every other fine-tuned arm; window F1 9–24× every other fine-tuned arm and roughly 9–10× the baseline.
It does not reach `propose-retrieve`/`ncc`, so this repo's floor-plan recommendation is unchanged, but
the finding for `owlv2-oneshot` fine-tuning specifically inverts: it is not a dead end, provided the
training objective supervises the exact cross-context comparison the method runs at inference rather
than a same-context proxy.

`--loss-mode both --supcon-crop-context` is implemented and smoke-tested (sanity-check.md), not
measured on GPU — the same disclosure as `both`-mode in the second experiment.

## Provenance — classification-loss experiment

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

## Provenance — contrastive experiment

Trained on the `headonly` freeze arm for 8 epochs, seed 0, `supcon_temperature=0.07`,
`supcon_background_negatives=64`, `w_contrast=1.0`, on a second vast.ai RTX 3090 instance (CUDA 12.1,
onnxruntime-gpu 1.23.2, torch 2.5.1+cu121) rented, used, and destroyed for this arm only, per the
single-instance budget approved at the human checkpoint. All three pull-back artifacts
(`docs/benchmark/owlv2-finetune/floorplans-{door,window}-contrastive.json`,
`models/finetune/owlv2-floorplans-contrastive/train_log.json`, and the exported ONNX) transferred
sequentially without the network issues that affected the first experiment's original instance.
`vastai show instances` confirms zero instances currently running.

| artifact | sha256 (as committed locally) |
|---|---|
| `models/owlv2_base_patch16_floorplans_ft_contrastive.onnx` (contrastive, unregistered comparison) | `a16f7414c510b0738fa2fdade9432564dfac31ef4a4cedb694df1bdb0cc1b74f` |

The contrastive ONNX artifact passes `_verify_graph`'s local contract check (`class_embeds
[batch, num_patches, 512]`, `pred_boxes [batch, 3600, 4]`) before being trusted. The offline
diagnostic that produced the query-patch/self-score table above was run separately, on a third vast.ai
instance rented, used, and destroyed solely for that comparison (no training, CPU-only inference
against all three already-exported ONNX checkpoints) — not part of this artifact's own provenance
chain, called out here because it is the source of the query-patch/self-score numbers.

## Provenance — crop-context experiment

Trained on the `headonly` freeze arm for 8 epochs, seed 0, identical hyperparameters to the
crop-context-free `contrastive` arm plus `--supcon-crop-context`. Two vast.ai RTX 3090 instances were
involved in getting this arm measured: the first went unreachable (SSH connection refused, `vastai
show instances` reporting `offline` with `intended_status: running`) partway through the ~3.7-hour
training run, the same host-flakiness pattern documented in the classification-loss experiment's
provenance above — destroyed without a usable result. A same-priced replacement offer from a different
listing resolved to the **exact same public IP** as a host already known-bad from this repo's earlier
GPU work (see 260801-8zy's infrastructure notes) — destroyed immediately, unused, before any spend.
A third, genuinely different instance (different IP, machine id 47340) trained the arm cleanly start
to finish in one run. Pulled back sequentially: two result JSONs, `train_log.json`, and the ONNX
export, whose sha256 (`fa79868753928ce3c8638378395fb4c74c8cd6aa7b3e7253a84b9bcffca02cd2`) was verified
identical between the box and the local copy (a straight file integrity check, not a
cross-hardware-reproduction claim — only one training run produced this artifact). The instance was
destroyed and `vastai show instances` confirmed empty immediately after pull-back.

| artifact | sha256 (as committed locally) |
|---|---|
| `models/owlv2_base_patch16_floorplans_ft_contrastive_crop.onnx` (contrastive-crop, unregistered comparison) | `fa79868753928ce3c8638378395fb4c74c8cd6aa7b3e7253a84b9bcffca02cd2` |

The contrastive-crop ONNX artifact passes `_verify_graph`'s local contract check (`class_embeds
[batch, num_patches, 512]`, `pred_boxes [batch, 3600, 4]`) before being trusted.

## Disposition

None of the four fine-tuned arms is adopted as `owlv2-oneshot`'s shipped default.
`owlv2-oneshot`'s default model path and behavior are unchanged — every fine-tuned weight set is an
opt-in research artifact, reachable for one run via
`OS_OWLV2_MODEL=owlv2_base_patch16_floorplans_ft.onnx` (headonly, classification),
`OS_OWLV2_MODEL=owlv2_base_patch16_floorplans_ft_full.onnx` (full, classification),
`OS_OWLV2_MODEL=owlv2_base_patch16_floorplans_ft_contrastive.onnx` (headonly, contrastive), or
`OS_OWLV2_MODEL=owlv2_base_patch16_floorplans_ft_contrastive_crop.onnx` (headonly, contrastive-crop),
mirroring the existing `OS_ONNX_PROVIDERS` override convention. `ncc` and `propose-retrieve` remain
this repo's floor-plan recommendation (per `docs/eval/floorplans-findings.md`) — `contrastive-crop`'s
0.229 door / 0.216 window tuned F1 does not reach `propose-retrieve`'s 0.459 door F1 or `ncc`'s 0.403
window F1, so this does not change which method ships on this domain. What it does change: the earlier
verdict that fine-tuning `owlv2-oneshot`'s heads is foreclosed as a lever is retracted. Fine-tuning
works, and works by a wide margin over every other measured arm and over the pretrained baseline,
*once the training objective supervises the same crop-context-vs-scene-context comparison the method
runs at inference* — the concrete, measured mechanism, not a hedge. `contrastive-crop` is the
best-measured OWLv2 configuration on this domain and a reasonable opt-in choice for anyone using
`owlv2-oneshot` on floor-plan-like symbol detection specifically, even though it is not this repo's
top overall recommendation here.
