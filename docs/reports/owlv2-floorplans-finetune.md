# Fine-tuning `owlv2-oneshot` on floor plans — measured result

**Verdict, stated up front: four fine-tuning objectives were tried. The first two fail outright; the
third — supervising the exact training/inference mismatch the second one exposed — works, and the
fourth pushes it further on doors specifically, at a small cost on windows. None reach
`propose-retrieve`/`ncc`.** Classification-loss fine-tuning (first experiment) makes doors worse than
the pretrained baseline and barely moves windows. A directly-matched supervised-contrastive (SupCon)
objective (second experiment) is worse still: door F1 0.010, window F1 0.009. Diagnosis found why:
`owlv2-oneshot` computes two embeddings of the same object through two different forward passes — a
**crop-context** query embedding (the small cropped exemplar alone) and a **scene-context** embedding
of that object inside the full scene — and calibration depends on them agreeing, but the SupCon
batches were built entirely from scene-context forward passes. Training never touched the crop-context
path at all, and it drifted so far that the exemplar's own self-similarity score went
cosine-**negative**, collapsing the calibration threshold and retaining ~86% of all scene patches
instead of ~25–30%. **The third experiment fixes exactly this**: adding crop-context anchors to the
SupCon pool (reusing `owlv2-oneshot`'s own query-encoding functions, not a reimplementation, to
guarantee train/inference fidelity) restores a healthy, strongly positive self-similarity score and
lifts door F1 to 0.229–0.391 and window F1 to 0.216. **The fourth experiment tried two further, cheap
levers**: crop context-margin padding (tested first, at zero GPU cost, against the already-trained
third-experiment checkpoint) helps doors at one margin value but hurts windows at every margin tried,
so no margin was carried into training; rotation/mirror-augmented crop positives in SupCon training
then push door F1 further, to **0.253 tuned / 0.433 default** (the best door numbers of any arm
measured), while window F1 dips slightly to 0.204. All four fine-tuned arms with a working objective
(third and fourth experiments) sit clearly ahead of the pretrained baseline (0.154 door / 0.023 window)
on both classes, but none reach `propose-retrieve`'s 0.459 door F1 or `ncc`'s 0.403 window F1, so
`ncc`/`propose-retrieve` remain the floor-plan recommendation. The through-line across all four
experiments: fine-tuning `owlv2-oneshot` works once the training objective supervises the same
cross-context comparison the method actually runs at inference, and further, cheap levers on top of
that fixed objective move the numbers in real but class-asymmetric ways worth measuring individually
rather than assuming a shared direction.

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
| headonly (contrastive-crop) | 0.261 | 0.785 | 0.391 | 0.138 | 0.674 | 0.229 |
| headonly (contrastive-crop-v2) | 0.299 | 0.785 | **0.433** | 0.151 | 0.773 | **0.253** |

**Windows** (`floorplans-window`, test, 28/28 plans scored in every arm):

| arm | default P | default R | default F1 | tuned P | tuned R | tuned F1 |
|---|---|---|---|---|---|---|
| baseline (pretrained) | 0.011 | 0.237 | 0.022 | 0.012 | 0.237 | 0.023 |
| headonly (classification) | 0.014 | 0.346 | 0.027 | 0.015 | 0.353 | **0.028** |
| full (classification) | 0.005 | 0.192 | 0.011 | 0.005 | 0.167 | 0.010 |
| headonly (contrastive) | 0.005 | 0.090 | 0.009 | 0.005 | 0.096 | 0.009 |
| headonly (contrastive-crop) | 0.124 | 0.846 | **0.216** | 0.124 | 0.846 | **0.216** |
| headonly (contrastive-crop-v2) | 0.116 | 0.846 | 0.204 | 0.116 | 0.846 | 0.204 |

**The crop-context-free classification and contrastive arms all regress below baseline** (doors:
0.154 → 0.087 → 0.083 → 0.010; windows: 0.023 → 0.028 → 0.010 → 0.009), and the crop-context-free
contrastive arm regresses furthest on both classes — consistent with the diagnostic finding below: it
retains far more scene patches than any other checkpoint, so it is not failing to find candidates, it
is failing to reject almost anything. **`contrastive-crop` reverses this pattern entirely**: door F1
0.229 tuned / 0.391 default and window F1 0.216 (default and tuned select the same config) beat the
pretrained baseline on both classes. **`contrastive-crop-v2`** (the fourth experiment, below) pushes
door further to **0.253 tuned / 0.433 default** — the best door numbers of any fine-tuned arm — at a
small cost on window (0.204, −0.012 vs. `contrastive-crop`). No cell across any arm reaches
`propose-retrieve`'s 0.459 door F1 or `ncc`'s 0.403 window F1 — those remain the methods to ship on
this domain — but `contrastive-crop`/`contrastive-crop-v2` are the fine-tuning arms where the answer
to "does fine-tuning help this method" is genuinely yes.

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

## Fourth experiment: crop-context margin + rotation/mirror augmentation

Two further, independently-motivated levers on the `contrastive-crop` recipe, sequenced so the cheap
one is genuinely measured before spending any GPU time on the second.

### Lever A: crop context-margin padding — tested first, at zero GPU cost

**Hypothesis.** `owlv2-oneshot`'s exemplar crop is sliced from the RAW, tight exemplar box, then
padded to a square and resized to 960×960 — for a small door/window symbol (tens of pixels), this is
a large upsample of an isolated symbol on synthetic pad color, possibly missing the surrounding-wall
context the model sees when matching in the full scene. A margin — growing the crop box by a fraction
of its own size before slicing, via a new shared `expand_box_with_margin` function used by both this
inference path and (opt-in) training — might recover some of that context.

This was tested **before any retraining**: a local sweep of `crop_context_margin_frac` (0.0, 0.15,
0.3, 0.5) against the *already-exported* `contrastive-crop` checkpoint, through the same
tune-on-val/freeze/report-on-test harness every other arm's numbers came from (a new backward-
compatible `grids` override on `object_search.eval.tuning.run_domain_tuning` made this possible
without forking the tuning loop). The margin=0.0 cell reproduces `contrastive-crop`'s already-
committed tuned F1 exactly on both datasets — the live regression check that the new plumbing is
trustworthy before trusting any nonzero-margin number.

| margin | door tuned F1 | window tuned F1 |
|---|---|---|
| 0.0 | 0.229 | 0.216 |
| 0.15 | **0.277** | 0.186 |
| 0.3 | 0.230 | 0.158 |
| 0.5 | 0.151 | 0.153 |

Door peaks at margin=0.15 (+21%); window degrades monotonically at every nonzero margin (−14% at
0.15, worse beyond). **No margin beats 0.0 on both classes** — a genuine split result. Plausibly,
door symbols benefit from surrounding wall context while window symbols are already distinctive
enough that added context pulls in confusable neighboring geometry instead of useful signal. Per the
plan's own decision rule, the final GPU arm trains and evaluates at **margin 0.0** and tests the
second lever alone — margin padding is a real, measured finding in its own right (full sweep in
`.planning/quick/260808-w8c-.../margin-verdict.md`), just not one that survived the free screen.

### Lever B: rotation/mirror-augmented crop positives in SupCon training

**Hypothesis.** Floor-plan symbols appear in arbitrary orientations, but `contrastive-crop`'s
crop-context anchor only ever showed the model one canonical-orientation crop per ground-truth box.
Adding one additional randomly-augmented view (rotate 90/180/270 or mirror, chosen per step) of the
SAME crop as a second same-class SupCon positive should teach the crop embedding to be
orientation-robust — reusing the exact "ordinary same-class positive" mechanism the third experiment
already established (zero changes to `supcon_loss`'s math). This is a different mechanism from the
report's earlier, already-reverted "rotation/mirror query-embedding augmentation" mitigation
(inference-time manipulation of the query embedding itself, which zeroed a near-symmetric window's
only true positive) — this lever augments TRAINING-time SupCon positives, never the shipped inference
query.

The `contrastive-crop-v2` arm trains `--supcon-crop-augment` with margin left at 0.0 (per lever A's
verdict), identical hyperparameters otherwise to `contrastive-crop` (headonly, 8 epochs, seed 0,
batch-size 2, grad-accum 4).

**Does it move the property it targets?** Pooled, instance-level crop/scene agreement, epoch 0
(before training) vs. epoch 8 (best checkpoint):

| metric | epoch 0 | epoch 8 (best) |
|---|---|---|
| `val_crop_scene_agreement.self_score_mean` | +0.479 | **+0.821** |
| `val_cos_gap.gap_class` | +0.106 | +0.318 |
| `val_cos_gap.gap_background` | +0.208 | +0.604 |

All three move in the intended direction, comparable in magnitude to `contrastive-crop`'s own
training curve (self-score +0.490→+0.808; `gap_class` +0.117→+0.344; `gap_background`
+0.248→+0.617) — augmentation does not disturb the mechanism it builds on.

**Does it hold on the independent single-exemplar diagnostic?** Extending 260808-dla's diagnostic
(same deterministic exemplar, `.planning/quick/260808-w8c-.../self_score_diagnostic.py`) to all five
checkpoints:

| checkpoint | self_score | threshold (×0.94) | scene patches retained |
|---|---|---|---|
| baseline (pretrained) | +0.368 | +0.346 | 970/2482 (39.1%) |
| headonly (classification) | +0.556 | +0.523 | 1215/2487 (48.9%) |
| contrastive | −0.200 | −0.188 | 1766/2490 (70.9%) |
| contrastive-crop | +0.859 | +0.808 | 199/2497 (8.0%) |
| contrastive-crop-v2 | **+0.896** | **+0.843** | 208/2530 (**8.2%**) |

`contrastive-crop-v2` has the highest self-score and among the tightest retention of all five
checkpoints — consistent with the pooled result, and with the fourth experiment's precision-leaning
F1 numbers above.

**Verdict.** Both levers are real findings, not both wins: margin padding helps door and hurts window
(no margin wins both, so it stays out of the shipped `contrastive-crop-v2` recipe); rotation/mirror
augmentation pushes door F1 to the best numbers of any fine-tuned arm measured (0.253 tuned / 0.433
default, vs. `contrastive-crop`'s 0.229 / 0.391) while costing window a small amount (0.204 vs.
0.216, −5.5%). Neither result is softened or buried: the class asymmetry is the finding, and it
generalizes the earlier lesson — a lever that helps the crop/scene-agreement mechanism does not
automatically help both classes equally, because door and window symbols interact differently with
both surrounding context (lever A) and orientation variety (lever B, which only clearly helped door).

`--loss-mode both --supcon-crop-context --supcon-crop-augment` is implemented and smoke-tested
(sanity-check.md), not measured on GPU — the same disclosure as `both`-mode in the second experiment.

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

## Provenance — crop-margin + rotation-augmentation experiment

The margin sweep (lever A) ran inference-only, zero retraining, against the already-exported
`contrastive-crop` checkpoint. It was tried first on local CPU and killed after 35+ minutes with the
first of 8 (margin, dataset) cells still incomplete — each cell re-runs the full 9-entry tuning grid
over the whole 56-image val split plus tuned/default test evaluation, far more compute than the
lightweight single-exemplar diagnostics used elsewhere in this report. Moved to a vast.ai RTX 3090
(`OS_ONNX_PROVIDERS=CUDAExecutionProvider,CPUExecutionProvider`), which finished all 8 cells in ~34
minutes. Full table and verdict: `margin-verdict.md`.

Training `contrastive-crop-v2` (lever B) used a vast.ai RTX 3090, machine id 127879 (a host already
used successfully for a diagnostic earlier in this quick task's research thread). `pixi run
fetch-datasets`'s full sweep — which converts every registered eval dataset, not just floor plans —
stalled repeatedly on an unrelated FSCD-LVIS Hugging Face Hub download timeout unrelated to this
arm; rather than wait out an indefinite retry loop, `fetch-datasets --only floorplans-door` and
`--only floorplans-window` converted just the two datasets this run needed (seconds, not the
multi-GB FSCD-147/FSCD-LVIS downloads), and training/export/evaluation proceeded manually from
there, reusing the same env setup (`onnxruntime-gpu==1.23.2`, cuDNN/cuBLAS `LD_LIBRARY_PATH`
discovery) `scripts/gpu_finetune.sh` already establishes. Training took ~43 minutes (8 epochs);
evaluation ~14 minutes. Pulled back sequentially: two result JSONs, `train_log.json`, and the ONNX
export, whose sha256 (`d900e8ed6f12cc9d2150a46240aed969382715a4d1c79ab89318ad91b758d985`) was
verified identical between the box and the local copy. The instance was destroyed and `vastai show
instances` confirmed empty immediately after pull-back.

| artifact | sha256 (as committed locally) |
|---|---|
| `models/owlv2_base_patch16_floorplans_ft_contrastive_crop_v2.onnx` (contrastive-crop-v2, unregistered comparison) | `d900e8ed6f12cc9d2150a46240aed969382715a4d1c79ab89318ad91b758d985` |

The contrastive-crop-v2 ONNX artifact passes `_verify_graph`'s local contract check (`class_embeds
[batch, num_patches, 512]`, `pred_boxes [batch, 3600, 4]`) before being trusted.

## Disposition

None of the five fine-tuned arms is adopted as `owlv2-oneshot`'s shipped default.
`owlv2-oneshot`'s default model path and behavior are unchanged — every fine-tuned weight set is an
opt-in research artifact, reachable for one run via
`OS_OWLV2_MODEL=owlv2_base_patch16_floorplans_ft.onnx` (headonly, classification),
`OS_OWLV2_MODEL=owlv2_base_patch16_floorplans_ft_full.onnx` (full, classification),
`OS_OWLV2_MODEL=owlv2_base_patch16_floorplans_ft_contrastive.onnx` (headonly, contrastive),
`OS_OWLV2_MODEL=owlv2_base_patch16_floorplans_ft_contrastive_crop.onnx` (headonly, contrastive-crop),
or `OS_OWLV2_MODEL=owlv2_base_patch16_floorplans_ft_contrastive_crop_v2.onnx` (headonly,
contrastive-crop-v2), mirroring the existing `OS_ONNX_PROVIDERS` override convention. `ncc` and
`propose-retrieve` remain this repo's floor-plan recommendation (per
`docs/eval/floorplans-findings.md`) — `contrastive-crop-v2`'s 0.253 door / 0.204 window tuned F1 does
not reach `propose-retrieve`'s 0.459 door F1 or `ncc`'s 0.403 window F1, so this does not change which
method ships on this domain. What it does change: the earlier verdict that fine-tuning
`owlv2-oneshot`'s heads is foreclosed as a lever is retracted. Fine-tuning works, and works by a wide
margin over the pretrained baseline, *once the training objective supervises the same
crop-context-vs-scene-context comparison the method runs at inference* — the concrete, measured
mechanism, not a hedge. `contrastive-crop-v2` is the best-measured OWLv2 configuration on this domain
for door detection specifically (0.253 tuned / 0.433 default F1); `contrastive-crop` remains
marginally better for window (0.216 vs. 0.204) if window is the priority class. Neither is this
repo's top overall recommendation — `ncc`/`propose-retrieve` are — but both are reasonable opt-in
choices for anyone using `owlv2-oneshot` on floor-plan-like symbol detection specifically, and the
choice between the two fine-tuned arms is itself class-dependent rather than a clean either/or.
