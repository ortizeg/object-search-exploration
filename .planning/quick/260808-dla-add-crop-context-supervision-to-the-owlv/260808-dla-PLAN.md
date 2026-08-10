---
quick_id: 260808-dla
type: execute
# task 4 is a blocking human checkpoint (vast.ai spend)
autonomous: false
requirements: [EVAL-09, EVAL-23, EVAL-24]
files_modified:
  - src/object_search/train/supcon.py
  - src/object_search/search/owlv2_oneshot.py
  - scripts/finetune_owlv2.py
  - scripts/gpu_finetune.sh
  - scripts/build_owlv2_finetune_comparison.py
  - tests/test_train_supcon.py
  - tests/test_owlv2_oneshot.py
  - tests/test_train_owlv2_targets.py
  - docs/reports/owlv2-floorplans-finetune.md
  - .planning/STATE.md
must_haves:
  truths:
    - "`--loss-mode contrastive` WITHOUT `--supcon-crop-context` remains behavior-preserving: its per-epoch train/val loss and `val_cos_gap` numbers are unchanged from a preflight fixture captured before this task's edits (the `epochs` array and `best_val_loss`, not `config`, which legitimately gains new fields). `--loss-mode focal` is unaffected and independently re-verified the same way."
    - "A crop-context anchor is built per training image by cropping the RAW scene pixels at a randomly-picked ground-truth box's PIXEL coordinates (via `boxes_to_pixels`, never the crop's own coordinates), preprocessed by the SAME `owlv2_preprocess_tensor` function the scene path and `owlv2-oneshot`'s inference path both use, and run through the SAME `select_query_patch_index` function `owlv2_oneshot.search()` calls at inference -- a shared, refactored function, not a reimplementation -- to pick the single most-distinctive covering patch."
    - "The selected crop-context patch's `class_embeds` row is appended to the SupCon pool as an ORDINARY same-class positive (never `negative_only`), with ZERO changes to `supcon_loss`/`supcon_loss_torch`'s math -- it is pulled toward same-class scene anchors and pushed away from different-class/background scene patches purely by the existing label-based mechanism."
    - "A new torch-free diagnostic, `crop_scene_agreement`, measures the INSTANCE-LEVEL cosine similarity between each crop-context anchor and the SPECIFIC scene-context patch the Hungarian matcher assigned to that same ground-truth box -- the same pairing `self_score` measures at inference -- logged per-epoch on val only (`val_crop_scene_agreement`), including an epoch-0 pre-training reference, and entirely ABSENT (not null) when crop-context supervision is off or the mode is `focal`."
    - "`select_query_embedding`'s existing public behavior and test (`test_select_query_embedding_picks_the_most_distinctive_covering_patch`) are unchanged after the refactor that extracts `select_query_patch_index`."
    - "A degenerate ground-truth box (rejected by `boxes_to_pixels` after pixel rounding) never crashes training: the crop-context builder retries a different box in that image, then skips the image's crop-context anchor for that step and logs at DEBUG -- never an exception, never a NaN."
    - "The local overfit sanity check (MPS/CPU, 4 images, several epochs) measures and reports FOUR numbers before any GPU money is spent: the contrastive loss falling, `gap_class`/`gap_background` widening, the box losses not blowing up, AND the NEW `val_crop_scene_agreement.self_score_mean` moving toward positive from 260805-hg1's -0.297 reference point -- the fourth is the whole point of this task and must be measured, not assumed or rounded up."
    - "The new `contrastive-crop` arm is trained once (headonly freeze, the SAME epochs/seed/hyperparameters as 260805-hg1's `contrastive` arm: 8 epochs, seed 0, batch-size 2, grad-accum 4) on ONE vast.ai instance, exported, and evaluated on the SAME 28-plan `floorplans-door`/`floorplans-window` test splits through the SAME `run_domain_tuning` harness as every other arm. The already-measured baseline/headonly/full/contrastive arms are NOT re-run or overwritten."
    - "`scripts/gpu_finetune.sh`'s bare/default invocation (`ARMS` unset) still reproduces the ORIGINAL three-arm run exactly; `ARMS=\"contrastive-crop\"` trains, exports, and evaluates only the new arm."
    - "The report's third-experiment section states, in its own words, whether the fix closes the crop/scene divergence gap: the pooled `val_crop_scene_agreement` before/after, AND the single-exemplar `self_score` before/after for all four checkpoints via a committed, reproducible local diagnostic script, plus the F1 numbers next to the existing rows in both door and window tables. A negative result is exactly as reportable as a positive one."
    - "The vast.ai instance id is recorded, destroyed, and the destruction confirmed with `vastai show instances`; the local `datasets/`/`models/` symlinks and every throwaway `models/finetune/_*` directory this task creates are removed before it closes."
  artifacts:
    - src/object_search/train/supcon.py
    - tests/test_train_supcon.py
    - src/object_search/search/owlv2_oneshot.py
    - scripts/finetune_owlv2.py
    - scripts/gpu_finetune.sh
    - docs/reports/owlv2-floorplans-finetune.md
    - .planning/quick/260808-dla-add-crop-context-supervision-to-the-owlv/self_score_diagnostic.py
    - docs/benchmark/owlv2-finetune/floorplans-door-contrastive-crop.json
    - docs/benchmark/owlv2-finetune/floorplans-window-contrastive-crop.json
  key_links:
    - "`_contrastive_rows` gathers the crop-context row from `select_query_patch_index(crop_class_embeds[i], crop_pred_boxes[i], config.supcon_query_iou_frac)` -> index -> `crop_class_embeds[i, index]` (torch, differentiable) -> appended to `ContrastiveRows.anchors`/`.labels` alongside the existing scene-matched anchors -> pooled by the UNCHANGED `_pooled_supcon`/`supcon_loss_torch`. If the index selection or the crop's pixel-space cropping is wrong, the crop anchor silently supervises the WRONG patch, training the model to agree crop-context and scene-context on the wrong thing while the loss still falls."
    - "`boxes_to_pixels(target.boxes[box_idx:box_idx+1], orig_w, orig_h)` -> pixel `BBox` -> `image[box.y:box.y2, box.x:box.x2]` raw crop -> `owlv2_preprocess_tensor` -> `_forward_batch` -> `crop_class_embeds`/`crop_pred_boxes`. This is the SAME pixel-to-tensor pipeline `owlv2-oneshot.search()` runs for its own query crop; any divergence here is exactly the training/inference-preprocessing-drift risk the objective calls out as the single most important correctness risk."
    - "`Owlv2HungarianMatcher.forward`'s `(source_idx, target_idx)` for THIS image (already computed for the scene anchors) -> `source_idx[target_idx == box_idx]` -> the scene patch paired with the crop anchor for `crop_scene_agreement`'s instance-level self-score diagnostic. Reusing the already-computed `indices` (no second matcher call) keeps the diagnostic free."
    - "`--loss-mode contrastive --supcon-crop-context` -> headonly checkpoint `models/finetune/owlv2-floorplans-contrastive-crop` -> `export_owlv2.py --checkpoint` -> `owlv2_base_patch16_floorplans_ft_contrastive_crop.onnx` -> `OS_OWLV2_MODEL` -> `run_domain_tuning` -> `docs/benchmark/owlv2-finetune/floorplans-{door,window}-contrastive-crop.json` -> `build_owlv2_finetune_comparison.py` 5th arm -> the report's third-experiment row."
---

<objective>
260805-hg1 added a SupCon loss over `owlv2-oneshot`'s `class_embeds`, trained on scene-context
forward passes only, and measured a sharp negative result (door/window F1 0.010/0.009). A follow-up
diagnostic found the mechanism: `owlv2-oneshot`'s inference-time self-similarity calibration depends
on a CROP-context query embedding (encoding the small cropped exemplar alone) agreeing with the
SCENE-context embedding of that same region (`self_score`), and the SupCon training loop never
touches the crop-context forward pass at all. For the contrastive checkpoint `self_score` went from
+0.71 (pretrained) to -0.297, flipping the calibration threshold negative and retaining ~86% of all
scene patches instead of ~25-30%.

This task adds crop-encoded anchors to the SupCon contrastive batch: for each training image, in
addition to the existing scene-context anchors and background negatives, ALSO run the exemplar crop
(cropped from a ground-truth box, preprocessed EXACTLY as `owlv2_oneshot.py`'s inference path
preprocesses the query crop) through the model's image encoder to get a crop-context embedding, and
include it in the SupCon pool as an ordinary same-class positive. The goal is for training to
explicitly teach the model that crop-context and scene-context embeddings of the same object should
be cosine-close -- the exact property calibration depends on and that plain scene-to-scene SupCon
never touched -- and to re-measure whether that closes the gap.

Purpose: this is the "concrete next lever" 260805-hg1's disposition named, not more variations on
scene-only training. It is a targeted mechanism fix, not a new objective.

Output: a torch-free NumPy diagnostic + a refactored, SHARED query-patch-selection helper, tested;
an opt-in `--supcon-crop-context` flag layered on the existing `--loss-mode contrastive`/`both` with
the previous behavior provably unchanged when it is off; one GPU arm trained, exported, and measured
on the same harness as every other arm; and a report section stating plainly whether the fix works.
</objective>

<execution_context>
@$HOME/.claude/gsd-core/workflows/execute-plan.md
@$HOME/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@.claude/CLAUDE.md
@.planning/STATE.md
@docs/reports/owlv2-floorplans-finetune.md
@scripts/finetune_owlv2.py
@src/object_search/train/supcon.py
@src/object_search/train/owlv2_targets.py
@src/object_search/search/owlv2_oneshot.py
@src/object_search/search/common/calibration.py
@scripts/gpu_finetune.sh
@scripts/build_owlv2_finetune_comparison.py
</context>

<verified_facts>
Established by reading the source before this plan was written. Treat as facts; do not re-derive.
Items marked TO CONFIRM are the executor's job to check against the installed `transformers`, and
are called out because getting them wrong is silent rather than loud.

1. `class_predictor(image_feats, query_embeds, query_mask)`'s returned `class_embeds`
   (`image_class_embeds`) is the `class_head.dense0(image_feats)` projection -- computed purely from
   `image_feats`. 260805-hg1 already relies on this being the IDENTICAL space `owlv2-oneshot`'s
   ONNX-exported `class_predictor(image_feats, query_embeds=None)` scores at inference
   (`_forward_batch`'s docstring: "It is the space owlv2-oneshot scores in at inference"). This is
   the fact that makes reusing `_forward_batch` for a CROP forward (which also calls
   `class_predictor` with real text `query_embeds`) numerically train/inference-consistent, up to
   weight differences from training.
2. `owlv2-oneshot`'s crop-context query encode crops the exemplar box directly from the RAW BGR
   scene array (`image[crop_box.y:y2, crop_box.x:x2]`, uint8, no scene-level preprocessing applied
   first) and passes THAT raw sub-array through `owlv2_preprocess_tensor` -- the crop gets its OWN
   independent pad-to-square-then-resize based on the CROP's own aspect ratio, not the scene's.
   Training must replicate this exactly.
3. `ImageTargets.boxes` are normalized over the SCENE's padded-square side (`max(scene_H, scene_W)`),
   never the crop's. Converting a GT box back to scene PIXEL coordinates for cropping requires
   `boxes_to_pixels(boxes, orig_w, orig_h)` (already defined, pure, in `owlv2_oneshot.py`), using the
   scene's own `orig_w`/`orig_h` (read via `cv2.imread`, exactly as `_load_pixel_values` already
   does).
4. `select_query_embedding`'s existing public signature and behavior (`test_owlv2_oneshot.py::
   test_select_query_embedding_picks_the_most_distinctive_covering_patch`) must be preserved
   byte-for-byte after extracting its index-selection logic into a reusable helper.
5. `Owlv2HungarianMatcher.forward` returns, per image, `(source_idx, target_idx)` where `target_idx`
   indexes the SAME per-image GT-box ordering as `ImageTargets.class_labels`/`.boxes`. The crop's
   chosen box index (`box_idx`, target-local) can therefore be looked up directly in the `indices`
   list `_contrastive_rows` ALREADY computes for the scene anchors -- no second matcher call needed.
6. `FinetuneConfig` is frozen + `extra="forbid"`; new fields must carry a `description` (an existing
   test in `tests/test_train_owlv2_targets.py` asserts every field is documented).
7. The already-measured `contrastive` arm's artifacts
   (`docs/benchmark/owlv2-finetune/floorplans-{door,window}-contrastive.json`,
   `models/finetune/owlv2-floorplans-contrastive/train_log.json`,
   `models/owlv2_base_patch16_floorplans_ft_contrastive.onnx`) are committed/present in this worktree
   and must NOT be re-run or overwritten. The new arm uses a distinct name: `contrastive-crop`.
8. `_epoch_record`/`_write_train_log` already establish the pattern for an optional, nullable
   per-epoch diagnostic key (`val_cos_gap`) that is entirely ABSENT -- not `null` -- in `focal` mode's
   `epochs` array. The same discipline applies to the new `val_crop_scene_agreement` key.
9. **TO CONFIRM:** `Owlv2ForObjectDetection.class_predictor`'s `image_class_embeds` and
   `box_predictor`'s output are genuinely independent of `query_embeds`/`query_mask` (fact 1's
   premise). Read the installed `transformers` source for both methods before trusting that a crop's
   `_forward_batch(runtime, crop_pixel_values)` output is train/inference-consistent. If either
   secretly depends on the text queries, STOP and report -- the entire premise of reusing
   `_forward_batch` for the crop is then wrong.
</verified_facts>

<decisions>
Locked before implementation. The executor implements these; it does not re-litigate them. Each is
recorded, with its reasoning, in `scripts/finetune_owlv2.py`'s module docstring (and a short
cross-reference in `supcon.py`'s, for the new diagnostic only -- `supcon_loss`/`supcon_loss_torch`
themselves are NOT modified by this task).

**D-dla-01 -- crop-context anchors are ORDINARY same-class positives in the existing SupCon pool.
Zero changes to `supcon_loss`/`supcon_loss_torch`'s math.** The crop-context embedding for a training
image's picked GT box is appended to `ContrastiveRows.anchors`/`.labels` alongside the existing
scene-context matched anchors. `supcon_loss_torch`'s label-based positive/negative machinery already
pulls same-labeled rows together and pushes different-labeled/background rows apart -- exactly "pulled
toward its own scene-context GT-box patch and other same-class scene patches, pushed away from
different-class and background scene patches," per the task's own framing. No special-casing.

**D-dla-02 -- ONE crop-context anchor per training image per micro-batch pass, not one per GT box.**
The box is chosen by `rng.integers(0, n_boxes)` from the SAME seeded generator already threaded
through `_contrastive_rows`, consumed AFTER background sampling so a `--supcon-crop-context`-disabled
run's rng stream (and therefore its background samples) is byte-identical to today's. Reasoning:
floor plans average ~20 boxes/image (3962 boxes / 197 images); a crop-context forward is a full
independent 960x960 ViT pass (unlike scene anchors, which piggyback for free on the one scene forward
already computed), so one crop per GT box would roughly 10-20x the model's forward-pass compute per
step. One crop per image roughly DOUBLES it instead (one extra batched crop forward alongside the
existing scene forward), and pooling across `--grad-accum` micro-batches (D-hg1-04) plus 8 epochs of
re-sampling gives broad box/class coverage across the run without the 10-20x cost. Rejected: one crop
per GT box (cost-prohibitive at this box density); always the first box (no coverage rotation).

**D-dla-03 -- crop-context supervision is OPT-IN, layered on the existing `--loss-mode
contrastive`/`both`, never changing their default meaning.** A new `supcon_crop_context: bool =
False` field and `--supcon-crop-context` CLI flag. Reasoning: 260805-hg1's committed `contrastive`
numbers and artifacts must remain reproducible from the current script state -- the same
behavior-preservation discipline `--loss-mode focal` already gets (D-hg1-05/T-hg1-02). Silently
changing what `--loss-mode contrastive` produces would retroactively make the already-measured,
already-reported numbers non-reproducible. The new GPU arm is invoked with `--loss-mode contrastive
--supcon-crop-context` together and named `contrastive-crop`, distinct from `contrastive`, so neither
its script row, checkpoint directory, nor result JSONs collide with the existing arm's artifacts
(T-hg1-05 extended).

**D-dla-04 -- the crop query-patch selection reuses `owlv2_oneshot.py`'s OWN selection logic via a
shared, refactored function, not a training-side reimplementation.** `select_query_patch_index(
class_embeds, boxes_cxcywh, iou_frac) -> int` is extracted from `select_query_embedding` (which
becomes a thin wrapper over it); training imports and calls the SAME function. This is the strongest
available guarantee against the crop-preprocessing/selection-fidelity risk the task exists to close:
literal code reuse, not "written to be equivalent." Training also reuses `owlv2_preprocess_tensor`
and `boxes_to_pixels` for the same reason (both already shared with the scene path since 260801-8zy).

**D-dla-05 -- `supcon_query_iou_frac` (default 0.8) is a separate `FinetuneConfig` field mirroring
`Owlv2OneshotConfig.query_iou_frac`'s default, not a hard-coded constant.** So the training-time
selection threshold is visible in `train_log.json`'s logged config and independently sweepable -- but
its default is pinned EQUAL to the inference config's default, because training must select the query
patch the way inference does. A silently divergent default would train against a selection heuristic
different from the one that runs at inference -- one level up, the exact class of bug this task exists
to prevent.

**D-dla-06 -- a new torch-free diagnostic, `crop_scene_agreement`, measures the property this fix
targets DIRECTLY and INSTANCE-LEVEL.** For each instance where a crop-context anchor was built, its
cosine similarity against the SPECIFIC scene-context patch the Hungarian matcher assigned to that
SAME GT box (not a class average) -- the same instance-level pairing `self_score` measures at
inference. Logged per-epoch on val ONLY (mirroring `val_cos_gap`/D-hg1-06), including an epoch-0
pre-training reference, as `val_crop_scene_agreement`, entirely ABSENT when `supcon_crop_context` is
`False` or `loss_mode == "focal"` -- preserving both existing modes' `epochs` array shape exactly.

**D-dla-07 -- the report restates the headline single-exemplar `self_score` number via a committed,
reproducible local diagnostic script, not only the pooled `val_crop_scene_agreement` statistic.** The
task's own framing ("does it go positive again?") refers to that exact metric and table. The ORIGINAL
diagnostic's exact exemplar coordinates were never persisted (260805-hg1's follow-up ran ad hoc on a
since-destroyed instance), so the new diagnostic picks a FRESH, deterministic exemplar (the first
door-class GT box, by annotation list order, in the first file_name-sorted training image that has
one) and re-measures `self_score` for ALL FOUR checkpoints (baseline, classification-headonly,
contrastive, contrastive-crop) against that SAME fixed exemplar, entirely locally (CPU ONNX Runtime,
no GPU). The four numbers are honestly comparable to each other; they are NOT bit-identical to the
previously-reported ones (a different exemplar), and the report says so plainly rather than implying
a re-verification of the old numbers.
</decisions>

<tasks>

<task type="tracer" tdd="true">
  <name>Task 1: End-to-end crop-context slice -- shared selection helper, crop anchor wiring, one image, one step</name>
  <precondition>`datasets/_incoming/floorplans/` and `models/` are symlinked into this worktree from the main checkout at `../../object-search-exploration/` (both are gitignored and main-checkout-only), and `pixi install -e export` succeeds. Halt and report if the symlink targets do not exist.</precondition>
  <files>src/object_search/search/owlv2_oneshot.py, tests/test_owlv2_oneshot.py, src/object_search/train/supcon.py, tests/test_train_supcon.py, src/object_search/train/owlv2_targets.py, tests/test_train_owlv2_targets.py, scripts/finetune_owlv2.py</files>
  <read_first>
    scripts/finetune_owlv2.py (sections 1, 2, 4 -- `Owlv2HungarianMatcher`, `_forward_batch`,
    `_contrastive_rows`, `_pooled_supcon`, `_batch_loss`); src/object_search/search/owlv2_oneshot.py
    (`select_query_embedding`, `boxes_to_pixels`, `_iou_with_unit_box`, the module docstring's
    preprocessing and query-embedding-selection sections); the installed `transformers` source for
    `Owlv2ForObjectDetection.class_predictor` and `.box_predictor` (verified-fact 9 -- confirm both
    are independent of `query_embeds`/`query_mask` before trusting the crop-forward reuse; STOP and
    report if not).
  </read_first>
  <behavior>
    `src/object_search/search/owlv2_oneshot.py` refactor (tested in `tests/test_owlv2_oneshot.py`):
    - Extract `select_query_patch_index(class_embeds, boxes_cxcywh, iou_frac) -> int` (pure NumPy, no
      torch) holding the covering-patch IoU + distinctiveness-argmin logic currently inline in
      `select_query_embedding`. `select_query_embedding` becomes: get the index from
      `select_query_patch_index`, then index the L2-normalized embedding matrix and return that row.
      `test_select_query_embedding_picks_the_most_distinctive_covering_patch` passes UNCHANGED, with
      no edit to the test itself.
    - New test: `select_query_patch_index` on the SAME fixture as the existing
      `select_query_embedding` test returns the index whose normalized row equals
      `select_query_embedding`'s output (cross-check the two never drift). New test: falls back to
      the largest-area patch when no box overlaps the unit box at all (mirrors the existing
      zero-max-IoU fallback branch).

    `src/object_search/train/supcon.py`, tested in `tests/test_train_supcon.py`:
    - `crop_scene_agreement(crop_embeddings, scene_embeddings) -> dict[str, float | None]` with keys
      `self_score_mean`, `self_score_min`, `self_score_max`, `n_pairs`. Both inputs are `(n, d)`
      ALIGNED pairs (row i of one against row i of the other -- not a pairwise matrix). L2-normalize
      both sides via the existing shared `_l2_normalize`, then row-wise cosine
      (`(normalize(crop) * normalize(scene)).sum(axis=1)`). Every float key is `None` when `n_pairs
      == 0`, never `0.0` (the repo's nullable-metric rule).
      - A hand-computed 2-pair case with an exact expected mean/min/max.
      - Invariant to a permutation of the pool (both sides permuted together) and to a positive
        rescaling of either side independently (normalization is inside).
      - Rejects mismatched shapes (either input not `(n, d)`, or the two lengths/dims disagreeing).
      - `n_pairs == 0` (both inputs empty) reports every key `None`.

    `src/object_search/train/owlv2_targets.py::FinetuneConfig` gains two fields, each with a
    `description` (D-dla-03, D-dla-05):
    - `supcon_crop_context: bool = False`
    - `supcon_query_iou_frac: float = 0.8` (`ge=0.0`, `le=1.0`)
    New tests in `tests/test_train_owlv2_targets.py` for both fields' defaults and out-of-range
    rejection of `supcon_query_iou_frac`.
  </behavior>
  <action>
    Wire the crop-context anchor into `scripts/finetune_owlv2.py`'s existing training path (this
    tracer task makes it flow into the LOSS end to end; the val-side diagnostic logging is Task 2):

    1. Import `select_query_patch_index` and `boxes_to_pixels` from
       `object_search.search.owlv2_oneshot`; import `crop_scene_agreement` from
       `object_search.train.supcon` (unused by the loss in this task, imported for Task 2's wiring to
       land cleanly).
    2. New `_load_crop_pixel_values(image_path: Path, box: BBox) -> torch.Tensor`: `cv2.imread`, slice
       `image[box.y:box.y2, box.x:box.x2]`, `owlv2_preprocess_tensor`, `torch.from_numpy` -- mirrors
       `_load_pixel_values` exactly but on a cropped sub-array rather than the whole image.
    3. New `_pick_crop_box_index(target: ImageTargets, rng: np.random.Generator) -> int`: returns
       `int(rng.integers(0, target.boxes.shape[0]))`.
    4. Extend `ContrastiveRows` with two new fields: `crop_diag_crop: torch.Tensor` and
       `crop_diag_scene: torch.Tensor` (aligned `(n, 512)` pairs for the diagnostic ONLY, never fed
       to the loss). Default them to an empty `(0, 512)` tensor when no crop-context row is built.
    5. `_contrastive_rows` gains an `image_dir: Path` parameter. AFTER the existing scene-anchor and
       background-negative gather (unchanged, and consuming `rng` in the SAME order as before, per
       D-dla-02, so a `supcon_crop_context=False` run's rng stream and background samples are
       byte-identical to today's): when `runtime.config.supcon_crop_context` is `True` and
       `loss_mode != "focal"`, for each image in the micro-batch pick `box_idx` via
       `_pick_crop_box_index`, read the scene image once with `cv2.imread` to get `orig_w`/`orig_h`,
       convert that one box to a pixel `BBox` via
       `boxes_to_pixels(target.boxes[box_idx:box_idx+1], orig_w, orig_h)[0]` -- if `None` (a
       degenerate box after pixel rounding), retry with a different `box_idx` up to 3 times, then
       skip that image's crop-context anchor for this step and log at DEBUG (never raise). Build the
       crop tensor via `_load_crop_pixel_values`, batch ALL of this micro-batch's valid crops into ONE
       `torch.cat`, and run them through ONE extra `_forward_batch(runtime, crop_pixel_values)` call
       inside the same `torch.autocast` settings the scene forward uses. For each valid image: call
       `select_query_patch_index` on the DETACHED numpy view of that image's slice of
       `crop_class_embeds`/`crop_pred_boxes` with `config.supcon_query_iou_frac` -> an index -> gather
       the TORCH row `crop_class_embeds[i, index]` (keeps the gradient path into `class_head.dense0`)
       -> append to `anchors`/`labels` with label `target.class_labels[box_idx]`, exactly like a
       scene anchor, per D-dla-01. Also look up the matched scene patch for the SAME `box_idx` from
       the `indices` this function ALREADY computed for the scene anchors
       (`source_idx[target_idx == box_idx]`, skip the diagnostic pairing silently if no match is
       found) and stash the DETACHED `(crop row, scene row)` pair into `crop_diag_crop`/
       `crop_diag_scene` for Task 2 to consume.
    6. `_batch_loss` passes `image_dir` through to `_contrastive_rows` (already in scope there).
    7. Add `--supcon-crop-context` (`action="store_true"`, default `False`) and
       `--supcon-query-iou-frac` (float, default `0.8`) CLI flags, wired through `_config_from_args`.
    8. Module docstring: add the "crop-context extension" section recording D-dla-01 through D-dla-05
       with their reasoning (mirrors the existing docstring's decision-record style).

    Stub nothing else in this task: a real 1-image, 1-step `--loss-mode contrastive
    --supcon-crop-context` run must produce a real checkpoint with the crop anchor genuinely
    contributing to the loss.
  </action>
  <verify>
    <automated>cd "$(git rev-parse --show-toplevel)" && $HOME/.pixi/bin/pixi run pytest tests/test_train_supcon.py tests/test_owlv2_oneshot.py tests/test_train_owlv2_targets.py -v --no-cov && $HOME/.pixi/bin/pixi run -e export finetune-owlv2 --loss-mode contrastive --supcon-crop-context --limit-images 1 --epochs 1 --max-steps 1 --device cpu --out models/finetune/_tracer_crop_context && test -f models/finetune/_tracer_crop_context/train_log.json && $HOME/.pixi/bin/pixi run python -c "
import json,pathlib
log=json.loads(pathlib.Path('models/finetune/_tracer_crop_context/train_log.json').read_text())
assert log['config']['supcon_crop_context'] is True
print('tracer OK:', log['epochs'][-1]['train_loss_supcon'])
"</automated>
  </verify>
  <done>The `select_query_embedding` refactor keeps its existing test green and the new
  `select_query_patch_index` cross-check passes. `crop_scene_agreement`'s tests pass including the
  None-vs-0.0 discipline. A 1-image, 1-step `contrastive` + `--supcon-crop-context` run writes a real
  checkpoint with the crop anchor wired into the loss (non-zero `train_loss_supcon`).</done>
</task>

<task type="auto" tdd="true">
  <name>Task 2: Val-side diagnostic, degenerate-box robustness, and the two preservation fixtures</name>
  <files>scripts/finetune_owlv2.py, tests/test_train_supcon.py, .planning/quick/260808-dla-add-crop-context-supervision-to-the-owlv/preflight-focal-train-log.json, .planning/quick/260808-dla-add-crop-context-supervision-to-the-owlv/preflight-contrastive-train-log.json</files>
  <read_first>
    scripts/finetune_owlv2.py `_evaluate`, `_epoch_record`, `_write_train_log`, and `main`'s epoch-0
    reference block (D-hg1-06) as they stand after Task 1's edits.
  </read_first>
  <behavior>
    - `_evaluate` gains the val-side crop-context diagnostic: when
      `config.supcon_crop_context and config.loss_mode != "focal"`, pool `crop_diag_crop`/
      `crop_diag_scene` across the val split at the SAME `grad_accum`-boundary granularity the
      contrastive term already uses, then call `crop_scene_agreement` once over the pooled pairs.
      `None` components must thread through untouched -- never fabricated as `0.0`.
    - `_epoch_record` gains an optional `crop_scene: dict[str, float | None] | None` parameter;
      `record["val_crop_scene_agreement"] = crop_scene` only when not `None`, mirroring `cos_gap`'s
      existing handling exactly. `main()`'s epoch-0 pre-training block and the per-epoch loop both
      pass this through.
    - A new test in `tests/test_train_supcon.py` (or a script-level test if more appropriate) covers
      the degenerate-box retry-then-skip path: a synthetic box `boxes_to_pixels` rejects for every
      retry must not raise, and the image contributes no crop-context row that step.
  </behavior>
  <action>
    1. FIRST, before this task's `_evaluate`/`_epoch_record` edits: capture two preflight fixtures
       from the script as Task 1 left it (this protects Task 2's OWN edits, exactly as 260805-hg1
       protected focal mode against its training-loop edits):
       `$HOME/.pixi/bin/pixi run -e export finetune-owlv2 --limit-images 2 --epochs 1 --max-steps 1
       --device cpu --out models/finetune/_preflight2_focal`, copy its `train_log.json` to
       `preflight-focal-train-log.json`; and `$HOME/.pixi/bin/pixi run -e export finetune-owlv2
       --loss-mode contrastive --limit-images 2 --epochs 1 --max-steps 1 --device cpu --out
       models/finetune/_preflight2_contrastive` (`--supcon-crop-context` OFF), copy to
       `preflight-contrastive-train-log.json`. Do not proceed until both files exist.
    2. Implement the behaviour block above.
    3. Postflight: re-run the SAME two commands from step 1 against the now-edited script and assert
       byte-identical `epochs` arrays and `best_val_loss` against both preflight fixtures -- proving
       this task's val/epoch-record edits are behavior-preserving for BOTH `focal` and
       `--supcon-crop-context`-disabled `contrastive`.
    4. Update the module docstring's "crop-context extension" section (added in Task 1) with D-dla-06
       and its reasoning.
  </action>
  <verify>
    <automated>cd "$(git rev-parse --show-toplevel)" && $HOME/.pixi/bin/pixi run test && $HOME/.pixi/bin/pixi run -e export finetune-owlv2 --limit-images 2 --epochs 1 --max-steps 1 --device cpu --out models/finetune/_postflight2_focal && $HOME/.pixi/bin/pixi run -e export finetune-owlv2 --loss-mode contrastive --limit-images 2 --epochs 1 --max-steps 1 --device cpu --out models/finetune/_postflight2_contrastive && $HOME/.pixi/bin/pixi run python -c "
import json,pathlib
base='.planning/quick/260808-dla-add-crop-context-supervision-to-the-owlv'
for name,out in (('focal','_postflight2_focal'),('contrastive','_postflight2_contrastive')):
    pre=json.loads(pathlib.Path(f'{base}/preflight-{name}-train-log.json').read_text())
    post=json.loads(pathlib.Path(f'models/finetune/{out}/train_log.json').read_text())
    assert pre['epochs']==post['epochs'], f'{name} mode drifted'
    assert pre['best_val_loss']==post['best_val_loss']
print('both modes are behavior-preserving')
" && $HOME/.pixi/bin/pixi run -e export finetune-owlv2 --loss-mode contrastive --supcon-crop-context --limit-images 2 --epochs 1 --max-steps 1 --device cpu --out models/finetune/_postflight2_crop && $HOME/.pixi/bin/pixi run python -c "
import json,pathlib
log=json.loads(pathlib.Path('models/finetune/_postflight2_crop/train_log.json').read_text())
gap=log['epochs'][-1].get('val_crop_scene_agreement')
assert gap is not None, 'val_crop_scene_agreement missing when supcon_crop_context is on'
print('val_crop_scene_agreement present:', gap)
"</automated>
  </verify>
  <done>Both preservation fixtures match byte-for-byte. `val_crop_scene_agreement` appears (non-empty
  dict) in a `--supcon-crop-context` run and is entirely absent in `focal` and in `contrastive`
  without the flag. The degenerate-box retry-then-skip path is covered by a test and never raises.
  `pixi run test` holds the coverage floor.</done>
</task>

<task type="auto">
  <name>Task 3: Overfit sanity check and the four quality gates -- including the crop-context self-score, before any GPU money is spent</name>
  <files>.planning/quick/260808-dla-add-crop-context-supervision-to-the-owlv/sanity-check.md, scripts/finetune_owlv2.py</files>
  <action>
    A model that cannot overfit four images is wired wrong, and finding that out on a rented GPU is
    the expensive way to find out. Run the check locally first, exactly as 260805-hg1's Task 3 did,
    now with the crop-context flag on and a FOURTH number to measure:

    1. `$HOME/.pixi/bin/pixi run -e export finetune-owlv2 --loss-mode contrastive
       --supcon-crop-context --limit-images 4 --epochs 12 --batch-size 2 --grad-accum 2 --device mps
       --out models/finetune/_sanity_crop_context` (fall back to `--device cpu` if MPS errors; note
       which was used).
    2. Read `models/finetune/_sanity_crop_context/train_log.json` and report FOUR numbers, not
       impressions:
       - `train_loss_supcon` falls substantially from epoch 1 to epoch 12;
       - `val_cos_gap.gap_class` and `gap_background` at the last epoch exceed their epoch-0 values;
       - `train_loss_bbox`/`train_loss_giou` have not blown up (the box head must not be sacrificed);
       - NEW: `val_crop_scene_agreement.self_score_mean` at the last epoch is HIGHER than at epoch 0
         -- moving toward positive from 260805-hg1's -0.297 reference. On 4 images and 12 epochs it
         may not cross zero; report the actual number either way rather than rounding up. This check
         is the whole point of this task.
    3. One smoke run of `--loss-mode both --supcon-crop-context --limit-images 2 --epochs 1
       --max-steps 1 --device cpu` so the combination is demonstrably functional.
    4. Write `sanity-check.md` recording the command lines, the device used, all four numbers
       (epoch 1 vs epoch 12 / epoch 0 vs last, as applicable), and a one-line go/no-go verdict. If any
       of the first three checks fails, STOP and fix before Task 4. If the fourth (self-score) does
       NOT move toward positive at all, STOP and report -- that would mean the fix is not working
       before any GPU money is spent, which is exactly what this gate exists to catch.
    5. Run the four gates: `pixi run lint`, `pixi run format-check`, `pixi run typecheck`,
       `pixi run test`. All four clean, no new `# type: ignore` without a reason comment.
  </action>
  <verify>
    <automated>cd "$(git rev-parse --show-toplevel)" && $HOME/.pixi/bin/pixi run quality && test -f .planning/quick/260808-dla-add-crop-context-supervision-to-the-owlv/sanity-check.md && $HOME/.pixi/bin/pixi run python -c "
import json,pathlib
log=json.loads(pathlib.Path('models/finetune/_sanity_crop_context/train_log.json').read_text())
eps=log['epochs']
first,last=eps[0],eps[-1]
assert last['train_loss_supcon']<first['train_loss_supcon'], 'supcon term did not fall on a 4-image overfit'
gaps=[e for e in eps if e.get('val_cos_gap')]
assert gaps[-1]['val_cos_gap']['gap_class']>gaps[0]['val_cos_gap']['gap_class'], 'class cosine gap did not widen'
scores=[e for e in eps if e.get('val_crop_scene_agreement') and e['val_crop_scene_agreement'].get('self_score_mean') is not None]
assert len(scores)>=2, 'val_crop_scene_agreement.self_score_mean not measured at epoch 0 and at the last epoch'
assert scores[-1]['val_crop_scene_agreement']['self_score_mean']>scores[0]['val_crop_scene_agreement']['self_score_mean'], 'crop/scene self-score did not move toward positive'
print('sanity OK: supcon', first['train_loss_supcon'], '->', last['train_loss_supcon'], '| self_score', scores[0]['val_crop_scene_agreement']['self_score_mean'], '->', scores[-1]['val_crop_scene_agreement']['self_score_mean'])
"</automated>
  </verify>
  <done>All four gates are green. `sanity-check.md` records all four numbers and an explicit go/no-go.
  The contrastive term falls, the cosine gap widens, box losses stay stable, and the crop/scene
  self-score moves toward positive on a 4-image overfit -- the recipe is demonstrably learning the
  intended property before any GPU is rented.</done>
</task>

<task type="checkpoint:human-verify" gate="blocking">
  <name>Task 4: Human checkpoint -- approve the crop-context recipe and the vast.ai budget</name>
  <what-built>
    Crop-context anchors (`--supcon-crop-context`, layered on `--loss-mode contrastive`/`both`,
    default off so the existing recipe stays reproducible) added to the SupCon pool as ordinary
    same-class positives, built by reusing `owlv2-oneshot`'s own `select_query_patch_index` and
    `owlv2_preprocess_tensor` for exact train/inference fidelity (D-dla-04). A new
    `val_crop_scene_agreement` diagnostic measures the instance-level crop/scene self-score directly.
    Model-free tests pass, both `focal` and flag-off `contrastive` are proven behavior-preserving
    against preflight fixtures, and a 4-image overfit shows the contrastive term falling, the cosine
    gap widening, and the crop/scene self-score moving toward positive.
  </what-built>
  <how-to-verify>
    1. Read `.planning/quick/260808-dla-add-crop-context-supervision-to-the-owlv/sanity-check.md` --
       the four numbers and the go/no-go.
    2. Read `scripts/finetune_owlv2.py`'s module docstring: confirm D-dla-01..06 read as decisions you
       agree with, especially D-dla-02's box-per-image (not box-per-GT-box) cost tradeoff and
       D-dla-03's opt-in-flag reproducibility guard.
    3. Confirm the **planned GPU spend**: ONE vast.ai instance, ONE training arm
       (`contrastive-crop`: `headonly` + `--loss-mode contrastive --supcon-crop-context`, 8 epochs,
       seed 0, batch-size 2, grad-accum 4 -- identical hyperparameters to 260805-hg1's `contrastive`
       arm), export, and eval on `floorplans-door` + `floorplans-window` test splits. The existing
       baseline/headonly/full/contrastive arms are NOT re-run. Crop-context roughly DOUBLES the
       forward-pass compute of the original contrastive arm (an extra crop forward per micro-batch),
       so expect roughly 3-5 GPU-hours (~$1-3 on a 3090); hard stop and report if spend would exceed
       $10.
  </how-to-verify>
  <resume-signal>Type "approved" to rent the GPU, or describe what to change first.</resume-signal>
</task>

<task type="auto">
  <name>Task 5: vast.ai GPU run -- train the contrastive-crop arm, export, evaluate, pull back, destroy</name>
  <precondition>The `vastai` CLI is authenticated and the account has credit; task 4 returned "approved". Halt if either is untrue.</precondition>
  <files>scripts/gpu_finetune.sh</files>
  <action>
    1. Add a `contrastive-crop` row to `scripts/gpu_finetune.sh`'s arm tables, alongside the existing
       `baseline`/`headonly`/`full`/`contrastive` rows, WITHOUT changing the default `ARMS` value (a
       bare `bash scripts/gpu_finetune.sh` must still reproduce the original three-arm run exactly):
       `TRAIN_FLAGS[contrastive-crop]="--loss-mode contrastive --supcon-crop-context"`,
       `CKPT_DIR[contrastive-crop]="models/finetune/owlv2-floorplans-contrastive-crop"`,
       `ONNX_NAME[contrastive-crop]="owlv2_base_patch16_floorplans_ft_contrastive_crop.onnx"`. Add the
       matching entry to step 7/8's provenance `onnx_names` Python dict and to the header comment's
       artifact list.
    2. Rent ONE instance (record its id AND its `machine_id`/public IP). scp the floor-plan COCO
       export to the box.
    3. Run `ARMS="contrastive-crop" SEED=0 EPOCHS=8 BATCH_SIZE=2 GRAD_ACCUM=4 bash
       scripts/gpu_finetune.sh`. Watch that the CUDAExecutionProvider assertion passes before
       training, and that epoch-0 `val_cos_gap` AND epoch-0 `val_crop_scene_agreement` are both
       logged.
    4. Pull back SEQUENTIALLY -- never concurrent ssh/rsync sessions to the same box:
       `docs/benchmark/owlv2-finetune/floorplans-{door,window}-contrastive-crop.json`,
       `models/finetune/owlv2-floorplans-contrastive-crop/train_log.json`,
       `models/owlv2_base_patch16_floorplans_ft_contrastive_crop.onnx`.
    5. If the ONNX must be regenerated on a different box, verify by comparing `train_log.json` loss
       curves, NOT sha256 -- GPU floating point is not bit-identical across physical hardware -- and
       say so plainly in the report.
    6. Destroy that one instance: `vastai destroy instance <id>`, then confirm with
       `vastai show instances`. Record both the id and the confirmed-empty output.
  </action>
  <verify>
    <automated>cd "$(git rev-parse --show-toplevel)" && test -f docs/benchmark/owlv2-finetune/floorplans-door-contrastive-crop.json && test -f docs/benchmark/owlv2-finetune/floorplans-window-contrastive-crop.json && test -f models/finetune/owlv2-floorplans-contrastive-crop/train_log.json && $HOME/.pixi/bin/pixi run python -c "
import json,pathlib
for ds in ('door','window'):
    r=json.loads(pathlib.Path(f'docs/benchmark/owlv2-finetune/floorplans-{ds}-contrastive-crop.json').read_text())
    m=r['methods'][0]
    assert m['method']=='owlv2-oneshot'
    assert m['tuned_test']['f1'] is not None, f'{ds}: nothing scored'
    assert m['tuned_test']['n_scored']==28, f'{ds}: scored {m[\"tuned_test\"][\"n_scored\"]}/28 plans'
    print(ds, 'tuned F1', m['tuned_test']['f1'], 'over', m['tuned_test']['n_scored'], 'plans')
log=json.loads(pathlib.Path('models/finetune/owlv2-floorplans-contrastive-crop/train_log.json').read_text())
assert any(e.get('epoch')==0 for e in log['epochs']), 'no epoch-0 cosine-gap reference'
assert log['epochs'][0].get('val_crop_scene_agreement') is not None, 'no epoch-0 crop/scene self-score reference'
"</automated>
  </verify>
  <done>Both contrastive-crop result JSONs score all 28 test plans, the train log carries epoch-0
  `val_cos_gap` AND `val_crop_scene_agreement` references, `ARMS` unset still reproduces the original
  three-arm recipe, and the single rented instance is destroyed with `vastai show instances`
  confirming it.</done>
</task>

<task type="auto">
  <name>Task 6: The measured answer -- comparison table, self-score diagnostic, report, and close-out</name>
  <files>scripts/build_owlv2_finetune_comparison.py, docs/reports/owlv2-floorplans-finetune.md, .planning/quick/260808-dla-add-crop-context-supervision-to-the-owlv/self_score_diagnostic.py, .planning/quick/260808-dla-add-crop-context-supervision-to-the-owlv/260808-dla-SUMMARY.md, .planning/STATE.md</files>
  <action>
    1. `scripts/build_owlv2_finetune_comparison.py`: extend `_ARMS` to include `contrastive-crop`
       (5 arms x 2 datasets = 10 rows). The eight existing result JSONs are read, never rewritten.
    2. Write `.planning/quick/260808-dla-add-crop-context-supervision-to-the-owlv/
       self_score_diagnostic.py`: pick the exemplar deterministically per D-dla-07 (the first
       door-class GT box, by annotation list order, in the first file_name-sorted image of
       `datasets/_incoming/floorplans/train/_annotations.coco.json` that has one). For each of the
       four checkpoints (`owlv2_base_patch16.onnx`, `owlv2_base_patch16_floorplans_ft.onnx`,
       `owlv2_base_patch16_floorplans_ft_contrastive.onnx`,
       `owlv2_base_patch16_floorplans_ft_contrastive_crop.onnx`), set `OS_OWLV2_MODEL` and run
       `object_search.search.owlv2_oneshot.search()` (or the lower-level `embed_image` /
       `select_query_embedding` / `calibration.calibrate` calls directly) against that ONE
       exemplar-box-in-its-own-scene-image, and print `self_score` (the same quantity `calibrate()`
       reports in `calib.reason` for the `self-similarity` strategy). Runs entirely locally (CPU ONNX
       Runtime, no GPU). Capture its stdout for the report.
    3. Extend `docs/reports/owlv2-floorplans-finetune.md` per D-dla-06/D-dla-07:
       - rewrite the top verdict to cover all THREE experiments and be true of all three;
       - a new "Third experiment: crop-context supervision" section stating the hypothesis (crop- and
         scene-context embeddings of the same object must be trained to agree, not just assumed to),
         the recipe (D-dla-01 through 05, briefly, with the cost tradeoff of D-dla-02 stated plainly);
       - the pooled `val_crop_scene_agreement` epoch-0-vs-final numbers from Task 5's train log;
       - the FOUR-checkpoint `self_score` table from step 2, with the D-dla-07 disclosure that this
         is a freshly-measured exemplar, not a re-run of the original diagnostic's exact coordinates;
       - the `contrastive-crop` rows appended to both door and window F1 tables;
       - a plain statement of whether this closes the gap: does `self_score` go positive again, and
         does F1 track it. A negative result is exactly as reportable as a positive one -- do not
         soften it and do not bury it below the recipe.
    4. Close out: remove the `datasets/`/`models/` symlinks, delete every throwaway
       `models/finetune/_*` directory this task created (`_tracer_crop_context`,
       `_preflight2_focal`, `_preflight2_contrastive`, `_postflight2_focal`,
       `_postflight2_contrastive`, `_postflight2_crop`, `_sanity_crop_context`), and confirm `git
       status` shows no stray artifacts. Stage files individually -- never `git add .`. Confirm `git
       rev-parse --show-toplevel` ends in `object-search-exploration` before any commit.
    5. Write the SUMMARY and update `.planning/STATE.md`'s quick-task table and last-activity line
       with the measured verdict in one sentence.
  </action>
  <verify>
    <automated>cd "$(git rev-parse --show-toplevel)" && $HOME/.pixi/bin/pixi run python scripts/build_owlv2_finetune_comparison.py && $HOME/.pixi/bin/pixi run python -c "
import json,pathlib
rows=json.loads(pathlib.Path('docs/benchmark/owlv2-finetune-comparison.json').read_text())['rows']
arms={(r['dataset'],r['arm']) for r in rows}
assert len(rows)==10 and ('floorplans-door','contrastive-crop') in arms and ('floorplans-window','contrastive-crop') in arms, arms
report=pathlib.Path('docs/reports/owlv2-floorplans-finetune.md').read_text()
assert 'contrastive-crop' in report and 'self_score' in report and 'val_crop_scene_agreement' in report, 'report is missing the contrastive-crop rows or the crop/scene diagnostic'
print('10 rows, report updated')
" && $HOME/.pixi/bin/pixi run quality && $HOME/.pixi/bin/pixi run docs-build && test ! -e datasets/_incoming/floorplans && test ! -e models/finetune/_sanity_crop_context</automated>
  </verify>
  <done>The comparison carries ten rows including both `contrastive-crop` cells. The report's top
  verdict covers all three experiments, the crop/scene self-score numbers (pooled and single-exemplar)
  are present, and the answer -- positive or negative -- is stated plainly. All four gates plus
  `docs-build --strict` are green, symlinks and throwaway checkpoints are gone, and STATE.md records
  the verdict.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| Roboflow COCO export -> training targets | Third-party annotations become supervision; already guarded by `owlv2_targets`'s drop rules |
| local worktree -> rented vast.ai instance | Source, dataset, and credentials cross to a machine outside the developer's control |
| rented instance -> committed report numbers | Measured floats cross back and become claims in a committed document |
| raw scene pixels -> crop-context anchor | A pixel-space cropping/preprocessing mismatch supervises the wrong region silently |

## STRIDE Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation Plan |
|-----------|----------|-----------|----------|-------------|-----------------|
| T-dla-01 | Tampering | crop pixel-space cropping (`boxes_to_pixels` -> `_load_crop_pixel_values`) | high | mitigate | A wrong pixel box or a preprocessing divergence from `owlv2-oneshot`'s inference path trains the crop-context anchor against the WRONG region, silently, while the loss still falls. Mitigated by D-dla-04's literal code reuse (`select_query_patch_index`, `owlv2_preprocess_tensor`, `boxes_to_pixels` are the SAME functions inference calls) and by verified-fact 9's TO-CONFIRM source check before trusting the crop-forward reuse. |
| T-dla-02 | Repudiation | contrastive-mode regression (with the flag OFF) | high | mitigate | This task edits `_contrastive_rows`/`_evaluate`/`_epoch_record`, which the existing `contrastive` arm also runs through. Task 1 and Task 2 each capture a preflight fixture from the script BEFORE their own edits and assert the postflight `epochs`/`best_val_loss` are byte-identical, for BOTH `focal` and flag-off `contrastive`. |
| T-dla-03 | Tampering | existing `contrastive` arm's committed artifacts | high | mitigate | The new arm is named `contrastive-crop` throughout (checkpoint dir, ONNX filename, result JSONs) so it cannot collide with or overwrite `contrastive`'s already-committed files. Task 6's verify asserts ten rows load, which fails loudly if an existing file was clobbered. |
| T-dla-04 | Denial of service | vast.ai spend | medium | mitigate | ONE instance, ONE training arm, blocking human checkpoint before rental, recorded instance id, revised GPU-hour estimate stated (crop-context roughly doubles forward-pass compute), hard stop above $10, `vastai show instances` confirmation. |
| T-dla-05 | Information disclosure | vast.ai instance | medium | mitigate | HF_TOKEN is optional (everything downloaded is public) and the floor plans arrive by scp. No secret required on the box. Instance destroyed and confirmed in Task 5. |
| T-dla-06 | Spoofing | reproduced ONNX artifact | low | accept | GPU float ops are not bit-identical across physical hardware, so a regenerated checkpoint cannot be sha256-matched. Accepted and disclosed: equivalence is argued from loss-curve agreement. |
| T-dla-07 | Elevation of privilege | numerical overflow / crash on a degenerate crop box | medium | mitigate | A GT box that rounds to sub-pixel after `boxes_to_pixels` must not raise. Retry-then-skip with a DEBUG log, tested in Task 2. Reuses `supcon_loss`'s existing log-sum-exp stability (unchanged). |
| T-dla-08 | Tampering | the self-score diagnostic's exemplar choice | low | accept | The fresh exemplar (D-dla-07) is not the same one 260805-hg1's original ad hoc diagnostic used (never persisted). Disclosed explicitly in the report as a fresh, comparable-across-checkpoints measurement, not a re-verification of the old numbers. |
| T-dla-SC | Tampering | npm/pip/cargo installs | high | mitigate | This task adds NO new dependency. The only install is the already-pinned `onnxruntime-gpu==1.23.2` force-reinstall inherited verbatim from `gpu_finetune.sh`. Any new package would require a legitimacy audit first; there is none to run. |
</threat_model>

<verification>
- `pixi run lint`, `pixi run format-check`, `pixi run typecheck`, `pixi run test` all clean, coverage floor held.
- `select_query_embedding`'s existing test is unchanged and green after the `select_query_patch_index` refactor.
- `finetune-owlv2` focal mode AND flag-off contrastive mode: `epochs` array and `best_val_loss` identical to their preflight fixtures (Task 1 and Task 2, independently).
- 4-image overfit: contrastive term falls, `gap_class` widens, box losses stable, crop/scene self-score moves toward positive (Task 3's automated check).
- Both `contrastive-crop` result JSONs score 28/28 test plans through `run_domain_tuning`.
- `docs/benchmark/owlv2-finetune-comparison.json` carries ten rows; the eight pre-existing JSONs are byte-unchanged (`git status` clean for them).
- `pixi run docs-build` (strict) passes with the extended report.
- `vastai show instances` confirms no instance is left running.
</verification>

<success_criteria>
- Crop-context anchors exist behind `--supcon-crop-context`, layered on `--loss-mode
  contrastive`/`both`, with both existing modes provably unchanged when the flag is off.
- The crop-context anchor is built via the SAME functions `owlv2-oneshot`'s inference path uses
  (`select_query_patch_index`, `owlv2_preprocess_tensor`, `boxes_to_pixels`), not a reimplementation.
- A new instance-level diagnostic (`val_crop_scene_agreement`) and a single-exemplar `self_score`
  table directly measure whether the fix moves the property that broke -- crop/scene agreement --
  independently of whether F1 follows.
- The `contrastive-crop` arm is trained once on `headonly`, exported, and measured on the same
  28-plan test splits through the same harness as the existing four arms.
- The report states, in its first paragraph, whether crop-context supervision closes the gap.
- Nothing about `owlv2-oneshot`'s shipped default model path or behavior changed; all four
  previously-measured arms' numbers and artifacts are untouched.
- One instance rented, one destroyed, symlinks removed, worktree clean.
</success_criteria>

<output>
Create `.planning/quick/260808-dla-add-crop-context-supervision-to-the-owlv/260808-dla-SUMMARY.md` when done.
</output>
