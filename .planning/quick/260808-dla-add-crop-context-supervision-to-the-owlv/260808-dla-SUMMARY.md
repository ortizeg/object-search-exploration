---
id: 260808-dla
title: Add crop-context supervision to the OWLv2 floor-plans SupCon fine-tune
status: complete
created: 2026-08-08
completed: 2026-08-08
branch: worktree/radiant-lark
---

# Summary — crop-context SupCon fix: the contrastive-loss failure is fixed, not fundamental

Added crop-context anchors to the SupCon contrastive fine-tune recipe from 260805-hg1, trained one
`headonly` arm on the same 197-image floor-plans train split, exported to ONNX, and measured it on
the same 28-plan test splits through the same tuning harness as every other arm.

**Headline result: the fix works.** 260805-hg1 diagnosed why the crop-context-free contrastive arm
collapsed (F1 0.010/0.009): SupCon training only ever supervised scene-context forward passes, never
the crop-context query-encoding path `owlv2-oneshot` uses at inference, so the exemplar's
self-similarity score drifted cosine-negative and broke calibration. This task adds crop-encoded
anchors to the SupCon pool — built via the **same** `select_query_patch_index` /
`owlv2_preprocess_tensor` / `boxes_to_pixels` functions the inference path itself calls, not a
reimplementation, to close off training/inference preprocessing drift as a risk — and re-measures.
Pooled `val_crop_scene_agreement.self_score_mean` moves from +0.490 (epoch 0) to +0.808 (epoch 7,
best checkpoint) and never dips negative. An independent, freshly-chosen exemplar (not the same one
260805-hg1's ad hoc diagnostic used) shows the same pattern even more sharply: baseline +0.368,
classification-headonly +0.556, crop-context-free contrastive **−0.200**, contrastive-crop **+0.859**
— the highest and tightest-retaining (8.0% of scene patches vs. 39–71% for the other three) of all
four checkpoints. F1 tracks it: door 0.229 tuned / 0.391 default (vs. 0.010 crop-context-free, 0.154
pretrained baseline) and window 0.216 (vs. 0.009 crop-context-free, 0.023 pretrained baseline) — the
best numbers of any fine-tuned arm measured across both quick tasks, by 2–39×. It does not reach
`propose-retrieve`'s 0.459 door F1 or `ncc`'s 0.403 window F1, so this repo's floor-plan
recommendation is unchanged — but 260805-hg1's conclusion that fine-tuning `owlv2-oneshot` is
foreclosed on this domain is retracted: fine-tuning works, once the training objective supervises the
exact cross-context comparison the method runs at inference. Full recipe, cosine-gap and self-score
tables, and provenance: `docs/reports/owlv2-floorplans-finetune.md` ("Third experiment").

## Task 4 checkpoint resolution

Live human checkpoint, reviewed directly (not pre-approved): `sanity-check.md`'s four numbers — the
contrastive loss falling, the cosine gaps widening, the box losses stable, and (the number this task
exists to prove) `val_crop_scene_agreement.self_score_mean` moving from +0.536 to +0.704 on a 4-image
overfit, never dipping negative — were reviewed alongside all four quality gates (847 passed, 5
skipped, 93.87% coverage) before approving the GPU spend, under the same standing full-autonomy
authorization established for 260801-8zy and reconfirmed for 260805-hg1 and this follow-up task.

## Commits (atomic, branch `worktree/radiant-lark`)

1. `63cc2d1` — **pre-dispatch plan**: goal-backward plan with verified facts (including the
   TO-CONFIRM check that `class_predictor`/`box_predictor` are independent of `query_embeds` —
   confirmed against the installed `transformers` source, so reusing `_forward_batch` for the crop
   forward is train/inference-consistent), decisions D-dla-01 through D-dla-07, and a STRIDE threat
   register.
2. `11dd79e` — **Task 1 tracer**: extracted `select_query_patch_index` from
   `owlv2_oneshot.select_query_embedding` as a shared, literally-reused function (not a
   reimplementation) so training and inference pick the query patch identically; new torch-free
   `crop_scene_agreement` diagnostic in `supcon.py`; `FinetuneConfig` gains `supcon_crop_context`/
   `supcon_query_iou_frac`; crop-context anchor wired into `_contrastive_rows` as an ordinary
   same-class SupCon positive — verified end to end with a real 1-image, 1-step checkpoint.
3. `03e9f98` — **Task 2 val-side diagnostic**: `val_crop_scene_agreement` per-epoch (with an epoch-0
   pre-training reference), degenerate-box retry-then-skip robustness (never raises, never NaNs), and
   two preflight/postflight fixture comparisons proving both `focal` and flag-off `contrastive` stay
   byte-identical — protecting 260805-hg1's already-committed numbers from this task's edits to the
   same code paths.
4. `dcc2273` — **Task 3 sanity check**: 4-image MPS overfit plus a `both`-mode + crop-context smoke
   test; GO verdict, all four numbers passing with real margins, none borderline.
5. (Task 4, no commit — live checkpoint approval to proceed to GPU spend.)
6. `83ebedd` — **Task 5 arm-selectable script**: `scripts/gpu_finetune.sh`'s `contrastive-crop` row,
   distinct checkpoint dir and ONNX filename from `contrastive` so neither collides. GPU run itself
   produced no further commit (result JSONs/ONNX/train_log are gitignored, matching every prior arm).
7. (this commit) — **Task 6 comparison + report + close-out**: `scripts/build_owlv2_finetune_comparison.py`
   extended to 5 arms / 10 rows; `docs/reports/owlv2-floorplans-finetune.md` rewritten top verdict
   covering all three experiments, new "Third experiment" section with the recipe, pooled
   crop/scene-agreement table, independent single-exemplar self_score table (4 checkpoints), F1 rows
   in both door/window tables, provenance (sha256, the two discarded vast.ai instances, the
   successful third), and a disposition that retracts the prior "fine-tuning is foreclosed" verdict.
   New `self_score_diagnostic.py` (CPU-only, no GPU) for the independent single-exemplar table.
   Symlinks and throwaway checkpoint dirs removed. SUMMARY + STATE.md.

## Infrastructure notes

- **First vast.ai instance went unreachable mid-training** (SSH connection refused, `vastai show
  instances` reporting `actual_status: offline` with `intended_status: running`, after ~3.7 hours of
  uptime — squarely inside the training window) — the same host-flakiness pattern documented in
  260801-8zy's provenance notes. Destroyed without a usable result; no partial artifacts were
  recoverable since SSH was refused, not just slow.
- **A same-priced replacement offer resolved to the exact same public IP (137.175.76.24) as a host
  already documented as bad from 260801-8zy's GPU run** — destroyed immediately, unused, before any
  training started. This is the second time in this repo's history that a "different" vast.ai offer
  has turned out to be the identical physical host; checking `machine_id`/public IP before trusting a
  replacement instance is now a load-bearing step, not a nice-to-have.
- A third instance (genuinely different IP, machine id 47340) trained the arm cleanly in one run
  (~1h20m). Pulled back sequentially (never concurrent ssh/rsync connections to the same box, per the
  lesson from 260801-8zy); the 360MB ONNX file's sha256 matched exactly between box and local copy.
- Three vast.ai instances were rented in total this task (two destroyed unused, one used); all three
  destroyed and confirmed via `vastai show instances` before this task closed.
- The executing background agent for this task's Tasks 1-3 repeatedly stopped itself mid-step waiting
  on background jobs it started, expecting a completion notification that did not reliably arrive
  while running as a background subagent. Worked around by having the orchestrator directly poll and
  resume it, and by taking over Task 5's GPU orchestration directly (rather than through a subagent)
  given the real-money stakes and the demonstrated unreliability of nested background-job notifications.

## Verification

- `pixi run quality`: **847 passed, 5 skipped, 93.87% coverage** (floor 80%) — measured twice
  (Task 3's checkpoint gate and Task 6's close-out gate), both green, no drift.
- `pixi run docs-build --strict`: clean.
- `docs/benchmark/owlv2-finetune-comparison.json` carries 10 rows (2 datasets × 5 arms); the eight
  pre-existing result JSONs were read, never rewritten.
- Both `contrastive-crop` result JSONs score 28/28 test plans; `train_log.json` carries epoch-0
  `val_cos_gap` AND `val_crop_scene_agreement` references alongside all 8 post-epoch measurements.
- `select_query_embedding`'s existing test passed unchanged after the `select_query_patch_index`
  extraction; a new cross-check test asserts the two never drift.
- Both `focal` and flag-off `contrastive` modes proven byte-identical to their preflight fixtures
  after this task's edits to the shared `_contrastive_rows`/`_evaluate`/`_epoch_record` code paths.
- Worktree clean: `datasets`/`models` symlinks removed, no weight/checkpoint/dataset file tracked by
  git, no untracked strays.
- Three vast.ai instances rented this task; all three destroyed and confirmed gone via `vastai show
  instances`.

## Disposition

`contrastive-crop` is not adopted as `owlv2-oneshot`'s shipped default — it remains an opt-in research
artifact (`OS_OWLV2_MODEL=owlv2_base_patch16_floorplans_ft_contrastive_crop.onnx`), and `ncc` /
`propose-retrieve` remain this repo's floor-plan recommendation, since neither door nor window F1
reaches their 0.459 / 0.403 numbers. What changes is the standing verdict on `owlv2-oneshot`
fine-tuning itself: 260805-hg1's conclusion that it was foreclosed on this domain is retracted.
Crop-context supervision is the best-measured OWLv2 configuration on floor plans by a wide margin over
every other fine-tuned arm and over the pretrained baseline, and the mechanism generalizes beyond this
one method: any exemplar-based method whose training data comes from one context (scene) but whose
inference query comes from another (crop) needs to supervise that specific cross-context comparison
explicitly, or risk exactly this failure mode.
