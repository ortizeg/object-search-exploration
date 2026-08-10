---
id: 260808-w8c
title: Crop context-margin padding + rotation/mirror-augmented crop positives for OWLv2 floor-plans
status: complete
created: 2026-08-08
completed: 2026-08-09
branch: worktree/radiant-lark
---

# Summary — two cheap levers on contrastive-crop: one split, one a class-asymmetric win

Two independently-motivated improvements to 260808-dla's `contrastive-crop` recipe, sequenced so the
cheap one was genuinely measured before any GPU money funded the second.

**Lever A — crop context-margin padding (inference-only sweep, zero retraining).** Sweeping
`crop_context_margin_frac` (0.0/0.15/0.3/0.5) against the already-exported `contrastive-crop`
checkpoint through the same tune-on-val/freeze/report-on-test harness (a new backward-compatible
`grids` override on `run_domain_tuning` made this possible without forking the tuning loop; the
margin=0.0 cell reproduces the committed numbers exactly, confirming the plumbing). **Result: door
peaks at margin=0.15 (0.229→0.277, +21%) but window degrades monotonically at every nonzero margin
(0.216→0.186 at 0.15, worse beyond) — no margin beats 0.0 on both classes, so margin was left out of
the final training arm.** This is a real, reportable finding: a margin that pulls in surrounding
context helps door symbols but confuses window symbols with neighboring geometry.

**Lever B — rotation/mirror-augmented SupCon crop positives (trained as `contrastive-crop-v2`).**
Adding one additional randomly-rotated/mirrored view of the crop-context anchor as a second same-class
SupCon positive per training image, reusing the exact "ordinary positive" mechanism 260808-dla
established (zero changes to `supcon_loss`'s math). **Result: door F1 improves to 0.253 tuned / 0.433
default — the best door numbers of any fine-tuned arm measured — while window dips slightly to 0.204
(from 0.216).** The crop/scene self-similarity mechanism the whole recipe depends on holds and
slightly improves: pooled self-score moves +0.479→+0.821 over training, and an independent
single-exemplar diagnostic (extended to five checkpoints) puts `contrastive-crop-v2` at the highest
self-score of all of them (+0.896, vs +0.859 for `contrastive-crop`).

Neither `contrastive-crop` nor `contrastive-crop-v2` reaches `propose-retrieve`'s 0.459 door F1 or
`ncc`'s 0.403 window F1, so the floor-plan recommendation is unchanged. What changes is which
fine-tuned arm is "best": it is now class-dependent — `contrastive-crop-v2` for door,
`contrastive-crop` marginally for window — rather than a single clean winner. Full recipe, both
tables, and provenance: `docs/reports/owlv2-floorplans-finetune.md` ("Fourth experiment").

## Task 5 checkpoint resolution (live human review)

Live human checkpoint, same as 260808-dla's Task 4/5: `sanity-check.md`'s four numbers (contrastive
loss falling, cosine gaps widening, box losses stable, and the crop/scene self-score moving from
+0.525 to +0.757 with augmentation on) plus all four quality gates (859 passed after a real fix —
see below) were reviewed and approved before renting the GPU, under the same standing full-autonomy
authorization established across all four prior quick tasks in this research thread.

## Commits (atomic, branch `worktree/radiant-lark`)

1. `f74854e` — **pre-dispatch plan**: goal-backward plan with verified facts, 10 locked decisions
   (D-w8c-01–10), and a STRIDE threat register, matching 260808-dla's rigor.
2. `93dfe3c` — **Task 1 tracer**: shared `expand_box_with_margin` helper (used by both inference and,
   opt-in, training — the same fidelity guarantee D-dla-04 gave query-patch selection), new
   `Owlv2OneshotConfig.crop_context_margin_frac` field (default 0.0, behavior-preserving), wired into
   `search()`'s step 1. Four new unit tests plus a stub-driven test proving the same exemplar box
   yields a materially larger query crop only when the field is set.
3. `dc31db5` — **Task 2 (part 1)**: additive `grids` override on `object_search.eval.tuning`'s
   `_tune_methods_at_count`/`run_domain_tuning`, defaulting to `None` (today's exact code path,
   pinned by a regression test) given this module's high blast radius (every method's domain-tuning
   run goes through it).
4. `70af4b1` — **Task 2 (part 2)**: the local margin sweep (run on a vast.ai RTX 3090 after a local
   CPU attempt was killed after 35+ minutes with the first of 8 cells still incomplete) and its
   explicit verdict in `margin-verdict.md` — no margin beats 0.0 on both classes.
5. `98b3ed9` — **Task 3 tracer**: `--supcon-crop-augment` (rotation/mirror crop-positive
   augmentation, layered on `--supcon-crop-context`) and `--supcon-crop-margin-frac` (threading
   `expand_box_with_margin` into the training-side crop-context anchor), both default off/0.0. A
   preflight/postflight fixture comparison proves `--supcon-crop-context` alone stays byte-identical
   after these edits to the shared `_crop_context_rows` function.
6. `0150e3b` — **Task 4 sanity check**: 4-image MPS overfit with augmentation on (margin left at
   0.0 per the verdict) plus a `both`-mode + crop-augment smoke test. GO verdict. Also fixed a real
   test failure the quality gate caught: `crop_context_margin_frac` (added in Task 1) was missing
   from `docs/methods/owlv2-oneshot.md`'s config reference (DOC-04 gate).
7. (Task 5, no commit — live checkpoint approval to proceed to GPU spend.)
8. `ebd2fac` — **Task 6 arm-selectable script**: `scripts/gpu_finetune.sh`'s `contrastive-crop-v2`
   row (base flags plus `--supcon-crop-margin-frac` only when `OWLV2_MARGIN_FRAC` is nonzero — 0.0 by
   default), with the eval loop threading the same margin into `Owlv2OneshotConfig` via the `grids`
   override so the arm is scored with the crop it was trained to expect. GPU run itself produced no
   further commit (result JSONs/ONNX/train_log are gitignored, matching every prior arm).
9. (this commit) — **Task 7 comparison + report + close-out**:
   `scripts/build_owlv2_finetune_comparison.py` extended to 6 arms / 12 rows;
   `docs/reports/owlv2-floorplans-finetune.md` rewritten top verdict covering all four experiments,
   new "Fourth experiment" section with both levers' recipes, tables, and verdicts, the five-checkpoint
   self-score diagnostic, provenance, and a disposition stating the class-dependent choice plainly.
   New `self_score_diagnostic.py` (extends 260808-dla's, CPU-only, no GPU) for the five-checkpoint
   table. Symlinks and throwaway checkpoint dirs removed. SUMMARY + STATE.md.

## Infrastructure notes

- **The local CPU margin sweep was abandoned after 35+ minutes** with the first of 8 (margin,
  dataset) cells still incomplete — each cell re-runs a full 9-entry tuning grid over the whole
  56-image val split plus tuned/default test evaluation, unlike the lightweight single-exemplar
  diagnostics used elsewhere. Moved to a vast.ai RTX 3090 with `OS_ONNX_PROVIDERS=
  CUDAExecutionProvider,CPUExecutionProvider`, which finished all 8 cells in ~34 minutes.
- **A rental attempt on offer `43824373` resolved to the documented-bad IP `137.175.76.24`** —
  destroyed immediately, unused, before any setup. A different offer (machine 47014) worked cleanly.
- **`contrastive-crop-v2`'s training run stalled on `pixi run fetch-datasets`'s full sweep**, which
  converts every registered eval dataset (not just floor plans) and hit a repeated HF Hub read
  timeout downloading an unrelated FSCD-LVIS archive. Worked around by killing the stalled step and
  running `fetch-datasets --only floorplans-door` / `--only floorplans-window` instead (seconds, not
  the multi-GB unrelated downloads), then continuing the rest of `gpu_finetune.sh`'s steps manually
  with the same env setup. Training/export/eval then completed cleanly in ~57 minutes total.
- Three vast.ai instances were rented across this task in total (one discarded unused for the
  known-bad IP, one used for the margin sweep, one used for training); all three destroyed and
  confirmed via `vastai show instances`.

## Verification

- `pixi run quality`: **859 passed, 5 skipped, 93.89% coverage** (floor 80%) — measured twice (once
  catching the missing config-reference doc entry, once clean after the fix).
- `pixi run docs-build --strict`: clean.
- `docs/benchmark/owlv2-finetune-comparison.json` carries 12 rows (2 datasets × 6 arms); the ten
  pre-existing result JSONs were read, never rewritten.
- Both `contrastive-crop-v2` result JSONs score 28/28 test plans; `train_log.json` carries epoch-0
  `val_cos_gap` AND `val_crop_scene_agreement` references and `supcon_crop_augment: true` in its
  logged config.
- The margin sweep's `margin=0.0` cell reproduces `contrastive-crop`'s committed tuned F1 exactly on
  both datasets — the live regression check on the new `grids` plumbing.
- `--supcon-crop-context` alone (augment off, margin 0.0) proven byte-identical to its preflight
  fixture after Task 3's edits to the shared `_crop_context_rows` function.
- Worktree clean: `datasets`/`models` symlinks removed, no weight/checkpoint/dataset file tracked by
  git, no untracked strays.
- Three vast.ai instances rented this task; all three destroyed and confirmed gone via `vastai show
  instances`.

## Disposition

Neither `contrastive-crop` nor `contrastive-crop-v2` is adopted as `owlv2-oneshot`'s shipped
default — both remain opt-in research artifacts, and `ncc`/`propose-retrieve` remain this repo's
floor-plan recommendation, since neither reaches their 0.459 door / 0.403 window F1. The margin lever
(A) is measured and disclosed but not carried into any shipped recipe — a genuine negative-on-window
result. The augmentation lever (B) is a real, class-asymmetric improvement: `contrastive-crop-v2`
is now the best-measured OWLv2 configuration for door detection specifically (0.253 tuned / 0.433
default F1), while `contrastive-crop` remains marginally better for window (0.216 vs 0.204) if window
is the priority class. Anyone choosing between the two fine-tuned arms should pick based on which
class matters more for their use case, not assume one dominates the other.
