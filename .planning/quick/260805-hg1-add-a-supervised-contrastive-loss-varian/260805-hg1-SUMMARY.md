---
id: 260805-hg1
title: Add a supervised-contrastive loss variant to OWLv2 floor-plans fine-tuning
status: complete
created: 2026-08-05
completed: 2026-08-08
branch: worktree/radiant-lark
---

# Summary — SupCon contrastive loss: measured, negative result with an identified mechanism

Added a supervised-contrastive (SupCon, Khosla et al. 2020, `L_out` formulation) loss variant
(`--loss-mode contrastive`) to the OWLv2 floor-plans fine-tuning recipe from 260801-8zy, trained one
`headonly` arm on the same 197-image train split, exported to ONNX, and measured it on the same
28-plan test splits through the same tuning harness as the three already-measured arms.

**Headline result: the contrastive objective is worse than the classification-loss recipe it was
meant to improve on — door F1 0.010, window F1 0.009, both roughly an order of magnitude below the
pretrained baseline (0.154 / 0.023) and below the classification-loss `headonly` arm (0.087 /
0.028).** This is a sharper negative result than 260801-8zy's, not just a repeat of it: the SupCon
loss does demonstrably learn the intended property — `val_cos_gap.gap_class` more than triples
(0.122 → 0.415) and `gap_background` more than triples (0.248 → 0.840) over training, in the exact
L2-normalized `class_embeds` space `owlv2-oneshot` scores cosine similarity over at inference — but
F1 collapsed anyway. A follow-up offline diagnostic (comparing `select_query_embedding`'s behavior
across all three checkpoints on the same exemplar) found the mechanism: the SupCon batches are built
entirely from **scene-context** forward passes (matched anchors + background patches from full
floor-plan images), and the **crop-context** forward pass that produces the query embedding at
inference — from encoding the small cropped exemplar image alone — was never part of the training
loop's loss at all. For the contrastive checkpoint this cross-context consistency broke completely:
the exemplar's own `self_score` (its scene-side patch scored against its own crop-side query
embedding) goes **cosine-negative** (−0.297, vs. +0.71/+0.67 for baseline/classification), which
makes the self-similarity calibration threshold negative and retains ~86% of all scene patches
(3080/3600) instead of the ~25–30% the other checkpoints retain. Not "wrong patches outrank right
ones" — the calibration threshold stopped rejecting anything. Full recipe, cosine-gap table,
query-patch/self-score table, and provenance: `docs/reports/owlv2-floorplans-finetune.md`.

## Task 4 checkpoint resolution (live human review)

Unlike 260801-8zy's Task 3, this plan's Task 4 (GPU-spend approval) was a **live** human checkpoint,
not pre-approved: the sanity-check numbers (4-image overfit, `train_loss_supcon` 4.539 → 4.009,
`gap_class` +47%, `gap_background` +61%, box losses falling normally) were reviewed and approved
before renting the GPU, under the same standing full-autonomy authorization as 260801-8zy for
everything downstream of that approval.

## Follow-up: offline diagnostic (new scope, separately authorized)

After the measured F1 collapse, the user asked for a diagnostic into why, specifically the
query-selection heuristic and the calibration threshold. Run on a fresh vast.ai instance (Python 3.12
via `uv venv`, since the repo's PEP 695 generic syntax needs 3.12+) comparing all three checkpoints'
`select_query_embedding` output and self-similarity calibration on the same floor-plan exemplar. The
result — the crop-vs-scene cross-context divergence described above — was reviewed with the user, who
then asked to prototype a fix (adding crop-context anchors to the contrastive batch) as further new
scope; that fix is not part of this quick task and is tracked separately.

## Commits (atomic, branch `worktree/radiant-lark`)

1. `c031eff` — **Task 1 tracer**: torch-free NumPy SupCon spec (`src/object_search/train/supcon.py`:
   `supcon_loss`, `background_patch_mask`, `patch_grid_size`), `FinetuneConfig` gains `loss_mode`/
   `supcon_temperature`/`w_contrast`, `scripts/finetune_owlv2.py` stops discarding `class_embeds`,
   adds the torch `supcon_loss_torch` mirror and `--self-check` (torch vs. NumPy agree to <1e-6, finite
   non-zero gradient) — verified with a real 1-image, 1-step contrastive checkpoint.
2. `ca62ccc` — **Task 2 real recipe**: background-patch denominator-only negatives
   (`sample_background_indices`), anchors and background pooled across the full effective batch
   (D-hg1-04, so rare classes like stairs don't get starved to zero at small per-micro-batch pools),
   per-epoch `val_cos_gap` diagnostics including an epoch-0 pre-training reference. Focal mode proven
   behavior-preserving against a preflight fixture captured from the unmodified script.
3. `4321eef` — **Task 3 sanity check**: 4-image MPS overfit run plus a `--loss-mode both` CPU smoke
   test; GO verdict recorded in `sanity-check.md` with the three required numbers, none borderline.
4. (Task 4, no commit — live checkpoint approval to proceed to GPU spend.)
5. `9ae356a` — **Task 5 arm-selectable script**: `scripts/gpu_finetune.sh` rewritten to drive arms
   from a lookup table (`ARMS` env var) instead of three hard-coded arms, adding the `contrastive` row
   without touching `baseline`/`headonly`/`full`. Trained on a vast.ai RTX 3090 (contrastive, headonly
   freeze, 8 epochs, seed 0), exported, evaluated on both test splits, pulled back sequentially,
   instance destroyed and confirmed via `vastai show instances`.
6. (this commit) — **Task 6 comparison + report + close-out**: `scripts/build_owlv2_finetune_comparison.py`
   extended to 4 arms / 8 rows; `docs/reports/owlv2-floorplans-finetune.md` rewritten top verdict
   covering both experiments, new "Second experiment" section with the recipe, cosine-gap table, the
   diagnostic's query-patch/self-score table and mechanism explanation, `both`-mode-implemented-not-
   measured note, contrastive provenance (sha256, second vast.ai instance, third instance for the
   diagnostic), disposition extended to cover the contrastive arm. Symlinks and throwaway checkpoint
   dirs removed. SUMMARY + STATE.md.

## Verification

- `pixi run quality`: **831 passed, 5 skipped, 93.81% coverage** (floor 80%). Lint/format/typecheck
  clean. (Real-model tests require the `models`/`datasets` symlinks in place; quality was run before
  the final symlink removal, per the plan's own verify ordering — removing them first produces 15
  spurious `FileNotFoundError` failures in the real-graph tests, not a code regression.)
- `pixi run docs-build --strict`: clean.
- `docs/benchmark/owlv2-finetune-comparison.json` carries 8 rows (2 datasets × 4 arms); the six
  pre-existing result JSONs were read, never rewritten.
- Both contrastive result JSONs score 28/28 test plans; `train_log.json` carries an epoch-0
  `val_cos_gap` reference alongside all 8 post-epoch measurements.
- Worktree clean: `datasets`/`models` symlinks removed, no weight/checkpoint/dataset file tracked by
  git, no untracked strays.
- One vast.ai instance for training (rented, used, destroyed, confirmed) plus one further instance for
  the follow-up diagnostic (rented, used, destroyed, confirmed) — `vastai show instances` returns
  empty.

## Disposition

Neither the classification-loss arms nor the contrastive arm is adopted as a default —
`owlv2-oneshot`'s shipped model path and default behavior are unchanged. All three fine-tuned weight
sets are opt-in research artifacts only, reachable via `OS_OWLV2_MODEL=<name>.onnx` for one run.
`ncc` and `propose-retrieve` remain the methods to ship on the floor-plan domain. This result forecloses
fine-tuning `owlv2-oneshot`'s heads on floor plans as a productive lever, classification or contrastive
objective alike, **unless** a future attempt specifically supervises the crop-context query encoding
path (not just scene-context patches) — the diagnostic's mechanism finding is the concrete next lever,
not more variations on scene-only training.
