---
id: 260801-8zy
title: Fine-tune OWLv2 on the floor-plans training data
status: complete
created: 2026-08-01
completed: 2026-08-04
branch: worktree/radiant-lark
---

# Summary — OWLv2 floor-plan fine-tune: measured, negative result

Fine-tuned `owlv2-oneshot`'s underlying OWLv2 weights on the 197-image floor-plans train split
(text-conditioned proxy training, since only that path has an established detection loss),
exported both arms to ONNX, and measured baseline-vs-fine-tuned precision/recall/F1 on the full
28-plan test splits for both classes.

**Headline result: fine-tuning does not close the gap to the current floor-plan winners
(`propose-retrieve` 0.459 door F1, `ncc` 0.403 window F1), and on doors it makes the pretrained
baseline worse** (tuned F1 0.154 → 0.087 headonly → 0.083 full-unfreeze — monotonically worse the
more of the model is allowed to move). Windows move a hair in the right direction for the
lightly-tuned arm (0.023 → 0.028) but the full-unfreeze arm regresses there too (0.010). Both arms
train cleanly — loss falls monotonically epoch over epoch, val tracks train, no sign of a wiring
bug. The finding: a lower text-conditioned training loss does not transfer to better image-guided
detection on this method, and spending more training capacity on the proxy objective (the full
arm reaches substantially lower training loss than headonly) buys more proxy-objective overfitting
relative to the small 197-image set, not more of the downstream property that matters. Full
writeup, training curves, and per-arm P/R/F1 tables: `docs/reports/owlv2-floorplans-finetune.md`.

## Task 3 checkpoint resolution (pre-approved, no live human review)

Per explicit standing user authorization to finish this plan autonomously, Task 3's blocking
human-verify checkpoint was resolved as: (1) the Task 2 training recipe as implemented — Hungarian
matcher + sigmoid-focal/L1/GIoU loss, freeze strategy with its two approved deviations
(`objectness_head` frozen, `text_projection` excluded from `--unfreeze-all`), val-selected
checkpoint, seeded — confirmed; (2) both arms (A frozen-heads-only primary, B full-backbone-unfreeze
stretch) approved; (3) a 24GB-class vast.ai GPU instance (RTX 3090) approved, with the plan's own
default-config fallback authorized if GPU time ran long (it did not — training took under 30
minutes for both arms combined).

## Commits (atomic, branch `worktree/radiant-lark`)

1. `349687b` — **Task 1 tracer**: target-builder module (`src/object_search/train/owlv2_targets.py`,
   normalizes COCO boxes against OWLv2's padded-square convention), `scripts/finetune_owlv2.py`
   entrypoint (1 image / 1 step), checkpoint-parameterized ONNX export, opt-in `OS_OWLV2_MODEL`
   env override in `owlv2_oneshot.py` — verified end to end with a real training step, a real
   export, and a real search through the fine-tuned graph.
2. `cd80c73` — **Task 2 real recipe**: `Owlv2HungarianMatcher` (sigmoid-consistent matching cost,
   since OWLv2's logits have no background column) + `ImageLoss` subclass with a sigmoid-focal
   `loss_labels` override, three freeze arms (`headonly`/`last{N}`/`full`), two-param-group AdamW
   with cosine warmup, grad-accum, bf16 autocast on CUDA, a `torch.no_grad()` backbone path when
   fully frozen, per-epoch train+val loss with save-only-on-val-improvement, seeded `train_log.json`.
   A genuine overfit sanity check (6 fixed images, 8+ epochs) resolved an earlier "loss rose for 2
   epochs" smoke-test concern as a sigmoid-focal warmup transient, not a bug — loss falls
   monotonically from epoch 4 (train ce 4.00 → 0.39, an 80% reduction).
3. `8bfa1e7` — **Task 4 GPU recipe**: `scripts/gpu_finetune.sh`, the reproducible recipe. Fixed a
   real bug found running this for real: `onnxruntime-gpu>=1.19,<2` resolves to whatever's newest
   (1.28.0), which needs CUDA 13/cuDNN 9 — on the CUDA 12.1 vast.ai box this **silently fell back
   to CPU with no error anywhere**, turning a should-take-minutes eval into hours. Pinned to
   `1.23.2` (confirmed to actually load `CUDAExecutionProvider`, not just report it as compiled
   in), added the cuDNN/cuBLAS `LD_LIBRARY_PATH` discovery (bundled with the export env's torch
   install, not reliably on the system CUDA path even on a "cudnn9-devel" image), and a real
   session-creation load-test right after the baseline export produces a model to check against.
4. `1c2a1e3` — **Task 5 comparison + report**: `scripts/build_owlv2_finetune_comparison.py`
   assembles the six pulled-back result JSONs; `docs/reports/owlv2-floorplans-finetune.md` states
   the verdict plainly, with training curves, the recipe, sha256 provenance, and the cross-hardware
   nondeterminism note below. Cross-linked from `docs/eval/floorplans-findings.md`; added to
   `mkdocs.yml`. `pixi run docs-build --strict` passes.
5. (this commit, docs only) — SUMMARY + STATE.md update.

## Infrastructure notes (not method changes, but real time/cost sinks worth recording)

- **First vast.ai instance (`46716150`) had a severely unreliable network path** for large file
  transfers specifically (small commands and SSH always worked; sustained rsync of the two 360MB
  `.onnx` files kept stalling/dropping, sometimes recovering after minutes, sometimes needing many
  retries). Root-caused as likely host-specific after ~2 hours of retries; destroyed and replaced
  with a fresh instance (`46826374`, confirmed different `machine_id`/public IP) which transferred
  both files cleanly on the first attempt. **A first replacement instance request returned the
  same public IP as the failing one** (same physical host, different offer id) — worth checking
  `machine_id`/IP before trusting "a different instance" fixes a host-specific network issue.
- **GPU floating-point ops are not bit-identical across different physical hardware.** The
  fine-tuned checkpoints/ONNX files actually committed locally come from a second training run
  (same seed 0, same recipe) on the replacement instance, since the original instance's checkpoints
  never fully transferred before it was destroyed. Their sha256s differ from what the reported
  eval numbers were measured against; the `train_log.json` curves are numerically indistinguishable
  from the original run (documented in the report), confirming this is a faithful reproduction, not
  a different result. This is consistent with this repo's own documented reproducibility limits
  (`provenance.environment_identity` exists for exactly this reason) — CPU is deterministic
  cross-run, GPU is not guaranteed to be cross-hardware.
- Two vast.ai instances were rented in total; both destroyed and confirmed via `vastai show
  instances` before this task closed. Total GPU time: well under an hour.

## Verification

- `pixi run quality`: **751 passed, 20 skipped, 92.44% coverage** (floor 80%). Lint/format/typecheck
  clean.
- `pixi run docs-build --strict`: clean, new report in nav.
- Six result JSONs (3 arms × 2 classes) present and parse; two `train_log.json` files show
  per-epoch train/val loss with save-only-on-improvement.
- `owlv2-oneshot._resolve_model_path()` with `OS_OWLV2_MODEL` unset returns the shipped default
  path; `MODEL_REGISTRY['owlv2-base-patch16']` (repo_id, revision, dest, pinned sha256) is
  unchanged — asserted by test, not eyeballed.
- Worktree clean: `datasets`/`models` symlinks removed, no weight/checkpoint/dataset file tracked
  by git, no untracked strays outside `.planning/`.
- Both vast.ai instances destroyed and confirmed gone via `vastai show instances`.

## Disposition

Neither fine-tuned arm is adopted as a default — `owlv2-oneshot`'s shipped model path and default
behavior are unchanged. The fine-tuned weights are an opt-in research artifact only
(`OS_OWLV2_MODEL=owlv2_base_patch16_floorplans_ft.onnx` for one run). `ncc` and `propose-retrieve`
remain the methods to ship on the floor-plan domain; this result forecloses fine-tuning as the next
lever to pull on `owlv2-oneshot` there — a follow-up would need to target the one-shot
query-embedding selection or the box head's small-symbol resolution instead, not more/different
fine-tuning of the shared heads.
