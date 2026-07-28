---
id: 260727-fpe
title: Floor-plan domain evaluation + threshold tuning
status: complete
created: 2026-07-27
completed: 2026-07-27
branch: feat/floorplans-eval
---

# Summary — floor-plan domain evaluation + per-method threshold tuning

Onboarded the Roboflow **floor-plans-500** COCO dataset as the target-domain research surface and
added a per-method domain threshold-tuning pass, all as an extension of the existing `eval/` harness
(no core-schema change). The Elsevier paper's dataset was inaccessible/nonexistent, so its "metrics"
reduced to the harness's existing suite (P/R/F1@0.5, COCO AP/AP50/AP75, counting MAE/RMSE/NAE).

## What shipped (branch `feat/floorplans-eval`, 8 commits)

1. `docs(fpe)` — plan.
2. `feat(fpe)` — `eval/converters/floorplans.py` (`convert_floorplans`: COCO xywh→BBox,
   class-filtered, seeded exemplars, one split at a time) + two `manual-download` multi-split
   `DatasetSpec`s (`floorplans-door`, `floorplans-window`) via a `normalize_floorplans` fetch branch
   reusing the HF `NormalizedDataset` + `image_sources` provenance path.
3. `feat(fpe)` — committed native split manifests (`val 56 / test 28`, train empty).
4. `feat(fpe)` — additive optional `config` param threaded through
   `run_research_benchmark`/`_run_one_research` (defaults unchanged).
5. `feat(fpe)` — `eval/tuning.py`: tune each method's one acceptance knob on val (argmax F1@0.5),
   freeze, report tuned-vs-default on test; `tune-floorplans` CLI + pixi task.
6. `test(fpe)` — tiny COCO fixture (door/window/stairs distractor) + model-free (ncc) tests:
   converter, class filtering, registry fetch, native manifests, end-to-end tracer, tuning argmax.
7. `chore(fpe)` — `scripts/gpu_bench.sh` extended: manual scp drop, floor plans in the sweep, tuning
   pass.
8. `docs(fpe)` — `docs/eval/research-datasets.md`: dataset section, tuning protocol + per-method knob
   table, Roboflow export-drop instructions.

## Verification

- `pixi run lint` / `typecheck` clean; pre-commit green on every commit.
- **Full suite: 696 passed, 5 skipped, coverage 93.74%** (≥80% gate). New files: floorplans 89%,
  tuning 91%, datasets 86%.
- Converter validated on the real download (door val 56/test 28 = 760 boxes; window = 619). Config
  threading + `tune_method` argmax validated on real data (fast proxy config).

## Design decisions

- **Per-class single-class datasets** (door, window) over the same plans — no `GroundTruth` change.
- **Val + test only** converted (exemplar methods do no training); native val strategy.
- **One knob per method** for tuning (retain_frac / min_inliers / similarity_floor), hand-listed and
  editable; tuned config is always an instance of the method's own frozen `config_model`.

## Follow-ups (not in scope here)

- Run the real GPU sweep + tuning on vast.ai (`scripts/gpu_bench.sh` after scp'ing the export) to get
  the tuned-vs-default leaderboard and pick the best method for floor plans.
- Optional: a committed HTML render of the floor-plan tuning table (like `report-research`).
- Optional: generic class-aware GT (`label` per box) if true multi-class-per-image eval is wanted.
