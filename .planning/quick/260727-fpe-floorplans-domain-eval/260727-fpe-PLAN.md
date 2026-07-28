---
id: 260727-fpe
title: Floor-plan domain evaluation + threshold tuning
status: in-progress
created: 2026-07-27
owner: Enrique G. Ortiz
branch: feat/floorplans-eval
---

# Floor-plan domain evaluation (Roboflow floor-plans-500) + per-method threshold tuning

## Goal

Evaluate the six search methods on a **real, target-domain** dataset (architectural floor
plans), using the exemplar-search framing the product is built around: draw one `door` (or
`window`), find every other instance in the same plan. Then **tune each method's acceptance
threshold to this domain** (tune F1@IoU 0.5 on `val`, freeze, report tuned-vs-default on
`test`) so we can name the best method for floor plans and how much domain adaptation it
needs. Run the heavy sweep on a vast.ai GPU box.

The Elsevier paper's dataset does not exist / is inaccessible; its "metrics" collapse to the
harness's existing suite (P/R/F1@0.5, COCO AP/AP50/AP75, counting MAE/RMSE/NAE), which is what
a floor-plan detection paper reports anyway.

## Dataset (already downloaded)

`~/Downloads/floorPlans` — Roboflow floor-plans-500 export, **COCO format**, native splits
`train 197 / valid 56 / test 28`. Categories: `bathroom, door, perimeter, stairs, window`
(id 0 `floorplans` is an unused supercategory). We use **door** (1822/527/233 instances) and
**window** (1413/463/156) — both dense repeated symbols present in every image. COCO `bbox` is
`[x,y,w,h]` == repo `BBox` (round floats, clamp ≥0, drop <1px). `zone` is not in this export;
`bathroom/perimeter/stairs` are region-ish and excluded (single-per-room, variable extent).

## Design decisions

- **Per-class single-class datasets.** `GroundTruth` is single-class (CARPK discards its class
  column). Register two keys — `floorplans-door`, `floorplans-window` — each the same incoming
  tree filtered to one class. Zero core-schema change; exemplar=one door, GT=all doors.
- **Manual multi-split source.** Floor-plan images live per-split (not one flat `images/`), so
  fetch reuses the HF path's `NormalizedDataset` + `image_sources` provenance rather than the
  single-`images_subdir` walk. Only **val + test** are converted (exemplar methods need no
  training set); manifest `train` is empty, strategy `native`.
- **Config threading for tuning.** `run_research_benchmark`/`_run_one_research` use
  `spec.config_model()` (defaults only). Add an additive optional `config` param threaded to
  `run_multi_exemplar` (already accepts one). Backward-compatible (`None` → defaults).
- **Tuning knob per method** (the acceptance gate; small explicit grid, readable):
  - `ncc` / `mosse`: `retain_frac`
  - `sparse-geo`: `min_inliers`
  - `dino-dense`: `retain_frac`
  - `propose-retrieve`: `similarity_floor`
  - `owlv2-oneshot`: `retain_frac`
- **Metric = F1@IoU 0.5** for selection (reported alongside AP/AP50/AP75 + counting on test).

## Commits (atomic)

1. **Converter + registry** — `converters/floorplans.py` (`convert_floorplans` one split,
   class-filtered, seeded exemplar indices), export it; `datasets.py` two `DatasetSpec`s +
   `normalize_floorplans` + manual multi-split fetch branch.
2. **Splits + committed manifests** — `_VAL_STRATEGY` entries; real
   `dataset_splits/floorplans-{door,window}.split.json` (native, train=(), val, test).
3. **Benchmark config threading** — additive `config` param on `_run_one_research` +
   `run_research_benchmark`.
4. **Tuning harness** — `eval/tuning.py` (grids, `tune_method`, `run_domain_tuning`
   producing tuned-vs-default); CLI `tune-floorplans` + pixi task.
5. **vast.ai** — extend `scripts/gpu_bench.sh`: convert floorplans from `_incoming`, add both
   keys to the sweep, run the tuning pass.
6. **Fixtures + tests** — `tests/fixtures/research/floorplans/{valid,test}` tiny COCO;
   converter test, fetch test, end-to-end tracer (ncc, model-free), tuning test.
7. **Docs** — `docs/eval/research-datasets.md` floorplans entry + tuning protocol.

## Verification

- `pixi run lint`, `pixi run typecheck`, `pixi run test` (≥80% coverage) all green.
- New code covered by model-free (ncc) fixture tests — no ONNX weights needed in CI.
- Real sweep + tuning run on vast.ai; tuned-vs-default report pulled back.

## Success criteria

- `pixi run fetch-datasets` converts `floorplans-door`/`floorplans-window` from a dropped tree.
- `bench-research datasets=[floorplans-door,floorplans-window]` produces the full metric set.
- `tune-floorplans` picks per-method best-F1 config on val and reports tuned-vs-default on test.
