---
phase: 11-research-dataset-evaluation-harness
plan: 01
subsystem: eval
tags: [research-datasets, carpk, converters, provenance, coco-ap, counting-metrics, tracer]
status: complete
requirements: [EVAL-21, EVAL-22, EVAL-23, EVAL-24]
provides:
  - object_search.eval.datasets (DatasetSpec, DATASET_REGISTRY, fetch, provenance)
  - object_search.eval.converters.convert_carpk
  - object_search.eval.splits (ResearchSplitManifest, load_split_manifest, research_image_ids)
  - object_search.eval.labels.load_research_ground_truth
  - object_search.eval.metrics.average_precision_coco
  - object_search.eval.metrics.counting_errors
  - object_search.eval.benchmark.run_research_benchmark
  - pixi task fetch-datasets / CLI fetch-datasets
requires:
  - object_search.eval.metrics.average_precision (reused per IoU)
  - object_search.eval.labels._parse_sidecar (the single GT reader, D-10)
  - object_search.inference.models (mirrored for DatasetSpec/fetch)
affects: [.gitignore, pixi.toml, conf/datasets/]
key-files:
  created:
    - src/object_search/eval/datasets.py
    - src/object_search/eval/converters/__init__.py
    - src/object_search/eval/converters/carpk.py
    - src/object_search/eval/splits.py
    - conf/datasets/carpk.split.json
    - tests/fixtures/research/carpk/ (3 synthetic CARPK-native images + annotations + README)
    - tests/test_eval_metrics_coco.py
    - tests/test_research_carpk_tracer.py
    - tests/test_research_datasets_fetch.py
  modified:
    - src/object_search/eval/metrics.py
    - src/object_search/eval/labels.py
    - src/object_search/eval/benchmark.py
    - src/object_search/cli.py
    - pixi.toml
    - .gitignore
decisions:
  - "Task 1 checkpoint ratified as option-a (committed manifests under conf/datasets/), proceeded autonomously per the pre-ratified plan/CONTEXT."
  - "Additive exemplar_indices field on GroundTruth accepted (default falls back to (exemplar_index,))."
metrics:
  completed: 2026-07-26
  tasks: 2
  files: 15
status_detail: complete
---

# Phase 11 Plan 01: CARPK research-dataset tracer Summary

The research-dataset architecture is proven end-to-end on CARPK, entirely offline on a committed
synthetic fixture: fetch → provenance manifest → native-annotation converter → the existing
`*.gt.json` sidecar (single loader) → committed split manifest → `ncc` through the benchmark at
1 exemplar → one report row carrying the full literature-metric column set
(P/R/F1 + AP/AP50/AP75 + MAE/RMSE/NAE). This is the one-way on-disk contract the other three
datasets (11-02/11-03) build on.

## Checkpoint proceeded under (Task 1 — `checkpoint:decision`, one-way)

Task 1 ratifies the on-disk research-dataset contract before any real data is baked in. The
decision was already ratified in the plan and 11-CONTEXT (D-08 provenance/gitignore, D-10 the
single sidecar loader), so per the orchestrator's instruction I proceeded **autonomously under
option-a** without stopping for interactive input:

- **option-a (committed manifests under `conf/datasets/<dataset>.split.json`)** — chosen. Manifests
  live with the Hydra benchmark config, are committed and diffable, and are cleanly separated from
  the gitignored raw tree with no `.gitignore` negation.
- Additive `exemplar_indices` sidecar field — accepted (default falls back to `(exemplar_index,)`
  via `GroundTruth.effective_exemplar_indices`, so every existing single-exemplar sidecar is
  unchanged; FSCD-* three-exemplar sidecars will fit later).

Concrete layout baked in: raw data at `datasets/<dataset>/…` (gitignored) with a human-drop path
`datasets/_incoming/<dataset>/`; converted sidecars + co-located scenes at
`datasets/<dataset>/<split>/`; provenance at `datasets/provenance.json`; committed split manifests
at `conf/datasets/<dataset>.split.json`.

## Pending USER action — real CARPK (user_setup)

The real CARPK/PUCPR+ data is **licence-gated** (author terms-of-use, non-commercial research; no
unauthenticated direct-download URL). It was **not** fetched, and no real-dataset results are
claimed — the tracer proves the pipeline on the committed synthetic fixture only. To run the real
data locally:

1. `pixi run fetch-datasets --list` — prints the CARPK entry, its licence, and the drop path
   `datasets/_incoming/carpk/`.
2. Accept the licence at https://lafi.github.io/LPN/ and place the raw archive (or an extracted
   `Images/` + `Annotations/` tree) at `datasets/_incoming/carpk/`.
3. `pixi run fetch-datasets` — verifies/records SHA-256 + source + licence into
   `datasets/provenance.json`, converts to `datasets/carpk/test/`, never re-hosts or commits raw
   bytes.

This is a local-only step (never CI); the fixture path needs no network and no licence acceptance.

## Task-by-task

- **Task 1 (checkpoint:decision):** Ratified option-a + `exemplar_indices` as above; proceeded.
- **Task 2 (tracer, tdd):** All 15 files landed. RED tests written first (metrics-COCO,
  CARPK tracer, fetch/provenance/gitignore), then implemented to green:
  - `metrics.average_precision_coco` — COCO IoU sweep `[0.50..0.95]`, calls the **existing**
    `average_precision` per IoU so **AP50 == the pre-existing single-IoU-0.5 number** (asserted
    exactly; no drift). `metrics.counting_errors` — MAE/RMSE/NAE, NAE guards `true==0` explicitly.
  - `datasets.py` — mirrors `models.py`: frozen `DatasetSpec` (+ `requires_manual`,
    `incoming_subdir`), `DATASET_REGISTRY` (CARPK only), `fetch`/`fetch_all`/`verify_all`,
    `write_provenance_manifest` → `datasets/provenance.json`. Zip-slip guard (T-11-02) and the
    graceful-absence path (T-11-05) are tested.
  - `converters/carpk.py::convert_carpk` — native `x1 y1 x2 y2 class` → the existing GroundTruth
    schema, converting inclusive corners to half-open `BBox(x=x1,y=y1,w=x2-x1,h=y2-y1)` at the
    boundary; co-locates the scene beside the sidecar. **No second parser** — grep confirms one
    `_parse_sidecar`.
  - `labels.load_research_ground_truth` — thin `_parse_sidecar` wrapper tagged `source="research"`.
  - `splits.py` — frozen `ResearchSplitManifest` + loaders; `conf/datasets/carpk.split.json`
    committed as `val_strategy="test-only"`.
  - `benchmark.run_research_benchmark` — threads `dataset`/`split`/`exemplar_count` + `ap50`/`ap75`
    + per-image counts through additive `ImageResult` fields and reuses the same match/precision/
    candidate-log scoring; the chipset/CI path stays byte-identical (new fields default `None`).
  - `fetch-datasets` Typer command + pixi task; `/datasets/` added to `.gitignore`.

## Deviations from Plan

- **[Rule 1 — Bug] `.gitignore` rule anchored to `/datasets/`.** The plan said "add `datasets/`".
  An unanchored `datasets/` **also matched `conf/datasets/`**, silently ignoring the committed split
  manifest the whole option-a decision depends on. Fixed to `/datasets/` (repo-root anchored);
  verified `git check-ignore` now tracks `conf/datasets/` while still ignoring root `datasets/`. The
  `.gitignore` still contains the substring `datasets/`, so the acceptance test is satisfied.
- **Per-image `ap` on the research path is the COCO mean AP** (with `ap50`/`ap75` alongside), not
  the single-IoU-0.5 AP the chipset path stores in the same field. This is the literature's headline
  AP for research rows (D-09); the chipset path is untouched.

No architectural (Rule 4) changes; no auth gates.

## Known Stubs

None. The committed `conf/datasets/carpk.split.json` `test` list holds the **fixture** ids by design
(the real ids are generated on fetch); this is the ratified test-only tracer manifest, not a stub.

## Verification output

- `pixi run lint` → `All checks passed!`
- `pixi run format-check` → `125 files already formatted`
- `pixi run typecheck` → `Success: no issues found in 69 source files` (mypy strict, no new ignores)
- `pixi run test` → `505 passed, 19 skipped`; `Required test coverage of 80% reached. Total
  coverage: 88.78%`
- `pixi run quality` (umbrella: lint + format-check + typecheck + test) → green.
- Targeted `pytest -k "research or coco or counting_errors"` → `23 passed`.
- `pixi run fetch-datasets --list` → prints CARPK entry + licence + `datasets/_incoming/carpk` drop
  path, exit 0.
- `git ls-files datasets/` → empty (no raw dataset file tracked).
- Grep: exactly one `_parse_sidecar`; `converters/carpk.py` only `json.dumps` (writes), never parses
  a sidecar — D-10 one-loader holds.

## Self-Check: PASSED

- Created files exist: `src/object_search/eval/datasets.py`, `.../converters/carpk.py`,
  `.../splits.py`, `conf/datasets/carpk.split.json`, `tests/fixtures/research/carpk/` (verified on
  disk).
- New symbols import and run: full suite (505 passed) exercises `convert_carpk`,
  `average_precision_coco`, `counting_errors`, `run_research_benchmark`, `fetch`,
  `load_research_ground_truth`.
- Commits: not created by this executor (orchestrator commits via the atomic tool; changes left in
  the working tree as instructed).
