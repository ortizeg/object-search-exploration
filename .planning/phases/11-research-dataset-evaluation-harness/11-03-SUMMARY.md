---
phase: 11-research-dataset-evaluation-harness
plan: 03
subsystem: eval
tags: [research-datasets, exemplar-sampling, k-shot-late-fusion, benchmark-sweep, coco-ap, counting-metrics, report, doc-07]
status: complete
requirements: [EVAL-23, EVAL-24, DOC-07]
provides:
  - object_search.eval.sampling.sample_exemplars
  - object_search.eval.benchmark.run_multi_exemplar
  - object_search.eval.benchmark.run_research_sweep
  - object_search.eval.benchmark.main_research (bench-research Hydra entry)
  - BenchmarkConfig.datasets/splits/exemplar_counts/seed/research_root/research_out
  - scripts/build_research_report.py + pixi task report-research
  - docs/eval/research-datasets.md (DOC-07)
  - docs/reports/research-report.html (committed fixture smoke-run deliverable)
requires:
  - object_search.eval.sampling (Task 1, this plan)
  - object_search.eval.metrics.average_precision_coco (11-01)
  - object_search.eval.metrics.counting_errors (11-01)
  - object_search.eval.benchmark._run_one_research / _aggregate_research (11-01)
  - object_search.eval.splits.load_split_manifest / research_image_ids (11-01/02)
  - object_search.eval.labels.load_research_ground_truth + GroundTruth.exemplar_indices (11-01)
  - object_search.search.common.nms.nms (dedup, tie-break (-score, y, x))
  - object_search.search.registry.SearchFn (UNCHANGED)
affects: [conf/benchmark.yaml, pixi.toml, .gitignore]
key-files:
  created:
    - src/object_search/eval/sampling.py
    - scripts/build_research_report.py
    - docs/eval/research-datasets.md
    - docs/reports/research-report.html
    - tests/test_research_sampling.py
    - tests/test_research_benchmark_sweep.py
    - tests/test_research_report.py
  modified:
    - src/object_search/eval/benchmark.py
    - conf/benchmark.yaml
    - pixi.toml
    - .gitignore
decisions:
  - "Task 2 checkpoint:decision ratified as option-a (k-shot LATE FUSION in the eval layer via run_multi_exemplar; SearchFn and the four method files UNCHANGED; sampled exemplars stay scored in gt so the 1- and 3-exemplar recall denominators match). Pre-ratified in the execution prompt; proceeded autonomously as instructed."
  - "Added a real bench-research Hydra entry (main_research) + active research keys in conf/benchmark.yaml, so the config entries are a genuine control for run_research_sweep rather than an inert advertised knob (CLAUDE.md: a control that is advertised and inert is worse than no control)."
metrics:
  completed: 2026-07-26
  tasks: 4
  files: 11
status_detail: complete
---

# Phase 11 Plan 03: Seeded exemplar sampler, k-shot late-fusion sweep, report + DOC-07 Summary

The research harness is complete. A seeded exemplar sampler queries every method at **1 exemplar**
(the product's one-box operating point) and **3 exemplars** (the literature convention); a k-shot
**late-fusion** runner (`run_multi_exemplar`) runs the single-exemplar method once per exemplar and
unions + NMS-dedupes the detections, leaving `SearchFn` and all four method files untouched; the
benchmark sweeps `method x dataset x {1,3} x {val,test}` (CARPK/PUCPR+ test-only) reusing the proven
11-01 metric/aggregate machinery and emitting the full literature column set (P/R/F1 + AP/AP50/AP75 +
MAE/RMSE/NAE); a committed report table renders those numbers; and `docs/eval/research-datasets.md`
(DOC-07) documents every dataset, every metric, and the tune-on-val → report-on-test protocol.

Everything runs offline on the committed synthetic fixtures. **The real dataset images are
licence-gated and not fetched**, so the numbers in the committed report are the **fixture smoke-run**
(stated in a banner on the report page and in `research-datasets.md`); the real report regenerates via
`pixi run report-research` once a human accepts each licence and fetches the archives. No real-dataset
numbers are claimed or fabricated.

## Checkpoint proceeded under (Task 2 — `checkpoint:decision`, one-way)

Task 2 locks HOW a method is run at >1 exemplar and HOW the sampled exemplars are scored, before the
sweep bakes both into every 3-exemplar number. The decision was **pre-ratified in the execution
prompt** (option-a, k-shot late fusion), so per instruction I proceeded autonomously without stopping:

- **option-a — k-shot LATE FUSION in the eval layer (chosen).** `run_multi_exemplar(spec_fn, image,
  exemplars, config)` calls `spec_fn` once per sampled exemplar, UNIONs the resulting matches AND
  sub-threshold candidates across the k runs, then NMS-dedupes overlapping detections
  (`search/common/nms.py`, tie-break `(-score, y, x)`). The k=1 run is the special case (pass-through
  of the single call), so 1 and 3 exemplars share one code path. **`SearchFn` and the four method
  files are UNCHANGED** — grep-asserted in `test_research_benchmark_sweep.py`.
- **Exemplar scoring semantics.** The sampled exemplar boxes REMAIN in `gt.boxes` and are scored like
  any other instance, so `len(gt.boxes)` (the recall denominator) is identical at count=1 and
  count=3 — the 1-vs-3 numbers are directly comparable. A test asserts this denominator parity.

## Task-by-task

- **Task 1 — Seeded exemplar sampler (auto, tdd): DONE.** `eval/sampling.py::sample_exemplars(gt, *,
  count, seed)` builds one canonical ordering — native `exemplar_indices` first (FSCD-*/RPINE ship
  three; honoured in order, seed-independent), then a `np.random.default_rng(seed)` permutation of the
  remaining GT boxes — and slices it by `count`. This gives the **prefix property** (count=1 ==
  count=3[:1]) for free, seed-determinism, and native-honouring, all in one function. `count >
  len(boxes)` returns all boxes (documented, no index error); `count < 1` raises. Only RNG source is
  `np.random.default_rng` (never `cv2.setRNGSeed`, D-11). RED tests written first.

- **Task 3 — Full benchmark sweep + run_multi_exemplar (auto, tdd): DONE.** Added `run_multi_exemplar`
  (the ratified late-fusion runner) to `eval/benchmark.py`; extended `BenchmarkConfig` with
  `datasets`/`splits`/`exemplar_counts`/`seed`/`research_root`/`research_out`; rewired
  `_run_one_research` to score `run_multi_exemplar(spec.fn, scene, sample_exemplars(gt, count, seed),
  config)` (sampled exemplars stay in `gt.boxes`); added `run_research_sweep` iterating dataset ×
  split × exemplar_count × method, reusing `run_research_benchmark` per cell and emitting P/R/F1 +
  AP/AP50/AP75 + MAE/RMSE/NAE. **D-04** is enforced structurally: a test-only dataset's manifest has
  empty val ids, so no val cell is ever emitted for CARPK/PUCPR+. The **CI subset is unchanged**
  (chipset-only, `resolve_run_set` untouched); a test asserts `ci=true` still resolves to
  `('ncc','sparse-geo')` even with research datasets configured. Results are written to
  `docs/benchmark/research-results.json` (added to `.gitignore`).

- **Task 4 — Report table + DOC-07 survey (auto): DONE.** `scripts/build_research_report.py` mirrors
  `build_report.py`: reads `docs/benchmark/research-results.json`, renders one row per
  (method, dataset, exemplar_count, split) with columns P/R/F1/AP/AP50/AP75/MAE/RMSE/NAE (abstentions
  as `n/a`, never 0), names the 3-exemplar fusion "**k-shot late fusion**" in the caption, and guards a
  missing results file with an actionable message. Added the `report-research` pixi task.
  `docs/eval/research-datasets.md` (DOC-07) documents each dataset (purpose, source link, annotation
  type, split table, strengths/weaknesses/biases), the metric definitions (incl. the AP50 ==
  single-IoU-0.5 reconciling note), and the tune-on-val → report-on-test protocol, seeded val-carve,
  CARPK/PUCPR+ test-only probe, the 1-vs-3 + k-shot-late-fusion explanation, and the box-only
  exclusion of dot-only sets.

## Fixture smoke-run (the committed report)

`docs/reports/research-report.html` was generated from a fixture smoke-run over the committed
synthetic fixtures (ncc, all five dataset keys → 14 cells: rpine/fscd147/fscd_lvis val+test × {1,3},
carpk test-only × {1,3}; pucpr_plus has no committed fixture so contributes 0). The page carries a
prominent **"Offline fixture smoke-run"** banner stating the numbers are not real-dataset results and
how to regenerate the real report. The raw `research-results.json` is gitignored (regenerable,
environment-dependent), exactly like `docs/benchmark/results.json`.

## Deviations from Plan

- **[Rule 3 — completeness] Added a real `bench-research` Hydra entry (`main_research`) + active
  research keys in `conf/benchmark.yaml`, and a `bench-research` pixi task.** The plan asked for
  "research sweep entries in conf/benchmark.yaml, kept OUT of the ci=true subset". Adding those keys
  as *active* config with no consumer would be an inert advertised control, which CLAUDE.md explicitly
  forbids ("a control that is advertised and inert is worse than no control"). So the keys are wired to
  a genuine entry point (`main_research` → `run_research_sweep`), gated on `research_root` + fetched
  data (like `bench` gates on fetched models), and kept entirely out of the default/CI chipset path.
- **[Rule 2 — required] `.gitignore` updated** to ignore `docs/benchmark/research-results.json` (not
  in the plan's `files_modified`, but the Task 3 acceptance criterion requires `git check-ignore` to
  succeed). Mirrors the existing `docs/benchmark/results.json` rule.
- **[artifact] `docs/reports/research-report.html` committed** as the DOC-07 report deliverable
  (per `<artifacts_this_phase_produces>`), rendered from the fixture smoke-run with the caveat banner.

No architectural (Rule 4) changes; no auth gates.

## Known Stubs

None. `conf/datasets/pucpr_plus.split.json` carries an empty test list by design (test-only, no
committed fixture this wave — the ratified fixture-vs-real pattern from 11-01/02), so the sweep simply
emits no pucpr_plus cell offline; it is not a stub blocking the plan goal.

## Threat surface

No new trust boundaries beyond the plan's register. T-11-10 (test-for-tuning bias): the sweep marks
val vs test explicitly and CARPK/PUCPR+ emit zero val cells (D-04), asserted in
`test_carpk_is_test_only_no_val_cell`; protocol documented in DOC-07. T-11-11 (non-reproducible
exemplar choice): `sample_exemplars` is deterministic `np.random.default_rng(seed)`, 1-exemplar is a
prefix of 3, seed-stability asserted. T-11-12 (committed numbers untraceable): `research-results.json`
is gitignored and the report records `git_sha`; the committed report is explicitly labelled a fixture
smoke-run. T-11-13 (abstention rendered as 0): the report renders `n/a`, never `0` (asserted).

## Verification output (pasted real results)

- `pixi run lint` → `All checks passed!`
- `pixi run format-check` → `138 files already formatted`
- `pixi run typecheck` → `Success: no issues found in 73 source files` (mypy strict, no new ignores)
- `pixi run test` → `553 passed, 19 skipped`; `Required test coverage of 80% reached. Total coverage:
  88.63%`
- `pixi run quality` (umbrella: lint + format-check + typecheck + test) → green.
- New/modified eval-module coverage: `sampling.py` 89%, `benchmark.py` 87% (full suite), `metrics.py`
  100%, `splits.py` 88%, `labels.py` 90% — all above the 80% floor.
- Targeted: `pytest tests/test_research_sampling.py tests/test_research_benchmark_sweep.py
  tests/test_research_report.py tests/test_research_carpk_tracer.py tests/test_eval_benchmark.py` →
  **33 passed** (incl. the denominator-parity test
  `test_recall_denominator_identical_across_1_and_3` and the no-method-dispatch grep test
  `test_run_multi_exemplar_is_defined_in_eval_not_search`).
- `pixi run report-research` with no results file → prints the actionable "No research-results.json
  found … fetch-datasets … gitignored …" message and exits 1.
- Fixture smoke-run: `run_research_sweep` over the committed fixtures → **14 cells**; `report-research`
  wrote `docs/reports/research-report.html` (banner present, 14 data rows).
- `git check-ignore docs/benchmark/research-results.json` → ignored (returncode 0); the raw dump is
  not tracked.
- Grep: `grep -c default_rng src/object_search/eval/sampling.py` > 0; `run_multi_exemplar` is defined
  under `src/object_search/eval/` and is absent from every `src/object_search/search/*.py` method
  file.

## Self-Check: PASSED

- Created files exist on disk: `src/object_search/eval/sampling.py`, `scripts/build_research_report.py`,
  `docs/eval/research-datasets.md`, `docs/reports/research-report.html`, and the three test modules
  (verified via `git status`).
- New symbols import and run: the full suite (553 passed) exercises `sample_exemplars`,
  `run_multi_exemplar`, `run_research_sweep`, `build_research_report`, and the extended
  `BenchmarkConfig`; `main_research` imports and the config composes cleanly into `BenchmarkConfig`.
- Commits: not created by this executor (changes left in the working tree as instructed).
