---
phase: 08-evaluation-and-docs
plan: 01
subsystem: testing
tags: [evaluation, metrics, hydra, benchmark, bradley-terry, average-precision, chipset]

# Dependency graph
requires:
  - phase: 01-foundation
    provides: synthetic + chipset generators with exact ground-truth sidecars (EVAL-03/19)
  - phase: 03-store
    provides: SQLite store, derived-metric views, paired_comparisons table, Wilson interval
  - phase: 05-07
    provides: the four registered search methods (ncc, sparse-geo, dino-dense, propose-retrieve)
provides:
  - one ground-truth loader for every *.gt.json sidecar (achieved count, honest coverage)
  - detection metrics — greedy-IoU precision/recall/F1 (abstention-aware) and all-point AP
  - Hydra benchmark runner with per-slice reporting and a model-free CI subset
  - paired comparison (same box, all methods) writing win/loss/tie to the store
  - regularised Bradley-Terry with the complete-separation guard
affects: [08-02-docs-and-charts]

# Tech tracking
tech-stack:
  added: [hydra-core benchmark entrypoint, omegaconf]
  patterns:
    - "Hydra-free pure core (run_benchmark) behind a thin @hydra.main wrapper, so tests never fight argv seizure"
    - "Pydantic BenchmarkConfig validates OmegaConf's untyped DictConfig at the boundary"
    - "Abstention convention (None, never 0) carried from metrics through pooled aggregates"

key-files:
  created:
    - src/object_search/eval/labels.py
    - src/object_search/eval/metrics.py
    - src/object_search/eval/benchmark.py
    - src/object_search/eval/paired.py
    - src/object_search/eval/bradley_terry.py
    - conf/benchmark.yaml
    - tests/test_eval_metrics.py
    - tests/test_eval_benchmark.py
    - tests/test_eval_paired.py
    - tests/test_bradley_terry.py
  modified:
    - pixi.toml
    - src/object_search/cli.py
    - .github/workflows/ci.yml
    - .gitignore

key-decisions:
  - "GT loader returns a richer GroundTruth object (exemplar_index + canvas size + slice metadata), not a bare list[BBox]"
  - "AP is all-point interpolation (COCO-style), computed from the EVAL-08 candidate log; stated in the docstring"
  - "Comparison scalar for paired mode is F1; abstention maps to 0.0, a crash below that; ties kept distinct"
  - "Bradley-Terry regularised with EPS=0.5 pseudo-games, scale pinned to geometric-mean-1"
  - "results.json is gitignored (regenerable, environment-dependent); committed charts are plan 08-02"

patterns-established:
  - "One loader per artifact format: three sidecar kinds, one reader (EVAL-02)"
  - "Model-free CI benchmark subset: classical methods over the deterministic chipset, no fetch-models"

requirements-completed: [EVAL-02, EVAL-04, EVAL-05, EVAL-15]

coverage:
  - id: D1
    description: "Unified ground-truth loader reads every *.gt.json (achieved count, None on missing GT)"
    requirement: EVAL-02
    verification:
      - kind: unit
        ref: "tests/test_eval_metrics.py#test_labels_reads_real_chipset_sidecar_with_achieved_count"
        status: pass
      - kind: unit
        ref: "tests/test_eval_metrics.py#test_labels_returns_none_for_unlabelled_image"
        status: pass
    human_judgment: false
  - id: D2
    description: "Metrics — greedy-IoU precision/recall/F1 (precision None on abstention) + all-point AP from the candidate log"
    requirement: EVAL-04
    verification:
      - kind: unit
        ref: "tests/test_eval_metrics.py#test_precision_none_on_abstention"
        status: pass
      - kind: unit
        ref: "tests/test_eval_metrics.py#test_average_precision_hand_computed_all_point"
        status: pass
    human_judgment: false
  - id: D3
    description: "Hydra benchmark runner with per-slice breakdowns and a model-free CI subset"
    requirement: EVAL-04
    verification:
      - kind: integration
        ref: "tests/test_eval_benchmark.py#test_ci_benchmark_writes_results_with_per_slice_breakdowns"
        status: pass
      - kind: e2e
        ref: "pixi run bench-ci (CI job 'benchmark (model-free chipset subset)', pass)"
        status: pass
    human_judgment: false
  - id: D4
    description: "Paired comparison — same box through all methods, win/loss/tie into paired_comparisons"
    requirement: EVAL-05
    verification:
      - kind: integration
        ref: "tests/test_eval_paired.py#test_run_paired_records_winner_into_table"
        status: pass
      - kind: integration
        ref: "tests/test_eval_paired.py#test_run_paired_stores_ties_as_a_distinct_outcome"
        status: pass
    human_judgment: false
  - id: D5
    description: "Bradley-Terry with the complete-separation guard — undefeated method yields a finite score"
    requirement: EVAL-15
    verification:
      - kind: unit
        ref: "tests/test_bradley_terry.py#test_undefeated_method_yields_finite_score"
        status: pass
      - kind: unit
        ref: "tests/test_bradley_terry.py#test_strongly_connected_detects_disconnected_graph"
        status: pass
      - kind: unit
        ref: "tests/test_bradley_terry.py#test_scale_is_pinned_so_two_refits_are_comparable"
        status: pass
    human_judgment: false

# Metrics
duration: ~30min
completed: 2026-07-25
status: complete
---

# Phase 8 Plan 01: Evaluation Harness Summary

**Ground-truth loading, abstention-aware precision/recall/F1 with all-point AP, a Hydra benchmark with a model-free CI subset that surfaces the NCC-vs-sparse-geo crossover, paired comparison, and regularised Bradley-Terry that keeps an undefeated method finite.**

## Performance

- **Duration:** ~30 min
- **Started:** 2026-07-25T06:16Z (branched from main)
- **Completed:** 2026-07-25T06:46Z
- **Tasks:** 4
- **Files modified:** 16 (1901 insertions)

## Accomplishments

- **Fixed a latent defect:** `pixi run bench` pointed at `python -m object_search.cli benchmark`, which cannot work — `@hydra.main` seizes `sys.argv` and cannot compose with a Typer subcommand. Re-pointed at `python -m object_search.eval.benchmark`, added `bench-ci`, and turned the Typer `benchmark` command into a loud redirect.
- **One GT loader** (`labels.py`) for synthetic, chipset, and hand-labelled sidecars: reads the achieved instance count (never requested N), returns `None` (never `[]`) when unlabelled so coverage is honest, and carries the designated `exemplar_index` + canvas size + slice metadata the benchmark needs.
- **Metrics** (`metrics.py`): greedy-IoU matching with the EVAL-16 duplicate convention (2 boxes on 1 instance = 1 TP + 1 FP); precision `None` on abstention, never 0; **all-point-interpolation AP** from the EVAL-08 sub-threshold candidate log (hand-computed AP = 5/6 asserted).
- **Hydra benchmark** (`benchmark.py`): a `@hydra.main` sweep writing `docs/benchmark/results.json` with per-slice breakdowns by instance count, canvas size, and scale bucket. Pure `run_benchmark` core is Hydra-free so tests drive it without argv seizure. A **model-free CI subset** (`ci=true`: ncc + classical sparse-geo over the chipset, no weights) is wired into CI as a separate job that does **not** `fetch-models`.
- **Paired comparison** (`paired.py`): `run_paired` runs the same exemplar box through all methods in one call, scores each vs GT (F1), records win/loss/tie into `paired_comparisons`; ties kept distinct; no winner invented without GT.
- **Bradley-Terry** (`bradley_terry.py`): all three PITFALLS §8.3 layers — `strongly_connected` (deterministic traversal), EPS=0.5 pseudo-game regularisation, geometric-mean-1 scale pin — plus ranking-with-uncertainty (Fisher-diagonal stderr, per-pair counts, connectivity flag). An undefeated method yields a **finite** score.

## Task Commits

1. **Task 1: rewire bench + GT labels loader** — `2ccd675` (feat) — EVAL-02
2. **Task 2: metrics.py (precision/recall/F1 + all-point AP)** — `0e90ca5` (feat) — EVAL-04
3. **Task 3: Hydra benchmark runner + CI subset** — `56e7b54` (feat) — EVAL-04, EVAL-19
4. **Task 4: paired comparison + Bradley-Terry** — `6186a66` (feat) — EVAL-05, EVAL-15

## Files Created/Modified

- `src/object_search/eval/labels.py` — one loader for every ground-truth sidecar + scene resolvers
- `src/object_search/eval/metrics.py` — greedy-IoU precision/recall/F1 and all-point AP
- `src/object_search/eval/benchmark.py` — `@hydra.main` sweep, per-slice, model-free CI subset
- `src/object_search/eval/paired.py` — same-box paired comparison into `paired_comparisons`
- `src/object_search/eval/bradley_terry.py` — regularised BT with the complete-separation guard
- `conf/benchmark.yaml` — benchmark defaults (Hydra file/console logging disabled)
- `pixi.toml` — `bench` rewired to the eval module; `bench-ci` added
- `src/object_search/cli.py` — Typer `benchmark` now redirects to the Hydra entrypoint
- `.github/workflows/ci.yml` — new `benchmark` job running the model-free subset without weights
- `.gitignore` — `docs/benchmark/results.json` and `benchmark.log` (regenerable artifacts)
- `tests/test_eval_metrics.py`, `tests/test_eval_benchmark.py`, `tests/test_eval_paired.py`, `tests/test_bradley_terry.py`

## Decisions Made

- Richer `GroundTruth` return over a bare `list[BBox]` (needed for same-box comparison and per-slice reporting).
- AP = all-point interpolation from the candidate log; stated in the docstring.
- Paired comparison scalar = F1, abstention→0.0, crash below abstention; ties distinct.
- Bradley-Terry: EPS=0.5 pseudo-games, scale pinned to geometric-mean-1.
- `results.json` gitignored; committed charts are plan 08-02.

## Deviations from Plan

### Auto-fixed / adjustments

**1. [Rule 2 - Missing Critical] Richer GroundTruth object instead of `list[BBox]`**
- **Found during:** Task 1
- **Issue:** The plan's literal `load_ground_truth -> list[BBox] | None` would drop the designated `exemplar_index` (EVAL-19 requires every method queried with the *same* box) and the canvas size / slice metadata the per-slice report needs (EVAL-10).
- **Fix:** Returns a frozen `GroundTruth` exposing `boxes`, `achieved_count`, `exemplar`, `width/height`, `slice_metadata`. Honours the "achieved not requested" and "None when unlabelled" constraints exactly.
- **Committed in:** `2ccd675`

**2. [Rule 3 - Blocking] Hydra wrote a stray `benchmark.log` and would create `outputs/` dirs**
- **Found during:** Task 3
- **Issue:** `@hydra.main` sets up its own file logging (`${job_name}.log`) and a per-run output dir, polluting the repo and duplicating Loguru output.
- **Fix:** `conf/benchmark.yaml` disables `hydra/job_logging` + `hydra/hydra_logging`, sets `output_subdir: null`, `run.dir: .`, `job.chdir: false`; `benchmark.log` gitignored belt-and-braces.
- **Committed in:** `56e7b54`

**3. [Rule 2] Added `tests/test_eval_paired.py` (not in the plan's file list)**
- **Reason:** `paired.py` needs its own tests to hold the ≥80% coverage floor; the plan only named `test_bradley_terry.py` for Task 4.
- **Committed in:** `6186a66`

---

**Total deviations:** 3 (1 missing-critical interface, 1 blocking config, 1 test-coverage addition)
**Impact on plan:** All necessary for correctness or the coverage gate. No scope creep — the harness matches the plan's intent.

## Issues Encountered

- **The crossover is real and now visible.** `pixi run bench-ci` shows NCC at P=R=1.0 on the fixed-scale chipset while sparse-geo abstains on every image (tiny 24px chips yield <20 SIFT keypoints → precision `None`, an honest abstention, not a fabricated 0). Per-canvas latency ramps 19 ms (320×240) → 568 ms (2048×1536). Reported per-slice, as the literature predicts — the intended honest finding, not a bug.

## Verification

- All four gates green locally and in CI: ruff, ruff format --check, mypy strict, pytest. **Coverage 93.03%** (floor 80%). Edge cases tested: R=0 abstention → precision `None`; all-scored aggregates; single-method / undefeated Bradley-Terry finite score.
- CI (PR #15): `quality` pass (1m34s), `benchmark (model-free chipset subset)` pass (22s, no weights fetched).
- `pixi run bench` and `pixi run bench-ci` both run (rewired away from the Typer CLI).

## Next Phase Readiness

- The harness is ready for plan 08-02 (committed charts/tables from `results.json`, README, per-method docs, aggregated robustness backlog, `docs/MILESTONE-2.md`).
- The full benchmark (dino-dense, propose-retrieve) needs `pixi run fetch-models` and is run locally; the model-free subset covers CI.

## Self-Check: PASSED

---
*Phase: 08-evaluation-and-docs*
*Completed: 2026-07-25*
