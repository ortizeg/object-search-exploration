---
phase: 03-backend-api
plan: 01
subsystem: database
tags: [sqlite, sqlite3, pydantic, wilson-interval, evaluation, null-discipline]

requires:
  - phase: 01-foundations
    provides: "Rating/RunRecord/Provenance/SliceMetadata schemas, provenance.config_hash"
provides:
  - "SQLite store package (src/object_search/store/): connection factory, versioned migrations, run/candidate/rating persistence, derived-metric views, Wilson interval, per-method scoreboard"
  - "The null-discipline enforced at the DDL layer (no DEFAULT on count columns) and proven by round-trip tests"
  - "Derived metrics as NULL-propagating SQL views, never stored columns"
affects: [03-02-api, phase-04-frontend, phase-08-benchmark]

tech-stack:
  added: []
  patterns:
    - "One connection factory sets FK/WAL/busy_timeout on every connection and asserts FK is live"
    - "Versioned migrations via PRAGMA user_version, each step in an explicit BEGIN/COMMIT"
    - "Derived metrics live only in CAST-to-REAL, SUM-not-TOTAL, NULLIF-guarded views"

key-files:
  created:
    - src/object_search/store/db.py
    - src/object_search/store/schema.py
    - src/object_search/store/migrations.py
    - src/object_search/store/runs.py
    - src/object_search/store/ratings.py
    - src/object_search/store/wilson.py
    - src/object_search/store/stats.py
    - src/object_search/store/__init__.py
    - tests/test_store_schema.py
    - tests/test_store_ratings.py
    - tests/test_wilson.py
    - tests/test_store_stats.py
    - tests/store_helpers.py
  modified:
    - .planning/REQUIREMENTS.md

key-decisions:
  - "Count columns are INTEGER NULL with no DEFAULT; NULL vs 0 is the null-discipline and it is proven at the DDL layer via PRAGMA table_info"
  - "No precision/recall/f1/tp/fp/expected column on any table; derived metrics are query-only views"
  - "Wide-but-flat runs row (per the plan) with only diagnostics_json size-capped, rather than the split-payload table"
  - "Wilson closed forms at p=0 and p=1 pin the bounds exactly and kill the -0.0 artifact"

patterns-established:
  - "Real temp-file DB round-trip tests, never mocks: a mock would pass while the DDL carried a DEFAULT 0"
  - "FP resolution (per-match verdicts win over a bare count) is a view CASE, mirrored in Pydantic FPSource"

requirements-completed: [EVAL-01, EVAL-07, EVAL-08, EVAL-09, EVAL-10, EVAL-11, EVAL-12, EVAL-13, EVAL-14, EVAL-17]

coverage:
  - id: D1
    description: "Connection factory + versioned schema/migrations with the null-discipline at the DDL layer"
    requirement: "EVAL-01"
    verification:
      - kind: unit
        ref: "tests/test_store_schema.py::test_count_columns_have_no_default"
        status: pass
      - kind: unit
        ref: "tests/test_store_schema.py::test_failed_migration_rolls_back_atomically"
        status: pass
    human_judgment: false
  - id: D2
    description: "No derived metric is a stored column; metrics are NULL-propagating views"
    requirement: "EVAL-07"
    verification:
      - kind: unit
        ref: "tests/test_store_schema.py::test_no_derived_metric_is_a_stored_column"
        status: pass
      - kind: unit
        ref: "tests/test_store_stats.py::test_views_cast_before_every_metric_division"
        status: pass
    human_judgment: false
  - id: D3
    description: "Run + sub-threshold candidate + provenance + slice + latency persistence, lossless round-trip"
    requirement: "EVAL-08"
    verification:
      - kind: unit
        ref: "tests/test_store_schema.py::test_run_round_trips_losslessly"
        status: pass
    human_judgment: false
  - id: D4
    description: "Rating persistence preserving NULL vs 0; validate flags over-R and verdict discrepancy"
    requirement: "EVAL-17"
    verification:
      - kind: unit
        ref: "tests/test_store_ratings.py::test_bare_thumbs_up_stores_null_counts"
        status: pass
      - kind: unit
        ref: "tests/test_store_ratings.py::test_explicit_zero_is_distinct_from_null"
        status: pass
    human_judgment: false
  - id: D5
    description: "Wilson interval (computed z, closed-form edges, n=0 -> None) and the per-method scoreboard"
    requirement: "EVAL-14"
    verification:
      - kind: unit
        ref: "tests/test_wilson.py::test_ten_of_ten_closed_form"
        status: pass
      - kind: unit
        ref: "tests/test_store_stats.py::test_precision_is_a_real_fraction_not_zero"
        status: pass
    human_judgment: false

duration: 45min
completed: 2026-07-25
status: complete
---

# Phase 3 Plan 1: SQLite run/rating store Summary

**The evaluation data layer, with the null-discipline enforced at the DDL layer and the derived metrics living only in NULL-propagating SQL views — a bare thumbs-up run can never claim a perfect score.**

## Performance

- **Duration:** ~45 min
- **Completed:** 2026-07-25
- **Tasks:** 5/5
- **Files modified:** 13 created, 1 modified

## Accomplishments

- **Connection factory** (`db.py`): `PRAGMA foreign_keys=ON`, `WAL`, `busy_timeout` on every connection, with a hard assertion that FK enforcement came back live. `open_store` connects + migrates in one call.
- **Schema + migrations** (`schema.py`, `migrations.py`): tables for `runs`, `matches`, `candidates`, `ratings`, `match_verdicts`, and `paired_comparisons` (created now so Phase 8 is additive). `wrong_count`/`missed_count` are `INTEGER NULL` with **no** `DEFAULT`. Migrations use `PRAGMA user_version` and wrap each step in an explicit `BEGIN`/`COMMIT`; a deliberately broken migration test proves atomic rollback.
- **Run persistence** (`runs.py`): `insert_run`/`get_run` round-trip a `RunRecord` losslessly — matches, top-N candidates (EVAL-08), 11-field provenance (EVAL-09), nullable slice metadata that stays `None` (EVAL-10), three latency columns (EVAL-11), and size-capped diagnostics JSON that drops only the heatmap above 256 KB. `empty` and `error` outcomes both persist with zero matches (EVAL-12).
- **Rating persistence** (`ratings.py`): counts pass straight through — `None` stays SQL `NULL`, `0` stays `0`, no coercion anywhere. `validate_rating_against_run` enforces `0 <= wrong_count <= retrieved` and flags (never reconciles) a confirmed-verdict vs bare-count disagreement (EVAL-18 store half).
- **Wilson interval** (`wilson.py`): dependency-free, z via `NormalDist().inv_cdf`, `n=0 -> None`, exact closed forms at p=0 and p=1 that also kill the `-0.0` artifact.
- **Derived-metric views + scoreboard** (`stats.py`, views in `schema.py`): `run_fp`/`run_metrics` compute FP (per-match wins over bare count), TP/FN/expected/precision/recall with `CAST(... AS REAL)` before every division, `SUM` never `TOTAL`, `NULLIF` on every denominator, and NULL propagating. `scoreboard()` reports thumbs/precision/recall each with its own separate `n` (EVAL-13), latency p50/p90/p99, abstention/error counts, and the threshold-sweep-eligible count, ranking by the Wilson lower bound.

## Verification (real output)

```
== PRAGMA table_info(ratings) dflt_value for count columns ==
  wrong_count: dflt_value=None notnull=0
  missed_count: dflt_value=None notnull=0
== grep table DDL for derived-metric columns ==
  derived columns found in tables: NONE
== Wilson exact values ==
  z@95% = 1.9599639845400534
  0/10  = (0.0, 0.2775327998628891)
  10/10 = (0.722467200137111, 1.0)
  n=0   = None
```

- **Gates:** `lint`, `format-check`, `typecheck` (mypy strict), `test` all green. **242 passed, coverage 92.11%** (floor 80%). CI green on PR #5.
- Null-discipline proven at DDL (`PRAGMA table_info` no default), model (schema defaults `None`), and round-trip (bare thumbs-up stores NULL, explicit 0 reads back distinct).
- CAST works: a 3/4 case returns `precision == 0.75`, not integer-truncated `0`.
- A bare-thumbs run is excluded from the precision and recall `n`; abstentions (`retrieved=0`) are excluded from precision, not averaged as `0`.

## Deviations from Plan

**1. [Rule 1 - Platform] Wilson exact-value assertions relaxed to a 1e-12 tolerance**
- **Found during:** Task 4 CI run (passed locally on macOS arm64, failed on Linux x86_64).
- **Issue:** `NormalDist().inv_cdf` calls `math.log`/`math.sqrt`, whose last ULP is platform-libm dependent — macOS yields z=`...534`, Linux yields `...536`, and that propagates into the 0/10 and 10/10 closed forms. The plan asked for full-precision equality (`== 1.9599639845400534`), which is not portable.
- **Fix:** Pinned the three affected assertions (z@95%, 0/10, 10/10) to `pytest.approx(..., abs=1e-12)` — still 12 significant figures, three orders of magnitude below the ~4e-6 gap to the textbook 1.96. The `-0.0` lower bound and the exact `1.0` upper bound remain exact assertions; the source-grep test remains the guard against a hardcoded constant. The Wilson math itself is unchanged.
- **Files modified:** tests/test_wilson.py
- **Commit:** 577a81d

## Notes

- **EVAL-18** is partially satisfied here (`validate_rating_against_run` at the store layer); the API-level mutual-exclusion enforcement is 03-02, so its traceability row stays `Pending`.

## Self-Check: PASSED
