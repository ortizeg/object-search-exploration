---
phase: 03-backend-api
plan: 02
subsystem: api
tags: [fastapi, http, pydantic, null-discipline, wilson-interval, registry, testclient]

requires:
  - phase: 03-backend-api
    provides: "SQLite store (runs/ratings/candidates), derived-metric views, scoreboard, Wilson interval, validate_rating_against_run"
  - phase: 01-foundations
    provides: "Rating/RunRecord/Provenance/SearchResult schemas, method registry, provenance capture"
provides:
  - "FastAPI app (src/object_search/api/): create_app factory, lifespan session registry, five endpoint groups (/methods /images /search /ratings /stats) + /runs/{id}"
  - "The null discipline enforced end to end over HTTP: a bare thumbs-up stores SQL NULL and feeds no precision/recall aggregate"
  - "Typed structured errors (APIError envelope) instead of 500 stack traces; method exceptions persisted as outcome='error'"
  - "registry.unregister() — the symmetric partner of register_method, for test isolation"
affects: [phase-04-frontend, phase-08-benchmark]

tech-stack:
  added: []
  patterns:
    - "Request/response contracts reuse the domain schemas verbatim (Rating is the /ratings body) so the HTTP layer invents no second source of truth"
    - "The API names no method (API-01): /methods loops over registry.method_schemas(); a grep test enforces zero method names in the api/ package"
    - "One connection per request via connect(app.state.db_path); routes are thin over the store, all metric care stays in the SQL views"

key-files:
  created:
    - src/object_search/api/routes_ratings.py
    - src/object_search/api/routes_stats.py
    - tests/test_api_ratings.py
    - tests/test_api_stats.py
  modified:
    - src/object_search/api/app.py
    - src/object_search/api/schemas.py
    - src/object_search/search/registry.py
    - src/object_search/search/__init__.py
    - tests/test_api_search.py
    - .planning/REQUIREMENTS.md
    - .planning/ROADMAP.md

key-decisions:
  - "POST /ratings takes the domain Rating verbatim and does NOT coerce; a body without wrong_count/missed_count stores SQL NULL (EVAL-17)"
  - "A bounds violation or a confirmed-verdict/bare-count discrepancy is a 422 with the flag, never a silent clamp or reconcile (EVAL-18)"
  - "GET /stats is a thin pass-through to store.stats.scoreboard; every rate carries its own n and a Wilson interval, null when n=0"
  - "The always-raising test stub is scoped to an autouse fixture with unregister() cleanup rather than registered at import time"

patterns-established:
  - "End-to-end HTTP tests through the real TestClient, reading the store back via client.app.state.db_path to assert NULL vs 0 and query-layer derived metrics"
  - "Runs seeded via store_helpers for deterministic match counts; the rating and scoreboard reads always travel over HTTP"

requirements-completed: [API-01, API-02, API-03, API-04, API-05, API-06, API-07, API-08, EVAL-18]

coverage:
  - id: A1
    description: "POST /ratings stores NULL when counts are omitted and the run is excluded from precision/recall n"
    requirement: "EVAL-17"
    verification:
      - kind: integration
        ref: "tests/test_api_ratings.py::test_bare_thumbs_up_stores_null_counts_and_is_ignored"
        status: pass
    human_judgment: false
  - id: A2
    description: "wrong_count > retrieved is rejected 422 with the flag, not clamped; nothing is stored"
    requirement: "EVAL-18"
    verification:
      - kind: integration
        ref: "tests/test_api_ratings.py::test_wrong_count_greater_than_retrieved_is_422_not_clamped"
        status: pass
      - kind: integration
        ref: "tests/test_api_ratings.py::test_per_match_verdict_conflicting_with_wrong_count_is_flagged"
        status: pass
    human_judgment: false
  - id: A3
    description: "A per-match + missed_count rating yields P=0.75, R=0.6, expected=5 from the query layer, no derived column stored"
    requirement: "API-05"
    verification:
      - kind: integration
        ref: "tests/test_api_stats.py::test_per_match_and_missed_yields_correct_precision_recall_expected"
        status: pass
    human_judgment: false
  - id: A4
    description: "/stats carries n and a Wilson interval beside every rate; a method with n=0 thumbs renders the interval as null"
    requirement: "API-05"
    verification:
      - kind: integration
        ref: "tests/test_api_stats.py::test_every_rate_carries_n_and_a_wilson_interval"
        status: pass
      - kind: integration
        ref: "tests/test_api_stats.py::test_method_with_zero_thumbs_renders_null_interval_not_zero_one"
        status: pass
    human_judgment: false

duration: 40min
completed: 2026-07-25
status: complete
---

# Phase 3 Plan 2: FastAPI app — /methods /search /images /ratings /stats Summary

**The HTTP layer over the store and the registry, with the null discipline proven end to end through the real TestClient — a bare thumbs-up POSTed over HTTP stores SQL NULL and contributes to no precision or recall aggregate.**

## Performance

- **Duration:** ~40 min (Task 4 pickup; Tasks 1–3 delivered in prior commits on the branch)
- **Completed:** 2026-07-25
- **Tasks:** 4/4 (this session completed Task 4; Tasks 1–3 were already committed)
- **Files:** 4 created, 7 modified

## Accomplishments

Tasks 1–3 (prior commits `e69152b`, `178a930`, `4e48a0e`) delivered the app factory + lifespan session registry (API-07), `/methods` + `/images` with no hardcoded method knowledge (API-01, API-02, API-06), and `/search` with provenance/latency capture and typed structured errors (API-03, API-08). Task 4 completes the phase:

- **`POST /ratings`** (`routes_ratings.py`): takes the domain `Rating` verbatim as the request body — its count fields already default to `None`, and the route does **not** coerce, so a body omitting the counts writes SQL `NULL` (EVAL-17). It loads the rated run, reads `retrieved = len(matches)`, and applies `validate_rating_against_run`: `0 <= wrong_count <= retrieved`, and per-match verdicts winning over a bare count with the discrepancy **flagged** — either violation is a `422 rating_rejected` carrying the flag message, never a silent clamp or reconcile (EVAL-18). Unknown run → `404`.
- **`GET /stats`** (`routes_stats.py`): a thin pass-through to `store.stats.scoreboard`. All the honesty lives in the SQL views; the route only opens a connection and serialises `list[MethodStats]`, so each rate ships beside its own `n` and a Wilson interval, and a method with `n=0` thumbs renders the interval as `null` rather than `[0,1]`.
- **`RatingResponse`** added to `api/schemas.py`; both routers mounted in `create_app`.

## Verification (real output)

```
$ pixi run quality      # lint, format-check, mypy strict, pytest
ruff check ....................... Passed
ruff format --check .............. Passed
mypy src/ (strict) ............... Passed
263 passed, 1 warning in 25.34s
Required test coverage of 80% reached. Total coverage: 91.15%
```

The four Phase 3 success-criterion tests, all through the real `TestClient`:

- `test_bare_thumbs_up_stores_null_counts_and_is_ignored` — a thumbs-up POSTed with no counts stores `wrong_count IS NULL` / `missed_count IS NULL` (read back from the `ratings` row) and `/stats` shows `precision_n=0`, `recall_n=0` for that method.
- `test_per_match_and_missed_yields_correct_precision_recall_expected` — 1 wrong of 4 confirmed verdicts + `missed_count=2` gives `precision=0.75`, `recall=0.6` on `/stats` and `expected=5` from the `run_metrics` view, with `PRAGMA table_info` confirming no `precision/recall/f1/expected/tp/fp` column on any table.
- `test_every_rate_carries_n_and_a_wilson_interval` + `test_method_with_zero_thumbs_renders_null_interval_not_zero_one` — every rate carries `n` and finite Wilson bounds inside `[0,1]`; an unrated method reports `thumbs_n=0` and a `null` interval.
- `test_wrong_count_greater_than_retrieved_is_422_not_clamped` — `wrong_count=4` against a 3-box run is a `422 rating_rejected` naming both counts, and no rating row is written.

- **CI:** green on **PR #6** (`quality` check pass, 1m30s). PR left open, not merged.

## Deviations from Plan

**1. [Rule 1 - Bug] Fixed a cross-file test-isolation leak that broke the full suite**
- **Found during:** Task 4, running the full suite for the "earlier tasks' tests still pass" gate.
- **Issue:** `tests/test_api_search.py` (from Task 3) registered an always-raising `test-raiser` stub into the **global** method registry at import time. It never got removed, so `tests/test_samples.py::test_render_writes_a_dir_per_registered_method` — which enumerates every registered method and renders a sample for each — later tried to run the raiser, whose only job is to raise. The failure surfaced only under the full suite (`test_api_search` imported before `test_samples`), never per-file, which is why it slipped past Task 3.
- **Fix:** Added `registry.unregister()` (documented as the symmetric partner of `register_method`, for test isolation) and converted the raiser from an import-time registration into an autouse fixture in `test_api_search.py` that registers in setup and unregisters in teardown. Production registration behaviour is unchanged.
- **Files modified:** `src/object_search/search/registry.py`, `src/object_search/search/__init__.py`, `tests/test_api_search.py`
- **Commit:** `505b4f6`

**2. [Process] The isolation fix is bundled into the feature commit, not a separate atomic commit**
- Intended as two commits (feature + fix). The pre-commit `mypy` hook runs over the whole `src/` tree, so a fix-only commit that stashes the unstaged feature changes while the untracked `routes_ratings.py` (which imports the new `RatingResponse`) stays on disk fails type-checking. Committing the feature first — with the fix files already staged from the earlier attempt — swept both into `505b4f6`. The commit message documents both changes.

## Notes

- **API-01 grep guard:** the new `routes_ratings.py` / `routes_stats.py` name no method; the existing package-wide grep test in `test_api_methods.py` continues to pass.
- **No new stubs.** The lifespan session registry remains intentionally empty in Phase 3 (no ONNX-backed method is registered yet) — this is documented wiring for Phases 5–7, not a stub; `/ratings` and `/stats` are fully wired to the store.

## Self-Check: PASSED
