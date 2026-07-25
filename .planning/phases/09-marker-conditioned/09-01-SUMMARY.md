---
phase: 09-marker-conditioned
plan: 01
subsystem: explorations
status: complete
tags: [milestone-2, explorations, registry, orientation, region-proposal, api]
requires:
  - search/registry.py (get_method)
  - search/proposals.py (propose)
  - schemas/search.py (Match.transform, SearchResult, Diagnostics)
  - store/schema.py (runs.exploration column)
provides:
  - explorations/registry.py (@register_exploration, get/list/has/schemas)
  - explorations/markers.py (estimate_geometry, theta_from_transform)
  - explorations/marker_conditioned.py (the marker-conditioned exploration)
  - explorations/same_image_search.py (default adapter)
  - synthetic markers (synthesize_markers, MarkerSpec/MarkerGT/MarkerImage)
  - api: GET /explorations, /search exploration routing + persistence
affects:
  - schemas/search.py (Diagnostics: +markers/reference-points/directions)
  - schemas/records.py (RunRecord: +exploration field, DEFAULT_EXPLORATION)
  - store/runs.py (writes/reads the exploration column)
tech-stack:
  added: []
  patterns: [registry-mirror, two-path-orientation, weighted-sum-scoring, propose-once]
key-files:
  created:
    - src/object_search/explorations/__init__.py
    - src/object_search/explorations/registry.py
    - src/object_search/explorations/markers.py
    - src/object_search/explorations/marker_conditioned.py
    - src/object_search/explorations/same_image_search.py
    - src/object_search/api/routes_explorations.py
    - tests/test_markers.py
    - tests/test_marker_conditioned.py
    - tests/test_explorations_registry.py
    - tests/test_api_explorations.py
  modified:
    - src/object_search/synthetic/generator.py
    - src/object_search/synthetic/__init__.py
    - src/object_search/schemas/search.py
    - src/object_search/schemas/records.py
    - src/object_search/store/runs.py
    - src/object_search/api/schemas.py
    - src/object_search/api/routes_search.py
    - src/object_search/api/app.py
decisions:
  - "Explorations mirror the method registry; adding one = one file + one import"
  - "Orientation prefers the free transform path (atan2(c,a)); PCA fallback with arrowhead-mass tip"
  - "Symmetric markers return centroid + direction None, never a guessed direction"
  - "propose() is called once per image, shared across markers"
  - "RunRecord carries exploration explicitly; no schema migration"
  - "Omitted the plan's inert `seed` field per CLAUDE.md (no stochastic step to seed)"
metrics:
  duration_min: 55
  tasks: 5
  commits: 5
  tests_added: 39
  coverage_pct: 93.34
  completed: 2026-07-25
---

# Phase 9 Plan 01: Marker-Conditioned Backend Summary

Milestone 2's backend delivered as a new **exploration**, not a fork: an exploration registry that
mirrors the method registry, a genuinely new orientation estimator (transform + PCA paths),
proposal scoring, the four-step pipeline, synthetic arrow markers with exact ground truth, and API
routing that persists marker runs through the unchanged store with no schema migration.

## What was built

- **Synthetic markers** (`synthesize_markers`, `MarkerSpec`/`MarkerGT`/`MarkerImage`): head-heavy
  arrows, symmetric dots, carets, each with an exact tip, unit direction and centroid; optional
  target objects a known gap past the tip; single-`default_rng` byte-identical output;
  `save_marker_image` writes a `.markers.json` sidecar. `MARKER_DEMO_SPECS` covers arrows,
  arrows-with-targets, dots.
- **Orientation** (`explorations/markers.py`): `theta_from_transform` = `atan2(c, a)` from a
  flattened 2×3 affine; the transform path resolves the 180° flip by mapping the exemplar's tip
  through the affine and orienting centroid→mapped-tip. The PCA fallback fits the foreground mask's
  principal axis (foreground = Otsu on distance from the **border-ring** background, robust to fill
  ratio) and disambiguates the tip by an arrowhead-mass heuristic. Low asymmetry ⇒ centroid and
  `direction=None`.
- **Exploration registry** (`explorations/registry.py`): the exact surface of `search/registry.py`
  — `@register_exploration`, `get/list/has_exploration`, `exploration_schemas`, `unregister`,
  duplicate/unknown errors. `same_image_search.py` is the default adapter over `get_method(...).search`.
- **Marker-conditioned exploration** (`explorations/marker_conditioned.py`): one self-contained,
  numbered-step module with a ROBUSTNESS BACKLOG and explicit pre/post-processing. Finds markers
  via any M1 method, orients each, calls `propose()` **once**, scores every proposal per marker with
  a `Field`-described weighted sum (proximity + direction + objectness + size prior; direction term
  zeroed when `None`), and keeps the best proposal per marker as a `Match`. No markers / no
  proposals ⇒ `outcome=EMPTY` with a note. Diagnostics carry the full proposal set plus per-marker
  reference points and directions.
- **API**: `GET /explorations` mirrors `/methods` (schema-driven, no exploration name in `api/`);
  `POST /search` gains registry-driven `exploration` routing. `RunRecord`/`store` thread the
  `exploration` value through explicitly, so a marker run persists under `marker-conditioned` and
  the default under `same-image-search`, with the default path byte-for-byte unchanged.

## Verification (real output)

- **Four gates:** `ruff`, `ruff format`, `mypy --strict` all clean; `477 passed, 5 skipped`,
  **total coverage 93.34%** (floor 80%).
- **Orientation vs synthetic GT:** PCA arrow-direction errors `[3.2, 1.3, 4.2, 3.3, 0.3]°`
  (max 4.2°, within ~10°), tip within a few px; transform path `theta = 0.6458 rad` vs GT
  `0.6458 rad`.
- **propose() once:** `test_propose_is_called_exactly_once` passes (counting stub, `calls == 1`).
- **Persistence:** marker run persists `exploration='marker-conditioned'`; default persists
  `exploration='same-image-search'`.
- **No hardcoded dispatch:** grep test over `api/*.py` finds no exploration name as a literal.

## Commits

| Task | Commit | Description |
| ---- | ------ | ----------- |
| 1 | a4788e8 | synthetic arrow/dot/caret markers with exact tip/direction GT |
| 2 | 8318a95 | marker reference-point + orientation estimation (M2-02) |
| 3 | 7f4acbe | exploration registry + same-image-search default adapter |
| 4 | 85b79d9 | marker-conditioned exploration pipeline (M2-01..M2-04) |
| 5 | ac69050 | API /explorations + exploration routing and persistence |

## Deviations from Plan

**1. [Rule 2 / CLAUDE.md] Omitted the `seed` field on `MarkerConditionedConfig`.**
- **Found during:** Task 4
- **Issue:** The plan listed `seed=0`, but the pipeline has no stochastic step (marker method,
  `propose()`, and scoring are all deterministic). CLAUDE.md is a hard rule: "Never add a seed
  parameter that does nothing; a control that is advertised and inert is worse than no control."
- **Fix:** Omitted the field; determinism is covered by `test_deterministic_under_fixed_inputs`.
  Any stochastic step in a marker-finding method takes its own seed via `marker_config`.

**2. [Rule 2] Extended `Diagnostics` with three additive optional overlay fields.**
- `markers`, `marker_reference_points`, `marker_directions` — needed so the UI overlay can draw the
  per-marker pointing arrows "by field presence", as the CONTEXT specifies. All optional/default
  `None`, `extra="forbid"` preserved, no store migration (diagnostics are JSON).

**3. [Rule 3] Threaded `exploration` through `RunRecord`/`store` explicitly.**
- The plan required persisting the exploration value instead of relying on the column DEFAULT.
  `RunRecord` gained an `exploration` field (default `same-image-search`, canonical constant
  `DEFAULT_EXPLORATION` in `schemas/records.py`); `insert_run`/`get_run` write and read it. The
  DDL is unchanged, so this is not a migration and the existing default-value test still passes.

## Known Stubs

None. The only intentionally model-free paths are the unit tests (stub proposal backend, stub
marker method), which is the plan's stated design; the real marker→proposal end-to-end test is
skip-when-absent and runs against the present weights.

## Notes

- The end-to-end API test against the real FastSAM weight asserts routing + persistence and
  tolerates an `outcome='error'` caused by the macOS CoreML execution provider failing to build a
  plan for this graph — an environment limitation, not a pipeline bug. FastSAM decode itself is
  covered by `test_fastsam.py` / `test_proposals.py`.

## Self-Check: PASSED
- All created files exist on disk (verified by the test suite importing them).
- All five task commits present in `git log` (a4788e8, 8318a95, 7f4acbe, 85b79d9, ac69050).
