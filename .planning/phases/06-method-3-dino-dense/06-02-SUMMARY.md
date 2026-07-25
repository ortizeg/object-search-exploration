---
phase: 06-method-3-dino-dense
plan: 02
subsystem: search
tags: [dino-dense, dinov2, cosine-similarity, connected-components, calibration, method-3]
status: complete

# Dependency graph
requires:
  - phase: 06-method-3-dino-dense
    provides: DINOv2Inferencer.dense_tokens(image) -> (grid, scale_x, scale_y) from 06-01
  - phase: 01-foundations
    provides: SearchResult/Match/Candidate/Diagnostics schemas, registry, common.calibration, common.viz
provides:
  - "dino-dense search method registered in the registry (the third method, METHOD-05 complete)"
  - "DINOv2 dense-token prototype cosine-similarity search with high-res capped inference and map upsampling"
  - "reusable pure-numpy helpers (L2-normalize, prototype pooling, similarity map, map upsampling, component extraction)"
affects: [api-session-registry, propose-retrieve, phase-07, benchmark]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Module-level lazy-singleton inferencer: the SearchFn contract shares nothing but (image, exemplar, config), so the method loads and caches its own DINOv2Inferencer from the gitignored weight rather than reading app.state"
    - "Cosine-not-raw-dot: L2-normalize BOTH prototype and token grid before the dot, so DINOv2's high-norm artifact tokens cannot dominate by magnitude"
    - "Upsample the MAP (gh,gw) not the 384-d tokens, using the inferencer's scale factors so a token peak lands on its patch-centre pixel"
    - "connectedComponentsWithStats with an explicit label-0 (background) skip; sub-threshold components retained as EVAL-08 candidates; every clearing component survives (METHOD-12)"

key-files:
  created:
    - src/object_search/search/dino_dense.py
    - tests/test_dino_dense.py
    - docs/methods/dino-dense.md
    - docs/samples/dino-dense/ (cluttered-distractors, lattice-plain, lattice-touching, scatter-scaled + index.md)
  modified:
    - src/object_search/search/__init__.py
    - docs/ROBUSTNESS-BACKLOG.md
    - tests/test_api_app.py
    - .planning/REQUIREMENTS.md
    - .planning/ROADMAP.md

key-decisions:
  - "Both the prototype and the token grid are L2-normalized before the dot product (cosine), not an unnormalized dot dominated by token magnitude"
  - "The similarity MAP is bilinearly upsampled to input resolution using the inferencer's scale factors; the tokens are never upsampled"
  - "connectedComponentsWithStats label 0 (background) is skipped explicitly with a comment; label-0 emission would be one image-sized false positive"
  - "The scene input resolution is CAPPED at scene_max_side (default 1568) and the cap is LOGGED when it engages; scenes at/below the cap run native"
  - "Default calibration is gmm (two-mode fit on the similarity distribution); absolute cosine thresholds do not transfer across images for deep features"
  - "Absent weight degrades to outcome=error with a model_unavailable note, never a raise, so the sample renderer and API stay green"

requirements-completed: [METHOD-05]

# Coverage metadata
coverage:
  - id: M1
    description: "L2-normalization order: prototype and grid normalized before the dot (cosine)"
    requirement: "METHOD-05"
    verification:
      - kind: unit
        target: "tests/test_dino_dense.py::test_prototype_is_mean_pooled_then_l2_normalized, ::test_similarity_is_cosine_not_magnitude_dominated_raw_dot, ::test_l2_normalize_makes_unit_vectors_and_keeps_zero_zero"
  - id: M2
    description: "The similarity map is bilinearly upsampled using the inferencer's scale factors so a token peak aligns to its pixel"
    requirement: "METHOD-05"
    verification:
      - kind: unit
        target: "tests/test_dino_dense.py::test_upsample_aligns_a_known_token_peak_to_its_patch_pixel, ::test_upsample_uses_the_scale_factors_to_recover_input_size"
  - id: M3
    description: "connectedComponentsWithStats skips label 0 (background); min-area and cap-scale respected"
    requirement: "METHOD-05"
    verification:
      - kind: unit
        target: "tests/test_dino_dense.py::test_extract_components_skips_label_zero_background, ::test_extract_components_respects_min_area_and_cap_scale"
  - id: M4
    description: "dino-dense finds pose/rotation-varied instances that ncc misses (success criterion 1)"
    requirement: "METHOD-05"
    verification:
      - kind: integration
        target: "tests/test_dino_dense.py::test_dino_dense_beats_ncc_on_pose_variation (model-gated; verified locally dino recall 1.0 vs ncc 0.0)"
  - id: M5
    description: "Three calibration strategies produce different, inspectable thresholds (success criterion 3)"
    requirement: "METHOD-05"
    verification:
      - kind: unit
        target: "tests/test_dino_dense.py::test_three_calibration_strategies_yield_different_reasoned_thresholds"
  - id: M6
    description: "The resolution cap engages and is logged/recorded rather than silently truncating"
    requirement: "METHOD-05"
    verification:
      - kind: integration
        target: "tests/test_dino_dense.py::test_resolution_cap_engages_and_is_recorded (model-gated)"

metrics:
  duration_minutes: 40
  tasks_completed: 2
  files_created: 4
  files_modified: 5
  completed_date: 2026-07-25
---

# Phase 6 Plan 2: dino-dense — DINOv2 Dense-Token Prototype Matching Summary

Method 3 (`dino-dense`, METHOD-05) implemented as one self-contained module reusing the Phase 6
(1/2) `DINOv2Inferencer`: mean-pooled crop prototype → cosine similarity map → high-res capped
inference → bilinear map upsampling → gmm-calibrated threshold → connected components (label-0
skipped) → boxes, with sub-threshold candidates and a similarity heatmap. It completes Phase 6.

## What was built

- **`src/object_search/search/dino_dense.py`** — the method, top-to-bottom readable with numbered
  steps matching `docs/methods/dino-dense.md`, a `DinoDenseConfig` (frozen; `scene_max_side`,
  `calibration`, `threshold`, `candidate_margin`, `min_component_area`, `max_candidates`, `seed`,
  `retain_frac`), a ROBUSTNESS BACKLOG docstring, and explicit pre/post-processing. Registered via
  `@register_method` and one import line in `search/__init__.py`.
- **`tests/test_dino_dense.py`** — 13 model-free tests (gate CI) + 4 model-gated tests
  (skip-when-absent), 17 total, all passing locally with the weight fetched.
- **`docs/methods/dino-dense.md`**, an appended `docs/ROBUSTNESS-BACKLOG.md` section, and the
  regenerated `docs/samples/dino-dense/` gallery.
- **Deferred fix from 06-01:** `tests/test_api_app.py` no longer assumes an empty session
  registry — it asserts the registry matches on-disk weight presence, passing with and without the
  gitignored weight.

## How the Phase 6 success criteria were met

1. **dino-dense beats ncc on pose variation** — `test_dino_dense_beats_ncc_on_pose_variation`: on
   a fixture with an upright exemplar + two rotated copies, dino-dense (default gmm) recall `1.0`
   vs ncc recall `0.0`, and a concrete instance is covered by dino-dense but not ncc.
2. **Similarity heatmap renders** — produced and carried in `Diagnostics.similarity_heatmap`;
   visible in the regenerated samples. **The in-UI visual check is pending an orchestrator browser
   run** (see Manual verification).
3. **Three calibration strategies differ** — `test_three_calibration_strategies_...`: self-
   similarity, ratio, gmm give three distinct, reasoned thresholds on one map.
5. **Inferencer reuse** — the method reuses the 06-01 `DINOv2Inferencer` (one download, one
   preprocessing contract), the same instance Method 5 reuses in Phase 7.

## Verification (all four gates green, local)

- `ruff check` clean · `ruff format --check` clean · `mypy src/` strict: no issues (47 files).
- Full suite: **331 passed, coverage 92.92%** (floor 80%); `dino_dense.py` at **96%**.
- Model fetched: `pixi run fetch-models --only dinov2-small` installed
  `models/dinov2_small.onnx` (sha256 matched the pinned hash). **`n_register = 0`** for this model
  (derived, not hardcoded). The ncc-disagreement, end-to-end, and resolution-cap tests **ran and
  passed** locally.
- With the weight **absent**: `test_api_app` + `test_samples` + model-free `test_dino_dense` pass;
  the 4 model-required dino-dense tests skip — so CI (no weight) stays green.
- `pixi run samples`: dino-dense appears (registry iterates), `ok` on all four sample scenes with
  real heatmaps and multiple instances.
- Server: `GET /methods` → 200 with `dino-dense` and its full config schema; `GET /docs` → 200.

## Manual verification (pending orchestrator browser run)

`pixi run serve`, then: pick **dino-dense**, draw a box on an object that recurs with pose/lighting
variation, run Search, confirm instances are found that ncc would miss, toggle the diagnostics
overlay to confirm the **similarity heatmap** renders over the image, and switch the **calibration**
strategy in the config form to confirm the threshold visibly changes.

## Deviations from Plan

### 1. [Rule 4-adjacent — architectural constraint honoured] Inferencer via module-level lazy singleton, not `app.state`

- **Found during:** Task 1. The plan's step 1 said "get the DINOv2 inferencer from app.state (or
  construct for CLI)". The `SearchFn` contract is `(image, exemplar, config) -> SearchResult` and
  `routes_search` calls `spec.fn(image, exemplar, config)` — methods have no access to
  `app.state.sessions`, and the session registry stores a raw `ort.InferenceSession`, not a
  `DINOv2Inferencer`. Threading `app.state` into every method would change the shared protocol
  (an architectural change).
- **Resolution:** the method builds and caches a `DINOv2Inferencer` once via a module-level lazy
  singleton loaded from the gitignored weight. This reuses the one model across all queries, keeps
  the method self-contained (the primary convention), and needs no protocol change. The lifespan
  still warm-loads the weight at startup (and the fixed `test_api_app` asserts it).

### 2. [Rule 1 — correctness] `n_register` is 0 for this model, not 4

- The task brief stated "n_register=4 for this model". The actual
  `onnx-community/dinov2-small-ONNX` has **`n_register = 0`** (verified at load and asserted in
  `test_dinov2.py::test_construction_probes_and_pins_layout`). The "4" is what the synthetic
  `test_derive_layout_derives_register_count_not_hardcoded_one` case yields for a *with-registers*
  variant. The method doc records the accurate value: derived from the token count (=0 here; would
  be 4 for a with-registers variant), never hardcoded to `1`.

## Known Stubs

None. The method is fully wired: real DINOv2 inference, real cosine similarity, real connected
components, real calibration, real heatmap. The only non-search path is the honest
`outcome=error` return when the gitignored weight is absent.

## Out-of-scope note

`pixi run samples` regenerated `docs/samples/sparse-geo/` (Phase 5's method) as a side effect of
the registry-driven renderer; those files were left untracked and **not** committed here, as they
belong to Phase 5's documentation scope.

## Self-Check: PASSED
