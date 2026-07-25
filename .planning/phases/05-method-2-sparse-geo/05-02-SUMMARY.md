---
phase: 05-method-2-sparse-geo
plan: 02
subsystem: search
tags: [superpoint, onnx, learned-backend, sequential-ransac, library-review, frameless-keypoints, non-commercial-licence]

requires:
  - phase: 01-foundations
    provides: "ONNXInferencer (input-contract validation at load), ModelSpec registry + fetch-models, ExemplarBox/BBox, method registry, SearchResult/Diagnostics schemas"
  - phase: 05-method-2-sparse-geo
    provides: "05-01 classical sparse-geo: backend abstraction (_Backend/_Keypoints), Hough voting, NumPy per-peak RANSAC, sequential-ransac path, three voting modes with single-4dof raising on a frameless backend"
provides:
  - "SuperPointInferencer(ONNXInferencer) — grayscale [1,1,H,W] BT.601 luma, no mean/std, pad-to-8; int64 keypoints, L2-normalized 256-D descriptors, frameless (no scale/angle), 8-px border"
  - "'superpoint' wired into SparseGeoConfig.backend through the SAME code path; switching backend is config alone"
  - "Config-time rejection of single-4dof + superpoint (frameless), with translation-2dof the superpoint default via a before-validator"
  - "docs/library-reviews/superpoint.md — Trial verdict, MagicLeap non-commercial licence, frozen v1.0.0 asset caveat"
  - "sequential-ransac (METHOD-04b) verified with a dedicated test suite"
  - "docs/methods/sparse-geo.md completed for both backends + config reference; docs/ROBUSTNESS-BACKLOG.md sparse-geo section; regenerated docs/samples/sparse-geo/"
affects: [phase-08-evaluation]

tech-stack:
  added: []
  patterns:
    - "A learned backend joins the classical ones behind one _Backend interface; the module stays backend-agnostic because it only reads metric + has_frame"
    - "SuperPointInferencer overrides ONNXInferencer.preprocess for the single-channel grayscale-luma path the base RGB/mean-std path cannot express"
    - "Per-backend config defaults without a nullable schema field: a model_validator(mode=before) rewrites voting_mode for superpoint before validation"
    - "The frameless single-4dof rejection is a config-time (model-free) error, with the voting layer raising the same way as defence in depth"
    - "Descriptors already L2-normalized -> l2 metric via the existing brute-force NumPy kNN (cosine and squared-L2 agree on unit vectors); no re-normalization"

key-files:
  created:
    - src/object_search/inference/superpoint.py
    - tests/test_superpoint.py
    - tests/test_sparse_geo_sequential.py
    - docs/library-reviews/superpoint.md
  modified:
    - src/object_search/search/sparse_geo.py
    - src/object_search/inference/__init__.py
    - src/object_search/inference/models.py
    - docs/methods/sparse-geo.md
    - docs/ROBUSTNESS-BACKLOG.md
    - tests/test_api_app.py

decisions:
  - "Per-backend voting default implemented as a before-validator (superpoint -> translation-2dof) rather than making voting_mode nullable, keeping the JSON Schema the UI form is generated from a plain enum"
  - "single-4dof + superpoint raises at config construction (model-free), so the rule is gated in CI without the gitignored weight"
  - "Pinned superpoint.onnx sha256 from the first verified fetch (EVAL-09), turning the integrity check into a hard gate for an immutable release asset"
  - "SuperPointInferencer.preprocess pads bottom/right to a multiple of 8 (origin-preserving) instead of resizing, so keypoint coordinates need no remapping and no trailing rows/columns are silently floored"

metrics:
  duration_min: 70
  completed: 2026-07-25
  tasks: 4
  files_created: 4
  files_modified: 6
  coverage_total: "91.48%"
  coverage_module: "94% (sparse_geo.py)"

status: complete
---

# Phase 5 Plan 2: SuperPoint ONNX backend, sequential-RANSAC, docs Summary

Added the SuperPoint learned keypoint backend behind the existing `sparse-geo` interface — frameless keypoints, L2-normalized descriptors, config-only switching — verified the pre-existing sequential-RANSAC decomposition, and completed the method doc, robustness backlog, and sample gallery. This completes Phase 5.

## What was built

- **`SuperPointInferencer(ONNXInferencer)`** (`inference/superpoint.py`). Verified contract in the docstring (per the explicit-preprocessing constraint, exact numbers): input `image` f32 `[1,1,H,W]` single-channel **BT.601 grayscale**, range `[0,1]` (`/255`), **no mean/std**, pad-to-8 (origin-preserving, non-multiple sides else silently floored). Outputs `keypoints` **int64** `(x,y)`, `scores`, `descriptors` **already L2-normalized** (do not re-normalize). Effective **8-px** border; keypoints **frameless** (no scale/angle). Overrides `preprocess` for the grayscale-luma path the base RGB/mean-std path cannot express.
- **Backend wiring** (`search/sparse_geo.py`). `"superpoint"` added to `SparseGeoConfig.backend`; `_make_backend` loads the ONNX weight (frameless, `l2` metric) and `_detect` branches to `_detect_superpoint`, returning the identical `_Keypoints` shape so the rest of the module is unchanged. `single-4dof` + `superpoint` **raises at config time** (a `model_validator(mode="after")`); `translation-2dof` is the superpoint default (a `model_validator(mode="before")` rewrites it, keeping the schema a plain enum).
- **library-review** (`docs/library-reviews/superpoint.md`): **Trial** verdict, v1.0.0 release asset, MagicLeap **non-commercial research-only** weights (DERIVATIVES covers the ONNX; gitignored, never redistributed), the frozen-asset-not-reproducible-from-HEAD caveat, and rejected candidates.
- **sequential-RANSAC (METHOD-04b)** — already implemented in 05-01; **verified** with `tests/test_sparse_geo_sequential.py` (switches by config alone, recovers multiple distinct models on the 6-instance scene, deterministic under a fixed seed, comparable count to Hough, exemplar self-match labelled once).
- **Docs & samples** — `docs/methods/sparse-geo.md` gains a backends table, both backends' explicit pre/post-processing, SuperPoint output decoding, and a full config reference (NCC crossover already recorded as an expected finding). `docs/ROBUSTNESS-BACKLOG.md` gains the sparse-geo section mirrored from the module docstring. `pixi run samples` regenerated `docs/samples/sparse-geo/` via the registry-iterating renderer with **no renderer change**.

## Verification (real output, local osx-arm64)

- `pixi run lint` → `All checks passed!`
- `pixi run format-check` → `83 files already formatted`
- `pixi run typecheck` → `Success: no issues found in 47 source files`
- `pixi run test` → **331 passed, 3 skipped** (the 3 skips are the DINOv2 real-model tests; that weight is absent), **coverage 91.48%** (≥80% floor). `sparse_geo.py` at 94%.
- **Frameless single-4dof raise:** `SparseGeoConfig(backend="superpoint", voting_mode="single-4dof")` → raises a validation error naming the incompatibility; `SparseGeoConfig(backend="superpoint").voting_mode == "translation-2dof"`.
- **Model fetched:** `pixi run fetch-models --only superpoint` → installed 5.03 MiB, sha256 `234d12c9f523292efb34e0ca513b011050b0c052700da9c01787b9356a1138d2` (now pinned). Because the weight is present, the SuperPoint real-inference tests (int64 keypoints, ‖d‖ = 1, variable count, end-to-end) ran and passed; in CI (weights gitignored) they skip and the model-free contract/mode tests still gate the rule.

## Human-verify checkpoint — pending an orchestrator browser run

Not blocked (per instructions). Server smoke-checked: `pixi run serve` up; `GET /methods` → **200** (schema includes the `superpoint` backend), `GET /` → 307 → `/app/` → **200**. The **visual** criteria remain for a person / the orchestrator to confirm in the browser:

1. Pick `sparse-geo`, draw a box on one textured instance in an image with 6+ textured instances, run Search → **multiple distinct boxes** returned (not one).
2. Toggle the diagnostics overlay → the **Hough peaks and correspondences** are visible.
3. Switch the backend config from `sift` to `superpoint` → it still runs through the **same** path.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 — Bug] Made the Phase 3 `test_api_app` lifespan test hermetic**
- **Found during:** running the full suite after fetching the SuperPoint weight.
- **Issue:** `test_lifespan_migrates_store_and_builds_empty_session_registry` asserted `app.state.sessions == {}` using the real `models/` dir. The API lifespan loads whatever ONNX weights are present (API-07); making `superpoint.onnx` genuinely loadable meant the test failed the moment a developer fetched the weight. In CI (gitignored, empty `models/`) it passed, but the test was non-hermetic.
- **Fix:** point `lifespan_module.models_dir` at an empty temp dir in that test, so it is deterministic regardless of local fetches; the empty-load-list intent is still asserted.
- **Files modified:** `tests/test_api_app.py`
- **Commit:** a288f87

**2. [Rule 2 — Provenance] Pinned the SuperPoint sha256**
- **Issue:** the ModelSpec carried `sha256=None`; the v1.0.0 asset is immutable, so recording its hash makes the fetch integrity check a hard gate (EVAL-09).
- **Fix:** pinned `234d12c9…` from the first verified fetch.
- **Files modified:** `src/object_search/inference/models.py`
- **Commit:** 0337697

## Known Stubs

None. The SuperPoint backend runs a real ONNX model end-to-end; there are no placeholder data paths.

## Note on the sample gallery

The regenerated `docs/samples/sparse-geo/` panels show `outcome=empty` on all four manifest images. This is **not** a defect: the shared sample manifest (designed for NCC) uses smooth synthetic `DEMO_SPECS` scenes on which the exemplar crops yield only 0–3 SIFT keypoints, so `sparse-geo` correctly **abstains with a diagnostic note** (the METHOD-04c low-keypoint guard firing), rather than returning a silent empty. It is a faithful demonstration of the guard.

## Self-Check: PASSED
