---
phase: 02-method-1-shared-primitives
plan: 01
subsystem: search/common
tags: [nms, peaks, calibration, viz, scipy, sklearn, matplotlib, opencv, reproducibility]
requires:
  - phase: 01-02
    provides: frozen Pydantic schemas (BBox, ExemplarBox, Match, Point, HeatmapPayload)
provides:
  - deterministic greedy IoU NMS with (-score, y, x) tie-breaking (nms.py)
  - extract_peaks with nms / local-max / watershed strategies (peaks.py)
  - calibrate with fixed / self-similarity / ratio / gmm strategies + inspectable reason
  - headless viz: match overlays, similarity heatmap PNG, keypoint/correspondence panels
affects:
  - Phase 2 plan 02 (ncc) may import these offerings or inline its own variants
  - Phase 5 (sparse-geo) reuses draw_correspondences and the peak/calibration offerings
  - Phase 6 (dino-dense) uses watershed peaks on dense similarity maps
tech-stack:
  added:
    scipy: ">=1.14 (maximum_filter, distance_transform_edt, label)"
    scikit-learn: ">=1.5 (GaussianMixture for gmm calibration)"
    matplotlib-base: ">=3.9 (Agg-forced colormap for heatmaps)"
  patterns:
    - "Offerings, not requirements: search/common/__init__ does not eagerly import submodules"
    - "Every peak/nms path sorts by (-score, y, x) for run-to-run byte identity"
    - "Calibration returns threshold AND a reason string so a cut is inspectable"
    - "viz forces matplotlib Agg at import; cv2 for overlays, matplotlib only for colormaps"
key-files:
  created:
    - src/object_search/search/common/__init__.py
    - src/object_search/search/common/nms.py
    - src/object_search/search/common/peaks.py
    - src/object_search/search/common/calibration.py
    - src/object_search/search/common/viz.py
    - tests/test_nms.py
    - tests/test_peaks.py
    - tests/test_calibration.py
    - tests/test_viz.py
  modified:
    - .planning/REQUIREMENTS.md
decisions:
  - "nms suppression boundary is strict '>' — a pair at exactly iou_threshold is kept, pinned by test"
  - "local-max footprint = round(template * frac), forced odd and >= 3x3"
  - "gmm degeneracy guard: means < 2 pooled-within-std apart OR a starved component => fall back to ratio"
  - "heatmap payload reports the true (or caller-supplied) vmin/vmax the colormap spanned"
metrics:
  duration: ~40m
  completed: 2026-07-25
  tasks: 4
  files: 10
status: complete
---

# Phase 2 Plan 1: Shared Primitives Summary

Deterministic NMS, size-aware peak extraction (local-max beats plain NMS on touching
instances), inspectable threshold calibration (self-similarity / ratio / gmm with a degeneracy
fallback), and headless visualization — the four `search/common/` **offerings** later methods
may import or ignore.

## What was built

- **`nms.py`** — `nms(boxes, scores, iou_threshold) -> list[int]` returning kept indices.
  Imposes the total order `(-score, y, x)` before greedy suppression so tied scores (constant
  on synthetic lattices) never reorder run-to-run (PITFALLS.md 6.3). Strict `>` suppression
  boundary, pinned by a test.
- **`peaks.py`** — `extract_peaks(response, *, strategy, template_w, template_h, floor, ...)`.
  Three strategies via a small `if/elif`: `nms` (merges touching instances — the control),
  `local-max` (default; `scipy.ndimage.maximum_filter` footprint derived from crop size),
  `watershed` (distance-transform markers, label 0 skipped as background). Non-finite values
  sanitised so a NaN can never surface as a peak.
- **`calibration.py`** — `calibrate(scores, *, strategy, ...) -> CalibrationResult`. The result
  carries `threshold` **and** a human `reason`. `self-similarity` raises (naming itself) when
  `self_score` is missing. `gmm` uses `GaussianMixture(random_state=seed)` — a genuine seed —
  and a degeneracy guard that falls back to `ratio` with `degenerate=True` on a single-mode fit.
- **`viz.py`** — matplotlib `Agg` forced at import; cv2 for box/point overlays. `draw_matches`
  renders the exemplar distinctly (magenta, thicker) from ordinary matches. `heatmap_png_b64`
  returns a `HeatmapPayload` with the true `vmin/vmax`. Plus `draw_keypoints`,
  `draw_correspondences`, `compose_panel` for Phase 5 and the sample renderer.

## Verification (real output)

```
lint          — All checks passed!
format-check  — 38 files already formatted
typecheck     — Success: no issues found in 22 source files
test          — 166 passed in 3.05s; Total coverage: 90.85% (floor 80%)
```

Per-module coverage of the new files: nms 94%, peaks 93%, calibration 89%, viz 94%.

**Phase 2 success criterion 2** (local-max separates what nms merges):
```
tests/test_peaks.py::test_local_max_separates_touching_instances_that_nms_merges PASSED
```
On one touching-bump map, `nms` returns 1 peak and `local-max` returns 2.

**Phase 2 success criterion 4** (different, inspectable thresholds):
```
tests/test_calibration.py::test_three_strategies_differ_on_a_bimodal_input PASSED
tests/test_calibration.py::test_gmm_flags_a_unimodal_distribution_as_degenerate_and_falls_back PASSED
```
The three strategies cut at distinct values on one bimodal input, each with a stated reason;
`gmm` flags a unimodal distribution as degenerate and falls back to `ratio`.

## Deviations from Plan

None — plan executed as written.

One implementation note (not a deviation): the load-bearing peaks test uses an explicit
`suppression_radius_frac=0.25` with a 48-px template so the nms-merge / local-max-separate
margin is comfortable rather than sitting on the IoU boundary. The demonstration is unchanged;
the default `frac` remains `0.5` in the module.

## Commits

- `ce0d676` feat(02-01): deterministic greedy IoU NMS offering (METHOD-07)
- `8f9345c` feat(02-01): peak extraction offering — nms / local-max / watershed (METHOD-08)
- `a7b9cbe` feat(02-01): threshold calibration offering — fixed / self-similarity / ratio / gmm
- `b794c7e` feat(02-01): headless viz offering — overlays, heatmap PNG, panels

## Known Stubs

None. `draw_correspondences` draws `src` on the crop panel and `dst` on the scene panel; the
richer exemplar-origin-aware rendering (subtracting the exemplar box origin) is a Phase 5
concern when the method that produces correspondences exists. This is documented in the
function docstring, not a silent stub.

## Self-Check: PASSED

All four modules and four test files exist on disk; all four task commits are in `git log`.
