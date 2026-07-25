---
phase: 05-method-2-sparse-geo
plan: 01
subsystem: search
tags: [sift, akaze, orb, hough-voting, ransac, keypoint-matching, multi-instance, determinism]

requires:
  - phase: 01-foundations
    provides: "ExemplarBox/BBox half-open geometry, method registry (@register_method), SearchResult/Match/Candidate/Diagnostics/HoughPeak/Correspondence schemas"
  - phase: 02-method-1-ncc
    provides: "Method-module conventions (numbered steps, ROBUSTNESS BACKLOG, is_exemplar labelling, EMPTY-with-note guard)"
provides:
  - "search/sparse_geo.py — Method 2 classical path, one self-contained module registered as 'sparse-geo'"
  - "SIFT/AKAZE/ORB backend abstraction with the distance metric fixed by the backend (L2 vs Hamming), never a config field"
  - "Many-to-many top-k matching with the standard Lowe ratio test DISABLED; optional k+1 ratio; k_ceiling_hit diagnostic"
  - "Generalized Hough voting: soft binning (16 bins/vote), circular theta, scale-dependent location bins in a dict; three voting modes with single-4dof raising on a frameless backend"
  - "NumPy per-peak RANSAC seeded from np.random.default_rng(config.seed); proper+reflected 2-point fits; scale + mirror degeneracy rejection"
  - "sequential-ransac decomposition (METHOD-04b) behind the same interface, so the decomposition config field is a real control"
  - "docs/methods/sparse-geo.md mirroring the numbered steps, pre/post-processing, and ROBUSTNESS BACKLOG"
affects: [phase-05-method-2-sparse-geo-plan-02]

tech-stack:
  added: []
  patterns:
    - "The whole algorithm reads top-to-bottom in one file; # 1..# 9 in search() match docs/methods/sparse-geo.md one-for-one"
    - "Descriptor kNN is brute-force NumPy (L2 expansion / popcount-table Hamming), so matching is deterministic and independent of cv2's BFMatcher"
    - "Pose votes live in a dict keyed by (x,y,scale,theta) because the location bin width is scale-dependent (Lowe §7.3)"
    - "A 2-point sample fits BOTH proper and reflected similarities, so mirror rejection via negative determinant is non-vacuous rather than a control that can only pass"
    - "Reproducibility comes from np.random.default_rng(config.seed), never cv2.setRNGSeed (which has no effect on OpenCV RANSAC)"

key-files:
  created:
    - src/object_search/search/sparse_geo.py
    - tests/test_sparse_geo_matching.py
    - tests/test_sparse_geo_hough.py
    - tests/test_sparse_geo_ransac.py
    - docs/methods/sparse-geo.md
  modified:
    - src/object_search/search/__init__.py

decisions:
  - "AKAZE is configured for its float KAZE descriptor (DESCRIPTOR_KAZE), so — as the method contract states — SIFT and AKAZE are both L2 and only ORB is Hamming"
  - "Implemented sequential-ransac (METHOD-04b) rather than expose an inert decomposition control; formally left METHOD-04b as Pending since it is not in this plan's requirement set"
  - "Built the single file across three atomic layer commits (matching -> voting -> RANSAC/search); the low-keypoint guard's outcome=EMPTY end-to-end assertion lands with search in layer 3"

metrics:
  duration_min: 55
  completed: 2026-07-25
  tasks: 3
  files_created: 5
  files_modified: 1
  coverage_total: "91.72%"
  coverage_module: "93% (sparse_geo.py)"

status: complete
---

# Phase 5 Plan 1: sparse-geo classical — matching, Hough voting, per-peak RANSAC — Summary

Method 2's classical path built as one self-contained, top-to-bottom-readable module: SIFT/AKAZE/ORB
backends, many-to-many top-k matching with the standard Lowe ratio test **disabled**, generalized
Hough voting with soft binning and three voting modes, and NumPy per-peak RANSAC with scale + mirror
degeneracy rejection. Recovers **many** geometric models (one per instance), never a single best.

## What was built

- **Config + backend abstraction** — `SparseGeoConfig` (frozen, every field described for the UI
  form). SIFT is the default backend; the descriptor **distance metric is a property of the
  backend** (SIFT/AKAZE float L2, ORB Hamming), not a config field.
- **Matching (ratio test disabled)** — top-`k` scene neighbours taken unconditionally; only the
  optional k+1 ratio is available; `k_ceiling_hit` surfaces truncated instance sets. Brute-force
  NumPy kNN keeps it deterministic.
- **Hough voting** — soft binning into the 2 nearest bins per dimension (16 in 4-DoF), circular
  theta wrap, Lowe's §7.3 widths, and **scale-dependent location bins stored in a dict**. Three
  voting modes (METHOD-04a); `single-4dof` **raises** on a frameless backend.
- **Per-peak RANSAC** — 4-DoF similarity fit in NumPy seeded from `np.random.default_rng(config.seed)`;
  each 2-point sample fits proper **and** reflected models; degeneracy rejection is scale +
  determinant-sign only (shear/aspect are vacuous for 4-DoF, with a comment saying so).
- **Assembly** — exemplar self-match labelled `is_exemplar`; sub-threshold peaks kept as candidates
  (EVAL-08); multiple distinct models returned (METHOD-12). Hough (default) and sequential-ransac
  decompositions behind one interface. Registered as `sparse-geo`; documented in
  `docs/methods/sparse-geo.md`.

## Verification (real output)

All four quality gates green; coverage **91.72%** total, **93%** for `sparse_geo.py`; **305 tests
pass**. Voting and RANSAC helpers are unit-tested independently of the end-to-end path.

- **Counterfactual ratio-test proof** — with 6 near-identical instances per crop keypoint, the
  standard Lowe ratio (best/second < 0.8) keeps `0` correspondences while the top-k path keeps all
  `n_crop × k`.
- **Frameless `single-4dof` raises** — `ValueError` matching `single-4dof`.
- **Seed determinism + different-seed-differs** — seed 0 twice → identical `sample_log` and model
  matrix; seed 0 vs seed 1 → different `sample_log`; clean similarity recovered (≥15 inliers, det>0,
  scale ≈ 1.2).
- **Low-keypoint guard** — flat scene → `outcome=EMPTY`, note contains "texture".
- **6-instance scene** — `outcome=OK`, **6 distinct boxes**, exactly **1** labelled `is_exemplar`,
  every match carries a flattened 2×3 transform. Mirror-transformed correspondences fit a det<0
  model and are rejected.

## Requirements

METHOD-02, METHOD-04, METHOD-04a, METHOD-04c ticked. METHOD-12 honoured (multiple distinct models).
METHOD-04b (sequential-RANSAC) is implemented and tested but left **Pending** in the register — it
is not in this plan's requirement set; the phase can claim it formally alongside 05-02.

## Deviations from Plan

### Auto-added functionality (Rule 2)

**1. Implemented the `sequential-ransac` decomposition (METHOD-04b).** The plan's config spec
includes a `decomposition` field with a `sequential-ransac` option; exposing it without an
implementation would be an advertised-inert control, which the project explicitly forbids. Built it
as a real alternative behind the same interface. Files: `src/object_search/search/sparse_geo.py`.
Commit: b474f13.

**2. Created `docs/methods/sparse-geo.md`.** Not in the plan's `files_modified`, but the method-module
convention mandates a method doc that mirrors the numbered steps and the ROBUSTNESS BACKLOG. Commit: b474f13.

### Sequencing note

The single module was built across the three task commits by layer (matching → voting →
RANSAC/search). The low-keypoint guard's `outcome=EMPTY` end-to-end assertion therefore lands with
`search` in the layer-3 test (`test_sparse_geo_ransac.py`); the guard predicate and its note are
unit-tested in layer 1. No functional deviation.

## Known Stubs

None. The classical path is fully wired; SuperPoint backend and UI diagnostics are the scope of 05-02.

## Self-Check: PASSED

- `src/object_search/search/sparse_geo.py` — FOUND
- `tests/test_sparse_geo_matching.py`, `tests/test_sparse_geo_hough.py`, `tests/test_sparse_geo_ransac.py` — FOUND
- `docs/methods/sparse-geo.md` — FOUND
- Commits ee08dfc, 8839787, b474f13 — FOUND on `phase-05/sparse-geo-classical`
- PR: https://github.com/ortizeg/object-search-exploration/pull/10 (open, CI `quality` pass, not merged)
