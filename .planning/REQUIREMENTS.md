# Requirements: Object Search Exploration

**Defined:** 2026-07-24
**Core Value:** Given one hand-drawn exemplar box, return all matching instances in the
image — through any of four interchangeable methods — and accumulate enough evidence
(subjective ratings plus objective precision/recall) to say which method actually works, on
which kind of image, and at what latency.

> Requirement IDs are inherited verbatim from `.planning/IDEA.md` §7 so that documentation,
> commit messages, PR bodies, and code comments all reference the same identifiers. Do not
> renumber them.

## v1 Requirements

### Infrastructure (INFRA)

- [x] **INFRA-01**: Pixi environment, Python 3.12, all commands runnable via `pixi run`
- [x] **INFRA-02**: src-layout package with `py.typed`; `pyproject.toml` is the single
      source of truth for package metadata

- [x] **INFRA-03**: Ruff (line-length 100) and MyPy strict both run clean over the package
- [x] **INFRA-04**: Pre-commit hooks installed before the first commit
- [x] **INFRA-05**: Loguru only for logging — no `print()`, no stdlib `logging`, enforced by
      lint

- [x] **INFRA-06**: Pytest with a ≥80% coverage gate that fails the build below the floor
- [ ] **INFRA-07**: GitHub repo with branch protection on `main` and CI running
      lint / typecheck / test on every PR

- [x] **INFRA-08**: Frozen Pydantic schemas for every inter-layer contract
      (`ExemplarBox`, `Match`, `SearchResult`, `RunRecord`, `Rating`)

- [x] **INFRA-09**: `ONNXInferencer` base performs **init-time dtype and shape validation** —
      a mismatched model raises at construction, not at first frame

- [x] **INFRA-10**: `SearchMethod` protocol plus decorator registry — adding a method touches
      exactly one new file plus one import

- [x] **INFRA-11**: `pixi run fetch-models` downloads or exports every ONNX model; weights
      are gitignored and the export step is scripted and reproducible, never manual

### Methods (METHOD)

- [x] **METHOD-01**: Method 1 `ncc` — `cv2.matchTemplate` with `TM_CCOEFF_NORMED`, image
      pyramid for scale search (per-peak level index so output box size is correct), and an
      optional rotated-template bank (default angle set `[0]`)

- [x] **METHOD-02**: Method 2 `sparse-geo` classical backend — OpenCV SIFT / AKAZE / ORB,
      no weights, no ONNX

- [ ] **METHOD-03**: Method 2 learned backend — SuperPoint via ONNX Runtime, behind the same
      backend interface as the classical detectors

- [x] **METHOD-04**: Method 2 many-to-many top-k kNN matching with the standard Lowe ratio
      test **disabled** (optional k+1 ratio test instead), generalized Hough pose voting with
      soft binning, and per-peak RANSAC with degeneracy rejection

- [x] **METHOD-04a**: Method 2 voting modes `single-4dof` / `translation-2dof` /
      `pairwise-4dof`, selectable by config, because SuperPoint keypoints carry no scale or
      orientation and single-correspondence 4-DoF voting is invalid for that backend

- [ ] **METHOD-04b**: Method 2 sequential-RANSAC decomposition available as a pluggable
      alternative to Hough voting, behind the same interface

- [x] **METHOD-04c**: Method 2 emits an explicit low-keypoint diagnostic rather than an empty
      result when the exemplar lacks texture; the exemplar self-match is labelled as the
      exemplar, neither double-counted nor discarded

- [x] **METHOD-05**: Method 3 `dino-dense` — DINOv2 ONNX dense patch tokens, mean-pooled crop
      prototype, cosine similarity map, threshold, connected components → boxes, with the
      scene run at high input resolution and the similarity map bilinearly upsampled

- [ ] **METHOD-06**: Method 5 `propose-retrieve` — FastSAM/MobileSAM ONNX class-agnostic
      proposals, DINOv2 region embeddings from the same inferencer as Method 3, cosine
      nearest-neighbour retrieval with threshold and NMS; proposal and embedding stages are
      **independently callable units**

- [x] **METHOD-07**: Shared threshold calibration (`search/common/calibration.py`) offering
      `self-similarity`, `ratio`, and `gmm` strategies — imported by choice, never mandated

- [x] **METHOD-08**: Shared peak extraction (`search/common/peaks.py`) offering `nms`,
      `local-max` (default, suppression radius tied to crop size), and `watershed` strategies

- [x] **METHOD-09**: Every method returns a `diagnostics` payload the UI can render
      (similarity map, keypoint correspondences, Hough peaks, proposal set)

- [x] **METHOD-10**: Every method module carries a `ROBUSTNESS BACKLOG` docstring section,
      mirrored into `docs/methods/<name>.md`

- [x] **METHOD-11**: Every method documents its pre-processing and post-processing explicitly,
      in both the module docstring and `docs/methods/<name>.md`

- [x] **METHOD-12**: Multiple instances per image are assumed throughout; no method may
      short-circuit to a single best match

### API (API)

- [x] **API-01**: `GET /methods` returns each method's name, description, and config JSON
      Schema, with zero method names hardcoded in the API layer

- [x] **API-02**: `POST /search` takes image id + exemplar box + method + config and returns
      a `SearchResult`

- [x] **API-03**: Every search is persisted as a `RunRecord` (image, box, method, config hash
      and JSON, matches, latency, timestamp)

- [x] **API-04**: `POST /ratings` records a rating against a run
- [x] **API-05**: `GET /stats` returns the per-method scoreboard
- [x] **API-06**: `GET /images` lists demo images; an upload endpoint accepts ad-hoc images
- [x] **API-07**: ONNX sessions are loaded once at startup via `lifespan` and reused across
      requests

- [x] **API-08**: Structured error handling — a method that raises returns a typed error
      response, not a 500 stack trace

### UI (UI)

- [x] **UI-01**: Method **and** its config are selected **before** the box is drawn, so every
      rating is attributable to an exact method + config

- [x] **UI-02**: Canvas box drawing with zoom/pan, redraw, and clear
- [x] **UI-03**: Results overlaid on the image with per-match scores
- [x] **UI-04**: Toggleable diagnostics overlay (similarity heatmap, keypoints, proposals)
- [x] **UI-05**: Tiered rating widget — thumbs up/down (required, one click); wrong matches
      via either per-match verdicts on the overlay or a bare `wrong_count`; `missed_count`;
      unratable/skip; free-text note. **No 1–5 star scale.**

- [x] **UI-06**: Stats dashboard showing per-method thumbs-up rate, precision/recall where
      ground truth exists, and latency percentiles

- [x] **UI-07**: Config form generated from the method's JSON Schema, so a new method needs
      zero frontend changes

- [x] **UI-08**: Count fields render **empty, never prepopulated with 0**, with one-click
      "all correct" / "none missed" buttons that write an explicit `0`; per-match verdicts
      require an explicit confirm action before they count as assessed

### Store & Evaluation (EVAL)

- [x] **EVAL-01**: SQLite store for runs and ratings, with a versioned and migratable schema
- [ ] **EVAL-02**: Ground-truth box labels for the demo image set
- [x] **EVAL-03**: Synthetic image generator with **exact** ground truth — lattices, clutter,
      distractors, scale and rotation variation

- [ ] **EVAL-04**: Benchmark runner covering every method × every image × default config,
      producing precision, recall, F1, AP, and latency

- [ ] **EVAL-05**: Paired comparison mode — the **same** exemplar box run through all four
      methods, so ratings are directly comparable rather than confounded by different boxes

- [ ] **EVAL-06**: Benchmark results rendered as committed charts and tables
- [x] **EVAL-07**: **Store raw judgments only; never store a derived metric as a column.**
      Precision, recall, F1, and expected-count are computed in queries/views from
      `retrieved`, per-match verdicts, and `missed_count`

- [x] **EVAL-08**: **Log sub-threshold candidates** — every run persists the top-N candidates
      (N ≈ 50) with raw scores *and* the applied threshold, so a threshold sweep and full PR
      curve can be reconstructed offline from ratings already collected

- [x] **EVAL-09**: Every run records provenance — git SHA, model file hash, config hash,
      method version — so ratings from before and after a change are never pooled

- [x] **EVAL-10**: Every run records slice metadata — true instance count, instance scale
      range, rotation range, clutter level, exemplar keypoint count — exact for synthetic
      images, best-effort otherwise

- [x] **EVAL-11**: Latency logged as a breakdown (preprocess / inference / postprocess), not
      a single number

- [x] **EVAL-12**: Empty results and method errors are recorded as **distinct outcomes**, not
      as zero-precision runs

- [x] **EVAL-13**: `rating_completeness` is a first-class field —
      `none` / `precision-only` / `recall-only` / `complete` — plus whether `FP` came from
      per-match verdicts or a bare count. Aggregates state which subset each metric was
      computed over, and report threshold-sweep sample size separately from precision sample
      size

- [x] **EVAL-14**: Stats dashboard reports **n and confidence intervals** (Wilson interval for
      thumbs-up rate) alongside every rate

- [ ] **EVAL-15**: Paired comparisons produce a win/loss/tie record and a Bradley-Terry (or
      Elo) ranking, not just a comparison of independent means

- [x] **EVAL-16**: The duplicate/fragment convention is defined once and shown in the UI —
      two boxes on one instance = 1 TP + 1 FP

- [x] **EVAL-17**: **All human count fields are nullable and stored empty until entered.**
      `null` ("not assessed") is never coerced to `0` ("assessed, none") at any layer — form
      default, API default, or DB column default. A rating submitted without touching the
      counts must not register as perfect precision and recall

- [x] **EVAL-18**: `wrong_count` accepted as a fast alternative to per-match verdicts,
      mutually exclusive with them, validated `0 ≤ wrong_count ≤ R`. If both are present,
      per-match wins and the discrepancy is flagged, not silently reconciled

- [x] **EVAL-19**: **Chip-insertion benchmark set** — 10 generated images, each with a
      *different* randomly-generated chip pasted `N ∈ {5, 10, 15}` times at random
      **non-overlapping** positions on a white background, across 10 canvas sizes ramping from
      small to very large. Ground truth is exact and known by construction, so precision,
      recall, and AP are computable per method with no human rating and no hand-labelling.
      This is the parameter-tuning and method-comparison harness: an objective, zero-cost
      signal that can be re-run after every config change.

### Demo Assets & Docs (DOC)

- [x] **DOC-01**: Demo image set — basketball broadcast frames from the sibling project,
      permissively-licensed generic repeated-instance photos (shelf, PCB, parking lot,
      tiles), and generated synthetic images — with a `LICENSES.md` recording provenance

- [x] **DOC-02**: **Pre-rendered sample runs committed to disk for every method** — a fixed
      exemplar box per demo image, run through each method, results rendered as images under
      `docs/samples/<method>/`, regenerable by one CLI command

- [ ] **DOC-03**: README showing the sample runs for all four methods side by side
- [ ] **DOC-04**: Per-method documentation page — algorithm, pre/post-processing, config
      reference, known failure modes, robustness backlog

- [ ] **DOC-05**: `docs/ROBUSTNESS-BACKLOG.md` aggregating every method's backlog
- [ ] **DOC-06**: `docs/MILESTONE-2.md` specifying the marker-conditioned region proposal
      feature and which Milestone 1 components it reuses

## v2 Requirements

Deferred to a future milestone. Tracked but not in the current roadmap.

### Milestone 2 — Marker-Conditioned Region Proposal

- **M2-01**: Find every instance of a marker (arrow, dot, caret, highlighter blob) by reusing
  any Milestone 1 method wholesale

- **M2-02**: Estimate each marker's reference point and orientation — arrow tip and direction,
  or centroid with no direction for a symmetric marker; orientation from PCA on the marker
  mask or recovered from the similarity/affine transform Method 2 already fits per instance

- **M2-03**: Propose objects near each marker by reusing Method 5's proposal stage directly
- **M2-04**: Score and pick the best proposal by distance from the reference point, alignment
  with the marker's direction, objectness, and a size prior

- **M2-05**: Second UI mode, selected by the exploration-mode selector above the method
  selector

### Deferred Methods

- **DEF-01**: Method 4 — exemplar-conditioned detectors (T-Rex2, CountGD) and counters
  (FamNet, BMNet+, SAFECount, CounTR, LOCA, CACViT). Re-check before committing; this corner
  of the field moves quickly

- **DEF-02**: Method 6 — one-shot personalized segmentation (PerSAM/PerSAM-F, Matcher,
  SegGPT/Painter, SAM 2 memory-bank propagation). Natural Milestone 3 once Method 5's SAM
  proposal stage exists

- **DEF-03**: Lattice fitting as post-detection verification — for grid-arranged instances,
  recovers misses and kills false positives more effectively than tuning the detector

## Out of Scope

| Feature | Reason |
|---------|--------|
| Cross-image / corpus search | Search is confined to a single image in Milestone 1. The Phase 7 embedding store is shaped so this is a later addition, not a rewrite. |
| Training or fine-tuning any model | All models are pretrained and frozen. |
| Video / temporal search | Single still images only. |
| Segmentation masks as primary output | Boxes are the output contract. Masks arrive with deferred Method 6. |
| Multi-user auth, deployment, scaling | Local single-user demo. |
| Real-time performance guarantees | Latency is measured, not guaranteed. |
| FAISS | Hundreds of proposals in one image do not need an ANN index. Adopt when corpus search arrives. |
| LightGlue / SuperGlue | Assignment-based matchers assume one-to-one correspondence — exactly wrong for repeated instances. |
| LoFTR / RoMa dense matching | ONNX export is awkward (variable keypoint counts defeat static shapes; only partial community exports exist). Research spike, not a scheduled task. |
| PyTorch inference path | ONNX Runtime for every learned model, per the local-first and portability constraints. |
| Gradio UI | Box drawing is the core interaction; it would sit behind a third-party community component with no clean seam for Milestone 2's second mode. |
| 1–5 star rating scale | Star scales drift within a session and are not comparable across methods. |
| Stored derived metrics (precision/recall columns) | They go stale on rating edits and invite the null-coercion bug the evaluation design exists to prevent. |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| INFRA-01 | Phase 1 | Complete |
| INFRA-02 | Phase 1 | Complete |
| INFRA-03 | Phase 1 | Complete |
| INFRA-04 | Phase 1 | Complete |
| INFRA-05 | Phase 1 | Complete |
| INFRA-06 | Phase 1 | Complete |
| INFRA-07 | Phase 1 | Pending |
| INFRA-08 | Phase 1 | Complete |
| INFRA-09 | Phase 1 | Complete |
| INFRA-10 | Phase 1 | Complete |
| INFRA-11 | Phase 1 | Complete |
| EVAL-03 | Phase 1 | Complete |
| DOC-01 | Phase 1 | Complete |
| METHOD-01 | Phase 2 | Complete |
| METHOD-07 | Phase 2 | Complete |
| METHOD-08 | Phase 2 | Complete |
| METHOD-09 | Phase 2 | Complete |
| METHOD-10 | Phase 2 | Complete |
| METHOD-11 | Phase 2 | Complete |
| METHOD-12 | Phase 2 | Complete |
| DOC-02 | Phase 2 | Complete |
| API-01 | Phase 3 | Complete |
| API-02 | Phase 3 | Complete |
| API-03 | Phase 3 | Complete |
| API-04 | Phase 3 | Complete |
| API-05 | Phase 3 | Complete |
| API-06 | Phase 3 | Complete |
| API-07 | Phase 3 | Complete |
| API-08 | Phase 3 | Complete |
| EVAL-01 | Phase 3 | Complete |
| EVAL-07 | Phase 3 | Complete |
| EVAL-08 | Phase 3 | Complete |
| EVAL-09 | Phase 3 | Complete |
| EVAL-10 | Phase 3 | Complete |
| EVAL-11 | Phase 3 | Complete |
| EVAL-12 | Phase 3 | Complete |
| EVAL-13 | Phase 3 | Complete |
| EVAL-14 | Phase 3 | Complete |
| EVAL-17 | Phase 3 | Complete |
| EVAL-18 | Phase 3 | Complete |
| EVAL-19 | Phase 1 (generator) + Phase 8 (benchmark consumption) | Phase 1 generator complete; Phase 8 consumption pending |
| UI-01 | Phase 4 | Complete |
| UI-02 | Phase 4 | Complete |
| UI-03 | Phase 4 | Complete |
| UI-04 | Phase 4 | Complete |
| UI-05 | Phase 4 | Complete |
| UI-06 | Phase 4 | Complete |
| UI-07 | Phase 4 | Complete |
| UI-08 | Phase 4 | Complete |
| EVAL-16 | Phase 4 | Complete |
| METHOD-02 | Phase 5 | Complete |
| METHOD-03 | Phase 5 | Pending |
| METHOD-04 | Phase 5 | Complete |
| METHOD-04a | Phase 5 | Complete |
| METHOD-04b | Phase 5 | Pending |
| METHOD-04c | Phase 5 | Complete |
| METHOD-05 | Phase 6 | Complete |
| METHOD-06 | Phase 7 | Pending |
| EVAL-02 | Phase 8 | Pending |
| EVAL-04 | Phase 8 | Pending |
| EVAL-05 | Phase 8 | Pending |
| EVAL-06 | Phase 8 | Pending |
| EVAL-15 | Phase 8 | Pending |
| DOC-03 | Phase 8 | Pending |
| DOC-04 | Phase 8 | Pending |
| DOC-05 | Phase 8 | Pending |
| DOC-06 | Phase 8 | Pending |

**Coverage:**

- v1 requirements: 67 total
- Mapped to phases: 67
- Unmapped: 0 ✓

> Note: METHOD-09/10/11/12 are established in Phase 2 (with Method 1 as the first
> instance) and are re-verified for each subsequent method in Phases 5, 6, and 7. They are
> traced to Phase 2 as the phase that creates the contract.

---
*Requirements defined: 2026-07-24*
*Last updated: 2026-07-24 after initial definition*
