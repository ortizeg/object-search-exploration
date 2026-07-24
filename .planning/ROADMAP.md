# Roadmap: Object Search Exploration

## Overview

The journey runs foundation → baseline method → API → UI → the three learned methods →
evaluation. The ordering is deliberate and is the roadmap's single most important decision:
**the API and UI land between method 1 and method 2**, so that every method after the first
is drawable, runnable, and ratable by a human on the day it lands. The alternative — build
all four methods, then a UI — would leave three methods untested by a human until the very
end, which is precisely the failure this project exists to avoid.

Method numbering follows the source research (1 → 2 → 3 → 5) so that code, documentation,
and the research survey stay aligned. Methods 4 and 6 from the research are deliberately
deferred; see `.planning/REQUIREMENTS.md` § v2.

Phases 5, 6, and 7 are mutually independent once Phase 4 lands and are the parallelization
opportunity — with the caveat that Phase 7 must follow Phase 6, because it reuses Phase 6's
DINOv2 ONNX inferencer rather than exporting a second copy.

Each phase ships as **2 pull requests** against a protected `main`, matching the two natural
checkpoints in each phase.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

- [ ] **Phase 1: Foundation** - Scaffold, quality gates, CI, schemas, registry, ONNXInferencer, demo assets
- [ ] **Phase 2: Method 1 + shared primitives** - `ncc`, calibration/peaks/nms/viz, sample-run renderer
- [ ] **Phase 3: Backend API** - FastAPI endpoints and the SQLite run + rating store
- [ ] **Phase 4: Web UI** - Canvas box drawing, schema-driven config, overlays, rating widget, stats
- [ ] **Phase 5: Method 2** - `sparse-geo` classical + SuperPoint, Hough voting, per-peak RANSAC
- [ ] **Phase 6: Method 3** - DINOv2 ONNX inferencer and `dino-dense` dense similarity
- [ ] **Phase 7: Method 5** - FastSAM proposals + DINOv2 region embeddings, `propose-retrieve`
- [ ] **Phase 8: Evaluation & docs** - Benchmark, paired comparison, charts, docs, Milestone 2 spec

## Phase Details

### Phase 1: Foundation
**Goal**: A green, protected, fully-gated repository with every shared contract in place —
so that adding a search method later is a single new file, and a wrong ONNX model fails at
load rather than at first frame.
**Depends on**: Nothing (first phase)
**Requirements**: INFRA-01, INFRA-02, INFRA-03, INFRA-04, INFRA-05, INFRA-06, INFRA-07, INFRA-08, INFRA-09, INFRA-10, INFRA-11, EVAL-03, DOC-01
**Success Criteria** (what must be TRUE):
  1. `pixi run lint`, `pixi run typecheck`, and `pixi run test` all pass locally and on a
     green CI run against a protected `main` that cannot be pushed to directly.
  2. An intentionally mismatched ONNX model raises at `ONNXInferencer` construction time,
     before any image is processed — proven by a test that constructs one and asserts the
     raise.
  3. The synthetic generator emits an image plus its exact ground-truth boxes, and the same
     seed reproduces the same image byte-for-byte.
  4. `pixi run fetch-models` runs end to end and is the only way weights ever arrive; no
     weight file is tracked in git.
  5. A demo image set exists on disk with `LICENSES.md` recording the provenance and licence
     of every image.
**Plans**: 2 plans

Plans:
- [ ] 01-01: Pixi/pyproject scaffold, Ruff + MyPy strict + pre-commit + Loguru, pytest with
      coverage gate, GitHub repo settings, CI workflow, branch protection
- [ ] 01-02: Frozen Pydantic schemas, `SearchMethod` protocol + registry, `ONNXInferencer`
      base with init-time dtype/shape validation, `fetch-models`, synthetic generator,
      demo asset set + `LICENSES.md`

### Phase 2: Method 1 + shared primitives
**Goal**: The honest zero-model baseline every learned method must clear, plus the two
cross-cutting primitives the research names as the real weak links (thresholding and peak
extraction) — and the first committed sample runs.
**Depends on**: Phase 1
**Requirements**: METHOD-01, METHOD-07, METHOD-08, METHOD-09, METHOD-10, METHOD-11, METHOD-12, DOC-02
**Success Criteria** (what must be TRUE):
  1. Drawing a box on a synthetic lattice image and running `ncc` from the CLI returns every
     instance with no duplicates.
  2. Swapping the peak strategy from `nms` to `local-max` measurably separates touching
     instances — demonstrated by a test asserting a higher instance count on a
     deliberately-touching lattice.
  3. Sample-run images exist on disk under `docs/samples/ncc/` and regenerate identically
     from one CLI command.
  4. All three calibration strategies (`self-similarity`, `ratio`, `gmm`) run on the same
     image and produce different, inspectable thresholds.
  5. `ncc` returns a `diagnostics` payload, carries a `ROBUSTNESS BACKLOG` docstring section,
     and documents its pre/post-processing in both the module docstring and
     `docs/methods/ncc.md`.
**Plans**: 2 plans

Plans:
- [ ] 02-01: `search/common/` — `calibration.py`, `peaks.py`, `nms.py`, `viz.py` with their
      selectable strategies and tests
- [ ] 02-02: `ncc.py` with pyramid scale search and optional rotation bank, the sample-run
      renderer CLI, first committed sample runs, `docs/methods/ncc.md`

### Phase 3: Backend API
**Goal**: Every search is an HTTP call that gets persisted with enough context — provenance,
latency breakdown, slice metadata, and sub-threshold candidates — that a rating collected
today still means something after the code changes.
**Depends on**: Phase 2
**Requirements**: API-01, API-02, API-03, API-04, API-05, API-06, API-07, API-08, EVAL-01, EVAL-07, EVAL-08, EVAL-09, EVAL-10, EVAL-11, EVAL-12, EVAL-13, EVAL-14, EVAL-17, EVAL-18
**Success Criteria** (what must be TRUE):
  1. `GET /methods` returns `ncc` with a complete config JSON Schema, with zero method names
     hardcoded anywhere in the API layer — proven by grepping the API package for method
     names and finding none.
  2. A search POSTed to the API is retrievable from the store with its config, provenance
     (git SHA, model hash, config hash), latency breakdown, and sub-threshold candidate
     scores.
  3. A rating with per-match verdicts and a missed count yields correct precision, recall,
     and inferred expected-count from the query layer — with no derived metric stored as a
     column.
  4. A rating submitted without touching the count fields stores `null`, not `0`, and
     contributes to neither precision nor recall aggregates — proven by a test that submits
     a bare thumbs-up and asserts the aggregates ignore it.
  5. `GET /stats` returns a scoreboard carrying `n` and a Wilson confidence interval
     alongside every rate.
  6. An empty result and a method error are recorded as distinct outcomes, neither of them
     as a zero-precision run.
**Plans**: 2 plans

Plans:
- [ ] 03-01: SQLite store — schema with nullable human-count columns and no defaults,
      migrations, raw-judgment tables, derived-metric views with NULL propagation, Wilson
      interval, sub-threshold candidate persistence
- [ ] 03-02: FastAPI app — `/methods` `/search` `/images` `/ratings` `/stats`, lifespan ONNX
      session loading, typed structured errors, provenance and latency-breakdown capture

### Phase 4: Web UI
**Goal**: A person can run the whole loop — pick a method and its config, draw a box, see
results and diagnostics, rate the run — without touching a terminal.
**Depends on**: Phase 3
**Requirements**: UI-01, UI-02, UI-03, UI-04, UI-05, UI-06, UI-07, UI-08, EVAL-16
**Success Criteria** (what must be TRUE):
  1. A person can open the app, pick a method, draw a box, see overlaid results, toggle the
     diagnostics overlay, and submit a rating that appears in `/stats` — without touching a
     terminal.
  2. The method and its config are locked in *before* the box can be drawn, so every rating
     is attributable to an exact method + config.
  3. The config form is generated entirely from the method's JSON Schema — adding a method
     requires zero frontend changes, proven by the form rendering for a method the frontend
     has never heard of.
  4. Count fields render empty, never prepopulated with `0`; the "all correct" and "none
     missed" buttons write an explicit `0`; per-match verdicts require an explicit confirm
     before they count as assessed.
  5. Boxes drawn on a zoomed and panned canvas land on the correct image pixels on a
     high-DPI display.
  6. The duplicate/fragment convention (two boxes on one instance = 1 TP + 1 FP) is visible
     in the rating UI.
**Plans**: 2 plans

Plans:
- [ ] 04-01: Canvas shell — image loading, zoom/pan, box drawing with correct
      DPR/zoom coordinate transforms, exploration-mode + method selector, schema-driven
      config form
- [ ] 04-02: Result and diagnostics overlays, tiered rating widget with the empty-count
      discipline, stats dashboard with n and confidence intervals

### Phase 5: Method 2
**Goal**: Recover *many* geometric models rather than one — Lowe's multi-object pipeline
with the ratio test deliberately disabled, because the thing it suppresses is exactly what
we are hunting.
**Depends on**: Phase 4
**Requirements**: METHOD-02, METHOD-03, METHOD-04, METHOD-04a, METHOD-04b, METHOD-04c
**Success Criteria** (what must be TRUE):
  1. On an image with 6+ instances of a textured object, `sparse-geo` returns multiple
     distinct geometric models, not one.
  2. Hough peaks are visible in the diagnostics overlay in the UI.
  3. Both the classical and SuperPoint backends run through the same code path and are
     switched by config alone.
  4. A test proves the standard Lowe ratio test would have suppressed the repeated instances
     that the k+1 variant keeps.
  5. On a low-texture crop the method emits its low-keypoint diagnostic instead of an empty
     result.
  6. All three voting modes run, and `single-4dof` is rejected with a clear error for the
     SuperPoint backend, whose keypoints carry no scale or orientation.
  7. Re-running the same search with the same seed produces byte-identical results despite
     RANSAC.
**Plans**: 2 plans

Plans:
- [ ] 05-01: Classical backend (SIFT/AKAZE/ORB), many-to-many top-k matching with the
      standard ratio test disabled and optional k+1 ratio, Hough voting with soft binning
      and the three voting modes, per-peak RANSAC with degeneracy rejection, exemplar
      self-match labelling, low-keypoint diagnostic
- [ ] 05-02: SuperPoint ONNX backend behind the same interface (gated through
      `library-review`), sequential-RANSAC decomposition alternative, `docs/methods/sparse-geo.md`,
      sample runs

### Phase 6: Method 3
**Goal**: The general-purpose default for "same object, moderate appearance variation" —
and the DINOv2 inferencer that Phase 7 will reuse rather than duplicate.
**Depends on**: Phase 4
**Requirements**: METHOD-05
**Success Criteria** (what must be TRUE):
  1. `dino-dense` finds instances that differ in pose or lighting from the exemplar, where
     `ncc` fails — demonstrated on a demo image where the two methods disagree.
  2. The similarity heatmap renders in the UI diagnostics overlay.
  3. All three calibration strategies produce different, inspectable thresholds on the same
     image.
  4. The DINOv2 inferencer exposes dense patch tokens (not just the CLS token), strips any
     register tokens before reshaping to a spatial grid, and documents its exact
     normalization, resize policy, layout, and output decoding in the inferencer docstring
     and `docs/methods/dino-dense.md`.
  5. `pixi run fetch-models` obtains the DINOv2 ONNX reproducibly, with the export scripted
     rather than manual.
**Plans**: 2 plans

Plans:
- [ ] 06-01: DINOv2 ONNX acquisition/export (gated through `library-review`),
      `DINOv2Inferencer` with dense-token output and documented pre/post-processing,
      `fetch-models` integration
- [ ] 06-02: `dino_dense.py` — prototype cosine similarity, high-res scene inference with
      bilinear upsampling, threshold, connected components, diagnostics, docs, sample runs

### Phase 7: Method 5
**Goal**: Boxes that hug object boundaries, built from independently callable proposal and
embedding stages — because Milestone 2 depends on calling exactly those two stages directly.
**Depends on**: Phase 6
**Requirements**: METHOD-06
**Success Criteria** (what must be TRUE):
  1. `propose-retrieve` returns boxes tightly aligned to object boundaries.
  2. The proposal stage and the embedding stage are each callable independently — verified by
     a test that calls them directly rather than through `search()`.
  3. The embedding stage uses the *same* `DINOv2Inferencer` instance contract as Method 3 —
     one model download, one preprocessing contract, proven by a test asserting no second
     DINOv2 model file is fetched.
  4. The proposal set renders in the UI diagnostics overlay.
  5. The proposal backend is switchable between FastSAM and MobileSAM by config alone, and
     any licence constraint on the chosen weights is recorded in `LICENSES.md`.
**Plans**: 2 plans

Plans:
- [ ] 07-01: FastSAM/MobileSAM ONNX proposal stage (gated through `library-review`) as an
      independently callable unit, with documented output decoding and `fetch-models`
      integration
- [ ] 07-02: `propose_retrieve.py` — region embedding stage, cosine NN retrieval with
      threshold and NMS, diagnostics, docs, sample runs

### Phase 8: Evaluation & docs
**Goal**: Turn "which method is better" into a number, with the crossover the research
predicts visible rather than hidden — and leave the repo readable to someone opening it
cold.
**Depends on**: Phase 5, Phase 6, Phase 7
**Requirements**: EVAL-02, EVAL-04, EVAL-05, EVAL-06, EVAL-15, DOC-03, DOC-04, DOC-05, DOC-06
**Success Criteria** (what must be TRUE):
  1. The benchmark produces a table of precision, recall, F1, AP, and latency for all four
     methods across all demo images.
  2. The README shows committed sample runs for every method, side by side.
  3. The paired-comparison mode runs one box through all four methods in a single request
     and records a win/loss/tie result.
  4. A Bradley-Terry (or Elo) ranking is fitted over the paired comparisons and degrades
     gracefully when one method never loses.
  5. The benchmark demonstrates the expected NCC-vs-sparse-geo crossover on small
     near-identical instances rather than hiding it.
  6. `docs/ROBUSTNESS-BACKLOG.md` aggregates every method's backlog, and `docs/MILESTONE-2.md`
     specifies the marker-conditioned proposal feature and the Milestone 1 components it
     reuses.
**Plans**: 2 plans

Plans:
- [ ] 08-01: Ground-truth labels for the demo set, benchmark runner (Hydra-driven CLI),
      paired-comparison mode, Bradley-Terry ranking, metrics module
- [ ] 08-02: Committed charts and tables, README with side-by-side sample runs, per-method
      doc pages, `docs/ROBUSTNESS-BACKLOG.md`, `docs/MILESTONE-2.md`

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8

Phases 5, 6, 7 are mutually independent given Phase 4, except that **7 depends on 6** for
the shared DINOv2 inferencer. Valid parallel schedule: 5 ∥ 6, then 7, then 8.

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Foundation | 0/2 | Not started | - |
| 2. Method 1 + shared primitives | 0/2 | Not started | - |
| 3. Backend API | 0/2 | Not started | - |
| 4. Web UI | 0/2 | Not started | - |
| 5. Method 2 | 0/2 | Not started | - |
| 6. Method 3 | 0/2 | Not started | - |
| 7. Method 5 | 0/2 | Not started | - |
| 8. Evaluation & docs | 0/2 | Not started | - |
