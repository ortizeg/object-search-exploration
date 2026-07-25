---
phase: 07-method-5-propose-retrieve
plan: 02
subsystem: search (Method 5 — propose-retrieve)
status: complete
tags: [propose-retrieve, dinov2, region-embeddings, cosine-nn, nms, milestone-2-seam, method-06]
requires:
  - FastSAMInferencer + propose() unit (07-01)
  - DINOv2Inferencer + dino_dense._get_inferencer() singleton (Phase 6)
  - common.calibration (gmm) and common.nms (deterministic) offerings
  - BBox / ExemplarBox / SearchResult / Diagnostics schemas (Phase 1)
provides:
  - embed_regions(image, boxes, config) -> (N, D) L2-normalized region embeddings (the 2nd unit)
  - propose_retrieve.search() composing propose() + embed_regions() + cosine NN + threshold + NMS
  - the "propose-retrieve" registered method (fourth and final Milestone 1 method)
affects:
  - Milestone 2 (marker-conditioned proposals reuse propose() and embed_regions() directly)
  - Phase 8 (benchmark runner sweeps this method alongside the other three)
tech-stack:
  added: []            # no new dependency — plain NumPy matmul, no FAISS (CONTEXT decision 7)
  patterns:
    - "compose two independently callable units; search() does nothing they cannot do alone"
    - "reuse Method 3's DINOv2 singleton — one model, one preprocessing contract, no 2nd fetch"
    - "cosine NN = plain NumPy matmul on L2-normalized (N, D); FAISS deferred to corpus scale"
    - "post-retrieval NMS collapses SAM over-segmentation; proposal count in diagnostics"
    - "latency attributes proposal vs embedding time separately (EVAL-11 finding)"
key-files:
  created:
    - src/object_search/search/propose_retrieve.py
    - docs/methods/propose-retrieve.md
    - tests/test_propose_retrieve.py
    - docs/samples/propose-retrieve/{cluttered-distractors,lattice-plain,lattice-touching,scatter-scaled}.png
    - docs/samples/propose-retrieve/index.md
  modified:
    - src/object_search/search/__init__.py
    - docs/ROBUSTNESS-BACKLOG.md
    - .planning/REQUIREMENTS.md
    - .planning/ROADMAP.md
decisions:
  - "embed_regions reuses dino_dense._get_inferencer() directly (one DINOv2, no second loader); the single intentional cross-method reference, documented as such"
  - "retrieval threshold via gmm when unset (absolute cosine cuts do not transfer across images); fixed retrieval_threshold passes through"
  - "inference_ms carries proposal+embedding sum; diagnostics.metrics reports proposal_ms/embedding_ms separately so the 'proposal dominates' finding is legible"
  - "boundary-alignment made a NUMBER: mean IoU vs exact chipset GT asserted > 0.70 (measures ~0.99)"
metrics:
  commits: 2
  tests_added: 18
  coverage_total: "92.87%"
  completed: 2026-07-25
---

# Phase 7 Plan 2: propose-retrieve — DINOv2 region embeddings, NN retrieval Summary

Method 5 `propose-retrieve` — **FastSAM class-agnostic proposals ranked by DINOv2 region-embedding
cosine nearest-neighbour**, built from two independently callable units so Milestone 2 can add an
exploration rather than fork the app. This completes Phase 7 and the fourth/final Milestone 1 method
(METHOD-06). Its selling point over `dino-dense`'s blobby components is **boundary alignment**, made
a measured number: mean IoU vs exact chipset ground truth = **0.99** (asserted `> 0.70`).

## What was built

- **`embed_regions(image, boxes, config) -> NDArray`** — the second independently callable unit
  (the Milestone 2 seam's embedding half). Crops each box, embeds via the **same DINOv2 backbone as
  Method 3** (`dino_dense._get_inferencer()` — one model, no second fetch), mean-pools the patch
  tokens, and L2-normalizes: one `(D,)` row per box. Knows nothing about proposals or exemplars; a
  sub-patch crop is up-sized to 14×14 so a small proposal still yields a token; an injected
  `inferencer=` makes the unit testable model-free.
- **`search()`** — numbered steps composing the two units: `propose()` → `embed_regions(proposals)`
  → `embed_regions([exemplar])` → cosine NN (**plain NumPy matmul, no FAISS**) → gmm/fixed threshold
  → **post-retrieval NMS** (`nms_iou`, collapses SAM over-segmentation; proposal count and
  `collapsed_by_nms` in diagnostics) → matches (with the exemplar's own region labelled) + EVAL-08
  sub-threshold candidates. `Diagnostics.proposals` carries the full proposal set for the UI
  overlay; `LatencyBreakdown` sums the two forward passes while `metrics` reports `proposal_ms` and
  `embedding_ms` **separately** (EVAL-11).
- **`ProposeRetrieveConfig`** (frozen, schema-driven form) — `proposal_backend`, `proposal_conf`,
  `retrieval_threshold`, `nms_iou`, `max_candidates`, `seed`.
- **Registration** — `@register_method("propose-retrieve")` + one import in `search/__init__.py`;
  the sample renderer picks it up with no code change.
- **Docs** — `docs/methods/propose-retrieve.md` (numbered steps, explicit FastSAM + DINOv2
  pre/post-processing, config reference, the AGPL sharing constraint, the MobileSAM
  non-implementation deviation, failure modes, ROBUSTNESS BACKLOG); the backlog section mirrored to
  `docs/ROBUSTNESS-BACKLOG.md`; `docs/samples/propose-retrieve/` regenerated.

## Verification (paste, real output)

- **Gates** — Ruff, Ruff-format, MyPy strict, pytest all green. **383 passed, 5 skipped**;
  coverage **92.87%** (≥80% floor; `propose_retrieve.py` at 98%).
- **Seam** — `embed_regions` and `propose` each called directly (stub inferencer / stub backend),
  each works standalone. ✅
- **Reuse / no second DINOv2** — spy-singleton test proves `embed_regions` routes through Method 3's
  DINOv2 singleton; registry holds exactly one `dinov2-*` key. ✅
- **IoU proof** — `mean_iou 0.9863`, 5/5 matches on the 1600×1200 chipset (asserted `> 0.70`; skips
  without weights). ✅
- **No FAISS** — import-scan test guards it. ✅
- **Latency split** — `metrics proposal_ms/embedding_ms: 201.09 / 44.70` (proposal dominates —
  EVAL-11). ✅
- **Served** — `pixi run serve`; `GET /methods` → HTTP 200 with `propose-retrieve` listed; real
  `POST /search` on `chipset-05` → HTTP 200, `outcome ok`, 5 matches, 12 proposals in
  `diagnostics.proposals`, latency 222 ms proposal vs 46 ms embedding.

## Deviations from Plan

**None affecting scope.** The plan's Task 2 IoU-proof test was authored alongside the Task 1 tests
(one test file), so it landed in the first commit rather than the second — a packaging choice, not a
scope change; both tasks' acceptance criteria are met.

Two **pre-existing, documented** deviations are carried forward and recorded (CLAUDE.md / CONTEXT):

- **MobileSAM is not a working second backend.** The ONNX SAM decoder takes one prompt per call, so
  "everything mode" is ~1024 sequential calls plus a ported `SamAutomaticMaskGenerator` — a phase of
  work, not a config swap. FastSAM ships as the single Milestone 1 backend behind the
  `ProposalBackend` protocol, keeping the seam open. Recorded in the method doc, PR body, and
  robustness backlog.
- **AGPL-3.0 (FastSAM).** The exported `.onnx` embeds the licence; publishing the repo or
  network-exposing the app fires §13. Recorded in `LICENSES.md`, the `fastsam-s` `ModelSpec`, and the
  method doc.

No Rule-1/2/3 auto-fixes were needed; the code passed the gates and the real-model tests on first
green.

## Known Stubs

None. Both stages run real inference; the method degrades to `outcome=error` (`model_unavailable`)
only when a gitignored weight is absent, which is honest reporting, not a stub.

## One cross-method reference (called out for the reviewer)

`propose_retrieve` imports `dino_dense._get_inferencer` — the one deliberate cross-method reference
in the repo. It exists because the DINOv2 backbone is genuinely shared state (CONTEXT decision 6:
one model, one preprocessing contract), not a per-method concern; there is structurally no second
DINOv2 loader, and a test asserts the reuse. Documented at the top of the module.

## Pending

The **visual UI check is deferred to an orchestrator browser run** (plan checkpoint, gate=blocking,
not blocked per execution instruction): pick `propose-retrieve`, draw a box on one object, run
Search, confirm the returned boxes hug object boundaries **tighter than `dino-dense`**, and toggle
the diagnostics overlay to confirm the **proposal set renders**. The HTTP `/search` above already
confirms the proposal set is carried in the response payload the overlay reads.

## Self-Check: PASSED

- `src/object_search/search/propose_retrieve.py` — FOUND
- `docs/methods/propose-retrieve.md` — FOUND
- `docs/samples/propose-retrieve/index.md` — FOUND
- commits `b542d3a` (feat) and `a632d8f` (docs) — present on `phase-07/propose-retrieve`
