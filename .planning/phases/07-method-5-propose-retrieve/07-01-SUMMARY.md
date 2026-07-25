---
phase: 07-method-5-propose-retrieve
plan: 01
subsystem: inference + search (Method 5 proposal stage)
status: complete
tags: [fastsam, onnx, proposals, yolov8-seg, agpl, milestone-2-seam]
requires:
  - ONNXInferencer (Phase 1) input-contract validation
  - the export pixi env + fastsam-s ModelSpec (Phase 1 task 7)
  - BBox / geometry schemas (Phase 1)
provides:
  - FastSAMInferencer (ONNX, YOLOv8-seg decode) -> list[Proposal]
  - Proposal (frozen: box, mask|None, objectness)
  - ProposalBackend protocol + propose() independently callable unit
affects:
  - 07-02 (embed_regions + propose_retrieve.py compose propose() with DINOv2 embedding)
tech-stack:
  added: []            # ultralytics/torch already isolated in the export env since Phase 1
  patterns:
    - "ONNX-only runtime; AGPL exporter quarantined to a separate pixi env"
    - "pure module-level decoding functions so CI gates the arithmetic model-free"
    - "ProposalBackend protocol = the Milestone 2 extension seam (single impl by design)"
key-files:
  created:
    - docs/library-reviews/fastsam.md
    - scripts/export_fastsam.py
    - src/object_search/inference/fastsam.py
    - src/object_search/search/proposals.py
    - tests/test_fastsam.py
    - tests/test_proposals.py
  modified:
    - pixi.toml
    - src/object_search/inference/__init__.py
decisions:
  - "output0's 37-channel dim exports SYMBOLIC under dynamic=True; only output1's 32 is static"
  - "masks decoded behind config.return_masks (boxes are the M1 contract); crop-to-box mandatory"
  - "NMS reimplemented on float arrays in fastsam.py rather than importing search/common/nms.py, to keep the inference layer independent of the search layer"
metrics:
  commits: 4
  files_changed: 8
  lines_added: 1226
  tests_added: 20
  coverage_total: "89.04%"
  completed: 2026-07-25
---

# Phase 7 Plan 1: FastSAM ONNX Proposal Stage Summary

The class-agnostic **proposal stage** for Method 5 (`propose-retrieve`), built as an
**independently callable unit** — `FastSAMInferencer` decodes FastSAM-s (ONNX, YOLOv8-seg) into
`list[Proposal]`, and `propose(image, config)` returns them behind a `ProposalBackend` protocol
that knows nothing about exemplars or retrieval. The AGPL exporter stays isolated in the
export-only pixi env; the runtime is torch-free.

## What was built

- **`FastSAMInferencer(ONNXInferencer)`** — docstring writes out the full decode (transpose
  `output0` → `[anchors, 37]`, split 4 box + 1 conf + 32 coeffs, confidence filter, deterministic
  `(-score, y, x)` NMS, `sigmoid(coeffs @ protos)` masks, **crop each mask to its own box**,
  upsample, undo letterbox). `predict → list[Proposal]`; `propose(image, config)` is the
  `ProposalBackend` entry point. Input spec: 1024² letterbox, fill 114, RGB, `/255`, no mean/std.
  All arithmetic lives in pure module functions so CI gates it model-free.
- **`proposals.py`** — `ProposalBackend` protocol (runtime-checkable, `config: BaseModel`) with
  FastSAM as the single implementation; `propose(image, config, *, backend)` is the Milestone 2
  seam; `default_backend()` builds the FastSAM backend and raises loudly when the weight is absent.
- **Export** — `scripts/export_fastsam.py` (reproducible 1024²/opset-17/dynamic export + graph
  verification); `export-fastsam` task under `[feature.export.tasks]`, runnable only in the
  torch+ultralytics env.
- **AGPL / MobileSAM records** — `docs/library-reviews/fastsam.md` (verdict **Trial**), completing
  the three AGPL records (with the pre-existing `LICENSES.md` row and `ModelSpec.license_note`);
  MobileSAM **Hold** + the documented non-implementation deviation.

## Verification

- **All four gates green** (`lint`, `format-check`, `typecheck` mypy-strict, `test`): **356 passed,
  15 skipped, coverage 89.04%** (≥80 floor). fastsam.py 87%, proposals.py 100%.
- **Model-free decoding** validated with synthetic `[1, 37, 21504]` / `[1, 32, 256, 256]` tensors;
  a **crop-to-box test** asserts mask pixels outside the box are exactly zero.
- **`propose()` callable standalone** — a test calls it directly (never via `search()`) with a stub
  backend on a committed chipset image and asserts boxes + objectness.
- **Runtime env torch-free** — `ultralytics`/`torch` appear only under `[feature.export]`.
- **Export ran and succeeded** — `pixi run -e export export-fastsam` produced `fastsam_s.onnx`
  (45.0 MB, sha256 `fa172569…`); PyTorch confirmed `((1, 37, 21504), (1, 32, 256, 256))`. With the
  weight present, **all real-model tests pass** (predict/propose against the actual model): the
  decode is validated end-to-end, not just model-free.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Export verification wrongly asserted `output0`'s channel dim as static**
- **Found during:** the best-effort export run (Task 1 verify).
- **Issue:** `scripts/export_fastsam.py::_verify_graph` asserted `output0` dim 1 == 37 statically.
  Under `dynamic=True` the exporter marks that dim symbolic (`Concatoutput0_dim_1`); only
  `output1`'s 32 is static. The export itself was correct (PyTorch logged `(1,37,21504)`), but the
  verifier reported a false failure.
- **Fix:** assert only what the graph pins (2 outputs, `output0` rank 3, `output1` rank 4 with
  static 32) and log the symbolic dims; the concrete 37/21504/256 stay pinned by the model-free
  decoding tests. Re-ran: export verifies clean and all real-model tests pass.
- **Files modified:** scripts/export_fastsam.py
- **Commit:** 0cdcfce

### Planned deviations (carried from CONTEXT, not execution surprises)

- **MobileSAM is not implemented as a second backend** — documented in
  `docs/library-reviews/fastsam.md`, the PR body, and the robustness backlog. The ONNX SAM decoder
  is one-prompt-per-call (~1024 calls + a ported auto-mask generator = a phase of work). The
  `ProposalBackend` protocol is the seam that keeps it a later, non-restructuring add.
- **Task 1 was partly pre-satisfied by Phase 1 task 7** — the `export` pixi env, the `fastsam-s`
  `ModelSpec` (with AGPL `license_note`), and the `LICENSES.md` AGPL row already existed. This plan
  added the `export-fastsam` task, the export script, and the library-review (the third AGPL
  record). No conflict; no rework.

## Known Stubs

None. The proposal stage is complete and callable; masks are decoded behind `config.return_masks`
(boxes are the Milestone 1 output contract, per IDEA.md §4 and the scope fence).

## Requirements

`METHOD-06` is **not** ticked — this plan delivers only the proposal stage; the embedding and
retrieval stages that complete METHOD-06 land in 07-02.

## Self-Check: PASSED

- Files exist: docs/library-reviews/fastsam.md, scripts/export_fastsam.py,
  src/object_search/inference/fastsam.py, src/object_search/search/proposals.py,
  tests/test_fastsam.py, tests/test_proposals.py — all present.
- Commits exist: 460ee62, d879e66, 06815f4, 0cdcfce — all in `git log`.
