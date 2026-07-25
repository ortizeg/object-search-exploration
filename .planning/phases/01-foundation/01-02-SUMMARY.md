---
phase: 01-foundation
plan: 02
subsystem: infrastructure
tags: [pydantic, onnx, onnxruntime, typer, registry, synthetic-data, opencv, licensing]
requires:
  - phase: 01-01
    provides: pixi env, quality gates, Loguru setup, CI, provenance helpers
provides:
  - frozen Pydantic schema set for every inter-layer contract (geometry, search, records)
  - SearchMethod decorator registry with JSON-Schema export (method_schemas)
  - ONNXInferencer with init-time input dtype/shape validation (ONNXContractError)
  - synthetic generator with exact ground truth and byte-identical determinism
  - chip-insertion benchmark set (10 images, non-overlapping, achieved-count ground truth)
  - fetch-models framework + populated MODEL_REGISTRY (dinov2/superpoint/fastsam)
  - Typer CLI (fetch-models, synth, chipset, render-samples, benchmark)
  - committed demo assets + LICENSES.md provenance
affects:
  - Phase 2 adds a method as one file + one import against this registry
  - Phases 5/6/7 build inferencers on ONNXInferencer and fetch weights via fetch-models
  - Phase 3 persists RunRecord/Rating; Phase 8 consumes the chip benchmark
tech-stack:
  added:
    onnx: "1.22.0 (dev feature — build tiny test models)"
    ultralytics: "export feature only (AGPL-3.0, torch-bearing)"
    typer: ">=0.12"
  patterns:
    - "PEP 695 generics for BaseInferencer[OutT] / ONNXInferencer[OutT] / PostProcessor"
    - "Typed ONNXInputSpec IS the preprocessing contract (structural, not prose)"
    - "Registry is the only indirection; one import per method in search/__init__.py"
    - "Derived values are properties, never stored fields, across every schema"
key-files:
  created:
    - src/object_search/search/registry.py
    - src/object_search/search/__init__.py
    - src/object_search/inference/base.py
    - src/object_search/inference/onnx_inferencer.py
    - src/object_search/inference/models.py
    - src/object_search/synthetic/generator.py
    - src/object_search/synthetic/chipset.py
    - src/object_search/cli.py
    - assets/demo/LICENSES.md
    - tests/test_registry.py
    - tests/test_onnx_inferencer.py
    - tests/test_synthetic.py
    - tests/test_chipset.py
    - tests/test_models.py
    - tests/test_cli.py
  modified:
    - pixi.toml
    - pixi.lock
    - pyproject.toml
key-decisions:
  - "ONNXInferencer validates INPUT contract only; output extents are symbolic on all three real models"
  - "Added an `export` pixi environment so torch/ultralytics never enter the runtime env"
  - "onnx added to the dev feature so the INFRA-09 test builds a real tiny model"
  - "chip benchmark records the ACHIEVED instance count, never the requested N"
requirements-completed: [INFRA-02, INFRA-08, INFRA-09, INFRA-10, INFRA-11, EVAL-03, EVAL-19, DOC-01]
status: complete
---

# Phase 1 Plan 2: Contracts, Registry, ONNXInferencer, Synthetic + Chip Data, Demo Assets Summary

Built every shared contract the rest of the project stands on — the frozen Pydantic schema
set, the decorator-based `SearchMethod` registry with JSON-Schema export, an `ONNXInferencer`
that rejects a mismatched model at construction, the `fetch-models` framework with all three
runtime-verified model specs, a deterministic synthetic generator with exact ground truth, the
EVAL-19 chip-insertion benchmark, and the committed demo asset set with full licence provenance.

## Accomplishments

- **Schemas (INFRA-08, tasks 1–3, already committed on this branch):** frozen geometry, search,
  and record models; nullable-count discipline (`wrong_count`/`missed_count` default `None`).
- **Registry (INFRA-10):** `register_method` decorator (duplicate name raises), `get_method`/
  `list_methods`/`method_schemas`; no method name hardcoded. `search/__init__.py` is the
  one-import-per-method seam for Phase 2.
- **ONNXInferencer (INFRA-09):** `ONNXInputSpec` as the typed preprocessing contract; init-time
  dtype + shape validation raising `ONNXContractError` (every mismatching dim reported) before
  any image is processed; output extents never asserted; `model_sha256` for provenance; PEP 695
  generic over the output type. Ported from the sibling design with no dependency on it.
- **Synthetic generator (EVAL-03):** single-`default_rng` determinism (byte-identical per seed),
  ground truth = AABB of the drawn rotated shape, lattice + scatter modes, distractors excluded,
  `DEMO_SPECS`, PNG + `.gt.json` sidecars.
- **Chip benchmark (EVAL-19):** 10 canvas sizes 320×240→6000×4000, one textured chip each pasted
  N∈{5,10,15} times at strictly non-overlapping positions on white; pairwise IoU exactly 0;
  achieved count recorded; deterministic; all PNGs < 2 MB (committed, none regenerate-only).
- **fetch-models + CLI (INFRA-11):** `MODEL_REGISTRY` populated with the three verified specs and
  their licences; `.part`-then-rename downloads with a sha256 gate; Typer CLI with loud
  `render-samples`/`benchmark` placeholders (exit non-zero).
- **Demo assets + LICENSES.md (DOC-01):** synthetic + chip sets + 3 downscaled basketball frames
  (source paths recorded, flagged non-redistributable), and a licence accounting for every file
  plus the model-weight terms.

## Deviations from Plan

### Auto-added / structural (Rules 2–3)

**1. [Rule 3 - Blocking] Added `onnx` to the dev feature.**
- **Found during:** Task 5. The INFRA-09 acceptance test must build a real tiny ONNX model; the
  `onnx` package was not installed (only `onnxruntime`). Added `onnx>=1.22,<2` to
  `[feature.dev.dependencies]`. `onnx` is the official first-party package — legitimate.

**2. [Rule 2 - Structural] Added the `export` pixi feature/environment.**
- FastSAM's ONNX export needs Ultralytics (torch, AGPL-3.0). Quarantined in an `export`
  environment so the default runtime env stays torch-free, making the "ONNX Runtime for every
  learned model" constraint structural. Deviation from plan 01-01's single-environment manifest.
  `ultralytics.*` added to the mypy missing-imports override (absent from the runtime env).

**3. Combined Tasks 7 and 7b into one commit** (sanctioned by the plan) so the chipset module
  existed for the CLI import and mypy at commit time.

### Pre-existing design deviation reaffirmed here

**4. Eleven-field `Provenance` (PITFALLS §6.6).** The record schema (task 3) carries
  python/numpy/cv2/onnxruntime versions, ORT providers, and `pixi_lock_sha256` in addition to the
  four fields the plan named, because a git SHA captures none of what actually moves the numbers
  (OpenCV 4.10 vs 5.0 affine results; CoreML-vs-CPU provider). Built via `Provenance.capture(...)`.

### Blocked / carried forward

**5. INFRA-07 branch protection — NOT applied (unavailable).** Enabling required status checks on
  `main` returns `403 "Upgrade to GitHub Pro or make this repository public"` for this private
  repo. Left **Pending** in REQUIREMENTS.md; CI still runs on PRs. Revisit when the repo is public
  or on a paid plan. This is the one Phase-1 requirement not satisfied.

## Verification (real output)

- `pixi run lint` → All checks passed! · `format-check` → 29 files already formatted ·
  `typecheck` → Success: no issues found in 17 source files · `test` → **135 passed, coverage
  90.34%** (≥80% gate reached).
- ONNX construction-raise: `test_wrong_shape_...` and `test_wrong_dtype_...` pass (raise at
  `__init__`, message naming actual vs expected).
- Synthetic same-seed `np.array_equal` → True; chipset-01 regeneration byte-identical → True;
  all-10 pairwise IoU==0 → True; achieved counts `[5,15,5,15,5,10,10,5,10,5]`.
- `pixi run fetch-models --list` → exit 0, lists all three models.
- `pixi run chipset` → 10 images + sidecars, all < 2 MB.
- `grep -rn "wrong_count: int = 0\|missed_count: int = 0" src/` → nothing.
- **CI on PR #2:** completed **success**.

## Known Stubs

None. `render-samples` and `benchmark` are intentional loud placeholders (exit non-zero, naming
the phase that implements them) — not silent stubs.

## Self-Check: PASSED

- All created source and asset files exist on disk (verified by the passing test suite and the
  committed asset listing).
- All six task commits present: 8a4ad93, 752aff2, ceba56d, 94942ba, e7b7533 (+ tasks 1–3 prior).
