---
phase: 06-method-3-dino-dense
plan: 01
subsystem: inference
tags: [dinov2, onnx, onnxruntime, dense-tokens, vit, patch-grid, cosine-similarity]

# Dependency graph
requires:
  - phase: 01-foundations
    provides: ONNXInferencer base with snap-to-multiple(14) preprocessing + INFRA-09 input validation, MODEL_REGISTRY + fetch-models framework
provides:
  - DINOv2Inferencer producing dense patch tokens with CLS + register tokens stripped and the grid proven non-transposed
  - dense_tokens(image) -> (grid[H//14, W//14, 384], scale_x, scale_y) API for Method 3 (06-02) and Method 5 (Phase 7)
  - load-time token-layout validation (_derive_layout + two-aspect-ratio _probe_layout) that turns a silent spatial shift into an ONNXContractError
  - Adopt library-review verdict for onnx-community/dinov2-small-ONNX + pinned sha256 integrity gate
affects: [dino_dense, propose-retrieve, phase-07, api-session-registry]

# Tech tracking
tech-stack:
  added: [onnx-community/dinov2-small-ONNX (Apache-2.0, pinned revision)]
  patterns:
    - "Output-contract validation at load: subclass probes the model at two swapped non-square aspect ratios and raises on disagreement (extends the base's input-only INFRA-09 discipline to the token layout)"
    - "Derive-don't-hardcode register slice: [1 + n_register:] with n_register = tokens - 1 - gh*gw"
    - "Prove-don't-assume grid orientation with a non-square off-centre fixture; identify the object by cosine to a prototype, robust to register-free ViT artifact tokens"

key-files:
  created:
    - src/object_search/inference/dinov2.py
    - tests/test_dinov2.py
    - docs/library-reviews/dinov2.md
    - .planning/phases/06-method-3-dino-dense/deferred-items.md
  modified:
    - src/object_search/inference/models.py
    - src/object_search/inference/__init__.py
    - assets/demo/LICENSES.md

key-decisions:
  - "n_register is DERIVED from the token count and the slice is written [1 + n_register:], never hardcoded to 1 — a with-registers variant would otherwise silently shift the whole feature map"
  - "The grid orientation is proven with a 448x896 off-centre fixture whose expected peak column (50) exceeds a transposed grid's 31-column range, making it structurally incapable of passing under a transpose"
  - "The object is located by cosine similarity to an orange prototype, NOT L2 deviation from the mean — dinov2-small has no register tokens and its high-norm artifact tokens dominate a deviation peak (verified empirically)"
  - "The first-fetch sha256 was recorded in MODEL_REGISTRY as a hard EVAL-09 integrity gate (deterministic because the HF revision is pinned)"

patterns-established:
  - "Load-time output-layout probe: run cheap swapped-aspect-ratio forward passes at construction to pin model constants (n_register, embed_dim) and raise on any contract violation"
  - "Two-tier inferencer tests: model-free arithmetic/contract tests gate the risky logic in CI; real-model behavioural tests skip-when-absent"

requirements-completed: []  # METHOD-05 is PARTIAL (inferencer half only); it completes in 06-02

# Coverage metadata
coverage:
  - id: D1
    description: "DINOv2Inferencer exposes dense patch tokens with CLS + register tokens stripped and the grid proven non-transposed"
    requirement: "METHOD-05"
    verification:
      - kind: unit
        ref: "tests/test_dinov2.py#test_grid_is_not_transposed_on_a_non_square_off_centre_fixture (real-model, skip-when-absent; ran+passed locally)"
        status: pass
      - kind: unit
        ref: "tests/test_dinov2.py#test_derive_layout_derives_register_count_not_hardcoded_one (model-free, CI)"
        status: pass
    human_judgment: false
  - id: D2
    description: "The token-count arithmetic is asserted at load and raises ONNXContractError on mismatch (INFRA-09 discipline)"
    requirement: "METHOD-05"
    verification:
      - kind: unit
        ref: "tests/test_dinov2.py#test_derive_layout_raises_on_negative_register_count (model-free, CI)"
        status: pass
      - kind: unit
        ref: "tests/test_dinov2.py#test_construction_probes_and_pins_layout (real-model, skip-when-absent)"
        status: pass
    human_judgment: false
  - id: D3
    description: "Input sides snapped to a multiple of 14 and the exact scale factors returned"
    requirement: "METHOD-05"
    verification:
      - kind: unit
        ref: "tests/test_dinov2.py#test_dense_tokens_returns_scale_factors_for_a_non_multiple_scene, test_input_spec_is_snap14_bicubic_rgb_no_crop"
        status: pass
    human_judgment: false
  - id: D4
    description: "DINOv2 gated by a recorded Adopt library-review verdict + pinned model registry entry"
    requirement: "METHOD-05"
    verification:
      - kind: other
        ref: "grep -q Adopt docs/library-reviews/dinov2.md && grep -q dinov2-small src/object_search/inference/models.py"
        status: pass
    human_judgment: false

# Metrics
duration: ~35min
completed: 2026-07-25
status: complete
---

# Phase 6 Plan 1: DINOv2 ONNX Inferencer — Dense Tokens Summary

**`DINOv2Inferencer` produces correctly-oriented dense patch tokens from
`onnx-community/dinov2-small-ONNX`, with snap-to-14, a derived CLS+register slice asserted at
load, and a non-transposed grid proven by a non-square off-centre fixture — the backbone Method 3
(06-02) and Method 5 (Phase 7) reuse.**

## Performance

- **Duration:** ~35 min
- **Tasks:** 2 completed
- **Files created/modified:** 7

## Accomplishments
- **`DINOv2Inferencer`** subclassing `ONNXInferencer`, carrying the exact verified contract in its
  docstring (input `pixel_values` f32 NCHW dynamic, RGB, scale `1/255`, mean `[0.485,0.456,0.406]`,
  std `[0.229,0.224,0.225]`, bicubic, **no centre-crop**; output `last_hidden_state
  [B, floor(H/14)*floor(W/14)+1, 384]`, index 0 = CLS).
- **Three silent-bug guards enforced:** snap-to-14 + returned scale factors; strip CLS+registers
  with a **derived** `[1 + n_register:]` slice asserted at load (`ONNXContractError` on a bad
  count); grid reshaped height-first and **proven non-transposed** by a 448×896 off-centre fixture.
- **`_probe_layout`** runs two swapped non-square aspect ratios at construction to pin
  `n_register`/`embed_dim` and raise on disagreement — output-layout validation layered on the
  base's input-only INFRA-09 check.
- **Adopt verdict** recorded in `docs/library-reviews/dinov2.md`; model fetched and its first-fetch
  **sha256 pinned** in `MODEL_REGISTRY` as a hard EVAL-09 gate.
- **Two-tier tests:** model-free arithmetic/contract tests gate the risky register slice in CI;
  real-model behavioural tests skip when the gitignored weight is absent.

## Task Commits

1. **Task 1: library-review Adopt verdict + pinned model registry entry + LICENSES** — `ee0d4a0` (docs)
2. **Task 2: DINOv2Inferencer — dense tokens with the verified contract** — `0fa9a39` (feat)

## Files Created/Modified
- `src/object_search/inference/dinov2.py` — the inferencer, `_derive_layout`, `_probe_layout`, `dense_tokens`
- `tests/test_dinov2.py` — model-free (CI) + real-model (skip-when-absent) tests, incl. the transpose proof
- `docs/library-reviews/dinov2.md` — Adopt verdict, Apache-2.0 (verified three ways), rejected candidates
- `src/object_search/inference/models.py` — recorded the first-fetch sha256 as a hard integrity gate
- `src/object_search/inference/__init__.py` — export `DINOv2Inferencer`
- `assets/demo/LICENSES.md` — cross-reference the library-review verdict
- `.planning/phases/06-method-3-dino-dense/deferred-items.md` — the out-of-scope API test coupling

## Decisions Made
See `key-decisions` in the frontmatter. The load-bearing one: the object in the transpose test is
located by **cosine to a prototype**, not L2 deviation from the mean — dinov2-small ships no
register tokens, so its high-norm artifact tokens win a deviation-based argmax (empirically the
deviation peak landed at row 29 / col 54, an artifact; the cosine peak landed at row 4 / col 49,
the actual block).

## Deviations from Plan

### Out-of-scope discovery (logged, not fixed)

**1. [Scope boundary] `test_api_app.py` session-registry assertion coupled to "no weights on disk"**
- **Found during:** Task 2, after `pixi run fetch-models --only dinov2-small`.
- **Issue:** `test_lifespan_migrates_store_and_builds_empty_session_registry` asserts
  `app.state.sessions == {}` ("Phase 3 ships no ONNX weights"). The lifespan loads a session for
  every weight present, so once the DINOv2 weight is fetched the assertion fails **locally**.
- **Why not fixed here:** In CI the weight is gitignored/absent, so the test passes there
  (verified: full suite 283 passed / 3 skipped / 90.66% coverage with the weight hidden). The fix
  belongs to **06-02**, which owns `api/` and wires `app.state.sessions["dinov2-small"]`. Touching
  it from 06-01 (inferencer only) would be scope creep.
- **Action:** logged in `deferred-items.md` for 06-02.

**Total deviations:** 0 auto-fixed, 1 out-of-scope item deferred.
**Impact on plan:** none — both tasks completed exactly as written; CI is green.

## Issues Encountered
- Two real-model tests failed on the first run: the transpose test used L2-deviation (hit an
  artifact token) and the snap test used wrong banker's-rounding math. Both fixed by empirically
  probing the real model (cosine-to-prototype localization; corrected 320→322 snap arithmetic).
- The `pixi` shell function needs `$PIXI_EXE`, which the sandboxed non-interactive shell does not
  set; ran the `~/.pixi/bin/pixi` binary directly. No effect on the artifacts.

## User Setup Required
None — the model is fetched by `pixi run fetch-models --only dinov2-small` (gitignored weight).

## Next Phase Readiness
- **06-02 (`dino_dense.py`)** can consume `DINOv2Inferencer.dense_tokens` directly: L2-normalize the
  grid, mean-pool the crop into a prototype, cosine against the grid, upsample, calibrate,
  connected components → boxes. It also owns the `test_api_app.py` session-registry update.
- **METHOD-05 is intentionally NOT ticked** — it completes in 06-02.
- No blockers.

## Self-Check: PASSED

- All created files present on disk (`dinov2.py`, `test_dinov2.py`, `dinov2.md`, `06-01-SUMMARY.md`, `deferred-items.md`).
- Both task commits present in history (`ee0d4a0`, `0fa9a39`).

---
*Phase: 06-method-3-dino-dense*
*Completed: 2026-07-25*
