---
phase: 09-marker-conditioned
plan: 02
subsystem: frontend + samples + docs
status: complete
tags: [milestone-2, explorations, ui, schema-driven, marker-overlay, sample-runs, docs]
requires:
  - api: GET /explorations, /search exploration routing (09-01)
  - explorations/marker_conditioned.py (run, MarkerConditionedConfig, backend kwarg)
  - schemas/search.py (Diagnostics: markers/reference-points/directions)
  - synthetic/generator.py (synthesize_markers, MARKER_DEMO_SPECS, save_marker_image)
  - frontend/js/form.js (buildForm), overlay.js, main.js
provides:
  - frontend/js/explorations.js (name-free exploration logic: enum injection, wrapper detection)
  - second schema-driven UI exploration mode (populated from GET /explorations)
  - presence-driven marker overlay (marker boxes, direction arrows, chosen-proposal connectors)
  - pixi run markers (assets/demo/markers/ + .markers.json sidecars)
  - docs/samples/marker-conditioned/ (committed, byte-identical sample runs)
  - docs/explorations/marker-conditioned.md
affects:
  - frontend/index.html (JS-populated #exploration + #method-control)
  - frontend/css/app.css (.control[hidden])
  - src/object_search/cli.py (markers command, render-samples marker gallery)
  - src/object_search/samples.py (render_marker_samples, exploration-registry loop)
  - docs/MILESTONE-2.md (specified -> built), README.md, docs/ROBUSTNESS-BACKLOG.md
tech-stack:
  added: []
  patterns: [schema-driven-form, structural-not-name-dispatch, method-ref-enum-injection, presence-driven-overlay, registry-iterating-renderer, stub-backend-byte-identical-test]
key-files:
  created:
    - frontend/js/explorations.js
    - tests/test_marker_samples.py
    - docs/explorations/marker-conditioned.md
    - assets/demo/markers/{arrows,arrows-with-targets,dots}.png (+ .markers.json)
    - docs/samples/marker-conditioned/{arrows,arrows-with-targets,dots}.png (+ index.md)
  modified:
    - frontend/js/{api,main,overlay}.js
    - frontend/index.html
    - frontend/css/app.css
    - src/object_search/cli.py
    - src/object_search/samples.py
    - pixi.toml
    - assets/demo/LICENSES.md
    - docs/MILESTONE-2.md
    - README.md
    - docs/ROBUSTNESS-BACKLOG.md
decisions:
  - "The config form is rebuilt from the selected exploration's JSON schema; the same-image (method-wrapper) exploration is detected structurally (a method-ref field + a nested config object), never by name, so no exploration name is used for dispatch."
  - "marker_method renders as a <select> by injecting the live GET /methods list as the field's enum in the frontend (data, not a literal) — no method name in frontend/, still grep-clean."
  - "The marker sample gallery uses ncc (model-free) as the marker finder and the CPU FastSAM provider for reproducibility; tests inject a deterministic stub backend so byte-identity is proven model-free."
metrics:
  duration_min: 95
  completed: 2026-07-25
  tasks: 3
  files: 22
  tests: "481 passed, 5 skipped, 93% coverage"
---

# Phase 9 Plan 02: Marker UI Mode, Demo Assets, Sample Runs, Docs Summary

Completes Milestone 2 by hosting the marker-conditioned exploration as a second, fully schema-driven UI mode, committing the synthetic marker demo assets and byte-identical sample runs, and flipping the docs from "specified" to "built".

## What shipped

**Task 1 — schema-driven second UI mode + marker overlay.** `#exploration` is populated from `GET /explorations`; selecting an exploration rebuilds the config form from *that exploration's* JSON schema (reusing `form.js buildForm`). New `frontend/js/explorations.js` holds the name-free logic: injecting the live method list as the `enum` of any method-reference field (so `marker_method` is a real `<select>`), and detecting the same-image method-wrapper by structure (a method-ref field plus a nested `config` object) so its method selector + per-method config form are preserved. `overlay.js` gained a presence-driven marker layer: marker boxes, a fixed-length pointing arrow from each reference point, and a dashed connector to the chosen proposal. No exploration or method name literal appears in `frontend/` (enforced by the existing `test_no_method_name_appears_in_frontend`; `frontend/js/*.js` is clean of `marker-conditioned`).

**Task 2 — marker demo assets + sample gallery.** `pixi run markers` writes `MARKER_DEMO_SPECS` into `assets/demo/markers/` with exact `.markers.json` GT sidecars (PNGs a few KB each). `samples.py` gained `render_marker_samples`, which iterates the **exploration registry** and renders the marker-conditioned exploration under `docs/samples/marker-conditioned/` (marker boxes, arrows, chosen proposals, connectors). A model-free stub-backend test asserts two renders are byte-identical; the committed gallery is rendered by `render-samples` with the CPU FastSAM provider (reproducible). `LICENSES.md` accounts for the marker images.

**Task 3 — docs.** New `docs/explorations/marker-conditioned.md` mirrors the module's numbered steps with explicit pre/post-processing (both orientation paths + the scoring formula), a schema-derived config reference, failure modes, and a ROBUSTNESS BACKLOG. `docs/MILESTONE-2.md` flipped to **built** with the seam table in past tense pointing at shipped files and a "what actually shipped vs the spec" note. README gained a Milestone 2 subsection with a marker sample. A marker section was appended to `docs/ROBUSTNESS-BACKLOG.md`.

## Verification (real output)

- `ruff check` → **All checks passed!**
- `ruff format --check` → **115 files already formatted**
- `mypy src/` (strict) → **Success: no issues found in 64 source files**
- `pytest` → **481 passed, 5 skipped, 93.01% coverage** (floor 80%)
- `GET /explorations` → `['marker-conditioned', 'same-image-search']`; marker schema exposes `marker_method` + the four `w_*` weights + `size_prior_frac`/`max_markers`; same-image schema is `{method, config}`.
- Server smoke: `/` → 307, `/app/` → 200, `/explorations` → 200.
- Byte-identical: re-running `pixi run markers` and re-rendering the marker gallery both produce **no git diff**.

## Deviations from Plan

**1. [Environment caveat — not code] FastSAM inference fails under the CoreML EP on this host.** A server-side marker `POST /search` persists as `outcome=error` (`CoreMLExecutionProvider ... Error in building plan`). This is pre-existing infrastructure behavior — ONNX sessions default to all available providers (`lifespan.py:67`, `default_backend(providers=None)`), and CoreML is broken for FastSAM here — affecting every FastSAM path, not introduced by this plan (UI/docs/assets). The **CPU provider works** and is what the committed gallery uses. Not auto-fixed: changing global provider selection is shared inference infrastructure outside this plan's scope and would touch the determinism guarantees of every learned method (scope boundary / Rule 4). Logged in `.planning/phases/09-marker-conditioned/deferred-items.md` with the CPU workaround.

**2. [Data/method-fit — documented] The committed `arrows` sample resolves only the exemplar-orientation instance.** The synthetic arrows are randomly rotated; `ncc` (the model-free gallery finder) is not rotation-invariant and classical `sparse-geo` abstains on the low-texture arrows (< 20 keypoints). Honestly documented in the exploration doc, the MILESTONE-2 "what shipped vs the spec" note, and the robustness backlog; the `dots` sample (5 markers resolved) and `arrows-with-targets` (arrow → chosen proposal on the pointed-at target) demonstrate the full pipeline.

## Human-verify checkpoint — pending an orchestrator browser run

Per plan instructions this checkpoint was **not blocked**. All auto tasks are implemented, tested, and committed; the server starts and `/explorations` lists both explorations. The interactive marker flow (pick "Marker-conditioned", draw a box around one arrow on a marker demo image, Search, and confirm every instance is found with a direction arrow and the pointed-at object boxed as the chosen proposal) is left for an orchestrator-driven browser run. The orchestrator should run on a host where the CoreML EP builds the FastSAM plan, or start the server with CPU-only providers (see Deviation 1).

## Known Stubs

None. `frontend/js/explorations.js`'s method-label fallback derives from server data (`GET /methods`), not a hardcoded value; no placeholder/empty-data paths were introduced.

## PR

https://github.com/ortizeg/object-search-exploration/pull/18 — "Milestone 2 (2/2): marker UI mode, demo assets, sample runs, docs". Left open (not merged) per instructions.

## Self-Check: PASSED

All created files exist on disk; all three task commits (60dfeab, 556cecc, dbf3771) are in the branch history.
