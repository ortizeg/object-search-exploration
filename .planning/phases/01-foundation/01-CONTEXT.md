# Phase 1 Context — Foundation

**Source:** `.planning/IDEA.md` (complete project brief) + user's stated ML-practitioner
standards. No `/gsd-discuss-phase` was run; the brief is detailed enough to be the context.

## Domain

Repository scaffolding and shared contracts for a computer-vision exploration harness. No
search algorithm is implemented in this phase — the phase exists so that Phase 2 onward can
add a method as one new file, and so that a wrong ONNX model fails at load rather than at
first frame.

## Locked Decisions

These are settled. Do not re-litigate them during planning or execution.

1. **Pixi only.** `pixi.toml` with `[workspace]` (not the legacy `[project]` table — pixi
   0.62.x). Python `3.12.*`. Every command is a pixi task.
2. **Proven version baseline, copied from the sibling `basketball-2d-to-3d` lockfile**, which
   resolves cleanly on osx-arm64 today:
   - `python 3.12.12`, `numpy 2.4.2`, `opencv 4.13.0` (conda-forge), `onnxruntime 1.23.2`
   - `onnxruntime` is pinned `==1.23.2` from **PyPI**, not conda-forge. 1.24.1 has macOS
     platform-tag issues with the pixi solver.
   - `opencv` comes from **conda-forge**, never PyPI `opencv-python-headless`.
3. **Platforms:** `["osx-arm64", "linux-64"]`. osx-arm64 is the dev machine; linux-64 is CI.
4. **src-layout**, package name `object_search`, with `py.typed`. `pyproject.toml` is the
   single source of truth for ruff/mypy/pytest config.
5. **Ruff line-length 100, MyPy strict, ≥80% coverage** — all three are hard gates in CI.
6. **Loguru only.** Ruff `T20` makes `print()` a lint failure. A custom lint check also bans
   `import logging`.
7. **`ONNXInferencer` design is ported** from
   `/Users/ortizeg/1Projects/⛹️‍♂️ Next Play/code/basketball-2d-to-3d/src/basketball_2d_to_3d/inference/`
   — init-time dtype/shape validation plus a post-processor strategy. **Port the design, do
   not add a dependency on that project.** The port generalizes the sibling's
   `list[Detection]` return type, because this project's inferencers return dense token
   grids and keypoint sets, not detections.
8. **Registry is a decorator**, one per method. Adding a method touches exactly one new file
   plus one import in `search/__init__.py`.
9. **Weights are gitignored** and arrive only via `pixi run fetch-models`.
10. **Frozen Pydantic v2 models** for every inter-layer contract.

## Canonical References

- `.planning/IDEA.md` §6 (architecture, package layout, the `SearchMethod` protocol), §7
  (requirement IDs), §8 (constraints)
- `.planning/research/STACK.md` — verified version/dependency-split research
- Sibling `ONNXInferencer`: `basketball-2d-to-3d/src/basketball_2d_to_3d/inference/onnx_inferencer.py`
- Sibling working manifest: `basketball-2d-to-3d/pixi.toml`

## Specifics

**Schema set** (all frozen, `model_config = ConfigDict(frozen=True)`):

- `ExemplarBox` — the user's drawn box, `x, y, w, h` in image pixels, plus optional label
- `Match` — a returned instance: box, score, optional `transform` (2×3 affine for Method 2),
  `is_exemplar: bool` (the self-match label from METHOD-04c)
- `Candidate` — a sub-threshold candidate: box, raw score. Distinct from `Match` because
  EVAL-08 requires persisting these separately with the applied threshold.
- `LatencyBreakdown` — `preprocess_ms`, `inference_ms`, `postprocess_ms`, `total_ms`
  (EVAL-11 — a single number is explicitly insufficient)
- `Diagnostics` — method-specific payload; `dict[str, Any]`-shaped at the schema boundary but
  with named optional fields for the things the UI knows how to render (similarity map,
  keypoints, correspondences, hough peaks, proposals)
- `SearchResult` — matches, candidates, threshold applied, latency, diagnostics, outcome
  (`ok` / `empty` / `error` — EVAL-12 needs these distinct), optional error payload
- `SliceMetadata` — EVAL-10: true instance count, scale range, rotation range, clutter level,
  exemplar keypoint count. Every field nullable (exact for synthetic, best-effort otherwise).
- `Provenance` — EVAL-09: git SHA, model hashes, config hash, method version
- `RunRecord`, `Rating` — persisted forms; the `Rating` count fields are `int | None` with
  **no default**, so a missing value is `None` and never `0` (EVAL-17)

**Synthetic generator** (EVAL-03) must emit, from a seed:
- a lattice mode (grid of identical instances, configurable spacing/jitter)
- a scatter mode with scale variation and rotation variation
- distractor objects (similar-but-different shapes) and clutter (background texture)
- exact ground-truth boxes as a sidecar JSON
- byte-identical output for the same seed

**Demo assets** (DOC-01): basketball frames from the sibling project's outputs, synthetic
images from the generator, and generic repeated-instance photos. `assets/demo/LICENSES.md`
records provenance for every file.

## Deferred

- Branch protection with **required status checks** cannot be configured until CI has run
  once and registered its check names. Plan 01-01 pushes CI first, then applies protection.
- Generic permissively-licensed photos (shelf/PCB/parking lot/tiles): if no
  clearly-licensed source can be obtained without a network fetch of unknown provenance,
  the synthetic generator plus basketball frames carry Phase 1, and the gap is recorded in
  `LICENSES.md` and revisited in Phase 8. Ground-truth *labels* for photos are Phase 8
  (EVAL-02) regardless.

## Scope Fence

**In:** scaffold, quality gates, CI, branch protection, schemas, registry, inference base,
`fetch-models` skeleton, synthetic generator, demo assets.

**Out:** any search method (Phase 2+), any actual ONNX model download (the models
themselves land in Phases 6/7 — `fetch-models` in Phase 1 is the framework plus a
verifiable no-op/registry, not the DINOv2 download), the API (Phase 3), the UI (Phase 4).

## Risk Summary

- **numpy 2.x vs onnxruntime 1.23.2** — the sibling lockfile proves this combination
  resolves and runs, so the risk is low, but `pixi install` must actually be executed in
  this phase rather than assumed.
- **MyPy strict against `onnxruntime` and `cv2`** — both ship incomplete or absent stubs.
  Handle with narrowly-scoped `[[tool.mypy.overrides]]` for those modules only, never a
  global `ignore_missing_imports = true`.
- **`disallow_any_explicit`** from the master-skill default conflicts with `Diagnostics`
  needing `dict[str, Any]`. Resolve by keeping the strict setting and using a narrow, named
  type alias with a documented `# type: ignore` at exactly one site, or by typing
  diagnostics as `Mapping[str, object]`. Prefer the latter — it keeps strict mode honest.
