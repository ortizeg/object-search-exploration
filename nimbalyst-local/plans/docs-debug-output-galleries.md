---
planStatus:
  planId: plan-docs-debug-output-galleries
  title: Debug-output galleries for floor-plans, textured, and real-objects
  status: in-review
  planType: improvement
  priority: medium
  owner: ortizeg
  stakeholders: []
  tags: [docs, mkdocs, eval, samples]
  created: "2026-08-10"
  startDate: "2026-08-10"
  updated: "2026-08-10T17:52:00.000Z"
  progress: 100
---

# Debug-output galleries for floor-plans, textured, and real-objects

## Implementation Progress

- [x] ~~Floor-plans overlay gallery~~ — dropped, see correction section below (no gap existed)
- [x] Generalize `src/object_search/samples.py` to load textured/real-objects images via `eval/labels.py` (`scene_path` + `load_ground_truth`) alongside the existing synthetic `DEMO_SPECS` path
- [x] Add the 6 regime-representative manifest entries (3 textured + 3 real-objects/apple)
- [x] Extend `tests/test_samples.py` for the new manifest entries, incl. byte-identical re-render
- [x] Run `pixi run samples` to regenerate `docs/samples/<method>/` for all 6 methods
- [x] Quality gates: ruff, mypy strict, pytest coverage floor, `pixi run docs-build` strict — full suite: 947 passed, 5 skipped (pre-existing, weight-gated SuperPoint/proposals tests, unrelated to this change), 94.01% coverage (floor 80%)
- [x] Manual sanity check of a handful of new panels per method
- [x] Update plan status to in-review, progress to 100

## Objective

Give a human reviewer enough visual, per-image debug output to find holes in the search
methods and the eval setup — not just aggregate metrics.

**Extend the per-method sample gallery to textured + real-objects.** Today
`docs/samples/<method>/index.md` exists for all 6 registered methods but only covers 4
synthetic regimes (`docs/samples.py`'s `SAMPLE_MANIFEST`, driven by `pixi run samples`).
Extend that same renderer to also cover the `textured` and `real-objects` datasets under
`assets/demo/`.

**Explicitly out of scope** (per user instruction): research datasets (RPINE / FSCD-147 /
FSCD-LVIS / CARPK) and the `basketball/` frames. Do not touch either.

## Dropped: floor-plans overlay gallery — correction, not just deprioritized

The original plan had a "Part 1" wiring `docs/benchmark/floorplans-overlays/` (24 PNGs) into
`docs/eval/floorplans-findings.md` as a committed gallery, on the assumption those PNGs were
already-committed, unlinked artifacts. **That assumption was wrong**, caught before any edit
was made: `docs/benchmark/floorplans-overlays/` is `.gitignore`d (`.gitignore:72`), and
`floorplans-findings.md` itself already explains why — the overlays embed the **licensed
floor-plan images**, so they're a local-only, regenerate-on-demand artifact
(`scripts/build_floorplans_report.py`), never committed. Embedding them would either 404 on
a fresh clone/CI build or require un-gitignoring licensed imagery, neither of which is
correct. The existing "Qualitative overlays (local, gitignored)" section in that file
already documents the regenerate-locally path correctly. There is no gap here — this item is
dropped, not deferred.

## Textured + real-objects sample galleries

### What already exists to build on

`src/object_search/eval/labels.py` already has everything needed to load these two
datasets — this is not new data-loading code, just reuse:

- `textured_image_ids() -> tuple[str, ...]` — 48 ids (16 each of `textured-{plain,varied,
  cluttered}-NN`)
- `real_objects_image_ids() -> tuple[str, ...]` — 30 ids (10 categories × {real-plain,
  real-varied, real-cluttered})
- `scene_path(image_id) -> Path | None` — resolves the image file
- `load_ground_truth(image_id, root=None) -> GroundTruth | None` — loads the `.gt.json`
  sidecar; `GroundTruth.exemplar` returns the designated `ExemplarBox` (same one the
  benchmark queries with), so **no new exemplar-box picking is needed** — reuse the existing
  designated exemplar per image, exactly like `pixi run bench` does.

### Code change

`src/object_search/samples.py`:
- `SAMPLE_MANIFEST` is currently a flat `dict[image_id, ExemplarBox]` sourced only from
  synthetic `DEMO_SPECS`. Generalize `_load_scene`/the manifest-building step so an
  `image_id` can resolve either via `DEMO_SPECS` (synthetic, current behavior, unchanged) or
  via `scene_path()` + `load_ground_truth()` (textured/real-objects, new). Keep the
  synthetic path's determinism guarantee (byte-identical re-render) — textured/real-objects
  images are static files on disk, so they're trivially byte-stable too; no risk there.
  For images with a scale/resize step, use the same read logic `eval/benchmark.py` uses.
- Extend the manifest to include the textured and real-objects image ids (subject to the
  coverage-depth decision below), each resolved to its designated exemplar via
  `GroundTruth.exemplar`.
- `render_samples()` writes one `index.md` per method today, covering the whole manifest in
  one flat table sorted by image_id — keep that structure (no new subdirectories), since
  image ids already have unambiguous prefixes (`textured-*`, `real-*` vs. the synthetic
  `lattice-*`/`scatter-*`/`cluttered-*`) and the existing "one file, one table" reader
  experience is worth preserving over-splitting.
- No change needed to `mkdocs.yml` nav — it already points at `samples/<method>/index.md`.

### Coverage depth — decided: regime-representative

One image per regime, matching the existing synthetic gallery's own philosophy exactly (one
image per regime, not every generated instance). 6 new manifest entries total:

```python
"textured-plain":       textured-plain-01       (assets/demo/textured/)
"textured-varied":      textured-varied-01      (assets/demo/textured/)
"textured-cluttered":   textured-cluttered-01   (assets/demo/textured/)
"real-plain-apple":     real-plain-apple        (assets/demo/real-objects/)
"real-varied-apple":    real-varied-apple       (assets/demo/real-objects/)
"real-cluttered-apple": real-cluttered-apple    (assets/demo/real-objects/)
```

Same category (`apple`) picked across all 3 real-objects regimes so the gallery also lets a
reviewer compare how regime alone changes a method's behavior on one fixed object. All 6
files confirmed present (image + `.gt.json` sidecar) on disk. 6 new images × 6 methods = 36
new panels — in line with the existing gallery's size (currently 4 × 6 = 24 panels).

### Compute

All 6 methods already render the existing 4-image × 6-method gallery locally on CPU via
`pixi run samples` in a few minutes (no GPU needed for `dino-dense`/`owlv2-oneshot` at this
volume — ONNX Runtime CPU EP). Scaling to 6 new images is still a CPU job; vast.ai is not
needed for this task (unlike the batch fine-tuning/sweep work in `.planning/quick/`, which is
a different order of magnitude of compute).

## Verification

- `pixi run samples` regenerates the extended gallery; confirmed byte-identical across two
  full 6-method re-renders (sha256 of every panel + index.md compared).
- `pixi run docs-build` (strict) stays green.
- Ruff (`check` + `format --check`) and MyPy strict clean on `samples.py` +
  `tests/test_samples.py`.
- `tests/test_samples.py` (9/9 passed, incl. 4 new tests for the manifest extension).
- Full suite + 80% coverage floor: running.
- Displayed 2 new panels (real-cluttered-apple, textured-varied-01) for `ncc` to
  eyeball-verify the overlays are sane, not degenerate.

## Incidental finding — mosse's committed gallery was already stale

Re-running `pixi run samples` (needed to add the new entries) also regenerated mosse's
**original 4 synthetic-regime rows**, and their values changed materially: e.g.
`cluttered-distractors` went from 51 matches @ threshold 0.0818 to 18 matches @ 0.3500. This
is **not caused by this change** — confirmed by checking the other 5 methods' original 4 rows,
none of which moved. `mosse`'s algorithm/config has iterated since its gallery panels were
last committed (matches the "MOSSE/ASEF ncc backend spike" work, e.g. the coarse-to-fine
verify + energy-floor change), and nobody re-ran `pixi run samples` after that landed. This
PR's regeneration incidentally fixes that drift as a side effect of the same command run for
the new coverage — flagged here rather than silently bundled.
