---
phase: 02-method-1-shared-primitives
plan: 02
subsystem: search/ncc + samples
tags: [ncc, matchTemplate, template-matching, pyramid, sample-renderer, opencv, determinism]
requires:
  - phase: 02-01
    provides: search/common offerings (peaks, calibration, nms, viz)
  - phase: 01-02
    provides: frozen schemas (ExemplarBox, Match, Candidate, SearchResult, Diagnostics)
  - phase: 01-02
    provides: registry (@register_method), synthetic DEMO_SPECS generator
provides:
  - "Method 1 ncc: self-contained matchTemplate over a scale pyramid, registered as 'ncc'"
  - "textureless-crop guard (outcome=EMPTY) for the measured flat-template trap"
  - "candidates with raw scores + threshold_applied (EVAL-08 offline sweep support)"
  - "registry-driven sample renderer (samples.py) + pixi run samples CLI exiting 0"
  - "committed docs/samples/ncc gallery that regenerates byte-identically"
  - "docs/methods/ncc.md and docs/ROBUSTNESS-BACKLOG.md"
affects:
  - "Phase 3 (API) exercises ncc over HTTP; ncc is CLI-only in this phase"
  - "Phase 5/6/7 methods gain docs/samples/<method>/ galleries with zero renderer changes"
  - "Phase 8 threshold sweep consumes the raw-score candidate log"
tech-stack:
  added: {}
  patterns:
    - "One self-contained method file, numbered steps # 1..# 9 matching docs/methods/ncc.md"
    - "config: BaseModel in the search signature, narrowed via isinstance to match SearchFn"
    - "per-level z-score (median/MAD) de-biases the ~15x cross-level noise floor before pooling"
    - "renderer iterates list_methods(), so a new method self-registers into the gallery"
    - "deterministic sample output: cv2 writers only, no timestamps, latency kept off index.md"
key-files:
  created:
    - src/object_search/search/ncc.py
    - src/object_search/samples.py
    - tests/test_ncc.py
    - tests/test_samples.py
    - docs/methods/ncc.md
    - docs/ROBUSTNESS-BACKLOG.md
    - docs/samples/ncc/index.md
    - docs/samples/ncc/lattice-plain.png
    - docs/samples/ncc/lattice-touching.png
    - docs/samples/ncc/scatter-scaled.png
    - docs/samples/ncc/cluttered-distractors.png
  modified:
    - src/object_search/search/__init__.py
    - src/object_search/cli.py
    - tests/test_cli.py
    - README.md
    - .planning/REQUIREMENTS.md
    - .planning/ROADMAP.md
decisions:
  - "textureless guard fires on crop std < 1e-6 BEFORE matchTemplate; platform-independent"
  - "pyramid rescales the scene and crops the template from the downscaled scene (self-match 1.0)"
  - "peaks extracted on the per-level z-map; raw score carried for threshold + candidate log"
  - "cross-level NMS prioritised by z-score (comparable) not raw score (size-biased)"
  - "index.md omits wall-clock latency so the whole gallery regenerates byte-identically"
  - "sample panels downscaled to <=1440px (deterministic) to stay under the 2MB large-file gate"
requirements: [METHOD-01, METHOD-09, METHOD-10, METHOD-11, METHOD-12, DOC-02]
metrics:
  duration_min: 40
  completed: 2026-07-25
  tasks: 3
  files_created: 11
  tests_total: 190
  coverage_pct: 91.5
status: complete
---

# Phase 2 Plan 02: Method 1 (`ncc`) + sample-run renderer Summary

Normalized cross-correlation template matching as one self-contained, top-to-bottom readable
module over a scale pyramid, plus the registry-driven sample-run renderer every later phase
reuses. Completes Phase 2 (success criteria 1, 3, 5; criteria 2 and 4 landed in plan 02-01).

## What was built

**Task 1 — `ncc.py` (METHOD-01/09/10/12).** A single file whose `search()` reads top to
bottom as nine numbered steps mirroring `docs/methods/ncc.md`. Every measured pitfall from the
research is handled where the reader can see it: the textureless-crop guard (std computed
first, `outcome=EMPTY` with a note before `matchTemplate` is ever called), the pyramid that
rescales the *scene* and crops the template from the downscaled scene (self-match stays 1.0),
full-scene correlation only, per-level z-scoring to de-bias the ~15× cross-level noise floor,
top-left box anchoring with no centre offset, and NaN→`-inf` before peak finding. It returns
sub-threshold `candidates` with raw scores plus `threshold_applied` (EVAL-08), labels the
exemplar's own region `is_exemplar=True`, and never short-circuits to a single best match
(METHOD-12 — a lattice of N returns N). Registered via one side-effect import in
`search/__init__.py` (INFRA-10).

**Task 2 — sample renderer (DOC-02).** `samples.py` drives a committed `SAMPLE_MANIFEST` over
the four synthetic demo specs and iterates the method registry, so a method added in a later
phase gains a full `docs/samples/<method>/` gallery with no change to the renderer. Output is
deterministic (cv2 writers, no timestamps, deterministic downscaling); `index.md` carries only
reproducible columns so the whole directory regenerates byte-for-byte. `pixi run samples` /
`cli render-samples` now exit 0 with `--method` and `--out` options, replacing the Phase-1
non-zero placeholder.

**Task 3 — docs (METHOD-11).** `docs/methods/ncc.md` headings mirror the module steps
one-for-one, with exact pre/post-processing sections, a config reference generated from the
JSON Schema (cannot drift), failure modes, embedded sample runs, and a `ROBUSTNESS BACKLOG`
mirrored verbatim into the new `docs/ROBUSTNESS-BACKLOG.md`. README now links the real gallery.

## Success criteria verified (real output)

1. **Lattice returns every instance, no duplicates.** `test_lattice_returns_every_instance_without_duplicates`
   asserts `len(matches) == len(gt)` and pairwise IoU `< nms_iou`; the committed `lattice-plain`
   sample reports **12/12**. (`lattice-touching` is solid same-colour rectangles — a textureless
   crop — so NCC honestly abstains via the step-1 guard; documented.)
3. **`docs/samples/ncc/` regenerates byte-identically.** `test_two_renders_are_byte_identical`
   renders twice and compares every file; re-running `pixi run samples` against the committed
   gallery produces an empty `git diff`.
5. **Diagnostics + ROBUSTNESS BACKLOG + pre/post-processing docs.** Present in both `ncc.py` and
   `docs/methods/ncc.md`; backlog mirrored to `docs/ROBUSTNESS-BACKLOG.md`.

## Gates

`ruff check` clean · `ruff format --check` clean · `mypy --strict` clean (24 files) ·
`pytest` 190 passed · coverage **91.50%** (`ncc.py` 94% on CI, floor 80%). CI (`quality`) green
on PR #4.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Flat-template NCC constant is platform-dependent, not version-dependent.**
- **Found during:** CI on Task 1's pinned-behaviour test.
- **Issue:** The research table (and my first test) assumed the zero-variance `TM_CCOEFF_NORMED`
  constant is a function of OpenCV version (4.10→1.0, 4.13/5.x→0.0). CI on Linux/x86_64 returned
  **1.0** on the same pinned opencv 4.13 where macOS/arm64 returns **0.0** — the constant is a
  build artefact.
- **Fix:** The test now pins the portable invariants that hold everywhere — the map is constant,
  carries no NaN/Inf, and its value is one of the known degenerate constants `{0.0, 1.0}` — never
  a plausible-looking correlation. The mandatory std guard is correct regardless, because it runs
  *before* `matchTemplate`.
- **Files modified:** `tests/test_ncc.py`
- **Commit:** 5992932

### Documented (non-blocking)

- **`index.md` omits latency.** Wall-clock latency cannot be byte-stable, and byte-identical
  regeneration is a success criterion, so latency is reported live on the CLI and left out of the
  committed table. Instance count, outcome, and threshold (all reproducible) remain.
- **Sample panels downscaled to ≤1440px.** The native three-tile clutter panel PNG-compressed to
  3.5 MB, over the repo's 2 MB large-file gate. Deterministic `INTER_AREA` downscaling preserves
  byte-identity while keeping the committed docs small.
- **`lattice-touching` sample is EMPTY.** Not a bug: its instances are solid same-colour
  rectangles (zero-variance crop), so NCC's mandatory guard abstains. It doubles as a visible
  demonstration of the guard on a real committed sample.

## Known Stubs

None. `scatter-scaled` finding 4/10 and `cluttered-distractors` finding 7/8 are honest baseline
behaviour (scale/rotation/clutter limits of raw NCC), not stubs — these are exactly the cases the
learned methods in later phases must beat.

## Self-Check: PASSED
- All created source/doc/sample files exist on disk (verified below).
- All four task commits present in git history (efe9b24, 5a8063e, 3932bed, 5992932).
- PR #4 open into `main`, CI `quality` green.
