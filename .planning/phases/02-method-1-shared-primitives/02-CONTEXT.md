# Phase 2 Context — Method 1 (`ncc`) + Shared Primitives

**Source:** `.planning/IDEA.md` §5 (Method 1, Cross-Cutting Concerns), §7 (requirement IDs),
plus `.planning/research/PITFALLS.md` (implementation traps, verified).

## Domain

The zero-model baseline — normalized cross-correlation template matching — plus the two
cross-cutting primitives the source research names as mattering *more than method choice*:
threshold calibration and peak extraction. Also the sample-run renderer, which every later
phase reuses to produce its committed `docs/samples/<method>/` output.

## Locked Decisions

1. **`ncc` is one self-contained module**, `src/object_search/search/ncc.py`, readable top to
   bottom, with numbered step comments matching `docs/methods/ncc.md`. It may import from
   `search/common/` but is not required to; readability wins over DRY.
2. **`cv2.matchTemplate` with `TM_CCOEFF_NORMED`.** Scale invariance via an image pyramid,
   keeping the level index per peak so the emitted box size is right. Rotation invariance via a
   rotated-template bank, **default angle set `[0]`** — off by default because it is a large
   constant-factor cost for a case that rarely needs it.
3. **`local-max` is the default peak strategy**, not `nms`. Plain NMS merges touching
   instances, which is the exact failure the research calls out. The suppression radius is tied
   to the crop size.
4. **Three calibration strategies, all real:** `self-similarity`, `ratio`, `gmm`.
   `gmm` uses `sklearn.mixture.GaussianMixture(n_components=2)` with a fixed `random_state`
   from config, and cuts between the two modes.
5. **Peak extraction and calibration are shared *offerings*** in `search/common/`, imported by
   choice. Nothing forces a method to use them.
6. **Sub-threshold candidates are produced by every method from Phase 2 onward** (EVAL-08).
   `ncc` returns the top ~50 peaks with raw scores plus the threshold that was applied, not
   only the accepted matches.
7. **The sample-run renderer is a CLI command**, `pixi run samples`, driven by a committed
   manifest of (image, exemplar box) pairs so runs are fixed and regenerable. One command
   regenerates everything; output is deterministic.
8. **Determinism — and what actually threatens it.** Research measured the usual suspects and
   found them irrelevant: OpenCV thread count, BLAS thread count, ONNX Runtime thread count, and
   argmax tie order all produced **bit-identical** results. Two corrections follow:
   - Do **not** claim thread pinning buys determinism. (Also: `cv2.setNumThreads(1)` is *silently
     ignored* on macOS GCD — only `0` has any effect — so a "reproducibility" call there would be
     doubly false.)
   - What genuinely bites: **set/dict iteration order**, **NMS tie-breaking**, **config-hash key
     order**, and library-version drift. Those are the things to pin. NMS therefore sorts by
     `(-score, y, x)` and the config hash serializes with sorted keys.
   The GMM is the one genuinely stochastic step here and takes `random_state` from config. The sample
   renderer asserts a byte-identical re-render, which is the real end-to-end guard.

## Canonical References

- `.planning/research/PITFALLS.md` — **required reading before implementing.** In particular:
  the `TM_CCOEFF_NORMED` zero-variance template trap, the response-map coordinate convention
  (`(W-w+1, H-h+1)`, top-left anchored), correct pyramid coordinate mapping, and
  rotated-template border artifacts.
- `.planning/IDEA.md` §5 Method 1 and Cross-Cutting Concerns
- `src/object_search/schemas/` — the frozen contracts from Phase 1
- `src/object_search/synthetic/generator.py` — `DEMO_SPECS`, especially `lattice-touching`,
  which is the fixture that proves `local-max` beats `nms`

## Specifics

**`NCCConfig`** (frozen Pydantic, drives the UI form):
`scales: tuple[float, ...]` (pyramid factors, default something like
`(0.75, 0.875, 1.0, 1.15, 1.3)`), `angles_deg: tuple[float, ...] = (0.0,)`,
`threshold: float | None = None` (None ⇒ use the calibrator),
`calibration: Literal["fixed", "self-similarity", "ratio", "gmm"] = "self-similarity"`,
`peaks: Literal["nms", "local-max", "watershed"] = "local-max"`,
`nms_iou: float = 0.3`, `suppression_radius_frac: float = 0.5`,
`max_candidates: int = 50`, `seed: int = 0`.

**Known traps that must be handled explicitly, not discovered later:**

- **Zero-variance template — and it does NOT produce NaN.** `TM_CCOEFF_NORMED` divides by the
  template's standard deviation, but research measured the actual behaviour and it is worse than
  NaN: **OpenCV 4.10 returns `1.0` at every single pixel** for a flat template, and OpenCV 5.0
  returns `0.0` everywhere. Both are undocumented, neither raises, and neither produces a NaN you
  could detect downstream. A flat crop therefore yields a *perfect match everywhere*, which after
  peak extraction becomes a confident-looking wall of false positives.
  Guard: compute the crop's std **before** calling `matchTemplate`; below a small epsilon,
  return `outcome=EMPTY` with a diagnostic note saying the exemplar has no texture for NCC. This
  is the Method 1 analogue of METHOD-04c, and it is mandatory, not defensive.
  Pin the observed behaviour in a test so an OpenCV bump that changes it is caught.
- **Response-map offset.** `matchTemplate` output has shape `(H-h+1, W-w+1)` and each value is
  anchored at the template's **top-left**, not its centre. Converting a peak index to a box is
  `BBox(x=peak_x, y=peak_y, w=tw, h=th)` at that level's scale — no centre offset. Getting this
  wrong shifts every box by half a template, which looks like "slightly bad matching" rather
  than a bug.
- **Always correlate over the FULL scene, never a cropped search region.** Research measured that
  restricting the search extent changes **73% of the returned floats**, and shifts the peak value
  from `0.99999994` to `1.0`. This is the single biggest threat to the "same input ⇒ identical
  results" constraint, because a later "optimization" that crops the search window would silently
  change every score. If a search-region restriction is ever added, it must be part of the config
  and therefore part of the config hash.
- **Pyramid: rescale the SCENE and crop the template from the DOWNSCALED scene.** This reverses
  the intuitive approach and the reversal is measured, not stylistic:
  - Downscaling the template independently of the scene drops the exemplar's own self-match from
    `1.0000` to `0.3071`, and does so **non-monotonically** — so it cannot be corrected by a
    per-level offset.
  - Cropping the template from the already-downscaled scene keeps the self-match at `1.0000`.
  - Additionally, the **spurious noise floor varies ~15× with template size** (0.577 at 8×8 vs
    0.039 at 128×128). A naive argmax across pyramid levels is therefore biased toward the
    smallest template. Either normalize per level against that level's own score distribution
    before comparing across levels, or restrict cross-level comparison to a narrow scale range —
    and say in the docstring which was chosen and that the bias is the reason.
  Record the level per peak so the emitted box size is correct.
- **Rotated-template borders.** Rotating a crop introduces empty corners. Correlating those
  zero corners against the scene biases the score. Use a rotated **mask** and OpenCV's masked
  matching, or inscribe the largest axis-aligned rectangle in the rotated crop. State which
  was chosen and why in the module docstring.
- **NaN handling.** Any NaN in the response map must be replaced or the peak search will return
  it as a maximum on some platforms.

**Cross-cutting modules:**

- `common/nms.py` — plain greedy IoU NMS over `(box, score)`. Deterministic tie-breaking (sort
  by `(-score, y, x)`, never by score alone, or equal scores reorder run to run).
- `common/peaks.py` — `extract_peaks(response, strategy, ...) -> list[(x, y, score)]`.
  `local-max` uses a maximum filter with a footprint tied to the crop size, then keeps points
  equal to the filtered value and above the floor. `watershed` uses the distance transform of
  the thresholded map. Each strategy is a plain function; the selector is a small dispatch at
  the top, not a class hierarchy.
- `common/calibration.py` — `calibrate(scores, strategy, ...) -> CalibrationResult` carrying the
  chosen threshold **and** the reasoning (which is what makes the Phase 2 success criterion
  "different, inspectable thresholds" verifiable).
- `common/viz.py` — matplotlib/cv2 rendering: draw matches with scores, draw the exemplar
  distinctly, render a similarity heatmap to a PNG, and compose a side-by-side panel. Used by
  the sample renderer and by the API's diagnostics payload (heatmap → base64 PNG).

## Deferred

- FFT-based correlation for large templates, log-polar / Fourier–Mellin joint rotation-scale
  invariance, and discriminative correlation filters (MOSSE/KCF) — all go in the module's
  `ROBUSTNESS BACKLOG` docstring section and `docs/ROBUSTNESS-BACKLOG.md`, not built.
- `watershed` peak extraction may ship as a working-but-secondary strategy; `local-max` and
  `nms` are the two that must be demonstrably different on the touching-lattice fixture.

## Scope Fence

**In:** `ncc.py`, `common/{nms,peaks,calibration,viz}.py`, the sample-run renderer CLI, the
first committed sample runs under `docs/samples/ncc/`, `docs/methods/ncc.md`.

**Out:** the API (Phase 3) — `ncc` is exercised from the CLI in this phase. Any other method.
Ground-truth labels for photos (Phase 8).

## Risk Summary

- **The touching-lattice fixture must actually produce touching instances.** If `lattice-touching`
  leaves gaps, the `nms` vs `local-max` comparison proves nothing. Verify the fixture visually
  (render it) before relying on it for the success criterion.
- **Coverage floor.** `viz.py` is rendering code and is easy to leave untested, which drags
  coverage below 80%. Test it by asserting output image dimensions and non-blankness rather
  than by pixel-comparing figures.
- **`gmm` on a degenerate score distribution** (all scores nearly equal) will fit two
  overlapping components and produce a meaningless cut. Detect low separation (e.g. component
  means closer than a fraction of the pooled std) and fall back with a recorded diagnostic
  rather than returning a confident-looking bad threshold.
