---
phase: 04-web-ui
plan: 02
subsystem: frontend
tags: [overlays, diagnostics, rating-widget, null-discipline, wilson, stats-dashboard, svg]

requires:
  - phase: 04-web-ui
    provides: "Canvas shell + viewport transform (screenToImage/imageToScreen), schema-driven config form, draw->search tracer, GET /image route"
  - phase: 03-backend-api
    provides: "POST /ratings (Rating schema, null-count discipline, bounds/verdict flags), GET /stats (MethodStats: per-metric n, Wilson interval, NULL-propagating precision/recall)"
provides:
  - "frontend/js/overlay.js — drawResults (zoom/pan-tracking match boxes + scores, distinct exemplar self-match, red wrong-verdict boxes), presence-driven diagnostics (heatmap/keypoints/correspondences/Hough peaks/proposals), pure hitTestMatch + presentDiagnosticFields"
  - "frontend/js/payload.js — buildRatingPayload, a PURE function preserving null(=not assessed) vs 0(=assessed,none); no or-zero/nullish-zero coercion (grep-guarded)"
  - "frontend/js/rating.js — tiered widget: thumbs (required) + two mutually exclusive precision modes + missed_count + unratable/skip + note; empty counts, explicit-0 buttons, confirm-verdicts gate, EVAL-16 text; no star scale"
  - "frontend/js/stats.js — plain-SVG scoreboard: thumbs-up rate with Wilson interval + n inline, ranked by the Wilson lower bound, n=0 rendered distinctly, precision/recall each with own n"
  - "tests/test_rating_contract.py — POSTs the exact bodies buildRatingPayload produces to the real /ratings and asserts stored NULL/0 + /stats aggregate; hard guard that payload.js has no or-zero/nullish-zero coercion"
affects: [phase-08-evaluation-docs]

tech-stack:
  added: []
  patterns:
    - "Overlays layer on one shared transform per frame: viewport.paintImage draws the image and leaves the ctx in image space; boxes stroke in image space (track zoom/pan), labels/dots draw in backing space at fixed size via viewport.imageToBacking"
    - "Diagnostics are presence-driven: DIAGNOSTIC_FIELDS names the Diagnostics shape, a toggle appears per field actually present, never per method name (grep-enforced no-method-name-in-frontend still passes)"
    - "The null-vs-zero discipline is a pure function (buildRatingPayload) proven from Python since there is no JS test runner; the phase's highest-risk line (or-zero coercion) is a hard CI grep test"
    - "The Wilson lower-bound ranking is re-applied client-side in stats.js so the ranking rule lives with the renderer, redundant with the backend's own ordering"

key-files:
  created:
    - frontend/js/overlay.js
    - frontend/js/payload.js
    - frontend/js/rating.js
    - frontend/js/stats.js
    - tests/test_rating_contract.py
  modified:
    - frontend/js/main.js
    - frontend/js/viewport.js
    - frontend/index.html
    - frontend/css/app.css
    - .planning/REQUIREMENTS.md
    - .planning/ROADMAP.md

key-decisions:
  - "buildRatingPayload kept pure and free of any or-zero/nullish-zero coercion; parseCount maps empty/whitespace/non-integer/negative to null (not assessed) and only an explicit '0' to 0 (assessed, none). Proven via tests/test_rating_contract.py against the real endpoint."
  - "Per-match verdicts and bare wrong_count are a single radio with two panels; the inactive panel's inputs are disabled (not just hidden) so the modes are exclusive at the source. Verdicts count only after Confirm verdicts sets verdicts_confirmed; editing after confirming re-opens the assessment."
  - "Per-match wrong indices live in the shared canvas state (state.wrongSet) so the overlay and the widget cannot disagree; canvas left-clicks in verdict mode hit-test matches (smallest-area wins) instead of starting a new query."
  - "Diagnostic overlays gated by field presence (Diagnostics field names), never method name; a method the UI has never seen still renders its heatmap/keypoints/etc. Default every present overlay ON so a run is legible."
  - "Stats rendered with plain SVG (no charting dependency, per the phase Deferred list). n=0 renders as an explicit 'no ratings yet — no information' rather than a [0,1] bar; precision/recall each render 'not assessed (n=0)' rather than a fabricated 0."

metrics:
  duration_min: 45
  completed: 2026-07-25
  tasks: 3
  files: 11
  tests_total: 277
  coverage_pct: 91.32

status: complete
---

# Phase 4 Plan 2: Result/Diagnostics Overlays, Tiered Rating Widget, Stats Dashboard Summary

The three pieces that turn the 04-01 canvas shell into a complete loop: result and
presence-driven diagnostics overlays, the tiered rating widget where UI-08's null-vs-zero
discipline lives, and the plain-SVG stats dashboard that reads honestly at small n. Built with
no npm and no method name anywhere in `frontend/`.

## What was built

- **`frontend/js/overlay.js`** — `drawResults` strokes each match box through the viewport's
  image-space transform (so boxes track zoom and pan), draws the exemplar self-match distinctly
  (gold + "self" label) and human-marked-wrong boxes in red, and labels each match with its
  score at a fixed size in backing space. `drawHeatmap` stretches the decoded base64 PNG under
  the boxes; `drawPointDiagnostics` renders keypoints, correspondences, Hough peaks and
  proposals. Which overlays exist is driven by the **presence** of the named `Diagnostics`
  fields, never by method name. Pure `presentDiagnosticFields` and `hitTestMatch` back the
  toggles and the per-match verdict clicks.
- **`frontend/js/payload.js`** — `buildRatingPayload(state)`, a pure function: an untouched
  count is omitted (stored as SQL NULL = "not assessed"), an explicit 0 is sent as 0 (assessed,
  none). `parseCount` maps empty/whitespace/non-integer/negative to `null` and only `"0"` to the
  integer 0. **No or-zero and no nullish-zero coercion anywhere** — the phase's single
  highest-risk line, now grep-guarded in CI.
- **`frontend/js/rating.js`** — the tiered widget: Tier 0 thumbs (required); Tier 1 wrong
  matches in two mutually exclusive modes (per-match verdicts clicked on the overlay, or a bare
  `wrong_count`), one visibly disabling the other; Tier 2 `missed_count`; an unratable/skip
  control; a note. Count inputs render **empty**; the "All correct (0)" / "None missed (0)"
  buttons write an explicit `0`; per-match verdicts stay unassessed until **Confirm verdicts**
  sets `verdicts_confirmed`. The **EVAL-16** convention text sits next to the widget. No 1–5 star
  scale.
- **`frontend/js/stats.js`** — the scoreboard in plain SVG. The thumbs-up rate is shown with its
  Wilson interval and n inline, the interval labelled; methods are ranked by the **Wilson lower
  bound** (re-sorted client-side); **n = 0 renders distinctly** ("no ratings yet — no
  information"), never a [0,1] bar; precision and recall each carry their **own separate n** and
  render "not assessed (n = 0)" rather than a fabricated 0. Abstention/error/sweep-eligible chips
  and total latency p50/p90/p99.
- **`frontend/js/main.js` + `viewport.js`** — `viewport.paintImage` (image draw, ctx left in
  image space) and `imageToBacking` let overlays layer on one shared per-frame transform. `main`
  composites the frame, builds the presence-driven toggles, mounts the rating widget per run,
  runs the panel tabs (Rating/Stats), routes canvas clicks to verdict marking in per-match mode,
  and refreshes stats after each submit.
- **`tests/test_rating_contract.py`** — POSTs the exact bodies the widget produces (a bare
  thumbs-up; a confirmed-verdicts body; an all-correct body with explicit 0) to the real
  `/ratings` endpoint and asserts the stored NULL/0 and the `/stats` aggregate; plus a hard test
  that `payload.js` contains no or-zero / nullish-zero coercion.

## Verification (real output)

Quality gates (Python side, all green):

```
ruff check src/ tests/          -> All checks passed!
ruff format --check src/ tests/ -> 74 files already formatted
mypy src/                       -> Success: no issues found in 44 source files
pytest tests/                   -> 277 passed; Total coverage: 91.32% (>=80% floor)
```

UI-08 grep (the highest-risk line) and the null-path test:

```
grep -n "|| 0\|?? 0" frontend/js/payload.js   -> (no matches)
pytest tests/test_rating_contract.py -q       -> 4 passed
  test_payload_js_has_no_zero_coercion              (|| 0 and ?? 0 absent from payload.js)
  test_bare_thumbs_up_body_stores_null_...          (bare thumbs-up -> wrong_count/missed_count NULL, precision_n/recall_n 0)
  test_confirmed_verdicts_body_feeds_per_match_...  (1 wrong of 3 -> precision 2/3, recall unavailable)
  test_all_correct_body_sends_explicit_zero         (wrong_count=0, missed_count=0 -> precision 1.0, recall 1.0, stored 0 not NULL)
```

All eight ES modules parse (`node --check`); the pure functions were exercised directly in node
(parseCount empty->null / "0"->0 / negative->null; buildRatingPayload bare-thumbs / explicit-0 /
unconfirmed-omit / confirmed-array; hitTestMatch smallest-area-wins; presentDiagnosticFields
excludes empty arrays).

Live server round trip (`uvicorn ... --port 8123`):

```
/                        -> 307 -> /app/
/app/  /app/js/overlay.js  /app/js/rating.js  /app/js/payload.js  /app/js/stats.js  -> 200
/methods  /stats         -> 200
POST /search (chipset-01, ncc)  -> 200; 5 matches; diagnostics present: similarity_heatmap
POST /ratings {run_id:1, thumbs_up:true}  -> 200 {rating_id:1}
GET /stats  -> ncc thumbs_n=1 n_up=1 ci_lower≈0.207 precision_n=0 recall_n=0 p50≈20.6ms
```

The 1/1 thumbs rating producing a Wilson lower bound of ≈0.207 (not 1.0) is the EVAL-14 ranking
behaviour working: a single lucky rating does not top the board.

## Manual full-loop verification (for the orchestrator's browser check)

Task 4 is a `checkpoint:human-verify` and was **not blocked** per the execution directive. The
endpoints, payloads, and the null/zero round trip are proven above; what a headless check cannot
prove is that the interaction feels right and the boxes land correctly under zoom/pan on a
high-DPI display. Drive a real browser:

1. `pixi run serve` (uvicorn on :8000). Open `http://localhost:8000/app`.
2. Pick the `ncc` method, choose `chipset/chipset-01.png`, and draw a box tightly around one
   chip. Confirm result boxes overlay with scores, and the **Overlays** toggles appear under the
   canvas (at least "Similarity heatmap"). Toggle the heatmap on/off; zoom (wheel) and pan
   (middle-drag) and confirm the boxes and heatmap stay locked to the image.
3. In the right panel (Rating tab): give a **thumbs-up**. Choose "Mark wrong boxes", click one
   result box on the canvas (it turns red and the match row highlights), press **Confirm
   verdicts**. Confirm the count inputs started **empty**, not 0. Press **Submit rating**.
4. Confirm "All correct (0)" and "None missed (0)" write a literal 0 into their inputs, and that
   leaving a count untouched leaves it blank.
5. Switch to the **Stats** tab (or note it auto-refreshes after submit): confirm `ncc` shows a
   thumbs-up rate with its **n** and a **Wilson interval**, and that a freshly-seeded method with
   no ratings would read "no ratings yet" rather than a full-width bar. Capture a screenshot for
   the PR.

## Deviations from Plan

**1. [Scope fence — out of scope to fix here] Latency shown as total percentiles, not a
preprocess/inference/postprocess split.**
- **Found during:** Task 3. The plan/context ask the dashboard to show latency percentiles
  "broken into preprocess/inference/postprocess". The `GET /stats` aggregate (`MethodStats`)
  exposes only total-latency percentiles (`latency_p50/p90/p99_ms`); the per-stage split exists
  on each run's `LatencyBreakdown` but is not aggregated into per-stage percentiles.
- **Decision:** Rendered the total p50/p90/p99, labelled "Latency (total ms)". Adding per-stage
  percentile aggregation is a change to the derived-metric layer, which the Phase 4 scope fence
  explicitly assigns to Phase 3 ("any change to the derived-metric semantics — Phase 3 owns
  them"). Deferred rather than reaching across the fence. No requirement (UI-06) mandates the
  split; it lists "latency percentiles".

Otherwise the plan executed as written.

## Notes

- Hough-peak diagnostics are rendered as vote-weighted markers at `(dx, dy)` (a translation in
  scene pixels, sized by vote weight) — an honest depiction of vote space rather than pretending
  a peak is an image box. Method 2 (the only producer of Hough peaks) lands in Phase 5, so this
  overlay is presence-driven and exercised live only then; it is not a stub.

## Known Stubs

None. Every overlay and every stats field is wired to real payload data; the rating loop
round-trips through the live `/ratings` and `/stats` endpoints (proven above).

## Self-Check: PASSED
