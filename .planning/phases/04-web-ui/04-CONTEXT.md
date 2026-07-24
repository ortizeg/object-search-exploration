# Phase 4 Context — Web UI

**Source:** `.planning/IDEA.md` §6 (frontend in the architecture diagram), §7 (UI-01…UI-08),
**§7a (the tiered rating design — the UI is where the null discipline is most easily broken)**,
§11 (the Milestone 2 seam), plus `.planning/research/PITFALLS.md` (canvas coordinate transforms).

## Domain

A static HTML/CSS/JS canvas app served by FastAPI. No build step, no framework, no npm — plain
ES modules. The reason is the same one that ruled out Gradio: box drawing is the core
interaction and it must be under direct control, and the app must stay trivially runnable years
from now with `pixi run serve`.

## Locked Decisions

1. **No npm, no bundler, no framework.** Vanilla ES modules in `frontend/`, served as static
   files by FastAPI. A `<script type="module">` entry point and a handful of small modules.
   Adding a toolchain here would mean a second dependency manager, which the project constraints
   forbid in spirit (Pixi only) and which adds nothing for a single-user local demo.
2. **Method and config are locked in BEFORE the box can be drawn** (UI-01). The canvas
   draw handler is disabled until a method is selected. This is not cosmetic — every rating must
   be attributable to an exact method+config, and letting a box be drawn first invites changing
   the method afterwards.
3. **The config form is generated entirely from the method's JSON Schema** (UI-07). No method is
   named in the frontend source. A test/verification step greps `frontend/` for each method name
   and asserts zero hits. Supported schema shapes: number (with min/max → range or number
   input), integer, boolean (checkbox), string enum (select), and array-of-number (comma-separated
   text parsed back to a list). `description` becomes the field help text, `default` the initial
   value, `title` the label.
4. **Count fields render EMPTY, never `0`** (UI-08). `<input type="number">` with `value=""`.
   One-click **"all correct"** and **"none missed"** buttons write an explicit `0`. An untouched
   field submits `null`, not `0`. **Do not add a `|| 0` anywhere in the submit path** — that
   single character is the whole bug.
5. **Per-match verdicts require an explicit confirm** (UI-08). The per-match panel starts with
   every box unmarked. Clicking a box toggles it to "wrong". `FP` stays `null` until the user
   presses **Confirm verdicts**. "Looked and approved" must be distinguishable from "never opened
   the panel".
6. **Per-match verdicts and the bare `wrong_count` are mutually exclusive modes** in the UI.
   Choosing one visibly disables the other, so the discrepancy case is prevented at the source
   rather than only flagged server-side.
7. **Thumbs up/down is required; everything else is optional.** Plus an explicit
   **unratable / skip** path so a badly drawn box or a genuinely ambiguous case never gets forced
   into the ratings. **No 1–5 star scale** — star scales drift within a session and are not
   comparable across methods.
8. **The duplicate/fragment convention is shown in the UI** (EVAL-16): "two boxes on one instance
   = 1 TP + 1 FP", displayed next to the rating widget, not buried in docs. Undocumented
   conventions make ratings inconsistent across sessions.
9. **An exploration-mode selector sits ABOVE the method selector** (Milestone 2 seam, §11). In
   Milestone 1 it has exactly one option, `Same-image search`. It exists so Milestone 2 adds an
   option rather than forking the app.
10. **Diagnostics overlay is toggleable** and renders whatever the payload contains — heatmap,
    keypoints, correspondences, Hough peaks, proposals — each independently toggleable and each
    driven by the presence of that field, never by the method name.

## Canonical References

- `.planning/research/PITFALLS.md` — the canvas coordinate-transform trap (CSS size vs backing
  store vs `devicePixelRatio` vs image natural size). **Required reading.**
- `.planning/IDEA.md` §7a — tiered rating, why counts start empty, the two FP paths
- The live `GET /methods` response — the single source of truth for what the form renders
- `interface-design`, `design-review`, and `design-deslop` skills (IDEA.md §10 maps them to
  this phase)

## Specifics

### The canvas coordinate trap — get this right first

Three coordinate spaces exist and conflating any two produces boxes that are subtly wrong:

- **CSS pixels** — what the mouse event reports, via `getBoundingClientRect()`
- **Backing-store pixels** — `canvas.width/height`, which should be
  `cssSize * devicePixelRatio` on a high-DPI display
- **Image pixels** — the coordinate space the API expects, and the only one that may be sent

Maintain one explicit transform (`scale` + `offsetX/offsetY` for zoom/pan) and two functions,
`screenToImage(px, py)` and `imageToScreen(ix, iy)`, that are each other's inverse. Everything
goes through them. A test-equivalent check: draw a box at a known image coordinate at
`zoom = 2.3`, `dpr = 2`, with a pan applied, and assert the box the API receives has the
expected image-space coordinates. On a Retina display an untransformed implementation is off by
exactly 2×, which reads as "the box is half the size I drew" — a recognisable symptom worth
stating in the code comment.

**Two measured gotchas that make the obvious implementation wrong:**

- **`event.offsetX` / `offsetY` are rounded to integers.** Research measured `124` where the true
  position was `123.5`. At high zoom that rounding is amplified by the inverse scale factor into a
  multi-pixel image-space error. **Use `event.clientX/clientY` minus `getBoundingClientRect()`**
  and keep the value as a float through the whole transform, rounding only at the final integer
  box conversion.
- **`scaleX` and `scaleY` are not equal.** For a canvas whose CSS size does not match its
  backing-store size — which is every responsive canvas — the horizontal and vertical ratios
  differ, so a single `scale` scalar skews the box. Compute them separately as
  `canvas.width / rect.width` and `canvas.height / rect.height` and apply each to its own axis.
  Using one ratio for both produces boxes that are correct in the middle of the canvas and
  progressively wrong toward one edge, which is easy to misread as sloppy drawing.

Both of these are Chromium-measured; treat the numbers as indicative and the structural fix
(`clientX` + `getBoundingClientRect` + per-axis scale + float math) as the requirement.

Never round until the final integer conversion, and clamp to the image bounds so a drag past the
edge cannot produce a negative or out-of-range box (the `BBox` validators would reject it and
the user would see an opaque 422).

### Layout

Single page, three regions: a left control rail (exploration mode, method select, generated
config form, image picker, Search button), the canvas centre stage (zoom/pan, draw, overlays,
overlay toggles), and a right panel that switches between the rating widget and the stats
dashboard.

### Stats dashboard

Per method: thumbs-up rate **with its Wilson interval and `n` rendered inline** — a rate from 4
ratings must not look like a rate from 400 (EVAL-14). Label what the interval means. Show
precision and recall each with their **own separate `n`**, plus abstention count, error count,
and latency percentiles broken into preprocess/inference/postprocess.

## Deferred

- Bradley-Terry ranking display and the paired-comparison UI — Phase 8.
- Any client-side charting library. Render the small number of bars/intervals with plain SVG or
  canvas; a chart dependency is not worth it here, and Phase 8's committed charts are generated
  server-side with matplotlib anyway.

## Scope Fence

**In:** `frontend/` (HTML, CSS, ES modules), the FastAPI static mount, canvas draw/zoom/pan,
schema-driven config form, result and diagnostics overlays, tiered rating widget, stats
dashboard.

**Out:** any new search method, any change to the derived-metric semantics (Phase 3 owns them),
the benchmark.

## Risk Summary

- **`|| 0` in the submit path** is the single highest-risk line in the phase. It looks like
  defensive coding and it destroys the null/zero distinction. Grep for it in review.
- **DPR/zoom transform bugs are silent** — the app looks fine and the boxes are wrong, which
  then reads as bad method performance and pollutes the ratings with unattributable failures.
  Verify the transform before anything else is built on top of it.
- **Coverage floor applies to the Python side only**; the frontend has no test runner and adding
  one would mean npm. Compensate by keeping logic-bearing JS (the transform, the schema→form
  mapping, the submit payload builder) in small pure functions, and by verifying the round trip
  through real HTTP requests in the Python test suite where possible.
- **Human verification is genuinely required** for the Phase 4 success criterion ("a person can
  open the app, draw a box, rate a run, without touching a terminal"). Automated checks can
  prove the endpoints and payloads; they cannot prove the interaction feels right. Drive the real
  browser to verify, and capture a screenshot as evidence in the PR.
