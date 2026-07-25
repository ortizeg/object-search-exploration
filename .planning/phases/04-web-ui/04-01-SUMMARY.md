---
phase: 04-web-ui
plan: 01
subsystem: frontend
tags: [canvas, coordinate-transform, es-modules, fastapi-static, json-schema-form, dpr]

requires:
  - phase: 03-backend-api
    provides: "FastAPI app + create_app factory, /methods /images /search endpoints, MethodInfo.config_schema (JSON Schema), resolve_image_path containment check"
  - phase: 01-foundations
    provides: "ExemplarBox/BBox half-open geometry convention, method registry"
provides:
  - "frontend/ vanilla-ES-module canvas shell served by FastAPI at /app (no npm, no bundler)"
  - "Viewport transform: screenToImage/imageToScreen as exact inverses + finalizeBox (frontend/js/viewport.js)"
  - "Schema-driven config form buildForm(configSchema) with no method name in frontend/ (frontend/js/form.js)"
  - "In-browser transform proof: frontend/dev/selfcheck.html + selfcheck.js (round-trip within 0.5px at dpr=2, zoom=2.3, panned)"
  - "GET /image?image_id=... raw scene-bytes route (api/static.py), path-containment reused from /search"
  - "The draw->search tracer: draw a box once a method is selected -> POST /search -> match count (frontend/js/main.js)"
affects: [phase-04-web-ui-plan-02]

tech-stack:
  added: []
  patterns:
    - "The coordinate transform is one pair of pure inverse functions; every screen/image conversion goes through them, floats kept until finalizeBox rounds exactly once (half-open box)"
    - "clientX/clientY + getBoundingClientRect, never the integer-rounded element-relative offset; scaleX and scaleY computed separately (PITFALLS §9.1/§9.3)"
    - "The config form is generated purely from the method's JSON Schema; a grep test enforces zero method names in frontend/ (UI-07)"
    - "Static serving isolated in api/static.py and wired into create_app; the raw-image route reuses resolve_image_path so one containment check protects both /search and /image"

key-files:
  created:
    - frontend/js/viewport.js
    - frontend/js/form.js
    - frontend/js/api.js
    - frontend/js/main.js
    - frontend/index.html
    - frontend/css/app.css
    - frontend/dev/selfcheck.html
    - frontend/dev/selfcheck.js
    - src/object_search/api/static.py
    - tests/test_frontend_static.py
  modified:
    - src/object_search/api/app.py
    - .planning/REQUIREMENTS.md
    - .planning/ROADMAP.md

key-decisions:
  - "Draw is gated behind method selection via drawingEnabled() (state.method !== null && image loaded); pointerdown returns early when false (UI-01)"
  - "The exemplar box is the single source of truth in IMAGE space: stored, sent, and stroked verbatim; never regenerated from screen coordinates (round-trip is not IEEE-754 exact)"
  - "zoom is expressed in backing-store px per image px so devicePixelRatio is absorbed entirely by scaleX/scaleY and never appears in the pointer math"
  - "Added GET /image raw-bytes route (deviation, Rule 3): the canvas needs real pixels; /images returns only metadata. Reuses resolve_image_path; traversal 404s"
  - "The exploration-mode selector sits ABOVE the method selector with one option (Same-image search) — the Milestone 2 seam; a source-order test pins it"

metrics:
  duration_min: 30
  completed: 2026-07-25
  tasks: 3
  files: 12
  tests_total: 273
  coverage_pct: 91.32

status: complete
---

# Phase 4 Plan 1: Canvas Shell, Coordinate Transform, Schema-Driven Config Form Summary

The FastAPI-served vanilla-ES-module canvas shell, the coordinate transform every later
Phase 4 interaction sits on (drawing, overlays, rating), and the schema-driven config form —
with drawing gated behind method selection. Built with no npm, no bundler, and no method name
anywhere in `frontend/`.

## What was built

- **`frontend/js/viewport.js`** — the three-space transform (CSS px / backing-store px / image
  px) as a pair of exact inverse pure functions `screenToImage` / `imageToScreen`, plus
  `finalizeBox` (normalise → clamp in float → integerise floor/ceil → re-clamp → reject) and a
  `Viewport` class owning zoom/pan, `fitContain` letterbox, wheel-zoom-about-cursor, and a
  transform-aware `render`. The two measured fixes are in place: `clientX/clientY` +
  `getBoundingClientRect` (never the integer-rounded element-relative offset), and `scaleX` /
  `scaleY` computed separately.
- **`frontend/dev/selfcheck.html` + `selfcheck.js`** — the in-browser assertion harness (the
  project has no JS test runner by design). It constructs a dpr=2, zoom=2.3, panned viewport,
  round-trips known image coordinates through both transforms, and renders PASS/FAIL, also
  asserting `scaleX !== scaleY` and a reverse-drag `finalizeBox`.
- **`frontend/js/form.js`** — `buildForm(configSchema)` generates inputs purely from JSON
  Schema (number→range/number, integer, boolean→checkbox, string-enum→select,
  array-of-number→comma text), resolving `$ref`/`anyOf` (Pydantic Optional) wrappers;
  `readValues()` returns a schema-shaped config.
- **`frontend/index.html` + `css/app.css`** — the shell: control rail (exploration mode ABOVE
  method, config host, image picker, Search), canvas centre stage, right-panel placeholder.
  The CSS pins the canvas contract (`border:0; padding:0; transform:none; touch-action:none`).
- **`frontend/js/api.js` + `main.js`** — fetch wrappers for the five endpoints and the tracer:
  load `/methods` + `/images`, render on the Viewport, gate drawing behind method selection,
  POST `/search` on box release, log the match count.
- **`src/object_search/api/static.py`** — mounts `frontend/` at `/app`, redirects `/` → `/app/`,
  and serves raw scene bytes at `/image` (containment via `resolve_image_path`). Wired into
  `create_app`.
- **`tests/test_frontend_static.py`** — 10 tests covering the redirect, index + module serving,
  the selfcheck harness, the raw-image route (serves + rejects traversal), the exploration-
  above-method source order, the draw gate code path, and the no-method-name-in-frontend grep.

## Verification (real output)

Quality gates (Python side):

```
ruff check src/ tests/        -> All checks passed!
ruff format --check src/ tests/ -> 73 files already formatted
mypy src/                     -> Success: no issues found in 44 source files
pytest tests/                 -> 273 passed; Total coverage: 91.32% (>=80% floor)
```

Transform-fix greps:

```
grep -c 'offsetX\|offsetY' frontend/js/viewport.js      -> 0
grep -c 'getBoundingClientRect' frontend/js/viewport.js -> 7
const scaleX = canvasWidth / rect.width;   (line 60, separate)
const scaleY = canvasHeight / rect.height; (line 61, separate)
grep -rn 'ncc' frontend/  -> 0 hits   (no registered method name leaks in)
```

Node round-trip of the pure transform at dpr=2, zoom=2.3, panned: worst error **5.68e-14 px**
(tolerance 0.5 px); `scaleX !== scaleY` true; reverse-drag `finalizeBox` → `{100,50,301,201}`.

Server smoke (curl against `uvicorn ... --port 8137`):

```
/                        -> 307 -> http://127.0.0.1:8137/app/
/app/                    -> 200
/app/dev/selfcheck.html  -> 200   (contains id="verdict")
/app/js/viewport.js      -> 200
/methods                 -> 200
/image?image_id=basketball/frame_000076.jpg -> 200 image/jpeg
```

## Manual verification steps (for the orchestrator's browser check)

The visual/DPR proof cannot be established headless. Drive a real browser:

1. `pixi run serve` (uvicorn on :8000).
2. Open `http://localhost:8000/app/dev/selfcheck.html` → confirm it renders **PASS** (the
   transform inverse proof; worst error should read ~1e-13 px).
3. Open `http://localhost:8000/app`. Confirm you **cannot** draw a box yet (status says "Pick a
   method first"); the Search button is disabled.
4. Pick the `ncc` method → confirm the config form appears and matches that method's fields
   (schema-driven), and that drawing is now enabled.
5. Choose a chipset image, zoom in (wheel), pan (middle-drag), and draw a box tightly around
   one chip. On release, confirm the console logs `[search] run <id> -> <n> matches {x,y,w,h}`
   and that the logged box coordinates land on the chip in **image** space (not offset or
   half-sized). This is the DPR/zoom proof.
6. Confirm no method name is visible in any served frontend source (already grep-enforced).

## Deviations from Plan

**1. [Rule 3 — Blocking] Added a raw scene-bytes route `GET /image?image_id=...`.**
- **Found during:** Task 2 — the canvas tracer must draw the actual scene pixels, but
  `GET /images` returns only metadata and no route served the bytes.
- **Fix:** Added `GET /image` in `api/static.py`, reusing `resolve_image_path` so the same
  path-containment check that protects `/search` protects it; a traversal id 404s.
- **Files:** `src/object_search/api/static.py`, `frontend/js/api.js` (`imageUrl`),
  `tests/test_frontend_static.py` (serves + rejects-traversal tests).
- **Commit:** f9120bf

**2. [Checkpoint deferred] Task 4 human-verify not blocked.**
- Per the execution directive, the browser/DPR visual checkpoint was not blocked on. All auto
  tasks are implemented, the Python static-serving test is added, the server was smoke-tested
  via curl (200s above), and the manual steps are recorded here for the orchestrator's
  browser-driven verification.

## Known Stubs

- Right-hand `#panel` is a placeholder ("Results, rating, and the scoreboard appear here")
  and the search result overlay is not drawn — both are **intentional** and owned by Plan
  04-02 (result/diagnostics overlays + rating widget + stats dashboard). The tracer logs the
  match count to the console to prove the round trip; it does not render matches.

## Self-Check: PASSED
