---
quick_id: 260726-lct
slug: set-up-mkdocs-material-docs-site-github-
date: 2026-07-26
status: complete
---

# Quick Task 260726-lct — MkDocs Material docs site + GitHub Pages + UI walkthrough

## Description

Render the existing `docs/` Markdown tree as a themed, searchable static site; publish it to
GitHub Pages; capture real screenshots of the running canvas UI; and write an interface
walkthrough page that embeds them.

## Context / decisions (from chat)

- Screenshots captured by **scripting Playwright** against the live app (not mocked, not manual).
- **Fetch inference models** as part of the task; also **pull latest `main`** first (newer docs
  there).
- Docs tooling (`mkdocs-material`) goes in the **default** env so `pixi run docs` works with no
  `-e`; Playwright is isolated in a dedicated **`capture`** env (pulls Chromium, maintainer-only).

## Tasks

1. **Sync + tooling.** Merge `origin/main`; add `mkdocs-material` (default env) and `playwright`
   (capture env) to `pixi.toml`; add `docs`, `docs-build`, `capture-ui`, `playwright-install`
   tasks; `pixi install`; fetch models.
2. **Site config.** Write `mkdocs.yml` (Material theme, light/dark toggle, search, nav over the
   real tree, `strict: true`) and `docs/index.md`. Fix 4 pre-existing out-of-tree links so
   `mkdocs build --strict` passes.
3. **Capture.** Write `scripts/capture_ui.py`; capture overview / method-selected /
   results-overlay / canvas-crop into `docs/assets/ui/`.
4. **Walkthrough + publish.** Write `docs/walkthrough.md` embedding the screenshots; add
   `.github/workflows/docs.yml` (build on PR, deploy from main); gitignore `site/`.

## Verify

- `pixi run docs-build` exits 0 with no strict warnings.
- Rendered home + walkthrough pages display correctly (theme, nav, embedded images) — verified
  by screenshotting the served site.
- All four UI screenshots show the intended state (12 matches + heatmap overlay on the results
  shot).
