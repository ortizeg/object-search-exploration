---
quick_id: 260726-lct
slug: set-up-mkdocs-material-docs-site-github-
date: 2026-07-26
status: complete
---

# Summary — MkDocs Material docs site + GitHub Pages + UI walkthrough

## What shipped

- **Docs site**: `mkdocs.yml` (Material theme, light/dark palette toggle, search, code-copy,
  full nav over `docs/`, `strict: true`) + `docs/index.md` landing page.
- **Interface walkthrough**: `docs/walkthrough.md` — a step-by-step UI guide embedding four
  real screenshots in `docs/assets/ui/` (overview, method-selected, results+overlay, canvas
  crop).
- **Screenshot capture**: `scripts/capture_ui.py` — Playwright drives the live app (selects
  `ncc`, drags a ground-truth exemplar on `synthetic/lattice-plain.png`, waits for the overlay)
  and writes deterministic 2× PNGs.
- **Publishing**: `.github/workflows/docs.yml` — builds with `--strict` on every PR, deploys
  `site/` to GitHub Pages from `main`.
- **Tooling**: `pixi.toml` gains `mkdocs-material` (default env → `pixi run docs` /
  `docs-build`) and `playwright` (isolated `capture` env → `pixi run -e capture capture-ui` /
  `playwright-install`); `pixi.lock` re-solved; `site/` gitignored.

## Deviations / notes

- Executed hands-on by the orchestrator rather than via a spawned gsd-executor: the task has a
  tight visual-verification loop (each screenshot had to be inspected and the capture re-run),
  which a text-returning subagent cannot close.
- Fixed 4 pre-existing links that pointed outside `docs/` (to `src/` and `assets/demo/`) — now
  absolute GitHub URLs — so `mkdocs build --strict` passes. These also resolve correctly on
  GitHub.
- FastSAM/OWLv2 ONNX exports still require the `export` env (`pixi run -e export fetch-models`);
  `dinov2-small` and `superpoint` were fetched. The walkthrough demo uses model-free `ncc`.
- **GitHub Pages must be enabled once** in repo Settings → Pages → Source = "GitHub Actions"
  for the deploy job to publish (one-time manual step, outside this repo's code).

## Verification

- `pixi run docs-build` → built with 0 strict warnings.
- Served site screenshotted: home + walkthrough render with theme, nav, ToC, and embedded
  images.
- Results screenshot confirmed to show all 12 matches with the similarity-heatmap overlay.
