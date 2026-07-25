# Polish Backlog

Cosmetic / UX items found during verification that do not affect data correctness. Surface
these in `docs/LIMITATIONS.md` during Phase 8 and fix opportunistically.

## UI

- **Canvas `#stage` overflows the viewport height.** Verified in a real browser on Phase 4:
  the canvas renders 759×1529 CSS px for an 800×600 image because the stage element stretches
  to a tall flex column taller than the viewport, so the fit-scaled image is letterboxed with
  large empty bands and the canvas extends below the fold. The coordinate transform is
  unaffected (proven to 5.68e-14 px round-trip with separate scaleX/scaleY), so draws still map
  to correct image pixels — this is layout polish only. Fix: constrain the stage to the
  available viewport height (e.g. `min-height: 0` on the flex child + `max-height` on the
  stage, or size the canvas box to the fit image's aspect). The `design-review` /
  `design-deslop` skills mapped to Phase 4 in IDEA.md §10 would catch this in a dedicated pass.
- **`/stats` latency shows total percentiles, not the preprocess/inference/postprocess split.**
  The breakdown IS stored (three columns), so Phase 8 has the data; only the dashboard
  aggregation is total. UI-06 requires "latency percentiles", which is satisfied; the split is
  a dashboard enhancement.
