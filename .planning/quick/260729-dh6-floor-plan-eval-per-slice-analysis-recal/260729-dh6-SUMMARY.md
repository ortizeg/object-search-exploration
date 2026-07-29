---
id: 260729-dh6
title: Floor-plan eval enrichment — per-slice analysis, aggressive tuning, dino-dense OOM fix
status: complete
created: 2026-07-29
completed: 2026-07-29
branch: feat/floorplans-slicing-tuning
---

# Summary — floor-plan eval enrichment (slicing + aggressive tuning + dino-dense OOM fix)

Follow-up to the merged PR #41. Adds per-slice result analysis, broadens the domain threshold
tuning, and fixes the `dino-dense` GPU-OOM — code + model-free tests only (the GPU re-run is a
manual vast.ai step performed after this lands).

## Commits (atomic, branch `feat/floorplans-slicing-tuning`)

1. `81b150b` — **matched-GT-index sibling** in `eval/metrics.py`. `match_predictions`'s return
   contract is unchanged; a new sibling reuses the same greedy loop to return *which* GT indices
   were matched, so recall can be bucketed by GT box size.
2. `be2dad7` — **per-slice research metrics**: additive `slices` block on `run_research_benchmark`
   / each `run_research_sweep` cell — `by_symbol_size` (recall by GT-box-area ÷ plan-area), `by_
   crowding` (F1 by instances-per-plan), `by_plan_resolution` (F1 by canvas size). Chipset/synthetic
   JSON stays byte-identical.
3. `748c722` — **broadened aggressive tuning grids** in `eval/tuning.py`: symbol-matched scale sets
   + wider `retain_frac` + a 2nd knob per method (verified real config fields: ncc/mosse
   `scales`+`retain_frac`+`nms_iou`, sparse-geo `min_inliers`+`nms_iou`, propose-retrieve
   `similarity_floor`+`nms_iou`, owlv2 `max_box_area_frac`+`query_iou_frac`), plus additive
   size-representative-exemplar selection and exemplar_count {1,3} options (behaviour-preserving
   defaults).
4. `8f8192c` — **dino-dense fixed-size letterbox** (opt-in `DinoDenseConfig.fixed_input_side`,
   multiple of 14, default `None` = unchanged): every scene letterboxed (uniform scale + constant
   pad) into one fixed NxN input so onnxruntime sees a single shape (no per-resolution CUDA arena
   growth); padding tokens masked; boxes map back via the one letterbox scale. Docstring + method
   doc updated.

## Verification

- `pixi run lint` clean, `pixi run typecheck` clean.
- **`pixi run test`: 714 passed, 5 skipped, 93.60% coverage** (≥80% floor holds). All new tests are
  model-free (`ncc`), so CI coverage holds without ONNX weights.

## Deviations (minor, sound)

- Task 2: the internal `gt_records` aggregation carrier is excluded from the `per_image` JSON dumps
  (kept internal) so the chipset/synthetic round-trip equality test stays byte-identical.
- Task 4: the padding sentinel is `-2.0` (not `-1.0`) because the candidate floor clamps to `-1.0`;
  `-2.0` guarantees padded pixels are never foreground.

## Next (manual)

- Aggressive GPU re-run on a fresh vast.ai box using the broadened grids (`fixed_input_side` set for
  dino-dense), producing the enriched, per-slice leaderboard. Confirm cost first (more configs).
