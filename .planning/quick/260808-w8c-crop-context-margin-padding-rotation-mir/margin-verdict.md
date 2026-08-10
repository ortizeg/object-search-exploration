# Task 2: crop context-margin sweep — verdict

Inference-only sweep (zero retraining) of `Owlv2OneshotConfig.crop_context_margin_frac` against the
already-exported `owlv2_base_patch16_floorplans_ft_contrastive_crop.onnx` checkpoint, crossed with
owlv2-oneshot's existing tuning grid (`max_box_area_frac` × `query_iou_frac`), through the same
`run_domain_tuning` tune-on-val/freeze/report-on-test methodology every other arm was measured with.

Run on a vast.ai RTX 3090 (CUDAExecutionProvider, onnxruntime-gpu 1.23.2) rather than locally — a
local CPU run of this same sweep was tried first and killed after 35+ minutes with the first cell
still incomplete (each of the 8 (margin, dataset) cells re-runs the full 9-entry grid over the whole
56-image val split plus tuned+default test evaluation on CPU, which is far more compute than the
earlier lightweight single-exemplar diagnostics). The GPU run completed all 8 cells in ~34 minutes.

## Regression check (D-w8c-04)

The `margin=0.0` cell must reproduce 260808-dla's already-committed `contrastive-crop` tuned F1
exactly on both datasets — it does:

- `floorplans-door` margin=0.0: **0.22902990517870164** (committed: 0.22902990517870164) — match.
- `floorplans-window` margin=0.0: **0.21586263286999183** (committed: 0.21586263286999183) — match.

The `grids` plumbing is trustworthy; the nonzero-margin numbers below are real measurements, not an
artifact of a silently-ignored override.

## The four margins × two datasets

| margin | door tuned F1 | window tuned F1 |
|---|---|---|
| 0.0 | 0.229 | 0.216 |
| 0.15 | **0.277** | 0.186 |
| 0.3 | 0.230 | 0.158 |
| 0.5 | 0.151 | 0.153 |

Door peaks at margin=0.15 (+21% over margin=0.0) then declines. Window degrades monotonically with
every nonzero margin tried (−14% at 0.15, growing worse through 0.5). No margin value beats 0.0 on
**both** classes simultaneously — a genuine split result, not rounded to a winner.

## Verdict

**No margin beats 0.0 on both classes.** Per D-w8c-05, the final GPU arm (`contrastive-crop-v2`)
trains and evaluates at **margin 0.0** and tests the rotation/mirror augmentation lever alone.

This is a real, reportable finding in its own right: crop context-margin padding is not a free win.
It measurably helps door detection at a moderate margin (0.15) — plausibly because door symbols
benefit from surrounding wall context — but measurably hurts window detection at every margin
tried, plausibly because window symbols are already distinctive enough that added context pulls in
confusable neighboring geometry (wall lines, other symbols) rather than useful signal. The
per-class asymmetry is itself worth recording in the report, separately from the go/no-go decision
for the combined GPU arm.
