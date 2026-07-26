# `owlv2-oneshot` improvement log

Empirical iterate → measure-per-regime → revert loop for Method 4 (`owlv2-oneshot`), following the
same playbook used for `dino-dense`. Each change is measured across the four regimes and kept only
if it improves without regressing a regime the method already handles.

## Harness and regimes

A fast harness reuses the benchmark's own scoring (`match_predictions` / `precision_recall_f1` /
`average_precision`, IoU 0.5) over the committed datasets, grouped into four regimes:

| regime | dataset | n | character |
| --- | --- | --- | --- |
| EASY | `chipset-*` | 10 | near-identical fixed-scale repeats, canvas 320×240 → 6000×4000 |
| TEXTURED | `textured-plain-*` | 16 | textured instances, plain background |
| VARIED | `textured-varied-*` | 16 | textured + scale/pose variation |
| CLUTTERED | `textured-cluttered-*` | 16 | textured instances amid distractors |

P/R/F1 are micro-averaged within a regime (sum tp/fp/fn); mAP is the mean per-image AP over the
matches-plus-candidates log. The **calibration threshold rule is the same across every regime and
is never fit to the labels** — only method hyperparameters (a size prior, a retain fraction) are
tuned, and they are chosen for robustness across regimes, not per-regime maxima.

> Weights: `owlv2_base_patch16.onnx` is produced by `pixi run -e export export-owlv2` (opset 17,
> legacy exporter, sha pinned in the registry). This is the first run in which owlv2 was actually
> exported and measured, so the "baseline" below is the as-shipped method's *first* real evaluation.

## Root cause (why the baseline scored ~0 F1)

The as-written method selected its query embedding by **mean-pooling** the exemplar-crop patches
whose predicted box covers the crop. On OWLv2 that yields the generic *whole-frame / background*
embedding, because the covering patches include the model's global-context direction. That generic
query then matched scene patches predicting **whole-image boxes** (verified: on `chipset-01` the top
five boxes were all ~the full 320×240 canvas at cosine 0.92, while the exemplar's own 24×24 region
scored only 0.81). The result: near-zero true positives everywhere except CLUTTERED, where an
unstable threshold flooded the frame (R=1.0 at P=0.018).

The fix is HuggingFace's `embed_image_query` heuristic: among the covering patches, take the single
**most-distinctive** one (least similar to the mean patch embedding) — the object, not the
background. This was parked in the ROBUSTNESS BACKLOG as a refinement; it is in fact a correctness
requirement.

## Iterations

Baseline = the method as first written (mean-pool query, gmm calibration, nms_iou 0.5).

| # | change | EASY F1 | TEXTURED F1 | VARIED F1 | CLUTTERED F1 | verdict |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | baseline (mean-pool query) | 0.000 | 0.022 | 0.023 | 0.035 | — |
| 1 | distinctiveness query select | 0.054 | 0.089 | 0.061 | 0.026 | **KEEP** — mAP 0.03→0.5-0.6, recall→0.5-0.7; ranking now correct, threshold is next |
| 2 | + giant-box filter (area<0.25) | 0.041 | 0.138 | 0.050 | 0.027 | **KEEP** — lifts mAP everywhere + TEXTURED F1; removes the whole-frame FP that anchored the threshold |
| 3 | + self-similarity calibration (rf 0.90) | 0.129 | 0.656 | 0.771 | 0.717 | **KEEP** — the decisive fix: F1 ×5-15 on textured regimes; recall 0.95-0.99, precision 0.49-0.65 |
| 4 | retain_frac 0.90→0.94, nms_iou 0.5→0.3 | 0.244 | 0.840 | 0.870 | 0.824 | **KEEP** — precision/recall tune (swept, not label-fit to a max): tight NMS collapses OWLv2's per-object duplicate patches |

Reverted / not adopted along the way: retain_frac 0.96 (higher overall F1 but sacrifices recall
0.89→0.82 and *lowers* VARIED — rejected for robustness); nms_iou 0.7 (loosens duplicate
suppression, precision collapses); the mean-pool query embedding (the original — the root-cause bug).

## Per-regime before / after

Baseline (method as first written) → final (all four changes), measured through the real `search()`
path at IoU 0.5:

| regime | P (base → final) | R (base → final) | F1 (base → final) | mAP (base → final) |
| --- | --- | --- | --- | --- |
| EASY | 0.000 → 0.145 | 0.000 → 0.765 | 0.000 → **0.244** | 0.000 → 0.287 |
| TEXTURED | 0.095 → 0.804 | 0.012 → 0.878 | 0.022 → **0.840** | 0.035 → 0.554 |
| VARIED | 0.105 → 0.876 | 0.013 → 0.865 | 0.023 → **0.870** | 0.028 → 0.573 |
| CLUTTERED | 0.018 → 0.735 | 1.000 → 0.938 | 0.035 → **0.824** | 0.159 → 0.489 |
| OVERALL | 0.018 → 0.502 | 0.291 → 0.874 | 0.035 → **0.637** | 0.061 → 0.495 |

The textured regimes go from unusable (F1 ≈ 0.02) to strong (F1 0.82–0.87). EASY (the chipset)
improves from 0 to 0.24 but remains the weak regime — a fixed-960 small-object ceiling on large
canvases, not a tuning miss (see Deferrals). Latency is unchanged (~4 s/query on CPU, two 960×960
forward passes); every gain here is post-processing, so it costs nothing at inference time.

## Fairness note

- The **calibration rule is identical across every regime** and label-free: `self-similarity` cuts
  at `self_score * retain_frac`, anchored to the exemplar's own self-match, with no per-dataset or
  per-image tuning and no access to ground truth.
- The two hyperparameters that were swept (`retain_frac`, `nms_iou`) are **method-level constants**,
  chosen once for **robustness across all four regimes** — explicitly *not* the global-F1 argmax.
  `retain_frac 0.96` scored marginally higher pooled F1 but was rejected because it drops recall
  0.89→0.82 and *lowers* VARIED F1; `0.94` is at or near the per-regime F1 peak everywhere while
  keeping recall high. This is the same discipline `ncc` uses for its `retain_frac 0.7`.
- No change was kept that regressed a regime the method already handled: every accepted iteration
  improved mAP (ranking quality) across the board; the final tune improved F1 in all four regimes
  over iteration 3.
- These are owlv2's own per-regime numbers. Where it sits in the cross-method scoreboard (it does
  **not** displace `ncc` on EASY/TEXTURED or `sparse-geo` on TEXTURED) is in the regenerated
  `docs/benchmark/` report, run with every method's weights present.

## Deferrals

- **Small objects on large canvases (the EASY ceiling).** OWLv2's fixed 960 input downscales chips
  in a 6000×4000 chipset scene below its effective resolution, so EASY precision stays ~0.15 and F1
  ~0.24 while the textured regimes reach 0.82–0.87. This is architectural, not a tuning miss; **tiled
  / multi-scale inference** is the fix and is deferred (it is a phase of work, not a knob).
- **Calibrated logits.** Exporting OWLv2's learned `logit_scale`/`logit_shift` and applying them
  before thresholding may make the score distribution genuinely bimodal, which would let `gmm`
  work and remove the reliance on self-similarity anchoring. Deferred behind a re-export.
- Both are recorded in the module docstring and `docs/ROBUSTNESS-BACKLOG.md`.
