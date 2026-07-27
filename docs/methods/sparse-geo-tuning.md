# Sparse-geo tuning log

A record of the iterative optimization of Method 2 (`sparse-geo`) against the objective
benchmark (`pixi run bench`, 60 labelled images: 10 chipset + 48 textured + 2 synthetic,
IoU 0.5). The method's defaults **are** what the benchmark runs (`spec.config_model()`), so
"tuning the method" here means moving its config defaults and adding one post-processing step.
Everything below was measured on the full 60-image set; changes that did not improve pooled
F1/AP without regressing a regime were reverted.

## Headline result

| metric | before | after | Δ |
|---|---|---|---|
| **F1** (pooled, micro-avg) | 0.720 | **0.823** | +0.103 |
| **mean AP** (macro-avg, EVAL-08 candidate log) | 0.629 | **0.751** | +0.122 |
| precision | 0.729 | **0.884** | +0.155 |
| recall | 0.711 | **0.770** | +0.059 |
| abstentions | 11 | **6** | −5 |

Per-regime F1 (none regressed):

| regime | before | after |
|---|---|---|
| chipset (fixed-scale, near-identical — NCC's turf) | 0.206 | 0.455 |
| synthetic (scatter-scaled, cluttered-distractors) | 0.000 | 0.286 |
| textured-plain | 0.994 | **1.000** |
| textured-cluttered | 0.748 | 0.863 |
| textured-varied | 0.615 | 0.760 |

The cross-method scoreboard (all four methods) is in
[`docs/benchmark/results.md`](../benchmark/results.md) and the rendered
[benchmark report](../reports/benchmark-report.html).

## Method

An in-process sweep harness scored the registered `search` over all 60 images with arbitrary
config overrides, reproducing the benchmark's exact metrics (`match_predictions`,
`precision_recall_f1`, `average_precision`) but in ~10 s instead of ~80 s, so dozens of configs
and combinations could be compared per minute. The winning config was then baked into the module
defaults and re-verified end-to-end through the real `pixi run bench` pipeline (identical
numbers). The harness scripts were scratch tooling and are not committed.

Diagnosis that drove the changes, from the per-image TP/FP/FN and per-regime pooling of the
baseline:

- **A large precision leak from duplicate boxes.** Hough de-duplicates in *pose* space (the 3⁴
  neighbourhood), but two peaks in different pose bins routinely mapped the exemplar to nearly the
  **same** scene box. The benchmark scores that as 1 TP + 1 FP (EVAL-16), so every duplicate was a
  pure precision loss. This was the single biggest lever.
- **Needless abstentions.** The `min_exemplar_keypoints=20` floor abstained on low-texture crops
  (chipset, synthetic) that could still support 4-DoF voting + RANSAC; an abstention is scored as
  recall-0, dragging pooled recall down.
- **Loose localization.** `ransac_thresh_px=5` admitted inaccurate fits that fell short of the
  IoU≥0.5 bar.
- **An over-strict Hough pre-filter.** `min_votes=3` discarded true instances before per-peak
  RANSAC (the real gate is `min_inliers`) ever saw them.

## Changes kept (ranked by impact)

1. **Spatial NMS on verified instances (new).** A final IoU non-maximum suppression over the
   verified boxes, keyed on inlier support, via the shared deterministic `nms` offering
   (`(-score, y, x)` tie-break, so the symmetric-lattice ties stay byte-identical). New config
   field `nms_iou` (default **0.4**). Precision 0.729 → ~0.86 on its own.
2. **`min_votes` 3 → 2.** Surfaces more true instances for RANSAC to verify; precision held
   (NMS + RANSAC still gate), recall and AP rose.
3. **`min_exemplar_keypoints` 20 → 8.** Dropped 3 abstentions; recall and AP up, precision
   unchanged. (The value saturates: any floor ≤ 8 gives the same result — the remaining 6
   abstentions are genuinely keypoint-starved crops.)
4. **`ransac_thresh_px` 5.0 → 3.0.** Tighter reprojection error → better localization → more
   boxes clear IoU 0.5.
5. **`k` 6 → 8.** Higher per-keypoint instance ceiling finds more repeats; NMS absorbs the extra
   duplicate cost.

## Changes tried and reverted (measured worse)

- **k+1 ratio test** (`use_kplus1_ratio`): catastrophic recall loss (F1 0.35, then 0.16 at
  ratio 0.8) — it discards exactly the repeated-instance correspondences the method exists to
  keep, as the module docstring warns.
- **`pairwise-4dof` voting**: precision collapsed to ~0.09.
- **AKAZE backend**: keypoint yield too low here (F1 0.09, 53 abstentions).
- **`min_votes=4`** as a precision route: unnecessary once NMS was in — it cost recall for a
  precision gain NMS already delivered.
- **`min_inliers` up/down, tighter `ransac_thresh` (2–2.5), `ransac_iters=400`**: neutral or
  slightly negative; not kept.

## What did NOT change, and why

The method's **character** is unchanged: SIFT default, standard Lowe ratio test still disabled,
generalized-Hough decomposition, NumPy per-peak RANSAC with the real `config.seed`, mirror/scale
degeneracy rejection, and the documented NCC-vs-sparse-geo crossover (sparse-geo remains the
scale-robust method; chipset stays hard and is *supposed* to be). These are tuning changes to the
existing pipeline, not a redesign — and every one is a principled default shift or a standard
detection post-processing step, not a benchmark-specific hack. Gains are spread across regimes.
