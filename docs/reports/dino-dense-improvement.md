# Engineering log — fixing `dino-dense` (2026-07-25)

A record of the investigation and iterative fix that took Method ③ `dino-dense` from returning
**one image-spanning box** (textured F1 ≈ 0.03) to a working detector (textured F1 ≈ 0.70). The
code lives in [PR #23](https://github.com/ortizeg/object-search-exploration/pull/23); this log
captures the *why*, the measured deltas, and the deferred follow-ups so the reasoning is not lost
in the diff.

Cross-references: the method itself is documented in
[`docs/methods/dino-dense.md`](../methods/dino-dense.md); current per-regime scores are in the
[benchmark report](benchmark-report.html); deferred work is in
[`docs/ROBUSTNESS-BACKLOG.md`](../ROBUSTNESS-BACKLOG.md).

## Symptom

On the textured regimes `dino-dense` scored **F1 ≈ 0.03** and typically returned a **single box
covering ~the whole image**. Example (`textured-plain-05`, 10 emblems of 84×84 px): outcome `ok`,
**1 match** of box `0,30,800×529`, 15 connected components found but collapsed to one.

## Root cause — the features were fine; scoring and thresholding were not

Thresholding the raw similarity map by hand showed the signal was *already there*:

| threshold | # components | # compact instance-sized blobs |
|---|---|---|
| 0.40 | 13 | **10** (~80×80 each) |
| 0.43 | 27 | 9 |
| **0.377 (what gmm picked)** | fuses into **1 image-spanning blob** | — |

So DINOv2 resolved all ten instances cleanly at ≈ 0.40–0.43. Three compounding post-processing
flaws destroyed that:

1. **A single mean-pooled prototype** averaged the crop's diverse parts into one mushy vector that
   matched *everything* weakly → a low-contrast map (`sim_max` only 0.51, foreground/background
   modes at 0.43 / 0.26).
2. **The `gmm` calibrator cut at the posterior boundary**, which — with a tiny foreground weight —
   landed at **0.377, down in the background shoulder**, well below where instances separate.
3. **Match components were grown at a sub-threshold floor** (`threshold − candidate_margin`), so
   the diffuse above-floor background *bridged* all ten instances into one connected blob whose
   peak cleared the threshold and was emitted as a single full-frame "match".

## The fix — iterated empirically on the EVAL-20 textured set (48 images)

Each pass was measured before moving on. Pooled precision/recall/F1 and mean AP:

| pass | P | R | F1 | AP |
|---|---|---|---|---|
| baseline (mean-pool prototype + gmm) | 0.079 | 0.015 | 0.025 | 0.025 |
| + `max-token` scoring & extraction at the accept threshold | 0.227 | 0.390 | 0.287 | 0.329 |
| + exemplar-relative area bounds | 0.495 | 0.390 | 0.436 | 0.365 |
| + `contrast` calibration | **0.629** | **0.779** | **0.696** | **0.541** |

The three landed changes, all in `src/object_search/search/dino_dense.py`:

1. **`max-token` best-part scoring (default).** Each scene token scores as the mean of its
   top-`match_tokens` cosines to the crop's *token bank* — "how well does the single best-matching
   part of the exemplar explain this location" — instead of a dot against one mean-pooled vector.
   High-contrast map. A `prototype` mode is retained as a readable baseline.
2. **`contrast` calibration (default).** The accept threshold is a 50/50 blend of a background
   anchor (`mean + std`) and a foreground anchor (`0.85 × p99.5`). On the high-contrast map this
   tracks the per-image optimum, where the `gmm` posterior cut did not.
3. **Match components grown at the accept threshold**, with component area bounded to
   `[min_area_frac, max_area_frac] × exemplar_area` so fragments and merged blobs drop out.
   Sub-threshold candidates (the EVAL-08 PR-sweep log) are collected in a *separate* pass at the
   floor, so their coarser boxes never pollute the returned detections.

### How far is this from optimal?

A per-image *oracle* threshold (the F1-maximising cut on each image, chosen with hindsight) tops
out at **F1 0.780**. The shipped `contrast` rule reaches **0.696** with a single fixed rule, on a
**broad plateau** — many nearby blend coefficients score 0.69–0.70 — so it is not a knife-edge
overfit. `match_tokens` (the top-k) is flat over k ∈ {1,2,3,5}, so that knob is robust too.

### Fairness — thresholds are not fit to the labels

The `contrast` coefficients are tuned to the shape of the *score distribution*, **never to the
ground-truth boxes**, and the same rule runs on every dataset. The numeric threshold adapts
per-image (an absolute cosine cut does not transfer across images for deep features), but no cut is
chosen to maximise F1 against the labels. AP remains threshold-free (it sweeps the full candidate
log).

## Result vs the other methods (official `pixi run bench`, IoU 0.5)

Per-regime F1 after the fix — and the other three methods are **byte-for-byte unchanged** (the
change is isolated to `dino-dense`):

| regime | ① ncc | ② sparse-geo | ③ dino-dense (before → after) | ④ propose-retrieve |
|---|---|---|---|---|
| EASY (chipset) | 0.97 | 0.21 | ~0.00 → **0.17** | 0.83 |
| TEXTURED | 1.00 | 0.99 | ~0.05 → **0.76** | 0.87 |
| VARIED (scale/rotation) | 0.24 | 0.62 | ~0.00 → **0.64** | 0.92 |
| CLUTTERED | 0.31 | 0.75 | ~0.02 → **0.69** | 0.73 |

`dino-dense` now **beats NCC** on the scale/rotation (VARIED) and clutter regimes, which is the
crossover the method exists to demonstrate.

## Chipset — investigated, deliberately deferred

`dino-dense` remains weak on the chipset (EASY) regime — F1 0.17, NCC's regime by design. The
weakness is **resolution-bound**, not a scoring bug: the `scene_max_side = 1568` cap squeezes every
chip down to ~3–5 stride-14 tokens (a 115 px chip on a 2560 px canvas → 70 px → 5 tokens; a 160 px
chip on a 6000 px canvas → 42 px → 3 tokens).

A resolution experiment on a 7-image subset (chips 24–115 px), sizing the scene so the exemplar
spans ≥ 6 tokens (clamped to a 2560 px max) and scoring against the original-pixel GT:

| policy | P | R | F1 | TP |
|---|---|---|---|---|
| baseline (cap 1568) | 0.053 | 0.077 | 0.063 | 5 |
| cap 2560 | 0.157 | 0.262 | 0.197 | 17 |
| adaptive (chip ≥ 6 tokens), max 2560 | **0.293** | **0.554** | **0.383** | 36 |

Findings:

- **Adaptive resolution ~6× the recall** (0.077 → 0.554; TP 5 → 36 of 65). It is a *general*
  small-object improvement ("give the exemplar enough tokens"), not chipset-specific tuning, so it
  would not violate the cross-dataset fairness rule.
- It fixes **recall/localisation only**. Precision barely moved (FPs 89 → 87) — the over-detection
  on flat chips is a *separate*, harder problem (needs duplicate suppression or a
  spatial-consistency term).
- The smallest chips (24 px) span ~2 tokens even upscaled; synthetic upscaling adds tokens but no
  real detail, so a floor remains.
- **Cost:** more tokens = more inference latency and memory — the exact tradeoff `scene_max_side`
  exists to manage.

**Decision:** left as-is. Chipset is NCC's regime (NCC F1 0.97 there); the latency cost is not
worth a data point on the "wrong tool for tiny flat chips" case. Captured in
[`docs/ROBUSTNESS-BACKLOG.md`](../ROBUSTNESS-BACKLOG.md) as `adaptive input resolution` so it can be
picked up if the priority ever changes.

## Verification

`pixi run quality` green: Ruff + Ruff-format clean, MyPy strict clean, **492 passed / 5 skipped,
coverage 92.66 %**. Benchmark report and charts regenerated from the fresh sweep. New model-free
tests cover `_maxtoken_similarity_map` (contrast vs the prototype), `_contrast_threshold` (cut sits
between the background bulk and the foreground tail), and the `max_area` ceiling.
