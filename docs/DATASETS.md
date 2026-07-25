# Datasets

Every benchmark dataset in this repo is **synthetic with exact ground truth by construction** —
each instance is pasted by the generator at a known rectangle, so precision, recall, and AP are
computable with no hand-labeling and no licensing. The exception is the basketball frames (real,
rating-only). Each set is deterministic from a seed and regenerates identically.

The sets are deliberately *stratified*: each isolates the regime where a particular method should
win, so the four-method comparison spans easy and hard cases rather than one flattering slice.
See the [benchmark report](reports/benchmark-report.html) for the per-regime results and
[`EVAL-DESIGN.md`](EVAL-DESIGN.md) for the full statistical protocol.

## Overview

| Dataset | Images | Exact GT | Regenerate | What it tests |
|---|---|---|---|---|
| **chipset** (EVAL-19) | 10 | ✅ | `pixi run chipset` | Near-identical, fixed-scale repeats — the NCC-favourable baseline |
| **textured** (EVAL-20) | 48 | ✅ | `pixi run textured` | Texture + scale/rotation/brightness variation — favours keypoints & deep features |
| **synthetic** (EVAL-03) | ~8 | ✅ | `pixi run synth` | Lattices, clutter, distractors, scale/rotation — general shape scenes |
| **markers** (Milestone 2) | 3 | ✅ (tip/dir) | `pixi run markers` | Arrow/dot markers for the marker-conditioned exploration |
| **basketball** | 3 | ✖ (rating-only) | — | Real broadcast frames with genuine texture; no box GT |

Licensing and provenance for every committed file are in [`assets/demo/LICENSES.md`](../assets/demo/LICENSES.md).

---

## chipset — the fixed-scale baseline (EVAL-19)

Ten images, canvas sizes ramping **320×240 → 6000×4000**. Each image gets **one distinct,
randomly-generated textured chip** pasted `N ∈ {5, 10, 15}` times at strictly non-overlapping
positions on a **white** background. Instances are **identical, axis-aligned, and fixed-scale** —
the regime where template matching (Method ① NCC) is genuinely hard to beat, and where the
keypoint method (② sparse-geo) starves for lack of texture and abstains.

- **What it tests:** near-identical repeated-instance detection at a single scale; over-detection
  on dense grids; latency scaling across a 300× range of canvas sizes.
- **Ground truth:** the paste rectangle of every chip, pairwise IoU 0. The sidecar records the
  **achieved** count (never the requested `N`), the chip seed, and the exemplar index.
- **In the report:** the **EASY** regime.
- Source: `object_search.synthetic.chipset`.

## textured — texture and appearance variation (EVAL-20)

Forty-eight images across **three regimes**, each a richly-textured "emblem" (a distinct
procedural object) pasted several times with exact, non-overlapping ground truth. Every emblem is
built to yield **≥ 20 SIFT keypoints** (min 27, mean 62 across the set) — the load-bearing
property that lets ② sparse-geo *engage* here instead of abstaining as it does on the flat chips.

| Regime | Variation | Favours |
|---|---|---|
| `textured-plain` | fixed scale & rotation | ② sparse-geo (keypoints, no confounds) |
| `textured-varied` | scale 0.6–1.6×, rotation ±35°, brightness ±25% | ③ dino-dense, ④ propose-retrieve |
| `textured-cluttered` | mild variation + noisy background + distractors | precision under clutter (all) |

- **What it tests:** the crossover the chipset cannot show — that ② sparse-geo and the deep-feature
  methods hold up under texture and geometric variation where ① NCC collapses.
- **Ground truth:** the axis-aligned bounding box of the **transformed** emblem (from the warped
  corners, so a rotated instance is boxed by its true extent), pairwise IoU 0, achieved-count
  recorded. Distractors (a *different* emblem) are drawn but excluded from GT — genuine
  false-positive bait. The sidecar also records the observed scale/rotation ranges.
- **In the report:** the **TEXTURED / VARIED / CLUTTERED** regimes.
- Source: `object_search.synthetic.textured`.

## synthetic — general shape scenes (EVAL-03)

The original generator: shapes (rect / triangle / circle / plus / chevron) laid out as a
**lattice** or **scatter**, with optional scale jitter, rotation jitter, position jitter, clutter,
and distractors. Used for the committed sample-run gallery and as scale-varied scenes in the full
sweep (`scatter-scaled`, `cluttered-distractors`).

- **What it tests:** touching-instance separation (the `lattice-touching` scene proves `local-max`
  peak extraction beats plain NMS), calibration behaviour, and moderate scale/rotation variation.
- **Ground truth:** the AABB of each drawn (possibly rotated) shape; distractors excluded.
- Source: `object_search.synthetic.generator`.

## markers — marker-conditioned exploration (Milestone 2)

Three images of repeated **markers** — arrows, dots, carets — each with a sidecar recording every
marker's exact **tip, direction, and centroid** (and, for the arrow-with-targets image, the object
each arrow points at). This is the ground truth for the marker-conditioned exploration, which
finds every marker and boxes the object it indicates.

- **What it tests:** marker instance detection plus orientation recovery (tip + direction) and the
  proposal-scoring that resolves what a marker points at.
- Source: `object_search.synthetic.generator` (`synthesize_markers`).

## basketball — real frames (rating-only)

Three real broadcast frames copied from the sibling `basketball-2d-to-3d` project. They carry
**genuine natural texture** (players, jerseys, court logos), which is where ② sparse-geo's keypoint
matching is at its best — but they have **no exact box ground truth**, so they are for qualitative
inspection and human rating (thumbs / per-match verdicts), not for precision/recall. Licence:
recorded in `LICENSES.md` as non-redistributable.

---

*All synthetic sets share one sidecar format (`<image_id>.gt.json`), so a single loader
(`object_search.eval.labels`) serves the metric layer and the benchmark. The chipset and textured
sets are consumed by `pixi run bench`; the report groups results by dataset/regime.*
