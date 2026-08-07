# Datasets

Every benchmark dataset in this repo is **exact ground truth by construction** — each instance is
pasted by the generator at a known rectangle, so precision, recall, and AP are computable with no
hand-labeling and no licensing. Most sets are fully synthetic (drawn/rendered pixels); the
**real-objects** set pastes real, segmented photographic cutouts onto real photographic
backgrounds, so ground truth stays exact while the pixels themselves are real. The exception is
the basketball frames (real, rating-only, no exact GT). Each set is deterministic from a seed and
regenerates identically.

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
| **real-objects** | 30 | ✅ | `pixi run fetch-real-photos` + `pixi run real-objects` | Real segmented object photos pasted onto real background photos, in three regimes — real texture/lighting, no synthetic render |
| **basketball** | 3 | ✖ (rating-only) | — | Real broadcast frames with genuine texture; no box GT |

Licensing and provenance for every committed file are in [`assets/demo/LICENSES.md`](https://github.com/ortizeg/object-search-exploration/blob/main/assets/demo/LICENSES.md).

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

## real-objects — real photo insertion

Every other set above is drawn or rendered by this repo; this one pastes **real, segmented object
photos** onto **real background photos**, so ground truth stays exact by construction while the
pixels themselves carry genuine photographic texture, lighting, and noise that no renderer
produces. It addresses the same underlying gap the "Generic repeated-instance photos" TODO in
`assets/demo/LICENSES.md` has carried since Phase 1 — real, clearly-licensed repeated-instance
photos — with a different, generic-object approach rather than the four specific categories that
TODO named (those stay open). Every source photo comes from Wikimedia Commons with an individually
recorded title/author/licence/source-URL, not a blanket dataset licence.

Ten object categories (9 everyday objects plus one deliberate stress object) × three
**regimes**, mirroring `textured.py`'s stratification exactly so the same orientation/scale
exploration exists on real pixels, not just a rendered emblem — **30 images**, large enough for
the plain/varied/cluttered comparison to be meaningful, small enough that the sweep cost stays
comparable to the existing chipset(10) + textured(48) footprint. (Four sourced object photos —
a coffee mug, a padlock, a pinecone, a rubber duck — were dropped after manual review, before the
final segmentation fixes below landed: each was a low local-contrast photo where FastSAM's
automatic mode produced a fragmented or wrong mask. A golf ball was sourced as a replacement to
hold the set at ten.)

| Regime | Background | Variation | Favours |
|---|---|---|---|
| `real-plain-*` | clean (`REAL_BACKGROUND_MANIFEST`) | fixed scale & rotation | ① ncc / ② sparse-geo (keypoints, no confounds) |
| `real-varied-*` | clean | scale 0.25–1.6×, rotation ±30° | ③ dino-dense, ⑤ propose-retrieve |
| `real-cluttered-*` | **busy** (`REAL_BUSY_BACKGROUND_MANIFEST`) | scale 0.3–1.5×, rotation ±20°, + a pasted-but-unrecorded distractor object | precision under real clutter (all) |

`real-cluttered-*` stacks three distinct sources of difficulty deliberately, mirroring
`textured.py`'s `noisy_background` regime: a genuinely busy real photo (leaf litter, a patterned
mosaic tile floor — high edge density, not just a uniform texture), scale/rotation jitter, and a
pasted-but-unrecorded distractor object. `plain`/`varied` stay on the clean, uniform backgrounds
(wood floor, concrete, grass, ...) so they isolate pose variation without also confounding it with
background difficulty — a plain background makes the pasted object trivially separable for every
method, which is a deliberate easy baseline, not the whole story.

`varied`/`cluttered`'s scale range spans roughly 5%–34% of the working canvas (55px–350px long
edge against the 1024px canvas), so every `varied`/`cluttered` image already tests large AND small
instances of the same object side by side, rather than needing a separate size-only regime — a
single image can have one 60px apple next to a 280px one. `plain` stays fixed at the cutout's
native size (no scale confound), matching the pose-isolation rationale above.

Each object is segmented out of its own source photo with FastSAM (automatic "everything" mode —
already Trial-approved, `docs/library-reviews/fastsam.md`). Getting a clean, tight, real-photo
cutout took several passes, each fixing a distinct measured failure: `_select_object_proposal`
weights FastSAM's own `objectness` score cubed (a plain product-photo backdrop routinely came back
as the single *largest*, best-centred proposal); an edge-to-edge-spanning proposal is excluded
outright (a second, independent "background included" signature objectness alone didn't catch);
`_select_merge_partners` unions in a touching, similarly-confident, *mostly disjoint* proposal
(FastSAM split a screwdriver into a handle proposal and a separate shaft proposal — mutually
adjacent, near-identical confidence — and the single-winner ranking above only kept one; the
"mostly disjoint" condition is load-bearing, since without it several near-duplicate/overlapping
proposals get wrongly merged in too, pulling in background); the thresholded mask is eroded by one
pass to trim the background-colour fringe a soft real edge always leaves, then only its largest
connected component survives (drops small disconnected noise, e.g. a ruler's alternating light/dark
squares crossing the alpha threshold as separate specks); and `extract_cutout` gates the result on
both mask solidity (catches fragmentation) and frame coverage (catches "background included")
before falling back through a small confidence-threshold ladder.

**Known remaining imperfection:** the ping-pong-ball cutout still carries a small dark wedge near
one edge, from a soft shadow in its source photo that FastSAM's own mask fused directly into the
ball (a single connected region, not a separate proposal or disconnected speck, so none of the
fixes above apply to it). Investigated and rejected as unfixable-without-regression: Otsu adaptive
thresholding (barely shifts the global threshold, the shadow population isn't separable that way),
morphological opening at several kernel sizes (the shadow region is too thick relative to the ball
to disconnect without also eroding the ball itself), two GrabCut seedings (didn't reject the shadow
color), and a luminance-ratio mask refinement (removed the shadow but also amputated the claw
hammer's dark rubber grip when tested against the rest of the set — rejected). A replacement source
photo was also evaluated and rejected: cleaner background, but FastSAM split the ball into two
overlapping half-proposals that no safe overlap/IoU threshold could merge without reopening the
tennis-ball/hockey-puck/chess-pawn/hammer false-merge regression (see `_select_merge_partners`'s
docstring). Left as a documented minor cosmetic artifact on one of ten objects rather than risking
the other nine.

Cutouts are cached once, then pasted onto a background photo several times per regime at strictly
non-overlapping positions. `real-cluttered-*` additionally pastes a *different* object category as
a distractor — drawn but never recorded, the same false-positive-bait convention as
chipset/textured.

- **What it tests:** the same orientation/scale crossover the textured regimes prove
  synthetically, but on real photographic pixels — including a deliberately **textureless,
  rotationally symmetric stress object** (a plain white ping-pong ball) expected to trip `ncc`'s
  low-variance guard and `sparse-geo`'s SIFT-keypoint floor exactly as the synthetic sets predict,
  but on a real photo rather than a flat rendered chip.
- **Where it sits in the difficulty ramp:** synthetic/chipset/textured are rendered and fully
  controlled; the research floor-plan set (`docs/eval/research-datasets.md`) is the real target
  domain — dense, small, low-contrast line-drawing symbols. This set is the proof-of-concept rung
  in between: real photographic pixels (texture, lighting, JPEG noise a renderer never produces),
  with `plain`/`varied` isolating pose variation the way the synthetic sets do and `cluttered`
  adding a genuinely busy real background. A method that only works on chipset/textured has not
  yet proven it survives real pixels at all; a method that also holds up here is a much better bet
  to generalize toward floor plans than one tuned on synthetic data alone.
- **Ground truth:** the AABB of the *pasted, warped* alpha mask (not the cutout's nominal size —
  same rule as textured's rotated-shape AABB), achieved-count recorded, distractors excluded.
- **In the report:** grouped with the full (non-CI) sweep, alongside chipset/textured.
- Source: `object_search.synthetic.real_insertion`. Regenerate with `pixi run fetch-real-photos`
  (downloads the licensed source photos) then `pixi run real-objects` (segments + composites;
  needs the `fastsam-s` weight, `pixi run -e export fetch-models --only fastsam-s`).
- **Dedicated report:** [`real-objects-report.html`](reports/real-objects-report.html) scores all
  six methods on this set alone (`pixi run bench-real-objects` then `pixi run report-real-objects`);
  [`real-objects-findings.md`](reports/real-objects-findings.md) compares those numbers against the
  synthetic benchmark and explains where and why they diverge.

## basketball — real frames (rating-only)

Three real broadcast frames copied from the sibling `basketball-2d-to-3d` project. They carry
**genuine natural texture** (players, jerseys, court logos), which is where ② sparse-geo's keypoint
matching is at its best — but they have **no exact box ground truth**, so they are for qualitative
inspection and human rating (thumbs / per-match verdicts), not for precision/recall. Licence:
recorded in `LICENSES.md` as non-redistributable.

---

*All synthetic sets share one sidecar format (`<image_id>.gt.json`), so a single loader
(`object_search.eval.labels`) serves the metric layer and the benchmark. The chipset, textured, and
real-objects sets are consumed by `pixi run bench`; the report groups results by dataset/regime.*
