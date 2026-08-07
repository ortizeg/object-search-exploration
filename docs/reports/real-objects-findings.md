# Real-objects vs synthetic findings — do the synthetic numbers predict real photos?

This records a real comparison of the six search methods on **real photographic pixels**
(`real-objects` — 30 real, segmented object photo cutouts pasted onto real background photos,
exact ground truth by construction) against the existing **fully synthetic** benchmark
(chipset + textured, 58 images, rendered/drawn pixels). Both are exemplar-search benchmarks with
matching regime structure (fixed pose → scale/rotation variation → clutter), so the comparison
asks a specific question: **do the synthetic per-regime numbers predict what happens on real
pixels, and where do they mislead?**

**How it was produced.** `pixi run bench` (unchanged default: chipset + textured +
real-objects + the two scale-varied synthetic scenes, 90 images × 6 methods) and
`pixi run bench-real-objects` (new, real-objects-only, 30 images × 6 methods) were both run
locally, CPU-pinned, method defaults (no tuning), IoU 0.5. Both swept every cell with full
coverage (`images_labelled == images_requested`, zero `error` outcomes on either run). The
numbers below pool `results.json`'s chipset+textured rows ("SYNTHETIC") against
`real-objects-results.json` in full ("REAL"); see [`real-objects-report.html`](real-objects-report.html)
for the bootstrap-CI'd version of the real-objects numbers alone, and
[`benchmark-report.html`](benchmark-report.html) for the synthetic side.

**Regime mapping.** `real-objects` has three regimes (`PLAIN`/`VARIED`/`CLUTTERED`); the
synthetic side has four (`EASY`/`TEXTURED`/`VARIED`/`CLUTTERED`). `real-plain-*` plays both the
`EASY` and `TEXTURED` roles at once — per `DATASETS.md`, every real cutout already carries genuine
photographic texture, so there is no flat-chip-vs-textured-emblem split to make on real pixels. The
tables below line `PLAIN` up against **both** `EASY` and `TEXTURED` for that reason.

## Headline: overall pooled (all regimes)

| method | synth F1 | synth AP | real F1 | real AP | ΔF1 | rank (synth → real) |
|---|---|---|---|---|---|---|
| `propose-retrieve` | 0.908 | 0.598 | **0.868** | 0.626 | −0.039 | 1st → 1st |
| `sparse-geo` | 0.833 | 0.772 | **0.786** | 0.740 | −0.047 | 2nd → 2nd |
| `mosse` | 0.809 | 0.795 | 0.544 | 0.541 | −0.265 | 3rd → 4th |
| `ncc` | 0.807 | 0.784 | 0.509 | 0.462 | −0.298 | 4th → 5th |
| `dino-dense` | 0.584 | 0.488 | 0.576 | 0.456 | **−0.008** | 6th → 3rd |
| `owlv2-oneshot` | 0.637 | 0.495 | **0.103** | 0.394 | **−0.534** | 5th → 6th |

Two things are immediately visible and neither is what a synthetic-only report would predict:

1. **The podium is stable, the bottom half reshuffles.** `propose-retrieve` and `sparse-geo` stay
   1st/2nd on both surfaces — the strongest evidence in this report that synthetic numbers *do*
   transfer for methods whose matching is scale-and-texture-robust by construction (region
   proposals, SIFT keypoints). Below that, `dino-dense` and `owlv2-oneshot` **trade places**:
   `dino-dense` is the synthetic benchmark's worst method (killed by the chipset's flat white
   background — see below) but lands mid-pack on real photos, while `owlv2-oneshot` is
   solidly mid-pack synthetically and **collapses** to a distant last on real pixels.
2. **`owlv2-oneshot`'s drop (ΔF1 −0.534) is an order of magnitude larger than any other method's**
   and is not explained by ordinary photographic noise — see §"The owlv2-oneshot collapse" below
   and [`owlv2-oneshot-real-objects-improvement.md`](owlv2-oneshot-real-objects-improvement.md).

## Per-regime comparison

**PLAIN (real) vs EASY / TEXTURED (synthetic) — clean background, fixed pose:**

| method | EASY F1 | TEXTURED F1 | PLAIN F1 (real) |
|---|---|---|---|
| `ncc` | 1.00 | 1.00 | 0.85 |
| `mosse` | 0.92 | 1.00 | 0.91 |
| `sparse-geo` | 0.45 | 1.00 | **0.99** |
| `dino-dense` | **0.17** | 0.76 | 0.60 |
| `propose-retrieve` | 0.93 | 0.96 | 0.98 |
| `owlv2-oneshot` | 0.24 | 0.84 | 0.20 |

**VARIED (both) — scale + rotation variation, clean background:**

| method | synth VARIED F1 | real VARIED F1 | Δ |
|---|---|---|---|
| `ncc` | 0.46 | 0.29 | −0.17 |
| `mosse` | 0.50 | 0.31 | −0.19 |
| `sparse-geo` | 0.76 | 0.67 | −0.09 |
| `dino-dense` | 0.64 | 0.54 | −0.10 |
| `propose-retrieve` | 0.94 | 0.72 | −0.22 |
| `owlv2-oneshot` | 0.87 | 0.08 | **−0.79** |

**CLUTTERED (both) — pose variation + a busy background + a distractor:**

| method | synth CLUTTERED F1 | real CLUTTERED F1 | Δ |
|---|---|---|---|
| `ncc` | 0.77 | 0.33 | −0.44 |
| `mosse` | 0.82 | 0.32 | −0.50 |
| `sparse-geo` | 0.86 | 0.66 | −0.20 |
| `dino-dense` | 0.69 | 0.60 | −0.09 |
| `propose-retrieve` | 0.82 | 0.87 | **+0.05** |
| `owlv2-oneshot` | 0.82 | 0.06 | **−0.76** |

## Why each method moves the way it does

### `sparse-geo` and `propose-retrieve` — the small, explained drops

Both stay within ~0.05–0.22 F1 of their synthetic numbers in every regime, and both are the two
methods whose matching mechanism does not depend on a fixed scale/appearance model:
`sparse-geo`'s SIFT keypoints are scale- and rotation-invariant by construction, and
`propose-retrieve`'s FastSAM region proposals + DINOv2 embedding are proposal-driven, not
template-scaled. `sparse-geo` actually improves on PLAIN (0.99 vs EASY's 0.45) because real
photo cutouts always carry texture — the EASY chipset's flat, single-colour chips are the one
regime `sparse-geo`'s ≥8-SIFT-keypoint floor is built to abstain on (6 of 90 synthetic
abstentions were on chipset; **zero** abstentions anywhere on real-objects, confirmed from
`real-objects-results.json`'s `n_abstentions`). The residual drops on VARIED/CLUTTERED are
ordinary: real backgrounds and JPEG compression add spurious/missed keypoint matches that a
rendered flat-colour background never produces, and `propose-retrieve`'s FastSAM proposals are
measurably less clean around a real object's soft edge than around a hard-antialiased synthetic
emblem boundary (occasional over/under-segmented boxes, visible in the report's overlay gallery).

### `ncc` and `mosse` — a known limitation, made worse by a wider scale range

Both collapse hardest on VARIED/CLUTTERED (ΔF1 −0.17 to −0.50), and the per-image data shows this
is a **recall** failure, not a precision one — e.g. `ncc` on `real-cluttered-claw-hammer`: 8 ground
truth instances, only **1** prediction. Both methods' default `scales` config is `(0.75, ..., 1.3)`
— explicitly documented in `ncc.py`'s docstring as "instances rotated or scaled past that are
missed." The recorded ground-truth `instance_scale_min`/`max` confirms why real hits this harder
than synthetic: `real-varied-apple` spans **0.25×–1.34×**, `real-cluttered-screwdriver` spans
**0.37×–1.44×**, both reaching well below the 0.75× floor — versus `textured-varied-05`'s
0.68×–1.43× and `textured-cluttered-05`'s 0.82×–1.19×, which mostly stay inside or just outside the
band. `real-objects`' scale range (documented in `DATASETS.md` as 0.25–1.6× to make "a single
image ... test large AND small instances ... side by side") is *deliberately* wider than
`textured`'s 0.6–1.6×, and it lands squarely on template matching's known, already-documented blind
spot — this is the **same crossover the synthetic VARIED regime already shows** (`ncc`/`mosse`
collapse relative to EASY/TEXTURED there too), just sharper because the real set's scale spread is
wider. This is a config/coverage limitation, not a new bug, so it does not get its own remediation
doc (§ below); the honest fix is a wider default `scales` bank, at the cost this repo's own `ncc`
engineering log already measured (wider scale pyramids raise CLUTTERED/VARIED recall a little but
cost latency and false-positive rate on the fixed-scale regimes).

### `dino-dense` — synthetic's worst method is real's middle-of-the-pack method

`dino-dense`'s EASY F1 (0.17) is the single worst cell in the whole synthetic benchmark: DINOv2's
dense patch features have nothing to discriminate on a flat, single-colour chip against a flat
white background — precision cratered (0.12) because near-uniform cosine similarity fires
everywhere. Every real photo carries genuine texture and lighting gradient even in the `PLAIN`
regime, so that specific failure mode does not exist on real pixels (PLAIN F1 0.60, 3.5× EASY).
The method still has one clean, repeatable real-photo weak point: it scores **0 of 8** on
`screwdriver` in **all three** real regimes (`real-plain-screwdriver`: 0 tp / 13 fp / 8 fn;
`real-varied-screwdriver`: 0 tp / 13 fp / 4 fn; `real-cluttered-screwdriver`: 0 tp / 5 fp / 8 fn) —
a long, thin, low-area object is exactly the case the method doc already names as its weakness
("coarse, stride-14 on tiny objects"). Net effect: a large synthetic-EASY-specific weakness
disappears and a smaller, already-documented thin-object weakness persists — the two roughly wash
out to the smallest ΔF1 (−0.008) of any method.

### The `owlv2-oneshot` collapse — real, large, and not explained by noise

`owlv2-oneshot`'s pooled precision falls from 0.502 (synthetic) to **0.057** (real) while recall
barely moves (0.874 → 0.593) — the method is still *finding* objects, it is drowning them in false
positives. The per-image breakdown shows this is **not** uniform photographic degradation but a
sharp, bimodal, **object-shape-dependent** failure:

| image | gt | predictions | outcome |
|---|---|---|---|
| `real-plain-apple` / `-orange` / `-golf-ball` / `-hockey-puck` / `-tennis-ball` / `-claw-hammer` | 4–8 | 4–8 | **perfect** (P=R=1.0) |
| `real-plain-c-clamp` | 5 | **45** | P=0.07 |
| `real-plain-chess-pawn` | 8 | **101** | P=0.01 |
| `real-plain-screwdriver` | 8 | **252** | P=0.01 |
| `real-varied-c-clamp` | 4 | **363** | P=0.01 |
| `real-varied-ping-pong-ball` | 8 | **377** | P=0.02 |

Round, blob-shaped objects (apple, orange, golf ball, hockey puck, tennis ball) are detected
essentially perfectly across every regime. Elongated or multi-part objects (c-clamp, chess pawn,
screwdriver, and claw-hammer once clutter is added) blow up to 45–363 predictions against 4–8
ground-truth instances **in every regime they appear in**, including the clean `PLAIN` background
— this rules out clutter or background noise as the cause. The one round object that also blows up
(`ping-pong-ball`, 377 predictions) does so only in `VARIED`, where it is scaled down toward the
low end of the range, suggesting a second, scale-driven trigger for a textureless object rather
than the same shape-driven one. This same signature — high recall, precision collapsing to
~0.01–0.11, with a similar cross-object split — is **already independently documented** in
`docs/eval/floorplans-findings.md` for the floor-plan domain, which is strong cross-domain evidence
that this is a systematic property of `owlv2-oneshot`'s box decoding/NMS, not a one-off artifact of
either dataset. A concrete hypothesis and remediation plan are in
[`owlv2-oneshot-real-objects-improvement.md`](owlv2-oneshot-real-objects-improvement.md).

## Ground-truth quality — measured, not assumed

`real-objects`' ground truth comes from FastSAM's automatic-mode soft-mask AABB (thresholded,
eroded, largest-component), not a rendered-exact box — `DATASETS.md` documents one known cosmetic
artifact (a shadow wedge fused into the `ping-pong-ball` cutout). Checked against the per-image
data: `ping-pong-ball` scores perfectly or near-perfectly for `sparse-geo`, `propose-retrieve`, and
`mosse` on `PLAIN`, and its `VARIED`/`CLUTTERED` misses match the same scale/precision patterns
every other object shows for the same methods. The documented GT imperfection is real but not a
measurable contributor to any method's score in this sweep.

## Recommendation

- **`sparse-geo` and `propose-retrieve` are the two methods whose synthetic ranking can be trusted
  on real photos** — both hold their relative position and most of their absolute score.
- **Do not read `dino-dense`'s synthetic EASY score as predictive of anything** — it is an artifact
  of the flat chipset background, not a real weakness; on real pixels the method is solidly
  mid-pack.
- **`owlv2-oneshot`'s synthetic numbers meaningfully overstate its real-world reliability** — do
  not ship it against real photographic input without addressing the over-detection collapse (see
  the linked remediation plan).
- **`ncc`/`mosse` are trustworthy only within their documented ±25–30% scale window** — the
  existing per-regime synthetic numbers already say this; real photos just make the consequence
  more visible because the real `varied`/`cluttered` regimes were deliberately built with a wider
  scale spread.

## Reproducing

```
pixi run bench                    # full sweep incl. real-objects rows -> docs/benchmark/results.json
pixi run bench-real-objects       # real-objects-only -> docs/benchmark/real-objects-results.json
pixi run report                   # regenerates docs/reports/benchmark-report.html (unchanged scope)
pixi run report-real-objects      # regenerates docs/reports/real-objects-report.html
```

Both result files are gitignored (regenerable); this doc and the two `.html` reports are the
committed record.
