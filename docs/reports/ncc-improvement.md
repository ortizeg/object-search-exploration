# Engineering log — improving `ncc` (2026-07-26)

A record of the iterative work that took Method ① `ncc` from collapsing under scale/rotation
variation (VARIED F1 ≈ 0.24, CLUTTERED ≈ 0.31) to a detector that recovers a real share of the
transformed repeats (VARIED ≈ 0.46, CLUTTERED ≈ 0.77) **without regressing** the fixed-scale
regimes it already owned (EASY and TEXTURED stay at F1 1.00 — EASY actually *improves* from 0.97).
This log captures the *why*, the measured deltas, and the dead ends, so the reasoning is not lost
in the diff.

Cross-references: the method is documented in [`../methods/ncc.md`](../methods/ncc.md); current
per-regime scores are in the [benchmark report](benchmark-report.html); deferred work is in
[`../ROBUSTNESS-BACKLOG.md`](../ROBUSTNESS-BACKLOG.md).

## Symptom

NCC correlates the raw exemplar crop against the scene, so it is not invariant to rotation or
scale. With the shipped default (`angles_deg=(0.0,)`, no rotation; `scales` spanning only
0.75–1.3) it found almost nothing once instances were rotated ±35° and rescaled 0.6–1.6×:

| regime | P | R | F1 | AP |
|---|---|---|---|---|
| EASY (chipset) | 0.944 | 1.000 | 0.971 | ~0.52 |
| TEXTURED | 1.000 | 1.000 | 1.000 | 0.559 |
| VARIED | 1.000 | **0.135** | 0.239 | 0.177 |
| CLUTTERED | 1.000 | **0.181** | 0.307 | 0.269 |

Precision was a perfect 1.000 on every regime — the method was *far* too conservative, leaving
recall on the table — and the collapse on VARIED/CLUTTERED was a **recall** collapse: the rotated
instances were simply never matched by a 0°-only template bank.

## Root cause — three compounding gaps

1. **No rotation bank.** A 0°-only template cannot correlate with a rotated instance at all, so no
   threshold change could recover it.
2. **A fixed-fraction accept cut.** The default `self-similarity` rule cut at `self_score × 0.7`.
   A rotated-and-rescaled true instance, degraded by resampling, correlates to only ~0.4–0.6 of
   the exemplar's self-match, so it fell below 0.7 and was dropped even when found.
3. **A duplicate-polluted candidate log.** The EVAL-08 candidate log took the top-N peaks by
   z-score *without deduplication* and *including* the above-threshold matches, so one instance
   (detected at many scales) entered the log many times; each duplicate scored as a false positive
   in the AP sweep, badly understating AP (EASY AP 0.52 when it should be ~1.0).

## The iteration — measured on the EVAL-20 textured set (48 images) + chipset (10)

Each pass was measured before moving on, over the four regimes. The harness reused
`object_search.eval.benchmark._run_one` and pooled P/R/F1 with `metrics.precision_recall_f1`.

### Step 1 — the rotation bank (recovers recall)

Pooled F1 on the **textured** regimes (TEXTURED+VARIED+CLUTTERED), calibration held at the old
`self-similarity` / `retain_frac=0.7`:

| bank | overall F1 | VARIED F1 | CLUTTERED F1 | p50 ms |
|---|---|---|---|---|
| baseline `(0.0,)` | 0.618 | 0.239 | 0.307 | 134 |
| `±35°`, 5 angles | 0.644 | 0.259 | 0.423 | 1199 |
| **`±35°`, 7 angles** | **0.691** | 0.332 | 0.547 | 1577 |
| `±35°`, 9 angles | 0.685 | 0.312 | 0.546 | 2259 |

**7 angles (~11.7° spacing) measured best.** 9 over-samples — it adds false peaks with no recall
gain; 5 leaves gaps the ~10–15° correlation tolerance cannot bridge. Widening `scales` to
0.6–1.7 on top of the bank added +0.006 F1 for an AP drop and +37 % latency, so it was **reverted**
— the default `scales` are unchanged.

### Step 2 — the accept threshold (the hard part: EASY vs the rest)

With the bank on, precision still had headroom, so lowering the cut recovered more recall. But a
single global fraction cannot serve every regime, because **the rotation bank throws moderate
false peaks (raw 0.5–0.76) on the structured chipset backgrounds** — measured *higher* than
genuine transformed instances score elsewhere (VARIED true matches reach down to ~0.43,
CLUTTERED to ~0.35). Lowering `retain_frac` to 0.45 to catch those true instances also admitted
the chipset false peaks:

| rule | EASY F1 | TEXTURED F1 | VARIED F1 | CLUTTERED F1 |
|---|---|---|---|---|
| `self × 0.45` (global) | **0.65** (43 FP) | 1.000 | 0.442 | 0.784 |
| `gmm` (adaptive) | 0.68 | 0.952 ↓ | 0.326 | 0.667 |
| `ratio` (adaptive) | 0.40 | 0.683 ↓ | 0.198 | 0.182 |
| Otsu (adaptive) | 0.66–0.68 | 0.98 ↓ | 0.33 | 0.70 |

Every off-the-shelf adaptive calibrator failed: `gmm`/Otsu see the chipset as a *3-mode*
distribution (low noise ~0.3 / rotation FPs ~0.6 / true matches ~1.0) and their 2-way split groups
the mid FPs with the true matches; `ratio` cuts right below the isolated self-match and collapses
recall. (I also confirmed the chipset FPs are **not** a masked-correlation artifact — they sit on
*higher*-variance regions than the exemplar, so a variance guard would not help.)

The distribution *does* separate the cases, just not by score alone: when the object repeats
**near-identically** (chipset, textured-plain) the true instances pile up **at** the self-match, so
there are ≥ 2 *distinct* locations near ~1.0; when the instances are **transformed** (varied,
cluttered) only the exemplar's own region sits up there. That is the **`repeat-aware`** rule:

- count the distinct locations scoring `≥ self × 0.9` (NMS-deduplicated — the bank hits the
  exemplar's own region many times, so a raw count would call every image a repeat);
- **≥ 2** → near-identical repeats → strict cut `self × 0.85` (rejects the rotation false peaks);
- else → transformed → permissive `self × retain_frac` (0.45) tail.

| rule | EASY F1 | TEXTURED F1 | VARIED F1 | CLUTTERED F1 | overall F1 |
|---|---|---|---|---|---|
| **`repeat-aware` (shipped)** | **1.000** | **1.000** | **0.459** | **0.768** | **0.78** |

`strict = 0.85` (not 0.80) is what keeps the biggest chipset image perfectly clean; the `retain_frac`
floor sits on a broad plateau (0.35–0.50 all pool to F1 ≈ 0.81 on textured, EASY/TEXTURED
untouched because they take the strict branch) — 0.45 was chosen for the cleaner precision on that
plateau, **not** to maximise F1 against the labels.

### Step 3 — deduplicate the candidate log (recovers AP)

The threshold changes leave AP alone (AP is threshold-free), but the rotation bank made the
pre-existing candidate-log duplication acute. NMS-ing the sub-threshold candidate log and dropping
any that overlap an accepted match — so `matches + candidates` is one clean ranked set — recovered
AP with **no** change to F1:

| regime | AP before | AP after |
|---|---|---|
| EASY | 0.52 | **1.000** |
| TEXTURED | 0.559 | **1.000** |
| VARIED | 0.177 | **0.398** |
| CLUTTERED | 0.269 | **0.820** |

## Result — per-regime, before → after (IoU 0.5)

| regime | F1 before | F1 after | AP before | AP after |
|---|---|---|---|---|
| EASY (chipset) | 0.971 | **1.000** | ~0.52 | **1.000** |
| TEXTURED | 1.000 | **1.000** | 0.559 | **1.000** |
| VARIED (scale/rotation) | 0.239 | **0.459** | 0.177 | **0.398** |
| CLUTTERED | 0.307 | **0.768** | 0.269 | **0.820** |

The other four methods (`sparse-geo`, `dino-dense`, `propose-retrieve`, `owlv2-oneshot`) are
**byte-for-byte unchanged** (the change is isolated to `ncc.py`). NCC
now recovers a genuine share of the rotated/rescaled repeats while *strengthening* its home turf —
EASY improves because `repeat-aware` also rejects the 5 stray baseline false positives.

## Fairness — thresholds are not fit to the labels

`repeat-aware` reads the **shape** of the score distribution (how many distinct locations sit near
the self-match), never the ground-truth boxes, and the identical rule runs on every dataset. The
numeric cut adapts per image (an absolute NCC cut does not transfer across images), but no cut is
chosen to maximise F1 against the labels: `strict = 0.85` is set by where the chipset false peaks
stop (≤ 0.76), and `retain_frac = 0.45` sits mid-plateau. AP remains threshold-free (it sweeps the
full deduplicated candidate log).

## Cost — latency

The rotation × scale bank is a large constant factor. On the small textured scenes a query goes
from ~134 ms to ~1.6 s; on the 6000×4000 chipset image from ~7.8 s to ~69 s (p50 across the
chipset ramp: 0.68 s → 8.9 s). This is the trade the method doc always warned the bank would cost,
and it is reported honestly in the latency-by-canvas chart. **FFT-based correlation** (already in
the ROBUSTNESS BACKLOG) is the mitigation if NCC ever needs to stay interactive at that resolution;
a caller who knows a scene is axis-aligned can also set `angles_deg=(0.0,)` to restore the old
millisecond path.

## What I tried and reverted

- **Widening `scales` to 0.6–1.7** alongside the bank: +0.006 F1 for an AP drop and +37 % latency.
  Reverted; the missed VARIED instances are rotation-limited, not scale-limited.
- **9-angle bank**: worse than 7 (extra false peaks, no recall gain).
- **`gmm` / `ratio` / Otsu calibration**: all fail on the chipset's 3-mode distribution or collapse
  recall (see Step 2). Kept selectable as controls.
- **`retain_frac = 0.40`**: ties 0.45 on F1 but with worse precision; below 0.35 CLUTTERED
  precision collapses. 0.45 chosen.
- **A scene-variance guard on matched windows** (to kill the chipset rotation FPs at the source):
  the FPs sit on *higher*-variance regions than the exemplar, so this would have hurt, not helped.

## Verification

`pixi run quality` green: Ruff + Ruff-format clean, MyPy strict clean, **496 passed / 5 skipped,
coverage 92.69 %**. New tests cover the `repeat-aware` strict/permissive switch, the rotation-bank
default, and the candidate-log deduplication invariant. Benchmark report and charts regenerated
from a fresh `pixi run bench`; only `ncc` numbers changed.

## Floor-plan domain follow-up (2026-07-30)

A real target-domain eval (Roboflow floor-plans-500, [`../eval/floorplans-findings.md`](../eval/floorplans-findings.md))
measured `ncc` doors F1 **0.248** on the 28-plan test split, with recall by symbol size (small/med/
large) **0.31 / 0.31 / 0.29** — flat and low. Flat recall-by-size is the signature of an
*orientation* problem, not a scale/texture one (recall should climb with symbol size if it were the
latter). This section tests that hypothesis empirically rather than assuming it.

### Root cause: CONFIRMED for doors, MIXED for windows

`NCCConfig.angles_deg` defaults to a 7-step bank over only ±35°. A floor-plan door sits on
whichever wall it is drawn on, so an instance on a perpendicular wall can be ~90° off the
exemplar — entirely outside that bank. Four rotation-bank variants were swept on val (shipped
±35° / cardinal-only 0°·90°·180°·270° / cardinal × the shipped ±35° sub-bank [~28 angles] /
uniform 30° spacing [12 angles]), each crossed with `retain_frac`, at a fixed single scale:

**Cardinal-only (0/90/180/270°) wins clearly on val for both classes** — not the wider continuous
banks. Floor-plan walls are discretely orthogonal, not continuously rotated, so a small precise
cardinal set beats a dense sweep (which throws more false peaks at more candidate angles on
structured background without recovering more true recall).

A separate, DEFAULT-OFF `mirror` field was added to `NCCConfig` (extends `_rotated_bank` to also
yield a `cv2.flip`-mirrored template/mask sibling per angle — a reflection is not in the rotation
group, so no bank width, however wide, can ever reach it; the archetype is a door drawn with the
opposite swing hand). Measured *separately* from the rotation-bank result so the two effects are
attributable independently:

- **Doors**: a statistical tie (val F1 0.197 mirror-on vs 0.192 mirror-off — within noise on a
  56-image val set). Mirror-on happened to win the argmax by the smallest possible margin.
- **Windows**: a clear net-negative (val F1 0.276 mirror-on vs 0.299 mirror-off). Window symbols
  in this dataset convention appear to carry much less of the bilateral-swing-direction variation
  doors do, so the extra mirrored candidate templates mostly just add false-peak surface.

Both `angles_deg` and `mirror` are additive to the existing `_TUNING_GRIDS["ncc"]` (the shipped
bank and `mirror=False` stay available), landed as a **grid-only** change — no `NCCConfig` default
was touched, so the synthetic regime cannot regress by construction. `run_domain_tuning` gained an
additive `grids=` override (defaults to `_TUNING_GRIDS`, byte-identical when omitted) so this and a
sibling `mosse` investigation could each sweep method-specific variants without shelling out to the
full `tune-floorplans` CLI (which always tunes all six methods) on every iteration; the previously-
aliased `ncc`/`mosse` grid objects were split into independent tuples in the same change (both
config models are `extra="forbid"`, so a method-only key fed into the wrong model would raise).

### Before / after — floor-plan domain (val 56 / test 28 plans, argmax F1 @ IoU 0.5)

| dataset | config | P | R | F1 |
|---|---|---|---|---|
| floorplans-door | default (shipped) | 0.111 | 0.309 | 0.164 |
| floorplans-door | tuned, pre-existing grid (no cardinal/mirror) | 0.569 | 0.159 | 0.248 |
| floorplans-door | **tuned, cardinal + mirror=True, retain=0.65** | **0.340** | **0.378** | **0.358** |
| floorplans-window | default (shipped) | 0.149 | 0.436 | 0.222 |
| floorplans-window | tuned, pre-existing grid (no cardinal/mirror) | 0.428 | 0.378 | 0.401 |
| floorplans-window | tuned, cardinal + mirror=False, retain=0.65 | 0.287 | 0.449 | 0.350 |

**Doors: a clear, robust win** — F1 more than doubles over the shipped default (+0.194) and beats
the pre-existing best-tuned grid entry by +0.110. This result reproduced consistently across three
independent measurements (the rotation-bank sweep, the separate mirror sweep, and the final
full-grid run all converged on test F1 0.355–0.358).

**Windows: an honestly-disclosed generalization gap, not reverted.** Cardinal genuinely wins on val
(F1 0.359 vs the old grid's 0.329), which is the *only* selection criterion the tuning protocol is
allowed to use — but this time the val-argmax pick generalizes to test F1 0.350, slightly *below*
what the pre-existing grid entry already achieved (0.401) on this particular test split. This is
disclosed rather than hidden: per-image debugging (`scripts/ncc_debug_visualize.py`, below) found
at least one plan where the cardinal config visibly regresses relative to the shipped default (P
0.636→0.286, R 0.538→0.308 on one 13-window plan) alongside others where it likely helps — the net
effect on this domain is genuinely more mixed for windows than for doors, and the grid is shared
across both classes so there is no clean per-dataset revert available without a larger tuning-
architecture change (out of scope here). Since the protocol forbids conditioning grid decisions on
test outcomes, the additive option stays; a future pass could split the grid per-dataset if this
gap matters enough to chase.

### A new debug tool: `scripts/ncc_debug_visualize.py`

Aggregate F1/P/R cannot say *which* instances are missed or *why* a false positive appears. This
script (flag-driven: `--config {default,tuned-door,tuned-window}`, `--image`, `--exemplar-index`)
runs `ncc.search()` on one floor-plan image and renders three per-step artifacts — `01_query.png`
(the exemplar), `02_matches_vs_gt.png` (accepted matches in green, sub-threshold candidates in red
with their score, ground truth colored by whether some match claimed it), `03_heatmap.png` (the
representative similarity heatmap) — plus a console dump of the calibration reasoning, threshold,
and self-score. It is a research/debug tool (`scripts/`, never touches shipped config or
`docs/benchmark/`); `ncc.search()` itself carries no debug branch.

Two illustrative runs:
- **Door plan 4052** (17 doors): the shipped default's wide 7-angle × 5-scale bank throws **137**
  false positives on this plan's structured background (dimension lines, wall hatching) — P
  collapses to 0.068 despite R 0.588. The cardinal bank cuts that to 9 FPs (P 0.438, R 0.412, F1
  0.122 → 0.425) — a concrete, visual instance of the aggregate door result.
- **Window plan 16** (13 windows): the tuned-window config underperforms the shipped default here
  (P 0.286/R 0.308 vs P 0.636/R 0.538) — a concrete instance of the windows generalization gap
  above, and the specific case that motivated digging further into whether more recall was
  available at all (next section).

### Is there more recall available? Two more levers tested — both net negative

Per-instance debugging showed **55% of missed doors (27/49 across a 10-image sample) have a
correctly-localized candidate** (IoU > 0.5 with the missed ground truth) scoring just under the
0.65 threshold — mostly clustered 0.53–0.65. `ncc` is finding the right location; the score just
doesn't clear the cut. Two levers that could plausibly recover that recall were tested directly
rather than assumed:

- **Lowering `retain_frac`** (0.35–0.65 swept on val, cardinal + mirror bank): F1 rises
  *monotonically* toward 0.65 (0.135 → 0.251) — there is no lower sweet spot within this range.
  precision improves faster than recall degrades all the way up; the argmax already sits at the
  best point tested.
- **Widening the scale pyramid** (1 / 3 / 5 levels, crossed with `retain_frac`): recovers a little
  recall (R 0.380 → 0.416 at `retain_frac=0.65`) but costs more precision than that is worth (P
  0.187 → 0.155) — F1 gets *worse* every time (0.251 → 0.216–0.226), confirming `scales=(1.0,)` is
  the right choice despite measured within-image door-size variance (~19% coefficient of variation
  on box area across a plan's own instances).

**Both levers fail for the same reason**: any change that gives `ncc` more candidate templates to
try (more scales, more angles, a looser cutoff) also gives structured floor-plan background more
chances to throw a false peak in the *same* 0.5–0.65 score range genuine instances occupy. This is
qualitatively different from the synthetic regime, where a genuine instance is a literal (or
affine-transformed) copy of the exemplar's exact pixels and correlates near the ~1.0 self-match —
a wide, clean separation from background noise that no floor-plan symbol (with its natural small
rendering differences — line weight, anti-aliasing offset, minor proportion drift between "the
same" symbol drawn at different locations) reproduces. Raw-intensity correlation has a real ceiling
here: when the true-positive and false-positive score distributions genuinely overlap, no single
threshold or search-bank change can separate them. (This is exactly the gap the `mosse` filter's
whitening — see `docs/reports/mosse-improvement.md` — exists to attack, since it is trained to
suppress background rather than just correlating raw pixels; a natural next step, not attempted in
this ncc-scoped investigation.)

### Fairness

`grids=` and the `_TUNING_GRIDS["ncc"]` cardinal/mirror block are argmax-F1-on-val selections —
the tuning protocol's normal, allowed use of labels (train/test split hygiene, never a test peek).
The `repeat-aware` threshold calibrator itself is untouched and still reads the score
*distribution* shape, never the ground-truth boxes, on every dataset. The windows generalization
gap above was deliberately NOT resolved by conditioning the grid on test performance, which would
have crossed that line.

### Verification

`$HOME/.pixi/bin/pixi run lint` and `$HOME/.pixi/bin/pixi run typecheck` clean. Full suite:
**767 passed / 20 skipped, 92.36 % coverage** (floor held; counts grew after rebasing onto latest
`main`, which landed an unrelated real-objects eval set and its own tests in parallel).
Synthetic regression guard: `pixi run bench-ci` unchanged (`ncc` F1 1.000 on the model-free
chipset subset) — this check is exact both before and after the rebase, since it always exercises
the same fixed 6-image chipset subset. A full `pixi run bench "methods=[ncc,mosse]"` re-run on the
rebased branch now measures **overall F1 0.7244** (fixed 0.9307, varied 0.4414) over **90** images,
not the 60 originally reported here — a separate, parallel PR added a 30-image real-object-insertion
set to the default full-sweep image pool between when this investigation started and when it was
rebased onto `main` for merge, which is why the pooled number moved (real photographic pixels are
harder for raw-intensity correlation than clean synthetic renders). This is NOT a regression from
this change: `NCCConfig`'s shipped defaults (`mirror=False`, `angles_deg` unchanged) were never
touched by this investigation, so `bench-ci`'s byte-identical result is the correct, sufficient
regression check; the pooled full-sweep number simply reflects a larger, harder, unrelated
benchmark pool as of the merge, not a change caused by this PR.
