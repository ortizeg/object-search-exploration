# 260730-vx3 — sparse-geo improvement: measurement log

Every number below is measured on this machine (CPU, osx-arm64) with the committed split
manifests and the seeded exemplar sampler. Aggregate metrics only — no plan pixels, no
per-image records (T-vx3-02). Raw JSONs live in `measurements/` (gitignored data never
enters them; they carry only the distilled metric blocks the tuning runner emits).

**Commands.**

```
pixi run tune-floorplans --dataset floorplans-{door,window} --methods sparse-geo
pixi run python .planning/quick/<this-dir>/regime_harness.py --label <label> [--overrides '<json>']
```

**Compute budget (Task 1 tracer).** A sparse-geo-only `tune-floorplans` sweep is
**1m37s** on doors and **1m58s** on windows (14-entry grid × 56 val plans, plus two
28-plan test passes). The synthetic-regime harness is **16s**. The whole
iterate/measure loop therefore runs locally — **no GPU box was provisioned**.

---

## Baseline

Captured **before any change to `sparse_geo.py`** (`git diff --stat src/` empty at the time
of capture; the only committed source change was the additive `--methods` CLI option, which
does not touch the method).

### Floor-plan target domain (test split, 28 plans, 1 exemplar, F1 @ IoU 0.5)

| class | config | F1 | P | R | AP50 | coverage | p50 ms |
|---|---|---|---|---|---|---|---|
| door | tuned (min_inliers=3, nms_iou=0.3) | 0.219 | 0.442 | 0.146 | 0.168 | 28/28 | 65.4 |
| door | default | 0.217 | 0.516 | 0.137 | 0.168 | 28/28 | 63.0 |
| door | *val F1 that selected the tuned config* | 0.254 | | | | | |
| window | tuned (min_inliers=3, nms_iou=0.3) | 0.309 | 0.627 | 0.205 | 0.293 | 28/28 | 60.9 |
| window | default | 0.290 | 0.659 | 0.186 | 0.293 | 28/28 | 60.0 |
| window | *val F1 that selected the tuned config* | 0.233 | | | | | |

**Reconciliation against the published table** (`docs/eval/floorplans-findings.md`, produced
by an aggressive multi-knob GPU sweep):

| | published | measured here | verdict |
|---|---|---|---|
| door tuned F1 | 0.219 | 0.2194 | reproduces |
| door default F1 | 0.217 | 0.2169 | reproduces |
| door P / R | 0.44 / 0.15 | 0.442 / 0.146 | reproduces |
| window tuned F1 | 0.309 | 0.3092 | reproduces |
| window default F1 | 0.290 | 0.2900 | reproduces |
| window P / R | 0.63 / 0.21 | 0.627 / 0.205 | reproduces |

The baseline reproduces to three decimal places on every cell, so deltas measured against it
are trustworthy. (The published sparse-geo row happened to land on the same
`min_inliers=3, nms_iou=0.3` operating point the committed grid selects here.)

### Synthetic regimes (the regression guard)

Sparse-geo at its **shipped defaults** over the committed benchmark image sets, pooled
micro-average per regime, via `regime_harness.py`.

| run | regime | P | R | F1 | AP | p50 ms |
|---|---|---|---|---|---|---|
| baseline | EASY (chipset, n=10) | 1.000 | 0.294 | 0.455 | 0.400 | 209.6 |
| baseline | TEXTURED (n=16) | 1.000 | 1.000 | 1.000 | 1.000 | 109.6 |
| baseline | VARIED (n=16) | 0.765 | 0.755 | 0.760 | 0.711 | 149.3 |
| baseline | CLUTTERED (n=16) | 0.858 | 0.869 | 0.863 | 0.836 | 136.2 |

### Committed full-benchmark reference (authoritative for the Task 4 regression check)

`docs/benchmark/results.md` / `docs/reports/benchmark-report.html`, generated at git SHA
`c450791`, 60 labelled images:

| method | precision | recall | F1 | mean AP | abstentions | errors | p50 ms |
|---|---|---|---|---|---|---|---|
| `sparse-geo` | 0.884 | 0.770 | 0.823 | 0.751 | 6 | 0 | 86.8 |

Recall by scale bucket: fixed-scale **0.773**, varied-scale **0.765**.

This is the number the final check compares against **if the shipped defaults change**. If
they do not, the stronger and cheaper claim is asserted instead: the default-config regime
numbers above are byte-identical after the change.

---

## Step A1 — diagnosis: where mirrored instances are actually lost

Run before touching `sparse_geo.py`, on **16 door val plans** at the shipped defaults, by
monkeypatching `_is_degenerate` and `_ransac_similarity` from a throwaway script
(`diagnose_mirror.py`) so the method file carries no diagnostic scaffolding.

| stage | measured |
|---|---|
| GT door instances over the 16 plans | 157 |
| exemplar keypoints (total) | 345 |
| correspondences (total) | 2 664 |
| crop keypoints contributing ≥1 correspondence | 333 |
| **Hough peaks hypothesized** | **55** |
| instances verified | 33 |

Degeneracy-gate outcomes over those 55 peaks, and the chirality of RANSAC's winner:

| outcome | count |
|---|---|
| accepted | 43 |
| rejected: **mirror (det<0)** | **2** |
| rejected: scale | 9 |
| RANSAC winner was **reflected** | 2 |
| RANSAC winner was proper | 52 |

**Finding.** `_is_degenerate`'s mirror branch fires **2 times in 55 peaks**. Relaxing it alone
could therefore recover at most ~2 detections across 16 plans — it is very nearly an inert
control. The funnel collapses much earlier: **only 55 peaks are ever hypothesized for 157 GT
doors**, and correspondences are plentiful (2 664). The loss is at the **voting** stage, exactly
as predicted: `_vote_single_4dof` and `_proper_similarity_2pt` computed only the
orientation-preserving branch, so a mirrored instance's correspondences predict a *wrong* centre,
scatter, and never accumulate into a peak — they never reach the degeneracy gate at all.

That is why hypothesis 1 was built as **one** end-to-end change (reflected pose votes **and** the
relaxed gate) rather than as two ablations.

### A follow-on finding from the end-to-end test: SIFT orientations are not mirror-consistent

On a controlled scene (one SIFT-rich tile, one identical copy, one horizontally mirrored copy),
instrumenting the correspondence set:

| target | correspondences landing inside it | of those, **geometrically correct** |
|---|---|---|
| identical copy | 94 | **33 (35%)** |
| mirrored copy | 78 | **13 (17%)** |

and the reflected pose angle `alpha = scene_angle + crop_angle` on those 13 correct
correspondences scatters across ~80–280° instead of clustering at the true 180°.

So even with the reflected branch wired, **`single-4dof` cannot cluster a mirrored instance**:
it derives a pose from one keypoint frame, and SIFT's orientation assignment on a mirrored patch
is not a predictable function of the original's. `pairwise-4dof` fits a pose from two point pairs
and never reads an orientation — and there the mirrored instance **is** recovered (IoU 0.98).
Both facts were pinned by tests in `tests/test_sparse_geo_mirror.py` during the investigation; that
file was removed with the reverted feature (see the verdict below), so the facts now live here and
in the report rather than in the suite.

---

## Hyp 1 (mirror) — at the shipped `single-4dof` voting mode

Grid: `allow_mirror=True` × `min_inliers ∈ {2,3,4,5,6,8,10}` × `nms_iou ∈ {0.3,0.5}` — the same
shape as the baseline grid, so the two are directly comparable.

| class | config | F1 | P | R | AP50 | coverage | p50 ms | ΔF1 vs baseline |
|---|---|---|---|---|---|---|---|---|
| door | baseline tuned (min_inliers=3, nms_iou=0.3) | 0.219 | 0.442 | 0.146 | 0.168 | 28/28 | 65.4 | — |
| door | **+allow_mirror** (min_inliers=3, nms_iou=0.3) | **0.216** | 0.356 | 0.155 | 0.161 | 28/28 | 60.9 | **−0.004** |
| window | baseline tuned (min_inliers=3, nms_iou=0.3) | 0.309 | 0.627 | 0.205 | 0.293 | 28/28 | 60.9 | — |
| window | **+allow_mirror** (min_inliers=3, nms_iou=0.3) | **0.290** | 0.492 | 0.205 | 0.286 | 28/28 | 49.8 | **−0.020** |

**The false-positive direction is exactly the predicted risk, and it is the whole story.**
Precision falls on both classes (door 0.442 → 0.356, window 0.627 → 0.492) while recall barely
moves (door +0.009, window ±0.000). Mirror acceptance at `single-4dof` admits bad reflected fits
without recovering true mirrored instances — because, per the A1 diagnosis, those instances never
form a peak in the first place. A tighter `min_inliers` did not compensate: the grid still
selected `min_inliers=3`, i.e. no operating point in the sweep traded the lost precision back.

**Verdict on hypothesis 1 at the default voting mode: DISPROVEN.** Not merely inert — mildly
harmful. It regresses floor-plan F1 on both classes.

---

## Hyp 1 (mirror) — at `pairwise-4dof`, the voting mode where it *can* work

The A1 follow-on finding says `single-4dof` structurally cannot cluster a mirrored instance
(SIFT orientations are not mirror-consistent), and that `pairwise-4dof` can (controlled scene,
IoU 0.98). So the honest test of hypothesis 1 is at `pairwise-4dof`.

That requires a **control**. Switching to `pairwise-4dof` is itself a change; without measuring
`pairwise-4dof` *without* mirror, any movement would be misattributed to mirror handling. Four
conditions were therefore run, all over the identical grid
`min_inliers ∈ {5,8,12,16,20} × nms_iou ∈ {0.3,0.5}` under `voting_mode="pairwise-4dof"`,
tuned on `val`, scored once on `test`:

| class | condition | val F1 | selected | test F1 | P | R | AP50 | cov | abst | p50 ms |
|---|---|---|---|---|---|---|---|---|---|---|
| door | *baseline* (single-4dof, no mirror) | 0.254 | mi=3, nms=0.3 | **0.219** | 0.442 | 0.146 | **0.168** | 28/28 | 14 | 65.4 |
| door | pairwise-4dof, **no** mirror *(control)* | 0.238 | mi=5, nms=0.3 | **0.231** | 0.323 | 0.180 | 0.096 | 28/28 | 14 | 113.6 |
| door | pairwise-4dof **+ allow_mirror** | 0.230 | mi=5, nms=0.3 | **0.223** | 0.257 | 0.197 | 0.062 | 28/28 | 14 | 202.6 |
| window | *baseline* (single-4dof, no mirror) | 0.233 | mi=3, nms=0.3 | **0.309** | 0.627 | 0.205 | **0.293** | 28/28 | 12 | 60.9 |
| window | pairwise-4dof, **no** mirror *(control)* | 0.222 | mi=5, nms=0.3 | **0.256** | 0.360 | 0.199 | 0.186 | 28/28 | 12 | 184.0 |
| window | pairwise-4dof **+ allow_mirror** | 0.213 | mi=5, nms=0.3 | **0.225** | 0.248 | 0.205 | 0.123 | 28/28 | 12 | 337.3 |

### Reading 1 — the isolated effect of `allow_mirror` (mirror vs. its own control)

This is the comparison hypothesis 1 actually stands or falls on: same voting mode, same grid,
same splits, mirror the only difference.

| class | ΔF1 (val) | ΔF1 (test) | ΔP | ΔR | ΔAP50 | Δp50 |
|---|---|---|---|---|---|---|
| door | **−0.008** | **−0.008** | −0.066 | +0.017 | −0.034 | +89 ms (1.78×) |
| window | **−0.009** | **−0.032** | −0.112 | +0.006 | −0.064 | +153 ms (1.83×) |

`allow_mirror` loses on **both classes**, on **both splits**, on **F1, precision and AP50**,
and roughly **doubles latency** (the reflected branch doubles the cast vote count and the
histogram occupancy). Recall gains are a rounding error (+0.017 / +0.006) and are bought with
2–4× more false positives.

The consistency across val *and* test rules out a test-split fluke. Mirror acceptance is
strictly harmful here even in the voting mode that can represent a mirrored pose at all.
The controlled-scene test in `tests/test_sparse_geo_mirror.py` shows the mechanism *works*
(a synthetic mirrored tile is recovered at IoU 0.98); it just does not pay on real floor
plans, because the reflected hypothesis space it opens is populated overwhelmingly by bad
fits rather than by genuine opposite-hand doors.

### Reading 2 — was the door "gain" ever attributable to mirror? No.

The only cell in the whole block that beats baseline F1 is **door, pairwise-4dof, *without*
mirror**: 0.219 → 0.231 (+0.012). The mirror row is *lower* than that control (0.223). So the
sliver of door F1 gain belongs entirely to **switching the voting mode**, not to mirror
handling. Hypothesis 1 has no positive evidence anywhere in the measurement set.

### Reading 3 — and even the control fails the revert filter

`pairwise-4dof` as a floor-plan recommendation is not rescuable either:

- **Doors:** +0.012 F1 comes with precision 0.442 → 0.323 (−27%) and **AP50 0.168 → 0.096
  (−43%)**. That is a ranking-quality collapse. The method emits more, worse-ordered boxes and
  scrapes a marginally higher F1 at one hand-picked operating point. A practitioner reading a
  PR curve rejects this trade; a +0.012 F1 headline does not survive a −0.072 AP50.
- **Windows:** an outright regression on every axis — F1 0.309 → 0.256, P 0.627 → 0.360,
  AP50 0.293 → 0.186.
- **Latency:** 1.7–3.0× baseline on both classes.

`voting_mode` is a pre-existing, already-documented knob, so there is nothing to revert for it —
it simply does not become a recommended floor-plan setting.

---

## Verdict on hypothesis 1 — DISPROVEN at both voting modes; fully REVERTED

| voting mode | door ΔF1 | window ΔF1 | precision | AP50 | verdict |
|---|---|---|---|---|---|
| `single-4dof` (shipped default) | −0.004 | −0.020 | falls on both | falls on both | disproven |
| `pairwise-4dof` (vs. its own control) | −0.008 | −0.032 | falls on both | falls on both | disproven |

Applying the revert filter from the plan (*keep only if it improves floor-plan test F1 AND does
not regress the synthetic regimes*): `allow_mirror` improves floor-plan test F1 in **zero of the
four class × voting-mode cells**. The plan is explicit that a disproven idea does not get parked
behind an off-by-default flag — CLAUDE.md forbids inert controls — so the entire surface comes
out:

| removed | what |
|---|---|
| `src/object_search/search/sparse_geo.py` | the `allow_mirror` config field; `_reflected_similarity_2pt`; `_vote_single_4dof_reflected`; the `reflect` field on `_Vote`; the 5th chirality bin dimension in `_accumulate_votes`; the `allow_mirror` parameter and branch in `_is_degenerate`; four docstring blocks |
| `tests/test_sparse_geo_mirror.py` | whole file (216 lines) |
| `docs/methods/sparse-geo.md` | the `allow_mirror` config-reference row |
| `_TUNING_GRIDS["sparse-geo"]` | nothing — no mirror entries were ever committed |

`_two_point_models` / `_model_from_complex(reflect=...)` **stay**: RANSAC has always fitted both
the proper and the reflected 2-point model and kept whichever has more inliers, and
`_is_degenerate` has always rejected a reflected winner. That is the pre-existing mirror
*rejection* contract, unchanged since Phase 5, and it is what makes the rejection non-vacuous.
Only the mirror *acceptance* path added by this task is removed.

**The shipped defaults are therefore byte-identical to the pre-task baseline** — which is the
cheaper and stronger regression claim the plan asks for (Task 4 step 7).

---

## Hyp 2 (superpoint) — the learned backend, measured against the ORIGINAL baseline

Measured against the **baseline** above (door 0.219 / window 0.309), *not* against hypothesis 1 —
hyp 1 is fully reverted and contributes nothing here. Nothing for hypothesis 2 was ever committed
to `src/`: every number below came from the throwaway `measure.py` driver with an explicit grid,
which reuses the project's own `tune_method` / `_evaluate`, so the code path producing these
numbers is identical to the shipped one.

### The feasibility probe (run before committing 56 val plans × a 10-entry grid to it)

The premise of hypothesis 2 is "a door symbol is a line plus an arc — too little DoG texture for
SIFT, so a learned detector should find more". Probed directly on 5 door plans
(`probe_superpoint.py`), counting **exemplar-crop** keypoints, which is the quantity the premise
is about:

| plan | GT doors | SIFT crop kp | SuperPoint crop kp | SIFT scene kp | SuperPoint scene kp |
|---|---|---|---|---|---|
| 109 | 12 | 0 | 1 | 291 | 466 |
| 110 | 12 | 2 | 1 | 711 | 550 |
| 119 | 7 | **33** | **12** | 2 170 | 5 986 |
| 120 | 8 | 6 | 8 | 813 | 839 |
| 121 | 4 | 0 | 3 | 368 | 291 |

**The premise is half-right and it does not matter.** SuperPoint does fire marginally on the
barren crops where SIFT finds literally nothing (0 → 1, 0 → 3, 6 → 8), but it finds **less than
half** as much on the one texture-rich crop (33 → 12). Crucially, `min_exemplar_keypoints=8` is
cleared by 1/5 crops under SIFT and 2/5 under SuperPoint — the abstention pattern is essentially
unchanged (**4/5 plans abstain under both backends**). Meanwhile SuperPoint yields ~2.8× the
*scene* keypoints on the textured plan (2 170 → 5 986), which is where the O(n²) `pairwise-4dof`
latency comes from. More scene keypoints, no more usable exemplar keypoints: the cost side of the
trade scales and the benefit side does not.

### The full sweep — 4 conditions, tuned on `val`, scored once on `test`

Both frameless-compatible voting modes were swept, not just `pairwise-4dof`, so the verdict is
not an artifact of one voting choice. (`single-4dof` raises at config time for a frameless
backend, by design.) Grids: `pairwise-4dof` × `min_inliers ∈ {5,8,12,16,20}` × `nms_iou ∈
{0.3,0.5}`; `translation-2dof` × `min_inliers ∈ {5,8,12}` × `nms_iou ∈ {0.3,0.5}`.

| class | condition | val F1 | selected | test F1 | P | R | AP50 | coverage | abst | p50 ms |
|---|---|---|---|---|---|---|---|---|---|---|
| door | *baseline* (SIFT, single-4dof) | 0.254 | mi=3, nms=0.3 | **0.219** | 0.442 | 0.146 | **0.168** | **28/28** | 14 | **65.4** |
| door | SuperPoint + `pairwise-4dof` | 0.195 | mi=5, nms=0.3 | **0.189** | 0.519 | 0.116 | 0.125 | 28/28 | 13 | 439.3 |
| door | SuperPoint + `translation-2dof` | 0.203 | mi=5, nms=0.3 | **0.195** | 0.788 | 0.112 | 0.184 | 28/28 | 13 | 348.5 |
| window | *baseline* (SIFT, single-4dof) | 0.233 | mi=3, nms=0.3 | **0.309** | 0.627 | 0.205 | **0.293** | **28/28** | 12 | **60.9** |
| window | SuperPoint + `pairwise-4dof` | 0.147 | mi=5, nms=0.3 | **0.213** | 0.488 | 0.136 | 0.104 | **26/28** | 18 | 339.5 |
| window | SuperPoint + `translation-2dof` | 0.156 | mi=5, nms=0.5 | **0.261** | 0.793 | 0.156 | 0.159 | **26/28** | 19 | 418.3 |

Deltas against baseline:

| class | condition | ΔF1 (val) | ΔF1 (test) | ΔP | ΔR | ΔAP50 | latency |
|---|---|---|---|---|---|---|---|
| door | SP + `pairwise-4dof` | **−0.059** | **−0.030** | +0.078 | −0.030 | **−0.043** | **6.72×** |
| door | SP + `translation-2dof` | **−0.051** | **−0.024** | +0.346 | −0.034 | +0.016 | **5.33×** |
| window | SP + `pairwise-4dof` | **−0.087** | **−0.096** | −0.140 | −0.069 | **−0.189** | **5.57×** |
| window | SP + `translation-2dof` | **−0.077** | **−0.048** | +0.166 | −0.049 | **−0.133** | **6.87×** |

**SuperPoint loses F1 in 4 of 4 cells, on both splits, in both voting modes.** The single positive
AP50 cell in the whole block (door + `translation-2dof`, 0.168 → 0.184) sits next to an F1 that is
*down* 0.024, so it does not rescue the hypothesis under the revert filter.

### The three things the plan said to watch

**1. Latency — disclosed either way.** 5.3–6.9× baseline p50 on every condition (65 ms → 349–439 ms
on doors; 61 ms → 339–418 ms on windows), with tail maxima of 4.1–9.4 s. Note that
`translation-2dof` is *not* cheaper than `pairwise-4dof` here despite avoiding the O(n²) pairing —
the SuperPoint ONNX forward pass on a full plan dominates both.

**2. Abstentions — materially worse on windows.** Counted as "plans that produced no output at all":

| class | condition | scored | abstained | plans producing output |
|---|---|---|---|---|
| door | baseline | 28 | 14 | **14** |
| door | SP + pairwise / translation | 28 | 13 / 13 | 15 / 15 |
| window | baseline | 28 | 12 | **16** |
| window | SP + pairwise | 26 | 18 | **8** |
| window | SP + translation | 26 | 19 | **7** |

Doors are a wash (+1 plan). Windows **halve**: 16 plans produced output at baseline, 7–8 under
SuperPoint. The recall loss on windows is not a threshold artifact — the method simply stops
answering on half the plans it used to answer on.

**3. Coverage — a hard failure on windows, and the mechanism is a crash, not an abstention.**
Window coverage drops **28/28 → 26/28** in *both* SuperPoint conditions (`n_errors: 2`). The cause
is specific and reproducible, from the run logs:

```
sparse-geo on research floorplans-window/... failed: [ONNXRuntimeError] : 1 : FAIL :
... CoreMLExecutionProvider ... Input (/Cast_12_output_0) has a dynamic shape ({-1,2})
but the runtime shape ({0,2}) has zero elements. This is not supported by the CoreML EP.
```

When SuperPoint detects **zero keypoints** on a sparse line-art crop or scene, the exported graph
hands a zero-row tensor to a node the CoreML execution provider cannot handle, and the query raises
instead of abstaining cleanly. This fired **92 times** across the `sp-t2-window` run (val grid plus
test). This is exactly the low-keypoint-yield regime hypothesis 2 was built to exploit — the
backend's failure mode on the target domain is a hard error, not a graceful empty result.

### Was the sweep unfair? The `min_inliers` grid floor, checked rather than waved away

The SuperPoint grids start at `min_inliers=5`, while the SIFT baseline's argmax selected
`min_inliers=3`. The val argmax pinned to the **grid floor (5) in all four conditions**, so the
sweep did not bracket the optimum from below. That is a real caveat, and it is bounded by the
baseline's own measured `min_inliers` sensitivity:

| class | SIFT val F1 @ mi=5 | @ mi=3 | gain from 5 → 3 | SP val gap to close |
|---|---|---|---|---|
| door | 0.2468 | 0.2542 | **+0.007** | 0.051 (t2) / 0.060 (pw) |
| window | 0.2143 | 0.2333 | **+0.019** | 0.077 (t2) / 0.087 (pw) |

Extending the SuperPoint grids down to `min_inliers=3` is worth roughly **+0.01 to +0.02 val F1**
by the baseline's own gradient — an order of magnitude short of the **0.051–0.087** gap it would
need to close. (Explicitly an extrapolation, not a measurement; it bounds the caveat rather than
eliminating it.) The val grids are also **monotonically degenerate upward**: F1 falls from 0.195 at
`min_inliers=5` to 0.029 at `min_inliers=20` on doors while precision saturates to **1.000 at
recall 0.015** — the classic "finds almost nothing but is right about it" collapse. There is no
hidden operating point in this grid.

### Verdict on hypothesis 2 — DISPROVEN; nothing to revert

Revert filter (*keep only if it measurably beats baseline F1, without an AP50/precision collapse,
without dropping below 28/28 coverage*):

| criterion | result |
|---|---|
| beats baseline test F1 | **NO** — loses in 4/4 cells (−0.024 to −0.096) |
| beats baseline val F1 | **NO** — loses in 4/4 cells (−0.051 to −0.087) |
| no AP50 collapse | **NO** — AP50 falls in 3/4 cells, by up to −0.189 |
| holds 28/28 coverage | **NO** — windows drop to 26/28 on a hard ONNX/CoreML crash |
| latency | 5.3–6.9× baseline, disclosed |

**Nothing comes out of the diff, because nothing ever went in.** Re-verified at the end of the
investigation:

| surface | state |
|---|---|
| `src/object_search/search/sparse_geo.py` | byte-identical to pre-task `df64af1` — `git diff` empty, SHA-256 `244a2dcd…7bd87883` on both |
| `_TUNING_GRIDS["sparse-geo"]` | still the plain baseline cross, `min_inliers ∈ (2,3,4,5,6,8,10) × nms_iou ∈ (0.3,0.5)` — no SuperPoint entries were ever added |
| `git diff --stat src/` | empty |

`backend` therefore still defaults to `"sift"`, which it would have regardless: the MagicLeap
weights are non-commercial research-only and gitignored, so SuperPoint could never have been the
shipped default even had it won.

---

## Combined (Step C) — not applicable

Step C combines "whichever of A and B individually helped". **Neither did.** Hypothesis 1 is
disproven at both voting modes and fully reverted (commit `8ab99a2`); hypothesis 2 is disproven in
all four class × voting-mode cells and was never committed. There is no surviving change to
combine, and no recommended floor-plan config other than the one already shipped.

**Final recommended config = the pre-task shipped default**, unchanged:
`backend="sift"`, `voting_mode="single-4dof"`, no `allow_mirror` field, tuned
`min_inliers=3, nms_iou=0.3` for the floor-plan domain (which is what the *existing* committed grid
already selects).

---

## Reverted — the complete list, with numbers

| hypothesis | best measured outcome | verdict | what was removed |
|---|---|---|---|
| 1 — mirror acceptance (`allow_mirror`) | door −0.004 / window −0.020 F1 at `single-4dof`; door −0.008 / window −0.032 vs its own control at `pairwise-4dof`; precision and AP50 down in every cell; ~2× latency | DISPROVEN at both voting modes | config field, `_reflected_similarity_2pt`, `_vote_single_4dof_reflected`, `_Vote.reflect`, the chirality bin, the `_is_degenerate` branch, `tests/test_sparse_geo_mirror.py` (216 lines), the doc row — all removed in commit `8ab99a2` |
| 2 — SuperPoint backend | door −0.024 (best of two voting modes) / window −0.048 F1; AP50 down in 3/4 cells; window coverage 28/28 → 26/28; 5.3–6.9× latency | DISPROVEN in 4/4 cells | **nothing — never committed**; measured entirely through the throwaway driver, `src/` untouched throughout |

---

## Regression guard — the shipped defaults are provably unchanged

Two independent checks, both run after all measurement was complete.

**1. The file is byte-identical.** `git diff df64af1 -- src/object_search/search/sparse_geo.py` is
empty, and the SHA-256 of the working-tree file equals the SHA-256 of the blob at `df64af1`:
`244a2dcd64eefaf292253c44bfeb637f5b361e06f0de91df692edac57bd87883`. `git diff --stat src/` is
empty across the whole source tree.

**2. The default-config synthetic-regime numbers reproduce exactly.** `regime_harness.py` re-run at
the shipped defaults, diffed field-by-field against `baseline-regimes.json` captured before any
change:

| regime | F1 | AP | tp / fp / fn | identical? |
|---|---|---|---|---|
| EASY | 0.4545 → 0.4545 | 0.4000 → 0.4000 | 25/0/60 → 25/0/60 | **yes** |
| TEXTURED | 1.0000 → 1.0000 | 1.0000 → 1.0000 | 164/0/0 → 164/0/0 | **yes** |
| VARIED | 0.7597 → 0.7597 | 0.7111 → 0.7111 | 117/36/38 → 117/36/38 | **yes** |
| CLUTTERED | 0.8634 → 0.8634 | 0.8360 → 0.8360 | 139/23/21 → 139/23/21 | **yes** |

Every precision / recall / F1 / AP / tp / fp / fn field matches exactly on all four regimes. Only
`p50_latency_ms` differs (roughly 3–4× higher on the re-run) — that is machine load and thermal
state on a laptop that had just finished a multi-hour ONNX sweep, not a code change; latency is not
a correctness field, and the identical tp/fp/fn counts are the actual proof the code path is
unchanged.

A full `pixi run bench` was deliberately **not** run: with the method file byte-identical it would
consume hours to reproduce numbers that cannot have moved, and the exact-match regime diff above is
the stronger claim the plan asks for (Task 4, step 7).
