# Engineering log — investigating `sparse-geo` on floor plans (2026-08-08)

A record of a **fully negative** investigation. Two independent hypotheses were raised against
Method ② `sparse-geo`'s flat door recall on the floor-plan target domain, each was measured in
isolation against a reproduced baseline, and **both were disproven**. Nothing shipped: the
recommended config is byte-for-byte the one that was already shipped before this work started.

That outcome is the deliverable. A method with no improvement report is not the same as a method
with a report that says "these two plausible levers were tried, here are the numbers, neither
paid off, and here is what that rules out". This log is the second thing.

Cross-references: the method is documented in [`../methods/sparse-geo.md`](../methods/sparse-geo.md);
the target-domain numbers come from [`../eval/floorplans-findings.md`](../eval/floorplans-findings.md);
deferred work is in [`../ROBUSTNESS-BACKLOG.md`](../ROBUSTNESS-BACKLOG.md). The sibling
template-correlation investigations are [`ncc-improvement.md`](ncc-improvement.md) and
[`mosse-improvement.md`](mosse-improvement.md) — see
[Why this investigation looks different](#why-this-investigation-looks-different-from-the-nccmosse-ones)
for the structural reason this one scoped to reflection rather than to a rotation bank, and why it
did not pay off the way those did.

## Symptom

On the Roboflow floor-plans-500 target domain, `sparse-geo` scores doors test F1 **0.219**
(P 0.44 / R 0.15), 3rd of six methods. The interesting part is not the rank — it is the shape of
the recall:

| method | small | medium | large |
|---|---|---|---|
| `sparse-geo` **(doors)** | **0.14** | **0.13** | **0.14** |
| `sparse-geo` (windows) | 0.13 | 0.25 | **0.40** |
| `ncc` (doors) | 0.31 | 0.31 | 0.29 |

**Windows behave exactly the way a keypoint method should**: recall climbs 0.13 → 0.40 with symbol
size, because a bigger symbol carries more texture and more keypoints. **Doors do not move at all.**
Flatness across size is the signature of a *structural* failure — something that fails identically
on a large door and a small one — rather than a "too few pixels" problem, which would show the
window-shaped curve. A door that is three times larger is still not found.

Two structural explanations were plausible enough to test, and they are not mutually exclusive:

1. **Mirror rejection.** `_is_degenerate()` unconditionally rejects any fitted similarity with
   `det < 0`. Door symbols are routinely drawn as genuine mirror images (same swing arc, opposite
   hinge hand), so for half the doors in a plan the *correct* fit may be exactly the fit that gate
   discards — and that would fail identically at every symbol size.
2. **Low keypoint yield on line-art.** A door symbol is a straight line plus a quarter-circle arc:
   almost no DoG-style blob texture for SIFT. A learned detector (`backend="superpoint"`) might
   find structure SIFT cannot.

## The measurement setup

Every number below is measured on CPU (osx-arm64) with the committed split manifests and the
seeded exemplar sampler, tuning on `val` (56 plans) and scoring once on `test` (28 plans), argmax
F1 @ IoU 0.5. A sparse-geo-only `tune-floorplans` sweep is **1m37s** on doors and **1m58s** on
windows, so the whole loop ran locally and **no GPU box was provisioned**.

The baseline was captured **before any source change** and reconciled against the published
findings table to three decimal places on every cell:

| | published | measured here | verdict |
|---|---|---|---|
| door tuned F1 | 0.219 | 0.2194 | reproduces |
| door P / R | 0.44 / 0.15 | 0.442 / 0.146 | reproduces |
| window tuned F1 | 0.309 | 0.3092 | reproduces |
| window P / R | 0.63 / 0.21 | 0.627 / 0.205 | reproduces |

A baseline that does not reproduce invalidates every delta measured against it, so this check came
first. Synthetic-regime numbers (EASY / TEXTURED / VARIED / CLUTTERED) were captured at the same
time as the regression guard.

## Root cause investigation — where mirrored doors actually die

**The mirror gate is not where they die.** Hypothesis 1's premise was diagnosed before it was
implemented, by instrumenting 16 door val plans at the shipped defaults (monkeypatching
`_is_degenerate` and `_ransac_similarity` from a throwaway script, so the method file carried no
diagnostic scaffolding):

| stage | measured |
|---|---|
| GT door instances over the 16 plans | 157 |
| correspondences | 2 664 |
| **Hough peaks hypothesized** | **55** |
| instances verified | 33 |

| degeneracy-gate outcome over those 55 peaks | count |
|---|---|
| accepted | 43 |
| rejected: **mirror (`det < 0`)** | **2** |
| rejected: scale | 9 |

**The mirror branch fires 2 times in 55 peaks.** Relaxing it alone could recover at most ~2
detections across 16 plans — very nearly an inert control. The funnel collapses far upstream:
**only 55 peaks are ever hypothesized for 157 ground-truth doors**, while correspondences are
plentiful (2 664). The loss is at the **voting** stage. `_vote_single_4dof` and
`_proper_similarity_2pt` compute only the orientation-preserving branch, so a mirrored instance's
correspondences predict a *wrong* centre, scatter, and never accumulate into a peak — they never
reach the degeneracy gate at all.

That is why hypothesis 1 was then built as **one** end-to-end change (reflected pose votes **and**
the relaxed gate) rather than as an isolated gate relaxation that the diagnosis had already shown
could not move the numbers.

### A second finding: SIFT orientations are not mirror-consistent

On a controlled scene (one SIFT-rich tile, one identical copy, one horizontally mirrored copy):

| target | correspondences landing inside it | of those, geometrically correct |
|---|---|---|
| identical copy | 94 | **33 (35%)** |
| mirrored copy | 78 | **13 (17%)** |

and the reflected pose angle `α = scene_angle + crop_angle` on those 13 correct correspondences
scatters across ~80–280° instead of clustering at the true 180°.

So even with the reflected branch wired, **`single-4dof` structurally cannot cluster a mirrored
instance**: it derives a pose from one keypoint frame, and SIFT's orientation assignment on a
mirrored patch is not a predictable function of the original's. `pairwise-4dof` fits a pose from
two point pairs and never reads an orientation — and there the mirrored instance **is** recovered
(IoU 0.98 on the controlled scene). This is what made the `pairwise-4dof` arm of hypothesis 1
necessary: testing mirror handling only in the voting mode that provably cannot represent it would
have been a rigged test.

## Hypothesis 1 — mirror acceptance

An `allow_mirror` flag was added to `SparseGeoConfig` (frozen, default `False`), threaded into
`_is_degenerate`, and — per the diagnosis — extended end-to-end with reflected pose votes. Then
measured.

### At `single-4dof` (the shipped voting mode)

Grid `allow_mirror=True × min_inliers ∈ {2,3,4,5,6,8,10} × nms_iou ∈ {0.3,0.5}`, the same shape as
the baseline grid so the two are directly comparable.

| class | config | F1 | P | R | AP50 | ΔF1 |
|---|---|---|---|---|---|---|
| door | baseline | 0.219 | 0.442 | 0.146 | 0.168 | — |
| door | **+`allow_mirror`** | 0.216 | 0.356 | 0.155 | 0.161 | **−0.004** |
| window | baseline | 0.309 | 0.627 | 0.205 | 0.293 | — |
| window | **+`allow_mirror`** | 0.290 | 0.492 | 0.205 | 0.286 | **−0.020** |

The false-positive direction is exactly the predicted risk and it is the whole story: precision
falls on both classes while recall barely moves (door +0.009, window ±0.000). A tighter
`min_inliers` did not compensate — the grid still selected `min_inliers=3`, i.e. no operating point
in the sweep traded the lost precision back.

### At `pairwise-4dof` (the voting mode where it *can* work) — with a control

Switching voting mode is itself a change, so a `pairwise-4dof`-without-mirror **control** was run;
without it, any movement would be misattributed to mirror handling. Four conditions, identical grid
`min_inliers ∈ {5,8,12,16,20} × nms_iou ∈ {0.3,0.5}`:

| class | condition | val F1 | test F1 | P | R | AP50 | p50 ms |
|---|---|---|---|---|---|---|---|
| door | baseline (single-4dof) | 0.254 | **0.219** | 0.442 | 0.146 | **0.168** | 65.4 |
| door | pairwise-4dof, **no** mirror *(control)* | 0.238 | **0.231** | 0.323 | 0.180 | 0.096 | 113.6 |
| door | pairwise-4dof **+ mirror** | 0.230 | **0.223** | 0.257 | 0.197 | 0.062 | 202.6 |
| window | baseline (single-4dof) | 0.233 | **0.309** | 0.627 | 0.205 | **0.293** | 60.9 |
| window | pairwise-4dof, **no** mirror *(control)* | 0.222 | **0.256** | 0.360 | 0.199 | 0.186 | 184.0 |
| window | pairwise-4dof **+ mirror** | 0.213 | **0.225** | 0.248 | 0.205 | 0.123 | 337.3 |

**The isolated effect of `allow_mirror`** (mirror vs. its own control — same voting mode, same
grid, same splits, mirror the only difference):

| class | ΔF1 (val) | ΔF1 (test) | ΔP | ΔR | ΔAP50 | Δp50 |
|---|---|---|---|---|---|---|
| door | −0.008 | −0.008 | −0.066 | +0.017 | −0.034 | +89 ms (1.78×) |
| window | −0.009 | −0.032 | −0.112 | +0.006 | −0.064 | +153 ms (1.83×) |

Mirror loses on **both classes, both splits, and on F1, precision and AP50 alike**, while roughly
doubling latency. Consistency across val *and* test rules out a test-split fluke.

**Was the door gain ever attributable to mirror? No.** The only cell that beats baseline F1 is
door, pairwise-4dof, *without* mirror (0.219 → 0.231). The mirror row is *lower* than that control.
The sliver of door gain belongs entirely to switching the voting mode.

**And the control fails too.** `pairwise-4dof` is not rescuable as a floor-plan recommendation: on
doors its +0.012 F1 comes with precision −27% and **AP50 0.168 → 0.096 (−43%)** — a ranking-quality
collapse, more and worse-ordered boxes scraping a marginally higher F1 at one hand-picked operating
point. On windows it is an outright regression on every axis. Latency is 1.7–3.0× baseline.
`voting_mode` is a pre-existing documented knob, so there is nothing to revert for it — it simply
does not become a recommended floor-plan setting.

**Verdict: DISPROVEN at both voting modes.** `allow_mirror` improved floor-plan test F1 in **zero
of four** class × voting-mode cells. Not merely inert — mildly harmful.

## Hypothesis 2 — the SuperPoint backend

Measured against the **original baseline**, not against hypothesis 1 (which was already reverted),
so the two are never conflated.

### The feasibility probe — the premise is half-right and it does not matter

Exemplar-crop keypoint counts on 5 door plans, which is the quantity the premise is actually about:

| plan | GT doors | SIFT crop kp | SuperPoint crop kp | SIFT scene kp | SuperPoint scene kp |
|---|---|---|---|---|---|
| 109 | 12 | 0 | 1 | 291 | 466 |
| 110 | 12 | 2 | 1 | 711 | 550 |
| 119 | 7 | **33** | **12** | 2 170 | 5 986 |
| 120 | 8 | 6 | 8 | 813 | 839 |
| 121 | 4 | 0 | 3 | 368 | 291 |

SuperPoint does fire marginally on the barren crops where SIFT finds nothing (0 → 1, 0 → 3), but
finds **less than half** as much on the one texture-rich crop (33 → 12). `min_exemplar_keypoints=8`
is cleared by 1/5 crops under SIFT and 2/5 under SuperPoint — **4/5 plans abstain under both
backends**. Meanwhile SuperPoint yields ~2.8× the *scene* keypoints on the textured plan, which is
where the O(n²) `pairwise-4dof` cost comes from. The cost side of the trade scales; the benefit
side does not.

### The full sweep — 4 conditions

Both frameless-compatible voting modes were swept, so the verdict is not an artifact of one voting
choice. (`single-4dof` raises at config time for a frameless backend, by design.)

| class | condition | val F1 | test F1 | P | R | AP50 | coverage | p50 ms |
|---|---|---|---|---|---|---|---|---|
| door | *baseline* (SIFT, single-4dof) | 0.254 | **0.219** | 0.442 | 0.146 | **0.168** | **28/28** | **65.4** |
| door | SuperPoint + `pairwise-4dof` | 0.195 | **0.189** | 0.519 | 0.116 | 0.125 | 28/28 | 439.3 |
| door | SuperPoint + `translation-2dof` | 0.203 | **0.195** | 0.788 | 0.112 | 0.184 | 28/28 | 348.5 |
| window | *baseline* (SIFT, single-4dof) | 0.233 | **0.309** | 0.627 | 0.205 | **0.293** | **28/28** | **60.9** |
| window | SuperPoint + `pairwise-4dof` | 0.147 | **0.213** | 0.488 | 0.136 | 0.104 | **26/28** | 339.5 |
| window | SuperPoint + `translation-2dof` | 0.156 | **0.261** | 0.793 | 0.156 | 0.159 | **26/28** | 418.3 |

| class | condition | ΔF1 (val) | ΔF1 (test) | ΔAP50 | latency |
|---|---|---|---|---|---|
| door | SP + `pairwise-4dof` | −0.059 | **−0.030** | −0.043 | **6.72×** |
| door | SP + `translation-2dof` | −0.051 | **−0.024** | +0.016 | **5.33×** |
| window | SP + `pairwise-4dof` | −0.087 | **−0.096** | −0.189 | **5.57×** |
| window | SP + `translation-2dof` | −0.077 | **−0.048** | −0.133 | **6.87×** |

**SuperPoint loses F1 in 4 of 4 cells, on both splits, in both voting modes.** The single positive
AP50 cell in the block sits next to an F1 that is down 0.024, so it does not rescue the hypothesis.

### Abstentions and a hard coverage failure

Plans producing any output at all:

| class | condition | scored | abstained | producing output |
|---|---|---|---|---|
| door | baseline | 28 | 14 | **14** |
| door | SP (either mode) | 28 | 13 | 15 |
| window | baseline | 28 | 12 | **16** |
| window | SP + pairwise | 26 | 18 | **8** |
| window | SP + translation | 26 | 19 | **7** |

Doors are a wash. Windows **halve** — the method stops answering on half the plans it used to
answer on, which is a recall loss no threshold can recover.

Window coverage drops **28/28 → 26/28** in both SuperPoint conditions, and the mechanism is a
**crash, not an abstention**:

```
[ONNXRuntimeError] : 1 : FAIL : ... CoreMLExecutionProvider ...
Input (/Cast_12_output_0) has a dynamic shape ({-1,2}) but the runtime shape ({0,2})
has zero elements. This is not supported by the CoreML EP.
```

When SuperPoint detects **zero keypoints** on a sparse line-art crop or scene, the exported graph
hands a zero-row tensor to a node the CoreML execution provider cannot handle, and the query raises
instead of abstaining cleanly. This fired **92 times** across one window run. The backend's failure
mode on precisely the low-texture regime hypothesis 2 was built to exploit is a hard error.

### Was the sweep unfair? The grid floor, checked rather than waved away

The SuperPoint grids start at `min_inliers=5` while the SIFT baseline selected `min_inliers=3`, and
the val argmax pinned to that floor in all four conditions — so the sweep did not bracket the
optimum from below. That caveat is bounded by the baseline's own measured `min_inliers` gradient:

| class | SIFT val F1 @ mi=5 | @ mi=3 | gain from 5 → 3 | SuperPoint val gap to close |
|---|---|---|---|---|
| door | 0.2468 | 0.2542 | **+0.007** | 0.051–0.060 |
| window | 0.2143 | 0.2333 | **+0.019** | 0.077–0.087 |

Extending the SuperPoint grids to `min_inliers=3` is worth roughly **+0.01–0.02 val F1** by the
baseline's own gradient — an order of magnitude short of the gap it would need to close. (Stated
explicitly as an extrapolation, not a measurement; it bounds the caveat rather than eliminating
it.) The val grids are also monotonically degenerate upward: door F1 falls 0.195 → 0.029 as
`min_inliers` goes 5 → 20, with precision saturating to **1.000 at recall 0.015** — the classic
"finds almost nothing but is right about it" collapse. There is no hidden operating point here.

**Verdict: DISPROVEN in all four cells** — fails on F1 (4/4), on AP50 (3/4), on coverage (windows),
and costs 5.3–6.9× latency.

## Combine — not applicable

Step C of the plan combines "whichever hypotheses individually helped". **Neither did.** There was
no surviving change to combine.

## Result — final / recommended config

**Unchanged from the pre-task shipped default.** No config field was added, no default was moved,
no tuning-grid entry was added:

| setting | value | note |
|---|---|---|
| `backend` | `sift` | never in question — the MagicLeap SuperPoint weights are non-commercial research-only and gitignored, so SuperPoint could not have been the shipped default even had it won |
| `voting_mode` | `single-4dof` | `pairwise-4dof` measured worse on windows and traded a −43% AP50 for +0.012 door F1 |
| mirror handling | **rejection**, as since Phase 5 | `_is_degenerate` still rejects `det < 0`; no `allow_mirror` field exists |
| floor-plan domain tuning | `min_inliers=3, nms_iou=0.3` | what the *existing* committed grid already selects |

`_two_point_models` / `_model_from_complex(reflect=...)` remain untouched: RANSAC has always fitted
both the proper and reflected 2-point model and kept whichever has more inliers, with
`_is_degenerate` rejecting a reflected winner. That pre-existing mirror *rejection* contract is what
makes the rejection non-vacuous, and only the mirror *acceptance* path this task added was removed.

## What I tried and reverted — with numbers

- **`allow_mirror` (hypothesis 1)** — reverted in full, commit `8ab99a2`. Measured −0.004 (door) /
  −0.020 (window) F1 at `single-4dof`, and −0.008 / −0.032 against its own control at
  `pairwise-4dof`, with precision and AP50 down in every cell and ~2× latency. Removed: the config
  field, `_reflected_similarity_2pt`, `_vote_single_4dof_reflected`, the `reflect` field on `_Vote`,
  the chirality bin dimension in `_accumulate_votes`, the `_is_degenerate` branch and parameter,
  four docstring blocks, `tests/test_sparse_geo_mirror.py` (216 lines), and the config-reference
  doc row. Per the plan and CLAUDE.md, a disproven idea is **not** parked behind an off-by-default
  flag — an advertised control that never helps is an inert control.
- **`voting_mode="pairwise-4dof"` as a floor-plan recommendation** — not adopted. Doors +0.012 F1
  but AP50 −43%; windows regress on every axis; 1.7–3.0× latency. Nothing to revert (it is a
  pre-existing documented knob); it simply is not recommended.
- **`backend="superpoint"` (hypothesis 2)** — never committed, so the "revert" is a genuine no-op
  in `src/`. Measured entirely through a throwaway driver that reuses the project's own
  `tune_method` / `_evaluate`. Best case −0.024 (door) / −0.048 (window) F1, AP50 down in 3/4
  cells, window coverage 28/28 → 26/28 on a hard ONNX/CoreML crash, 5.3–6.9× latency.

## Fairness — nothing here is fit to the labels

Tuning read `val` **only**; `test` was scored exactly once per frozen config. All selections are
argmax-F1-on-val, the tuning protocol's normal and allowed use of labels — never a test peek, and
no grid decision was conditioned on a test outcome. `val`/`test` membership comes from the
committed split manifests and was not regenerated. Acceptance rules (`min_inliers`, `nms_iou`, the
degeneracy gates) are keyed to the geometry of the fit and the shape of the vote distribution,
never to ground-truth boxes. The `pairwise-4dof` **control** exists specifically so that a change
in voting mode could not be silently credited to mirror handling.

## Cost

Latency was reported alongside every F1, including for the losing conditions. `allow_mirror`
roughly doubled p50 (the reflected branch doubles cast votes and histogram occupancy);
`pairwise-4dof` cost 1.7–3.0×; SuperPoint cost 5.3–6.9× with tail maxima of 4.1–9.4 s. On the
licence side, `models/superpoint.onnx` is MagicLeap **non-commercial research-only**, gitignored,
and fetched only by `pixi run fetch-models` — a permanent ceiling on that backend independent of
how it scores.

## Regression guard — the shipped defaults are provably unchanged

`src/object_search/search/sparse_geo.py` is **byte-identical** to the pre-task commit `df64af1`:
`git diff` is empty and the SHA-256 matches on both
(`244a2dcd64eefaf292253c44bfeb637f5b361e06f0de91df692edac57bd87883`). `git diff --stat src/` is
empty across the whole source tree.

Independently, the default-config synthetic-regime numbers were re-measured and diffed field by
field against the pre-change capture:

| regime | F1 | AP | tp / fp / fn | identical? |
|---|---|---|---|---|
| EASY | 0.4545 → 0.4545 | 0.4000 → 0.4000 | 25/0/60 → 25/0/60 | **yes** |
| TEXTURED | 1.0000 → 1.0000 | 1.0000 → 1.0000 | 164/0/0 → 164/0/0 | **yes** |
| VARIED | 0.7597 → 0.7597 | 0.7111 → 0.7111 | 117/36/38 → 117/36/38 | **yes** |
| CLUTTERED | 0.8634 → 0.8634 | 0.8360 → 0.8360 | 139/23/21 → 139/23/21 | **yes** |

Every precision / recall / F1 / AP / tp / fp / fn field matches exactly. A full `pixi run bench`
was deliberately not run: with the method file byte-identical it would spend hours reproducing
numbers that cannot have moved, and this exact-match diff is the stronger claim.

## Why this investigation looks different from the `ncc`/`mosse` ones

[`ncc-improvement.md`](ncc-improvement.md) and [`mosse-improvement.md`](mosse-improvement.md)
attacked the *same* flat-recall symptom on the *same* floor plans and both won decisively (doors
0.248 → 0.358 for `ncc`; a comparable cardinal-bank win for `mosse`). Their winning lever was an
explicit **cardinal rotation bank** (0/90/180/270°) plus an optional mirrored template.

That lever does not exist for `sparse-geo`, and the reason is structural rather than incidental:

- `ncc` and `mosse` are **template-correlation** methods. They correlate the exemplar's raw pixels
  (or a filter trained on them) against the scene, so they have **no built-in rotation invariance
  whatsoever** — a door on a perpendicular wall is simply unreachable unless a rotated copy of the
  template is explicitly manufactured and correlated. Their rotation bank is not a tuning nicety;
  it is the only mechanism by which a rotated instance can be found at all.
- `sparse-geo`'s SIFT keypoints are **rotation-invariant by construction**. Each keypoint carries a
  canonical orientation, descriptors are computed in that frame, and 4-DoF Hough voting recovers
  the relative rotation between exemplar and instance as an output. There is nothing to sweep: a
  door rotated 90° is already, in principle, matchable.

That is why this investigation correctly scoped to **reflection specifically**. A reflection is not
an element of the rotation group, so no amount of rotation invariance — built-in or banked — can
reach it. `ncc`/`mosse` needed *both* a rotation bank (which they lacked) and a mirror option;
`sparse-geo` already had rotation handled and only the reflection question was open. Scoping to
reflection was the right call; it just turned out that reflection is not where `sparse-geo`'s doors
are lost either.

And unlike `ncc` and `mosse`, **neither lever tested here paid off** — which is itself the honest,
reportable outcome for this method. The two cheap structural explanations for flat door recall are
now ruled out with measurements rather than left as plausible-sounding folklore.

## Deferred work and the open question

**The symptom remains unexplained.** Flat door recall-by-size (0.14 / 0.13 / 0.14) is *not*
accounted for by mirror rejection (the gate fires 2 times in 55 peaks) and *not* accounted for by
keypoint-detector choice (SuperPoint is worse on every measured axis). No diagnosis is offered here
that was not measured, and the honest state of the question is: open.

What *is* established and should anchor the next attempt: the funnel collapses at **peak
hypothesis** — 55 peaks for 157 ground-truth doors, from 2 664 correspondences. Correspondences are
plentiful; peaks are not. Whatever the real cause is, it lives between correspondence and peak.

Candidate directions worth investigating later — **explicitly speculation, not conclusions, and
none of them measured**:

- The `min_inliers` floor may interact badly with the sparse per-instance correspondence count on
  small stamped symbols: if a genuine door instance only ever attracts 2–4 correspondences, no
  inlier threshold that also suppresses false positives can accept it, and the flatness would then
  be a property of the *symbol*, not of its rendered size.
- Something about door-symbol geometry itself — the arc-plus-line form is close to self-similar
  under partial matching, which could make correspondences from *different* door instances
  interfere in the vote accumulator rather than reinforce within one instance.
- Per-instance funnel instrumentation (correspondences → votes → peak, attributed to individual
  ground-truth boxes rather than pooled) would distinguish these, and is the cheapest next
  measurement.

Carried forward in [`../ROBUSTNESS-BACKLOG.md`](../ROBUSTNESS-BACKLOG.md) and mirrored in the
`sparse_geo.py` docstring, unchanged by this investigation: multi-model fitting (J-linkage /
T-linkage), **DISK / ALIKED backends** (also the permissive-licence escape from SuperPoint's
non-commercial terms — and, given hypothesis 2's result, now a question about detector *quality on
line-art* rather than only about licensing), post-hoc orientation/scale assignment for frameless
keypoints, and LoFTR / RoMa dense matching for low-texture objects.

## Verification

`pixi run quality` green — Ruff and Ruff-format clean, MyPy strict clean, **784 passed / 16
skipped, coverage 92.78%** (floor is 80%). `pixi run docs-build` green (`mkdocs build --strict`)
with this report in the nav. No source file changed in the final state of this work, so the suite
and coverage are those of the pre-task tree; the regression guard above is the substantive check.
