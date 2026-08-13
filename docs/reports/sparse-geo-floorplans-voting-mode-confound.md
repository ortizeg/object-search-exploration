# Follow-up — is the floor-plans SuperPoint verdict a voting-mode confound? (2026-08-13)

[`sparse-geo-improvement.md`](sparse-geo-improvement.md) found `backend="superpoint"` DISPROVEN in
4/4 cells on floor-plans-500: F1 down 0.024–0.096, AP50 down in 3/4 cells, window coverage
28/28 → 26/28 on a hard ONNX/CoreML crash, 5.3–6.9× latency. Every one of those deltas compares
`sift/single-4dof` against `superpoint/{translation-2dof,pairwise-4dof}` — two variables at once.
`SparseGeoConfig` refuses `voting_mode="single-4dof"` for a frameless backend at construction
(`_reject_single_4dof_for_frameless_superpoint`), so SuperPoint was *forced* onto a different voting
mode and never had a same-mode SIFT baseline to be measured against.
[`sparse-geo-real-objects-superpoint-spike.md`](sparse-geo-real-objects-superpoint-spike.md) then
measured that missing control on a *different* domain (real photographic texture) and found
switching SIFT itself off `single-4dof` costs **0.055–0.079 F1** and **0.050–0.145 AP** on its own,
independent of backend — large enough to account for a real share of the floor-plans deltas. A
hypothesis about floor-plans data can only be settled on floor-plans data, so this follow-up runs
the same SIFT controls on floor-plans-500 itself.

## Method

[`scripts/sparse_geo_floorplan_voting_mode_experiment.py`](https://github.com/ortizeg/object-search-exploration/blob/main/scripts/sparse_geo_floorplan_voting_mode_experiment.py)
(committed at `39271eb`) reuses `object_search.eval.tuning.run_domain_tuning`'s `grids=` seam
unmodified — the exact harness and protocol that produced the published table: tune on val (56
plans), argmax F1 @ IoU 0.5, freeze, score test (28 plans) once. Two grids were swept, because a
second, unnamed confound sits alongside the voting-mode one:

- **COMMITTED grid** (`min_inliers` ∈ {2,3,4,5,6,8,10} × `nms_iou` ∈ {0.3,0.5}) — what the
  published `sift/single-4dof` baseline was tuned over.
- **SUPERPOINT-MATCHED grid** (`min_inliers` ∈ {5,8,12,16,20} × `nms_iou` ∈ {0.3,0.5}) — what the
  published SuperPoint rows, and the published `pairwise-4dof` no-mirror control from the mirror
  hypothesis, were tuned over. Its `min_inliers` floor of 5 is a disclosed caveat in the original
  report; matching it here is what makes the SuperPoint comparison grid-fair on *both* axes at once.

**Where it ran.** At the user's explicit direction, the sweep ran on a rented vast.ai CPU box
(offer `45169473`, $0.048/hr, 40 vCPUs) — this is classical SIFT/RANSAC/Hough-voting, no GPU-bound
step. Code shipped via `git archive HEAD | gzip` (no push to origin). The floor-plans-500 dataset is
a manual Roboflow drop with no auto-fetch; it was symlinked into this worktree from the primary
checkout's already-converted `datasets/floorplans-{door,window}/` and shipped to the box as a
separately dereferenced tarball (`tar -czh`, since `git archive` only carries tracked files and the
symlink is gitignored). Results were pulled back over `scp` and the instance was destroyed
immediately after. Total wall-clock: **6850s (~1h54m)** across 12 tuning runs; total cost **≈$0.09**.

**A `.gitignore` fix landed alongside this** (commit `1f76ff3`): a trailing-slash gitignore pattern
matches only a real directory, never a symlink (documented git behavior), so `/datasets/` alone did
not hide the symlinked-in dataset from `git status`. Added `/datasets` (no trailing slash) alongside
it — the same fix the `models/` precedent will eventually need if it is ever symlinked the same way.

No `SuperPoint` condition and no `sift/single-4dof` re-run on a third grid: those seven numbers
(the two `single-4dof` rows plus five `superpoint` rows) are already published and gated, and are
pulled forward as literals below rather than re-measured.

## Baseline reconciliation

Six published targets, checked before any delta was computed:

| target | published | measured | delta | agrees (≤3–4dp)? |
|---|---|---|---|---|
| committed, `single-4dof`, door — test F1 / P / R | 0.219 / 0.442 / 0.146 | 0.2194 / 0.442 / 0.146 | ~0 | yes |
| committed, `single-4dof`, window — test F1 / P / R | 0.309 / 0.627 / 0.205 | 0.3092 / 0.627 / 0.205 | ~0 | yes |
| sp-matched, `single-4dof`, door — val F1 | 0.2468 | 0.2468 | ~0 | yes |
| sp-matched, `single-4dof`, window — val F1 | 0.2143 | 0.2147 | +0.0004 | **no** (4dp) |
| sp-matched, `pairwise-4dof`, door (`allow_mirror`-era control) — val F1 / test F1 | 0.238 / 0.231 | 0.2361 / 0.2343 | −0.0019 / +0.0033 | **no** (3dp) |
| sp-matched, `pairwise-4dof`, window (`allow_mirror`-era control) — val F1 / test F1 | 0.222 / 0.256 | 0.2212 / 0.2520 | −0.0008 / −0.0040 | **no** (3dp) |

The two headline metrics this project leads with (committed-grid `single-4dof` test F1, the number
every other report cites) reproduce essentially exactly. The three flagged disagreements are all
0.0004–0.0040 in size — an order of magnitude below any of the effects this report measures — and
read as ordinary tie-breaking/rounding noise between this clean post-revert tree and the
`allow_mirror`-era tree the control rows were originally measured in, not a reproduction failure.
Disclosed per the ≤3dp bar rather than rounded away, as both prior reports do.

## The comparison table (superpoint-matched grid — the grid the SuperPoint rows were measured on)

Every SuperPoint row below is a **published literal** from `sparse-geo-improvement.md`; every SIFT
row is **newly measured** in this follow-up. Two delta columns: against the shipped
`sift/single-4dof` baseline (**committed grid**, 0.219/0.309 — the published comparison), and against
the **same-voting-mode SIFT control at this grid** (the fair comparison this follow-up creates).

### Door

| condition | source | test F1 | P | R | AP50 | coverage | p50 ms | Δ vs shipped baseline | Δ vs same-mode SIFT |
|---|---|---|---|---|---|---|---|---|---|
| `sift/single-4dof` | [published] committed grid | 0.219 | 0.442 | 0.146 | 0.168 | 28/28 | 65.4 | — | — |
| `sift/single-4dof` | [new] sp-matched grid | 0.2123 | 0.525 | 0.133 | 0.168 | 28/28 | 238 | −0.007 | — |
| `sift/translation-2dof` | [new] sp-matched grid | 0.1993 | 0.500 | 0.124 | 0.167 | 28/28 | 268 | −0.020 | — |
| `sift/pairwise-4dof` | [new] sp-matched grid | 0.2343 | 0.321 | 0.185 | 0.099 | 28/28 | 457 | +0.015 | — |
| `superpoint/translation-2dof` | [published] | 0.195 | 0.788 | 0.112 | 0.184 | 28/28 | 348.5 | −0.024 | **−0.004** |
| `superpoint/pairwise-4dof` | [published] | 0.189 | 0.519 | 0.116 | 0.125 | 28/28 | 439.3 | −0.030 | **−0.045** |

### Window

| condition | source | test F1 | P | R | AP50 | coverage | p50 ms | Δ vs shipped baseline | Δ vs same-mode SIFT |
|---|---|---|---|---|---|---|---|---|---|
| `sift/single-4dof` | [published] committed grid | 0.309 | 0.627 | 0.205 | 0.293 | 28/28 | 60.9 | — | — |
| `sift/single-4dof` | [new] sp-matched grid | 0.2944 | 0.707 | 0.186 | 0.293 | 28/28 | 220 | −0.015 | — |
| `sift/translation-2dof` | [new] sp-matched grid | 0.2600 | 0.591 | 0.167 | 0.287 | 28/28 | 259 | −0.049 | — |
| `sift/pairwise-4dof` | [new] sp-matched grid | 0.2520 | 0.344 | 0.199 | 0.186 | 28/28 | 814 | −0.057 | — |
| `superpoint/translation-2dof` | [published] | 0.261 | 0.793 | 0.156 | 0.159 | **26/28** | 418.3 | −0.048 | **+0.001** |
| `superpoint/pairwise-4dof` | [published] | 0.213 | 0.488 | 0.136 | 0.104 | **26/28** | 339.5 | −0.096 | **−0.039** |

The committed-grid block (both classes, all three SIFT conditions) shows the identical sign pattern
and is not reproduced in full here for space; door `pairwise-4dof` test F1 there is 0.236 (Δ+0.017
vs the committed-grid baseline), window `pairwise-4dof` is 0.244 (Δ−0.066) — consistent with the
sp-matched numbers above.

## The isolated voting-mode cost, SIFT-only, per class (sp-matched grid)

The direct floor-plans analogue of the real-objects spike's 0.055–0.079 F1 finding — the quantity
the verdict turns on:

| class | `single-4dof` → `translation-2dof` | `single-4dof` → `pairwise-4dof` |
|---|---|---|
| door | ΔF1 **−0.013**, ΔAP50 −0.001 | ΔF1 **+0.022**, ΔAP50 −0.069 |
| window | ΔF1 **−0.034**, ΔAP50 −0.006 | ΔF1 **−0.042**, ΔAP50 −0.107 |

Door and window diverge sharply. On **doors**, switching to `translation-2dof` costs almost nothing
in F1 and `pairwise-4dof` is a net *F1 gain* with SIFT — the shipped `single-4dof` default is not
even the best voting mode for this class, on this metric, once measured directly. On **windows**,
both non-`single-4dof` modes cost real F1 (−0.034, −0.042) and `pairwise-4dof` costs real AP50 on
both classes (−0.069 door, −0.107 window) regardless of backend — `pairwise-4dof`'s AP50 tax is a
property of the voting mode, not something either backend earns or escapes.

## Verdict: **PARTIALLY** — confirmed on doors, not on windows

The confound hypothesis does not hold uniformly; it holds well on one class and poorly on the other,
and the per-class split is itself the finding.

**Door — effectively CONFIRMED.** The same-mode SIFT control absorbs nearly all of the published
SuperPoint gap. At `translation-2dof`, voting mode alone accounts for 75% of the observed SuperPoint
loss (−0.013 of −0.017), leaving a residual backend-specific gap of only −0.004. At `pairwise-4dof`,
voting mode alone would have been a *gain* for SIFT (+0.022) — so the residual once that's backed
out is −0.045, meaning the true backend-specific gap at this cell is *larger* than the naively
published −0.030, not smaller, even though the confound framing turns out to be the right one to
apply here. And on AP50, at matched voting mode, SuperPoint is actually **ahead** of the same-mode
SIFT control in both modes (+0.017 at `translation-2dof`, +0.026 at `pairwise-4dof`) — a pattern
that echoes the real-objects spike's AP finding on a completely different domain.

**Window — REFUTED as the primary explanation, despite a real partial share.** At
`translation-2dof`, voting mode alone accounts for essentially the *entire* published gap (−0.034 of
−0.033 — 103%, i.e. the residual is a noise-sized +0.001). That part of the story is confirmed. But
at `pairwise-4dof`, voting mode alone explains only 52% (−0.042 of −0.081), leaving a real
backend-specific residual of −0.039 — and, independent of F1 entirely, AP50 stays clearly worse for
SuperPoint than the same-mode SIFT control in **both** window conditions (−0.128 at
`translation-2dof`, −0.082 at `pairwise-4dof`), and window is the class where the published report's
hard ONNX/CoreML crash and 28/28 → 26/28 coverage collapse actually occurred — a failure mode with no
voting-mode analogue at all.

**Net reading.** The published floor-plans SuperPoint verdict survives on windows (there is a real,
voting-mode-independent backend degradation there, on AP50 and on coverage, that this follow-up does
not overturn) but was substantially an artifact of the unmeasured voting-mode confound on doors
(where the residual backend effect, once isolated, is small on F1 and actually favorable on AP50).
Anyone reading `sparse-geo-improvement.md`'s door numbers as "SuperPoint is worse on floor-plans"
should read them instead as "the voting mode SuperPoint was forced onto is worse on floor-plans
doors" — a materially different, and more actionable, claim.

## Cost

The rental: offer `45169473` (machine `52447`, South Korea, $0.048/hr), 12 tuning runs, 6850s
(~1h54m) wall-clock, **≈$0.09** total. `pairwise-4dof` is the expensive voting mode independent of
backend: 2.4–3.5× `single-4dof`'s wall-clock in this sweep (committed-grid door 1441s vs 416s =
3.46×; window 960s vs 444s = 2.16×), consistent with the per-query p50 latency numbers in the tables
above (SIFT `pairwise-4dof` costs 4.1–13.4× the shipped `single-4dof` baseline's own p50, which is
itself a voting-mode tax, not a SuperPoint one). SuperPoint's originally-reported 5.3–6.9× latency
premium is measured against `single-4dof`; against the same-mode SIFT control it is far smaller —
roughly parity at door `pairwise-4dof` (439ms vs 457ms) and even *faster* at window `pairwise-4dof`
(339ms vs 814ms), though clearly slower at both `translation-2dof` cells (1.3–1.6×). Cost, like F1,
turns out to be substantially confounded with voting mode.

The licence ceiling from both prior reports is unchanged: `models/superpoint.onnx` is MagicLeap
**non-commercial research-only**, gitignored — this backend cannot become the shipped default on any
domain regardless of these numbers.

## Regression guard

`git status --porcelain src/ conf/` and `git diff --stat -- src/ conf/` are both empty. `backend`
still defaults to `sift`, `voting_mode` still to `single-4dof`; no tuning-grid entry, config field, or
default was touched. This is a spike, and the shipped defaults were never in play regardless of the
verdict above.

## Deferred work

- **The window backend residual is now the open question**, not the confound itself: −0.039 F1 and
  −0.08 to −0.13 AP50 remain unexplained by voting mode on windows specifically. The DISK / ALIKED
  question carried forward from both prior reports (a permissive-licence learned detector, tested on
  its own merits rather than as a SuperPoint licence workaround) is the natural next probe, scoped to
  windows.
- **The open floor-plans symptom is untouched by anything measured here.** Flat door recall-by-size
  (0.14/0.13/0.14, 55 Hough peaks for 157 ground-truth doors) — `sparse-geo-improvement.md`'s
  original open question — is neither explained nor further constrained by this follow-up; it was
  never a claim about backend or voting mode.
- **A door-specific recommendation is tempting but out of scope here.** This follow-up found
  `pairwise-4dof` beats `single-4dof` on door F1 with SIFT (+0.017 to +0.022) at the cost of AP50
  (−0.069 committed grid) and 2.2–3.5× latency — the same AP50-vs-F1 trade the original mirror
  investigation flagged for `pairwise-4dof` generally (window AP50 −43% for a +0.012 door F1 gain
  there). This is a real signal but a genuinely new tuning question, not a resolution of the
  confound hypothesis this report was scoped to answer, and is left for a dedicated investigation
  rather than folded in here.
