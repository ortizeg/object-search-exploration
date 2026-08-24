# Spike — does the floor-plans SuperPoint verdict hold on real photographs? (2026-08-12)

[`sparse-geo-improvement.md`](sparse-geo-improvement.md) tested `backend="superpoint"` against the
shipped `backend="sift"` default on the Roboflow floor-plans-500 target domain (door/window
detection in floor-plan line-art) and found it **DISPROVEN in all 4 tested cells** — F1 down
0.024–0.096, AP50 down in 3/4 cells, window coverage collapsing 28/28 → 26/28 on a hard
ONNX/CoreML crash, and 5.3–6.9× latency. That verdict was measured on a domain where both
detectors nearly starve for keypoints (4/5 sample exemplar crops abstained under *both* backends)
and where the crash's trigger — SuperPoint returning **zero** keypoints — is exactly the low-texture
failure mode the domain produces. This spike asks the obvious follow-up: is "SuperPoint loses" a
property of the *backend*, or a property of *sparse line-art*? It measures the same backend swap on
`real-objects` — 30 real photographic composite images (10 everyday objects × plain/varied/cluttered
regimes; see [`real-objects-findings.md`](real-objects-findings.md)) — where keypoint starvation is
not expected to happen at all.

## How it was produced

A throwaway driver, [`scripts/sparse_geo_real_objects_experiment.py`](https://github.com/ortizeg/object-search-exploration/blob/main/scripts/sparse_geo_real_objects_experiment.py)
(committed at `7fc33ae`), reuses the project's own scoring path verbatim: it builds a `MethodSpec`
variant via `dataclasses.replace(spec, config_model=functools.partial(SparseGeoConfig, backend=...,
voting_mode=...))` and monkeypatches `object_search.eval.benchmark.get_method` around an unmodified
call to `benchmark._run_one`, so scoring, AP-candidate logging, and abstention/error handling are
byte-identical to `pixi run bench-real-objects` by construction.

**Protocol parameters**, identical to `conf/benchmark-real-objects.yaml` so every number below is
directly comparable to the published `real-objects` figures: `iou_threshold=0.5` (a prediction
counts as a true positive at IoU ≥ 0.5, the same threshold every report in this project uses),
`exemplar_count=1` (one exemplar box per query), `seed=0` for the `np.random.default_rng`-based
exemplar sampler (D-11 — `cv2`'s RNG is never used for this, per the project's determinism rules),
and the default `"seeded-random"` exemplar-selection mode. No config field beyond `backend` and
`voting_mode` is overridden in any condition, so every other `SparseGeoConfig` field (e.g.
`min_inliers`, `nms_iou`) stays at its shipped default throughout this run.

**Where it ran.** At the user's request, the fetch + smoke + sweep steps ran on a rented vast.ai CPU
box (offer `47060238`, $0.048/hr, 12 vCPUs) rather than the local machine, to keep this spike off the
user's laptop. Code was shipped via `git archive HEAD | gzip` over `scp` — **not** `git push`, since
this worktree's branch isn't pushed to `origin` and pushing an in-progress branch needs separate
confirmation this task didn't have. The box installed the default pixi env, ran
`pixi run fetch-models --only superpoint` (sha256-gated, MagicLeap non-commercial weights), ran the
full sweep under the default CPU execution provider (no `onnxruntime-gpu` swap — deliberately kept
CPU-only for reproducibility, and this spike has no GPU-bound step to accelerate), and the results
were pulled back over `scp`. The instance was destroyed immediately after. Total rental: ~7 minutes
of actual compute across two attempts (see "A process note" below), well under $0.01.

No val/test split exists for `real-objects` (unlike floor-plans-500's val/test), so there is nothing
to tune: every condition below runs at `SparseGeoConfig` defaults with only `backend` and
`voting_mode` pinned, and no threshold is selected against these labels.

## The five conditions

| condition | backend | voting_mode | role |
|---|---|---|---|
| `sift/single-4dof` | sift | single-4dof | **shipped baseline** |
| `sift/translation-2dof` | sift | translation-2dof | control |
| `sift/pairwise-4dof` | sift | pairwise-4dof | control |
| `superpoint/translation-2dof` | superpoint | translation-2dof | test |
| `superpoint/pairwise-4dof` | superpoint | pairwise-4dof | test |

SuperPoint's keypoints are frameless (no scale/orientation), so `single-4dof` — which needs a single
correspondence's own frame to fit a similarity transform — is invalid for it; `SparseGeoConfig`
**raises at construction** for that pair (`_reject_single_4dof_for_frameless_superpoint`). It is
therefore never omitted by choice, only unreachable by design. That forces every SuperPoint run onto
a different voting mode than the shipped baseline — which is exactly the confound the two **SIFT
controls** exist to isolate. Without them, a SuperPoint delta measured only against
`sift/single-4dof` cannot distinguish "SuperPoint is worse" from "`translation-2dof`/`pairwise-4dof`
are worse than `single-4dof`, regardless of backend." (The floor-plans-500 report did not run this
control — see "What the SIFT controls reveal" below for why that matters.)

## Baseline reconciliation

The shipped-baseline condition was run first and reconciled against the published `real-objects`
numbers before any delta was computed, exactly as the floor-plans investigation did against its own
prior state.

| cell | published | measured | agrees (≤3dp)? |
|---|---|---|---|
| F1 | 0.786 | 0.7863 | yes |
| mean AP | 0.740 | 0.7398 | yes |
| PLAIN F1 | 0.99 | 0.9924 | yes |
| VARIED F1 | 0.67 | 0.6667 | yes |
| CLUTTERED F1 | 0.66 | 0.6606 | yes |
| n_images / n_scored / n_errors / n_abstentions | 30 / 30 / 0 / 0 | 30 / 30 / 0 / 0 | yes |
| precision | 0.833 | 0.8166 | **no** (Δ −0.016) |
| recall | 0.772 | 0.7582 | **no** (Δ −0.014) |

F1 and AP — the two metrics this project's reports lead with — reproduce to within 0.0003, and
every count field matches exactly. Precision and recall each disagree by ~0.015, which is larger
than the ≤3-decimal bar and is called out rather than rounded away, but it does not read as a
reproduction failure: it is the right size and direction to be a rounding/aggregation footnote in
[`real-objects-findings.md`](real-objects-findings.md)'s published table (which reports P/R
separately from the F1/AP headline table this spike's baseline matches exactly) rather than a
divergent run. Every downstream delta in this report is measured against **this run's own**
baseline condition, not the published figures, so the discrepancy does not propagate.

## Pooled results

| condition | P | R | F1 | AP | p50 ms | max ms | vs. same-mode SIFT (ΔF1 / ΔAP) | vs. shipped baseline (ΔF1 / ΔAP) |
|---|---|---|---|---|---|---|---|---|
| `sift/single-4dof` (baseline) | 0.817 | 0.758 | **0.786** | 0.740 | 1076 | 2547 | — | — |
| `sift/translation-2dof` | 0.705 | 0.709 | 0.707 | 0.690 | 1316 | 2743 | — | −0.079 / −0.050 |
| `sift/pairwise-4dof` | 0.754 | 0.709 | 0.731 | 0.595 | 2976 | 4074 | — | −0.055 / −0.145 |
| `superpoint/translation-2dof` | 0.699 | 0.742 | 0.720 | 0.731 | 3045 | 3908 | **+0.013 / +0.041** | −0.066 / −0.009 |
| `superpoint/pairwise-4dof` | 0.698 | 0.736 | 0.717 | 0.685 | 4743 | 5640 | −0.014 / **+0.090** | −0.069 / −0.055 |

## What the SIFT controls reveal

The two SIFT controls carry a finding of their own, independent of SuperPoint: switching **SIFT**
off `single-4dof` costs **0.055–0.079 F1** and **0.050–0.145 AP** by itself, before SuperPoint enters
the comparison at all. `single-4dof`'s 4-DoF fit from one correspondence is simply a stronger
estimator, on this domain, than 2-DoF translation voting or pairwise 4-DoF fitting from
correspondence pairs — regardless of which detector fed it. That means a large share of the
floor-plans-500 report's SuperPoint deltas (−0.024 to −0.096 F1, measured only against
`sift/single-4dof`) is plausibly attributable to the voting-mode switch SuperPoint is forced into,
not to SuperPoint's keypoints being worse. This spike's same-mode comparison is the fairer one, and
by that comparison **SuperPoint is roughly at parity with SIFT on real photographs, and ahead of it
on AP in both matched voting modes** (+0.041 at `translation-2dof`, +0.090 at `pairwise-4dof`).

## Per-regime

| condition | PLAIN F1 | VARIED F1 | CLUTTERED F1 |
|---|---|---|---|
| `sift/single-4dof` (baseline) | 0.992 | 0.667 | 0.661 |
| `sift/translation-2dof` | 0.977 | 0.545 | 0.565 |
| `sift/pairwise-4dof` | 0.908 | 0.654 | 0.571 |
| `superpoint/translation-2dof` | 0.992 | 0.545 | 0.597 |
| `superpoint/pairwise-4dof` | 0.949 | 0.516 | 0.655 |

No regime reverses the pooled story: SuperPoint tracks its same-mode SIFT control closely in every
regime (within ±0.05–0.06 F1, both directions), with no regime where it collapses the way the
floor-plans window condition did (28/28 → 26/28 coverage). PLAIN is where `single-4dof` most clearly
beats both non-single-4dof modes regardless of backend, consistent with the voting-mode finding
above — flat-pose objects are exactly where a single correct correspondence's own 4-DoF frame is
most reliable, and where throwing that frame away for 2-DoF/pairwise voting has the most to lose.

## Keypoint counts: real photographs vs. floor-plan line-art

| domain | detector | crop kp (min / median / max) | scene kp (min / median / max) | crops below `min_exemplar_keypoints=8` |
|---|---|---|---|---|
| floor-plans-500 (5-plan probe, prior report) | SIFT | 0 / 2 / 33 | 291 / 711 / 2170 | 4/5 |
| floor-plans-500 (5-plan probe, prior report) | SuperPoint | 1 / 3 / 12 | 291 / 550 / 5986 | 3/5 |
| real-objects (this spike, all 30) | SIFT | 17 / 144 / 576 | 924 / 4573 / 17252 | **0/30** |
| real-objects (this spike, all 30) | SuperPoint | 24 / 145 / 378 | 1060 / 3726 / 6569 | **0/30** |

This is the mechanism the whole comparison rests on. On floor-plan line-art, both detectors mostly
starve, and SuperPoint fires marginally more often than SIFT on the barren crops but yields under
half as many keypoints on the one texture-rich plan (33 → 12) — a genuine keypoint-quality gap. On
real photographs, **every single one of the 30 exemplar crops clears the `min_exemplar_keypoints=8`
acceptance floor under both detectors**, and the two backends' crop-keypoint medians are within one
keypoint of each other (144 vs 145). The starvation regime that drove the floor-plans result simply
does not occur on this domain.

## The zero-keypoint crash — does it reproduce?

**No warning-level log line fired in any of the 5 conditions across all 30 images**
(`warning_log_lines: 0` for every condition in `runs/summary.json`; `n_errors: 0`, `n_abstentions: 0`
everywhere). No crash, no abstention, no degraded outcome.

One caveat on what this does and doesn't prove: the floor-plans crash was a `CoreMLExecutionProvider`
bug on macOS, firing when SuperPoint returned a genuinely zero-row keypoint tensor. This spike ran on
a Linux vast.ai box under the CPU execution provider (`AzureExecutionProvider`/`CPUExecutionProvider`
— CoreML isn't loaded on Linux at all), so it cannot directly confirm or deny whether *CoreML
specifically* would mishandle a zero-keypoint tensor here. What it **does** establish is the
precondition question: the crash's trigger — SuperPoint detecting **zero** keypoints — never
occurred on any of the 30 real-object images (minimum observed was 24). The keypoint-starvation
regime that exposed the CoreML bug on floor-plans simply doesn't arise on rich photographic texture,
which makes the crash question moot for this domain regardless of execution provider.

## Verdict: **PARTIALLY DIVERGES**

The floor-plans-500 verdict does not hold as stated on real photographs, but it doesn't cleanly
invert either — the picture depends on which comparison you read:

- **Against the shipped `sift/single-4dof` baseline**, SuperPoint still loses on F1 in both voting
  modes (−0.066, −0.069) — the same *direction* as floor-plans, though a smaller magnitude
  (floor-plans: −0.024 to −0.096) and, unlike floor-plans, **without an AP or coverage collapse**
  (AP is down only 0.009–0.055 here vs. down in 3/4 floor-plans cells, and coverage stays 30/30 in
  every condition vs. floor-plans' window 28/28 → 26/28).
- **Against the fairer same-voting-mode SIFT controls** — which the floor-plans report did not run —
  SuperPoint is roughly at parity on F1 (+0.013, −0.014, both within noise) and **ahead on AP in
  both modes** (+0.041, +0.090).

Causal hypothesis, grounded in the measurements above: floor-plans-500's "SuperPoint loses" result
was driven by two things this spike's design deliberately separates. First, a real detector-quality
gap that is specific to sparse line-art — SuperPoint yielding under half of SIFT's keypoints on the
one texture-rich floor-plan sample, a gap that vanishes on real photographic texture (crop-keypoint
medians within one keypoint of each other here). Second, an unmeasured voting-mode confound — the
floor-plans comparison was `sift/single-4dof` vs. `superpoint/{translation,pairwise}-4dof` with no
same-mode SIFT control, and this spike's controls show the voting-mode switch alone costs
0.055–0.079 F1 with SIFT. Floor-plans-500 likely conflated both effects into one verdict; on real
photographs, only the second effect (voting-mode cost, backend-independent) remains, and it is the
smaller of the two. This is stated as hypothesis, not re-measured on floor-plans itself — doing so
(adding SIFT/translation-2dof and SIFT/pairwise-4dof controls to the floor-plans sweep) is the
natural next check and is carried into "Deferred work" below.

## Cost

Even in the conditions that lose, latency is reported honestly. SuperPoint costs less on real
photographs than it did on floor-plans: 1.6–2.3× at matched voting mode here
(`superpoint/translation-2dof` p50 3045ms vs. `sift/translation-2dof` p50 1316ms = 2.31×;
`superpoint/pairwise-4dof` p50 4743ms vs. `sift/pairwise-4dof` p50 2976ms = 1.59×), against
floor-plans' 5.3–6.9×. Against the shipped `single-4dof` baseline the multiplier is larger (2.83×,
4.41×) because `single-4dof` itself is the cheapest mode, independent of backend.

The licence ceiling from the prior report is unchanged and domain-independent: `models/superpoint.onnx`
is MagicLeap **non-commercial research-only**, gitignored, fetched only by
`pixi run fetch-models --only superpoint` — this backend cannot become the shipped default on *any*
domain regardless of how it scores, so nothing measured here changes that standing constraint.

## A process note on this run

Two vast.ai instances from an earlier, interrupted attempt at this same task were left running
unattended (a client-side polling bug — using `vastai show instance <id> --raw` singular, which
didn't parse as expected, instead of `vastai show instances --raw` plural) and were destroyed by the
orchestrator once found (~20 minutes / ~$0.03 combined, machines 129593 and 140986 — neither is this
project's known-bad denylisted host). The successful run in this report used a fresh instance
(offer `47060238`, machine `27850`) and was destroyed immediately after results were pulled back.
This is recorded here rather than left implicit because reproducibility and cost-honesty are both
explicit project values, and a spike report that pretends the first attempt didn't happen would
misstate what this actually cost.

## Regression guard

`git status --porcelain src/ conf/` is empty and `git diff --stat -- src/ conf/` is empty: nothing in
`src/` or `conf/` changed. `sparse-geo`'s shipped default stays `backend="sift"`. The only new files
are the driver script (`scripts/sparse_geo_real_objects_experiment.py`), this report, the raw run
artifacts under `.planning/quick/260811-p0l-spike-explore-the-superpoint-backend-for/runs/`, and the
`mkdocs.yml` nav entry pointing at this page.

## Deferred work

- ~~Re-run the floor-plans-500 sweep with the same-voting-mode SIFT controls added.~~ **Resolved
  2026-08-13** in [`sparse-geo-floorplans-voting-mode-confound.md`](sparse-geo-floorplans-voting-mode-confound.md):
  verdict **PARTIALLY** — the confound explains most to all of the door loss (SuperPoint even leads
  on AP50 at matched voting mode there), but a real backend-specific gap survives on windows.
- **DISK / ALIKED backends** (carried forward from `sparse-geo-improvement.md`, and from the
  robustness backlog): now informed by two data points instead of one — a permissive-licence
  learned detector is worth investigating on its own merits on both domains, not only as a
  SuperPoint licence workaround.
- **CoreML-specific verification of the crash-non-reproduction finding**, if the zero-keypoint
  question ever matters operationally on macOS with real (non-line-art) imagery: this spike answered
  the precondition question (keypoints never approach zero on real photos) but ran under the CPU
  execution provider throughout, not CoreML.

## Verification

`pixi run quality` and `pixi run docs-build` — see the executor's SUMMARY.md for this quick task for
the pass/fail record captured at commit time. `src/` and `conf/` are provably unchanged per the
regression guard above, so `sparse-geo`'s shipped `backend` default stays `sift` regardless of this
report's numbers.
