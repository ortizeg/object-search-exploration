# EXPERIMENTS — `propose-retrieve` on floor plans (quick task 260812-m8m)

Append-only lab notebook. One entry per experiment, never overwritten. Every number here comes from
a run of the committed `scripts/propose_retrieve_floorplans_experiment.py` against the committed
`dataset_splits/floorplans-door.split.json` split, and every entry records the git SHA and the exact
command that produced it. Estimates are labelled as estimates; nothing else is.

**Runtime for all entries below:** vast.ai contract 47510440 (RTX 3090 host, but **ONNX Runtime is
the CPU build** — `providers=['CPUExecutionProvider']`), `/root/repo` at the SHA noted per entry.
Raw JSON for every entry is under this directory's `runs/`.

---

## B0 — proposal-stage baseline (floorplans-door, val+test, `proposal_conf=0.4`)

- **SHA:** `0cb906d4eeeacd7930bf8fa7a0111b4ac689fda1` (+ the uncommitted harness under test)
- **Command:** `pixi run python scripts/propose_retrieve_floorplans_experiment.py b0 floorplans-door`
- **Artifact:** `runs/b0--floorplans-door.json`
- **Scope:** 84 plans (56 val + 28 test), 760 GT door boxes, FastSAM everything-mode proposals only
  — no embedding, no retrieval, no threshold.

A GT box counts as *proposed* when some proposal overlaps it at IoU ≥ 0.5 — the same IoU the
benchmark uses for a true positive, so this is an honest **ceiling** on final recall.

### Proposal-stage recall by crowding bucket

| doors/plan | n plans | mean n_gt | mean n_proposals | mean proposal recall | (excl. exemplar) | pooled |
|---|---|---|---|---|---|---|
| 1–3 (sparse) | 11 | 2.7 | 48.6 | **0.864** | 0.773 | 0.867 |
| 4–10 | 47 | 7.3 | 44.9 | **0.551** | 0.550 | 0.537 |
| 11+ (crowded) | 26 | 15.0 | 57.0 | **0.268** | 0.274 | 0.254 |
| all | 84 | 9.0 | 49.1 | 0.504 | 0.494 | 0.405 |

Two denominators are reported because they disagree and the diagnosis used the second one: the
**eval convention** keeps the sampled exemplar's own box in the denominator (what
`run_research_benchmark` does), the **diagnostic convention** drops it.

**Does this reproduce the diagnosed 0.74 / 0.51 / 0.27?** Yes, under the diagnostic convention:
**0.77 / 0.55 / 0.27** vs the diagnosed 0.74 / 0.51 / 0.27. The residual difference in the sparse
bucket is bucket membership — the scratch diagnostic bucketed plans on the exemplar-excluded count,
so a few 4-door plans it filed under `1–3` are filed under `4–10` here. The monotone collapse with
crowding — the finding the ablation is built on — reproduces exactly.

### Proposal-stage recall by symbol-size bucket

Cuts are `benchmark._symbol_size_bucket`'s (small < 0.4 %, medium < 1.6 %, large ≥ 1.6 % of plan
area), imported rather than restated so this table cannot drift from the committed
`by_symbol_size` recall table.

| symbol size | n GT | proposal recall |
|---|---|---|
| small | 366 | **0.279** |
| medium | 364 | 0.516 |
| large | 30 | 0.600 |

### Attribution — one of the two diagnosed failure modes is NOT supported

The plan's objective names two mechanisms: **(1)** a proposal budget that does not scale with
instance count, and **(2)** FastSAM's fixed 1024 letterbox shrinking symbols on large plans. They
make opposite predictions about plan size, so they can be separated:

| Pearson correlation (n = 84 plans) | value |
|---|---|
| proposal recall vs **n_gt** | **−0.537** |
| proposal recall vs plan long side | +0.190 |
| proposal recall vs plan area | +0.185 |
| **n_proposals** vs n_gt | +0.216 |
| **n_proposals** vs **plan area** | **+0.588** |

De-confounded as a crowding × plan-size cross-tab:

| crowding | plan long side | n plans | mean n_gt | mean n_proposals | mean proposal recall |
|---|---|---|---|---|---|
| 1–3 | ≤1024 | 7 | 2.9 | 54.6 | 0.905 |
| 1–3 | >1024 | 4 | 2.5 | 38.2 | 0.792 |
| 4–10 | ≤1024 | 33 | 7.2 | 43.0 | 0.481 |
| 4–10 | >1024 | 14 | 7.4 | 49.3 | **0.714** |
| 11+ | ≤1024 | 15 | 15.0 | 39.9 | **0.169** |
| 11+ | >1024 | 11 | 14.9 | 80.5 | **0.403** |

**Finding — this revises the plan's premise.** Failure mode (2) is *not* supported at the aggregate
level. Large plans do **better**, not worse, and they do better *within* every crowded bucket
(11+: 0.403 vs 0.169; 4–10: 0.714 vs 0.481), because they receive roughly twice the proposals
(80.5 vs 39.9). Plan-size mean n_gt is nearly identical across the two groups (9.6 vs 8.8), so this
is not a crowding confound.

The mechanism the data *does* support is a sharpened form of failure mode (1):

> **FastSAM's proposal budget scales with image AREA (r = +0.59), not with instance count
> (r = +0.22).** Crowding is what destroys recall (r = −0.54 vs n_gt). The worst cell in the whole
> dataset is a *crowded, small* plan — 11+ doors on a ≤1024 plan — at recall **0.169** with only
> 39.9 proposals for ~15 doors.

Failure mode (2) survives only as a single-plan anecdote, and that plan is a genuine outlier
(4000×1685, ~4× the long side of the next-largest plan). Four plans score proposal recall **0.000**,
and three of them are plans the 1024 letterbox *upscales*:

| plan | size | n_gt | n_proposals |
|---|---|---|---|
| `65_png.rf.y2pckOMkZpYvXHhC2AZE` | 4000×1685 | 19 | 83 |
| `4061_png.rf.4VhzLHqSK6GM2tgWBh6K` | 1170×742 | 11 | 50 |
| `155_png.rf.5XBwQ9IzguztSIBPEOnz` | 818×647 | 7 | 77 |
| `4_png.rf.g6FJloai36FKwgawnlGA` | 513×436 | 7 | 30 |

That a 513×436 plan with 7 doors and 30 proposals scores 0.000 is not a resolution problem — it is
FastSAM declining to segment CAD line-art door symbols as objects at all. Keep this in view for the
step-3 go/no-go: it is the single strongest argument for a non-FastSAM proposal source, and no
amount of tiling fixes a backend that does not consider the symbol an object.

**Consequence for step 1 (tiling).** Tiling was motivated as attacking both modes. The measured
mechanism says its *live* lever is **budget**, not magnification: a tile is a smaller area, and
FastSAM allocates proposals per area, so N tiles buy roughly N× the budget on the crowded plans that
need it. This makes tile **size** the parameter that matters (smaller = more budget) and predicts
the gain should concentrate in the crowded buckets. See R0.

---

## R0 — SAHI research note (specification for step 1)

- **Sources read:** Akyon et al., *Slicing Aided Hyper Inference and Fine-tuning for Small Object
  Detection*, ICIP 2022 (the SAHI paper); and the current `obss/sahi` implementation, read directly
  so the parameter defaults below are **verified from source rather than recalled** —
  `sahi/predict.py::get_sliced_prediction`, `sahi/slicing.py::get_slice_bboxes`,
  `sahi/postprocess/utils.py::calculate_bbox_ios`, `sahi/postprocess/combine.py::greedy_nmm`.

### What SAHI actually specifies (verified defaults)

| parameter | SAHI default | verified in |
|---|---|---|
| `overlap_height_ratio` / `overlap_width_ratio` | **0.2** | `predict.py:181-182` |
| `perform_standard_pred` (the full-image pass, "SAHI + FI") | **True** | `predict.py:183` |
| `postprocess_type` | **`GREEDYNMM`** | `predict.py:184` |
| `postprocess_match_metric` | **`IOS`** (intersection over smaller) | `predict.py:185` |
| `postprocess_match_threshold` | **0.5** | `predict.py:186` |
| slice geometry | last slice **clamped** to the image edge, not padded (`x_max = min(image_width, x_max)`) | `slicing.py:84-86` |
| overlap in pixels | `int(ratio * slice_size)` | `slicing.py:72-73` |
| IoS | `intersection / area of the smaller box`, matched on strict `>` | `utils.py:316-331`, `utils.py:355-356` |

Why IoS and not IoU is the load-bearing detail: a symbol truncated by a tile edge is *nearly
contained* in the whole-object box found by an overlapping tile or by the full-image pass. Contained
boxes have **high IoS and low IoU**, so plain IoU-NMS keeps both and the merged output carries a
duplicate fragment for every straddling instance. This is why the plan specifies a method-local
IoS merge instead of importing `search/common/nms.py` (which is IoU-based).

### Parameters chosen for THIS domain, and why

Measured symbol/plan scale on `floorplans-door` val+test (760 boxes, 84 plans; from the committed
GT sidecars):

- door box longest side: **median 55 px**, p10 24 px, min 13 px, max 271 px
- box area as a fraction of plan area: p10 0.15 %, **median 0.41 %**, p90 0.99 %
- plan long side: **median 832 px**, max 4000 px

FastSAM letterboxes every input to a fixed **1024** square, so a tile of side *S* magnifies each
symbol by `1024 / S`. The tile count is a function of plan area, computed by
`tile_count_forecast()` over the committed plan dimensions (arithmetic, no model):

| tile_side | overlap | mean tiles/plan | median | max | plans left **untiled** |
|---|---|---|---|---|---|
| 512 | 0.2 | 7.9 | 4 | 80 | **8 / 84** |
| 512 | 0.3 | 9.6 | 4 | 88 | 8 / 84 |
| 768 | 0.2 | 4.1 | 2 | 35 | 36 / 84 |
| 768 | 0.3 | 4.4 | 2 | 48 | 36 / 84 |
| 1024 | 0.2 | 2.4 | 1 | 20 | **55 / 84** |
| 1024 | 0.3 | 2.5 | 1 | 24 | 55 / 84 |

**Chosen values, each with its reason:**

- **`tile_side = 512` as the primary, with 768 and 1024 swept as controls.** This is the single most
  consequential choice and the forecast decides it: at `tile_side = 1024` the median plan yields
  **exactly one tile** and **65 % of plans (55/84) are untouched**, so a 1024 tile is close to a
  no-op on this dataset and could not possibly move the aggregate. At 512 only 10 % of plans are
  untouched, and each tile is upscaled 2× by the fixed letterbox. 1024 is kept in the sweep
  precisely as the near-null control — if it "wins", tiling is not what is helping.
- **`tile_overlap = 0.2`** (SAHI's verified default), with 0.3 as a control. The overlap band at
  512/0.2 is `int(0.2 × 512) = 102 px`, which exceeds the **p90 door longest side (107 px ≈ the
  band)** and is ~2× the median door (55 px) — so ~90 % of doors can sit wholly inside the band and
  be seen untruncated by at least one tile. The 271 px maximum door cannot, which is exactly what
  the full-image pass is for.
- **`tile_include_full_image = True`** (SAHI's `perform_standard_pred`, default True). Symbols too
  large for any single tile's overlap band, and the whole-plan context, come back from the untiled
  pass. Cost is exactly one extra forward pass per plan.
- **Merge by greedy NMM on IoS at 0.5** (SAHI's verified triple), in the project's canonical
  deterministic order `(-objectness, y, x)`, with `max_proposals` applied **after** the merge so the
  budget is global rather than per tile.

### One caveat recorded up front

SAHI's premise is *small objects lost to downscaling* — its lever is magnification. B0 measured that
this dataset's dominant failure is **budget**, not magnification. The two levers happen to point the
same way (smaller tiles buy both), so "SAHI-style tiling" remains the right step-1 instrument, but
the *reason* it should work here differs from the paper's, and the T1b sweep must therefore report
proposal-stage recall by **crowding** bucket as its primary read — not only by symbol size.

---

## B3 — cost probe (CPU, idle box)

- **SHA:** `0cb906d4eeeacd7930bf8fa7a0111b4ac689fda1`
- **Command:** `pixi run python scripts/propose_retrieve_floorplans_experiment.py b3`
- **Artifact:** `runs/b3--cost-probe.json`
- **Measured on an otherwise idle box**, before the B0/B1/B2 runs were launched, so these are clean
  single-process wall-clocks.

### Measured (3 val plans, `propose-retrieve` at defaults)

| plan | size | n_proposals | proposal stage | full search | proposal_ms | embedding_ms |
|---|---|---|---|---|---|---|
| `109_png…` | 603×451 | 27 | 1.23 s | 3.53 s | 1485 | 1600 |
| `110_png…` | 427×521 | 40 | 1.28 s | 3.38 s | 1378 | 1912 |
| `119_png…` | 1478×958 | 28 | 1.36 s | 15.62 s | 1400 | **14206** |
| **mean** | | | **1.29 s** | **7.51 s** | ~1421 | ~5906 |

**One 56-plan val pass ≈ 7.0 min.** The committed `propose-retrieve` tuning grid is
`similarity_floor (6) × nms_iou (2) = 12 trials`, so a full B1-style val sweep is **≈ 84 min per
dataset**, plus two 28-plan test reads.

**The finding this probe produced:** on floor plans the **embedding stage dominates, not the
proposal stage** — 5.9 s of the 7.5 s mean, and 14.2 s of 15.6 s on the 1478×958 plan. That
contradicts the general claim in `propose_retrieve.py`'s *Latency (EVAL-11)* section ("the proposal
stage dominates"), which was established on the chipset/textured regimes. It matters here because
tiling raises the **merged proposal count**, and embedding cost scales linearly with that count — so
tiling's true cost is not just its extra FastSAM passes.

### Extrapolated tiled cost — ARITHMETIC, NOT A MEASUREMENT

Nothing tiled has been built or run. These multiply the *measured* proposal stage by the *forecast*
tile count and hold the embedding stage **fixed**, which makes every figure a **lower bound**
(tiling also increases the proposal count feeding the embedding stage).

| geometry | mean tiles | est. s/plan | est. per 56-plan val trial |
|---|---|---|---|
| side 512, overlap 0.2 | 7.9 | ~16.4 s | **~15 min** |
| side 512, overlap 0.3 | 9.6 | ~18.6 s | ~17 min |
| side 768, overlap 0.2 | 4.1 | ~11.5 s | ~11 min |
| side 1024, overlap 0.2 | 2.4 | ~9.3 s | ~9 min |
| (flat 4× / 9× / 16× tile assumption, for reference) | 4 / 9 / 16 | 11.4 / 17.9 / 26.9 s | 10.6 / 16.7 / 25.1 min |

**What this implies for the ablation as planned** (all lower bounds, CPU):

- **T1b**, the proposal-stage-only sweep (`3 tile_side × 2 overlap × 2 full-image = 12` configs over
  56 val plans, no embedding): at side 512 ≈ 7.9 tiles × 1.29 s × 56 ≈ **9.5 min per config**,
  so ≈ **1.5–2 h** for the 12-config sweep.
- **T1c**, full tuned-vs-default on the winning geometry: ≈ **15 min per val trial**; the committed
  12-trial grid is therefore ≈ **3 h**, plus test reads.
- **T2**, the `proposal_conf` sweep (5 values) on top of the step-1 winner, with a lower gate
  *raising* proposal counts and hence embedding cost: ≈ **1.5 h+**, likely more.
- **Total remaining ablation on CPU: roughly 6–10 h of wall-clock**, at ~$0.13/h.

---

## B3-CORRECTION — the cost probe was biased low by ~7.7×, and why

Appended after B1 began producing a measured whole-split rate. **B3 above is left exactly as
recorded**; this entry supersedes its per-pass projection and states why, per the notebook's
append-only rule.

| quantity | B3 probe (3 plans, idle box) | measured over all 56 val plans (B1, in flight) |
|---|---|---|
| seconds per plan, full search | 7.51 s | **~57.6 s** |
| one 56-plan val pass | 7.0 min | **~53.8 min** |
| 12-trial val sweep | ~84 min | **~10.8 h** |

Two causes, both real, neither a rounding error:

1. **Sampling bias — a bug in the committed probe, now fixed.** `cost_probe` took
   `research_image_ids(...)[:n_plans]`, i.e. the first three plans in *manifest (filename) order*.
   Those three have a mean area of **0.54× the val mean** (636 781 px² vs 1 178 382 px²). Cost is
   super-linear in plan area — a larger plan yields more proposals *and* larger crops, and **every
   proposal is embedded by its own DINOv2 forward pass** — so a small-plan sample is biased low, not
   merely noisy. Fixed in the committed script: the probe now samples by **area quantile** across
   the split. A cost probe that cannot size a sweep is worse than no probe, because it gets quoted
   as if it could — which is exactly what happened here.
2. **CPU contention.** The B1 rate is measured with two `b1` processes (and briefly `b2`/`b0`)
   sharing the box, each taking ~4 cores. The idle-box probe had the machine to itself.

**Authoritative planning number: ~54 min per 56-plan val pass on this box's CPU, ~58 s/plan.**
The corrected probe has *not* been re-run on an idle box (doing so now would be contended and
therefore no better); that clean re-run belongs at the start of Task 2.

### Revised cost of the remaining ablation (built on the measured rate, labelled as estimates)

- **T1b**, proposal-stage-only sweep (no embedding): the proposal stage measures **1.44 s/plan**
  (B0). At `tile_side=512` (7.9 tiles/plan) ≈ 11.4 s/plan ≈ **10.6 min per config**, ≈ **2.1 h** for
  the 12-config sweep. This step is affordable on CPU.
- **T1c**, full tuned-vs-default with tiling: baseline is ~54 min/val trial, and tiling adds both
  the extra FastSAM passes (+~9 min/pass) *and* an embedding-stage increase proportional to the
  merged proposal count. If tiling roughly doubles proposals, a val trial lands near **~100 min**,
  so a 12-trial grid is **~20 h**.
- **T2**, the `proposal_conf` sweep (5 values), where a *lower* gate deliberately raises the
  proposal count and therefore the dominant embedding cost: **several hours more**.
- **Order-of-magnitude for the rest of the plan on CPU as written: ~30–50 h.**

### The mitigating fact — this workload parallelises trivially

Measured while three experiments ran concurrently: each process holds only **~3–4.6 cores**
(`%CPU` 267–395) on a **72-core** box, with **load average 8–12**. The box is ~85 % idle during a
"slow" sweep. Every tuning trial is an independent process reading the same read-only data, so the
sweep is embarrassingly parallel: ~15–20 concurrent trials would fit. The 12-trial val sweep that
takes ~10.8 h sequentially is roughly **~1 h** if the trials are run as 12 concurrent processes.
Neither `run_domain_tuning` nor this harness does that today — it is a scheduling change (launch
per-trial processes), not an algorithmic one. **This materially changes the CPU-vs-GPU trade and is
the single most decision-relevant fact in this entry.**

An additional, separate lever (noted, not acted on): `embed_regions` embeds each proposal with its
own DINOv2 forward pass — ~50 sequential single-crop sessions per plan. Batching them is an obvious
large win for the dominant cost, but it is a `src/` change outside this plan's scope and would need
its own before/after equivalence check.

---

## B1 — session-local final baseline (tuned vs default) — **IN FLIGHT, NOT COMPLETE**

- **SHA:** `0cb906d4eeeacd7930bf8fa7a0111b4ac689fda1`
- **Commands:** `pixi run python scripts/propose_retrieve_floorplans_experiment.py b1 floorplans-door`
  and `… b1 floorplans-window` (launched concurrently, `nohup`, 20:21 box time)
- **Artifacts (on completion):** `runs/b1--floorplans-door.json`, `runs/b1--floorplans-window.json`

**Status at the time of the Task-1 commit: incomplete.** Recorded honestly rather than omitted or
estimated. Progress is measured exactly, not guessed: the gmm degenerate-calibration DEBUG line is
deterministic per (plan, exemplar) and independent of `similarity_floor`/`nms_iou` (both are applied
*after* calibration), so the sequence of `modes …` values repeats **exactly every 18 lines = one
56-plan val pass** — confirmed by the first value recurring at lines 1, 19, 37, 55, 73.

| | doors | windows |
|---|---|---|
| elapsed | 4 h 38 m | 4 h 38 m |
| degenerate lines | 93 | 92 |
| **val trials complete** | **5.2 of 12** | **5.1 of 12** |
| implied rate | 53.8 min/val pass | 54.5 min/val pass |
| projected total (12 trials + 2 test reads) | **≈ 11.6 h** | ≈ 11.7 h |

Both processes are healthy (`nohup`-detached, ~4.6 cores each, CPU time accumulating; they survived
an SSH launcher being killed) and are **left running** — the work already banked is real and the box
costs ~$0.13/h. B1's completed numbers, and the "does the committed 0.459 doors row reproduce?"
question, are appended here as **B1-final** when the runs land.

**Expected, and pre-registered here so it cannot be rationalised later:** the committed doors row
(F1 0.459 / P 0.55 / R 0.39) was produced **on GPU** with **coverage 13/14** — a partial test read —
whereas this run is **CPU over all 28 test plans**. A difference is therefore expected, and per the
plan the *session-local* number becomes the delta reference for every later comparison.

---

## B2 — guardrail baseline (chipset / textured / synthetic regimes) — COMPLETE

- **SHA:** `0cb906d4eeeacd7930bf8fa7a0111b4ac689fda1`
- **Command:** `pixi run python scripts/propose_retrieve_floorplans_experiment.py b2`
- **Artifacts:** `runs/b2--regimes.json`, `runs/b2--regimes-raw.json`
- **Scope:** 90 labelled images, `propose-retrieve` at **DEFAULT** config, **0 errors, 90/90 scored**
  (so the guardrail is complete, not narrowed). Wall clock 45 min (contended).

| regime | n | precision | recall | **F1** | committed (`propose-retrieve-improvement.md`) |
|---|---|---|---|---|---|
| EASY (chipset) | 10 | 0.883 | 0.976 | **0.927** | 0.93 ✓ |
| TEXTURED (plain) | 16 | 0.921 | 1.000 | **0.959** | 0.96 ✓ |
| VARIED (scale/rotation) | 16 | 0.942 | 0.935 | **0.939** | 0.94 ✓ |
| CLUTTERED | 16 | 0.734 | 0.931 | **0.821** | 0.82 ✓ |
| synthetic | 2 | 1.000 | 0.833 | **0.909** | 0.91 ✓ |
| real-objects | 30 | 0.954 | 0.797 | **0.868** | (not in that report — extra guardrail) |
| **overall** | 90 | 0.881 | 0.918 | **0.899** | |

**Every one of the five recorded regimes reproduces to within rounding.** This is the guardrail any
floor-plan change must leave untouched, and it is now a session-local measurement rather than a
quoted number. It also establishes that the CPU execution provider reproduces the committed
(GPU-era) regime numbers — which is why a B1 doors difference, if one appears, should be attributed
to the coverage difference (28/28 vs 13/14) before it is attributed to the runtime.

---

## B1-final — session-local final baseline (tuned vs default) — **COMPLETE**

**Supersedes the "B1 — IN FLIGHT, NOT COMPLETE" entry above.** That entry is left exactly as
recorded, per the append-only rule; the numbers below are the finished runs.

- **SHA:** `0cb906d4eeeacd7930bf8fa7a0111b4ac689fda1`
- **Commands:** `pixi run python scripts/propose_retrieve_floorplans_experiment.py b1 floorplans-door`
  and `… b1 floorplans-window` (launched concurrently, `nohup`)
- **Artifacts:** `runs/b1--floorplans-door.json`, `runs/b1--floorplans-window.json`
  (also `/root/b1_door.log`, `/root/b1_window.log` on the box)
- **Runtime:** CPU execution provider, all 28 test plans scored (`n_scored` 28/28), 1 exemplar.
- **Wall clock:** ~12 h 40 m each, run concurrently — matching the B3-CORRECTION projection
  (≈11.6 h), not the original B3 probe's ≈84 min. The corrected probe was right; the original was
  biased low by ~7.7×.

### floorplans-door

| | precision | recall | F1 |
|---|---|---|---|
| **VAL** (winning trial) | 0.591 | 0.307 | **0.404** |
| **TEST tuned** | 0.604 | 0.399 | **0.481** |
| **TEST default** | 0.604 | 0.399 | **0.481** |

Recall by symbol size — **val**: small 0.160, medium 0.459, large 0.750.
**test**: small 0.393, medium 0.415, large 0.286.

Every val trial (all 12, P and R separately — never F1 alone):

| similarity_floor | nms_iou | precision | recall | F1 |
|---|---|---|---|---|
| 0.40 | 0.3 | 0.166 | 0.353 | 0.226 |
| 0.40 | 0.5 | 0.156 | 0.355 | 0.217 |
| 0.50 | 0.3 | 0.200 | 0.349 | 0.254 |
| 0.50 | 0.5 | 0.187 | 0.351 | 0.244 |
| 0.60 | 0.3 | 0.307 | 0.347 | 0.326 |
| 0.60 | 0.5 | 0.289 | 0.347 | 0.315 |
| **0.70** | **0.3** | **0.591** | **0.307** | **0.404** ← argmax |
| 0.70 | 0.5 | 0.572 | 0.307 | 0.400 |
| 0.80 | 0.3 | 0.838 | 0.157 | 0.265 |
| 0.80 | 0.5 | 0.814 | 0.157 | 0.264 |
| 0.85 | 0.3 | 0.935 | 0.082 | 0.150 |
| 0.85 | 0.5 | 0.935 | 0.082 | 0.150 |

### floorplans-window

| | precision | recall | F1 |
|---|---|---|---|
| **VAL** (winning trial) | 0.119 | 0.054 | **0.074** |
| **TEST tuned** | 0.119 | 0.103 | **0.110** |
| **TEST default** | 0.119 | 0.103 | **0.110** |

Recall by symbol size — **val**: small 0.020, medium 0.140, large 0.500.
**test**: small 0.062, medium 0.185, large 0.000.
Val argmax is again `{similarity_floor: 0.7, nms_iou: 0.3}` (val F1 0.074, next best 0.073 at
nms 0.5, then 0.068 at floor 0.6).

### Does this reproduce the committed 0.459 doors row? No — it is HIGHER, and both numbers stand

| source | coverage | runtime | P | R | **F1** |
|---|---|---|---|---|---|
| committed `docs/eval/floorplans-findings.md` | 13/14 | GPU-era | 0.55 | 0.39 | **0.459** |
| **this session (B1-final)** | **28/28** | CPU | 0.604 | 0.399 | **0.481** |

This is the session-to-session measurement drift `docs/reports/dino-dense-floorplans-improvement.md`
documents for its own method, and it was **pre-registered** in the B1 entry above before the runs
landed. Per the plan, **0.481 is this session's delta reference** for every step-1/2/3 comparison;
0.459 is stated alongside it wherever the committed row is cited, never silently replaced.
B2 already established that the CPU EP reproduces the committed regime numbers to within rounding,
so the difference is attributable to coverage (28/28 vs 13/14), not to the execution provider.

### The finding that matters most for the rest of this plan

**For BOTH datasets the tuned and default TEST rows are byte-identical**, because the val-argmax
winner `{similarity_floor: 0.7, nms_iou: 0.3}` **IS** the shipped `ProposeRetrieveConfig` default.

> **The committed `similarity_floor × nms_iou` grid buys exactly nothing on this domain.** The
> retrieval-stage knobs are already at their floor-plan optimum. Twelve val trials × two datasets
> ≈ 25 h of CPU produced a delta of 0.000.

Combined with B0 — door TEST final recall 0.399 sits essentially **at** the pooled proposal-stage
ceiling of 0.405 — this is independent confirmation that **the proposal stage is the only remaining
lever**. The retrieval stage is not leaving recall on the table; there is nothing left for it to
retrieve.

Two consequences, both acted on:

1. **Later sweeps hold `similarity_floor`/`nms_iou` near 0.7/0.3 rather than re-crossing the full
   6×2 grid**, since that cross is measured-inert here. Where tiling changes the proposal
   *distribution* (more proposals ⇒ more chances for a moderate-cosine false positive), a narrow
   floor sweep around the optimum is still run — the inertness was measured on the *untiled*
   distribution and does not automatically transfer.
2. **Trials are run as parallel processes from here on.** B3-CORRECTION measured each trial holding
   only ~3–4.6 of the box's 72 cores (load average 8–12 with three concurrent runs — ~85 % idle).
   The checkpoint decision was **stay CPU-only and parallelise**: no `onnxruntime-gpu` surgery on
   the shared box, ~12 concurrent independent trial processes instead. That turns the ~11 h
   sequential sweep that produced this entry into ~1 h of wall clock for an equivalent one.

---

## T1a — tiling tracer on the plans B0 measured at proposal recall 0.000

- **Local SHA:** `41b8431` (the box's `/root/repo` is an rsync of the source tree, not a clone, so
  it has no `.git` and the harness records `git_sha: "unknown"` — the local SHA is stated here
  instead, as B0 already does).
- **Command:** `pixi run python scripts/propose_retrieve_floorplans_experiment.py ptrial
  --name t1a-tracer-s512 --splits val --tile-side 512 --tile-overlap 0.2 --tile-merge-ios 0.5`
  (`tile_include_full_image=True`)
- **Artifact:** `runs/t1a-tracer-s512.json`
- **Scope:** the four plans B0 measured at proposal recall **0.000**, plus `22_png` (the most crowded
  val plan, B0 recall 0.059). Proposal stage only — no embedding, no retrieval.

| plan | size | n_gt | B0 n_prop | B0 recall | tiles | pre-merge | **tiled n_prop** | **tiled recall** |
|---|---|---|---|---|---|---|---|---|
| `65_png.rf.y2pckOMkZpYvXHhC2AZE` | 4000×1685 | 19 | 83 | 0.000 | 41 | 422 | 187 | **0.000** |
| `4061_png.rf.4VhzLHqSK6GM2tgWBh6K` | 1170×742 | 11 | 50 | 0.000 | 7 | 197 | 73 | **0.273** |
| `155_png.rf.5XBwQ9IzguztSIBPEOnz` | 818×647 | 7 | 77 | 0.000 | 5 | 260 | 65 | **0.000** |
| `4_png.rf.g6FJloai36FKwgawnlGA` | 513×436 | 7 | 30 | 0.000 | 3 | 88 | **15** | **0.000** |
| `22_png.rf.soRS3vbM0bF4DeDClZai` | 482×507 | 17 | 61 | 0.059 | 1 | 61 | 61 | 0.059 |

### The tracer gate fired, and it is recorded rather than smoothed over

The plan's tracer criterion was explicit: *"ONE plan (the worst one from B0: 4000×1685, 18 doors,
proposal recall 0.00) end-to-end tiled, before any sweep. If its proposal recall does not move off
0.00, stop and diagnose the tiling geometry before spending a sweep."*

**It did not move.** `65_png` received **41 tiles** and **187 merged proposals** for 19 doors and
still matched **zero** of them at IoU ≥ 0.5. Three of the four zero-recall plans stayed at exactly
0.000. Only `4061_png` moved (0.000 → 0.273).

**Deviation, recorded honestly:** the 12-config T1b sweep was launched concurrently with this tracer
rather than gated behind it, so the sweep was spent before the gate could stop it. That was a
process error. It cost box time, not correctness — and T1b's result (below) independently confirms
what the gate was trying to say, so the finding is unaffected. The gate is re-honoured from here on:
no further sweep is launched until the mechanism T1a exposed is understood.

### Two mechanisms visible in five plans

1. **`22_png` (482×507) returned a byte-identical result to B0** — 1 tile, 61 proposals, recall
   0.059. Both dimensions are ≤ 512, so `_tile_origins` returned the single whole-image tile and
   `propose_tiled_with_stats`'s step-0 short circuit ran the untiled path. This is the **identity
   property empirically confirmed on real data**, which is the same property that makes the
   chipset/textured/synthetic guardrail a no-op by construction.
2. **`4_png` came back with FEWER proposals than the untiled baseline: 30 → 15**, from **88
   pre-merge**. Tiling ran three passes, produced 88 candidate proposals, and the IoS merge deleted
   **83 %** of them, landing below the untiled count. A proposal stage that runs 3× the forward
   passes and returns half the proposals is not a budget increase — it is a budget *reduction*.
   This is the thread T1b pulls.

`65_png` remains the strongest single argument for a non-FastSAM proposal source: 187 proposals over
19 doors at 41 different magnifications, zero matches. No tiling geometry fixes a backend that does
not consider a CAD door symbol an object.

---

## T1b — proposal-stage geometry sweep (floorplans-door **VAL**, 56 plans, 12 configs)

- **Local SHA:** `41b8431`
- **Commands:** `… ptrial --name t1b-s{512,768,1024}-o{02,03}-fi{0,1} --splits val --tile-side S
  --tile-overlap O [--no-full-image]`, twelve `nohup`-detached processes, each
  `taskset -c N-M`-pinned to 4 cores with `OMP_NUM_THREADS=4`.
- **Artifacts:** `runs/t1b-s*-o*-fi*.json` (12 files)
- **Baseline row** is B0 **recomputed on the val split alone**, from B0's own per-plan rows, so the
  comparison is like-for-like (B0's headline table pools val+test; T1b is val-only).

Proposal-stage recall, mean-per-plan within each crowding bucket:

| geometry | mean tiles | pre-merge | merged n_prop | 1–3 | 4–10 | **11+** | all (mean) | all (pooled) | small | medium | large |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **B0 baseline (untiled)** | 1.0 | — | 46.5 | 0.905 | 0.595 | **0.171** | 0.498 | 0.374 | 0.220 | 0.537 | 0.750 |
| 512 / 0.2 / FI=off | 7.9 | 145.3 | 61.4 | 0.857 | 0.531 | **0.180** | 0.459 | 0.353 | 0.216 | 0.493 | 0.750 |
| 512 / 0.2 / FI=on | 8.8 | 187.4 | 60.6 | 0.857 | 0.556 | **0.214** | 0.484 | 0.378 | 0.238 | 0.520 | 0.812 |
| 512 / 0.3 / FI=off | 9.9 | 178.1 | 62.0 | 0.857 | 0.546 | **0.186** | 0.469 | 0.361 | 0.223 | 0.498 | 0.812 |
| 512 / 0.3 / FI=on | 10.8 | 220.3 | 61.5 | 0.857 | 0.571 | **0.237** | 0.500 | 0.395 | 0.255 | 0.537 | 0.812 |
| 768 / 0.2 / FI=off | 4.3 | 132.6 | 52.8 | 0.810 | 0.548 | **0.225** | 0.477 | 0.378 | 0.241 | 0.524 | 0.688 |
| 768 / 0.2 / FI=on | 4.9 | 163.1 | 53.3 | 0.810 | 0.563 | **0.239** | 0.490 | 0.393 | 0.252 | 0.546 | 0.688 |
| 768 / 0.3 / FI=off | 4.4 | 134.0 | 53.2 | 0.810 | 0.566 | **0.225** | 0.487 | 0.383 | 0.245 | 0.533 | 0.688 |
| **768 / 0.3 / FI=on** | 5.0 | 164.5 | 53.6 | 0.810 | 0.580 | **0.247** ← best 11+ | 0.502 | 0.402 | 0.259 | 0.559 | 0.688 |
| **1024 / 0.2 / FI=off** | 2.5 | 94.1 | 52.9 | 0.905 | 0.583 | **0.224** | **0.508** ← best all | 0.391 | 0.234 | 0.559 | 0.750 |
| 1024 / 0.2 / FI=on | 2.9 | 114.4 | 53.0 | 0.905 | 0.580 | **0.222** | 0.506 | 0.389 | 0.234 | 0.555 | 0.750 |
| 1024 / 0.3 / FI=off | 2.5 | 99.4 | 53.4 | 0.905 | 0.581 | **0.212** | 0.503 | 0.385 | 0.227 | 0.555 | 0.750 |
| 1024 / 0.3 / FI=on | 2.9 | 119.8 | 53.4 | 0.905 | 0.578 | **0.210** | 0.501 | 0.383 | 0.227 | 0.550 | 0.750 |

### Finding 1 — the IoS merge is a BUDGET CLAMP, and it is the reason tiling did nothing

The premise from B0 was: *"N tiles buy roughly N× the budget."* Measured, they do not:

| geometry | FastSAM passes/plan | pre-merge total | post-merge total | merge kill rate | **merged budget vs untiled** |
|---|---|---|---|---|---|
| 1024 / 0.2 / FI=off | 2.5× | 5 267 | 2 961 | 43.8 % | **1.14×** |
| 768 / 0.3 / FI=on | 5.0× | 9 214 | 3 001 | 67.4 % | **1.15×** |
| 512 / 0.2 / FI=off | 7.9× | 8 137 | 3 436 | 57.8 % | **1.32×** |
| 512 / 0.3 / FI=on | 10.8× | 12 335 | 3 443 | 72.1 % | **1.32×** |

> **Across a 4.3× range in forward passes (2.5 → 10.8 per plan), the merged proposal budget is
> pinned in the range 1.14–1.33× baseline.** The harder you tile, the harder the merge deletes.
> Pre-merge counts scale with tile count exactly as predicted (5 267 → 12 335); the merge removes
> the entire increase.

The mechanism is `_merge_tiled_proposals`'s use of **IoS**. IoS is `intersection / min(area)`, so a
small box **fully contained** in a larger one scores IoS = **1.0** and is suppressed whenever the
larger box was kept first. FastSAM everything-mode routinely emits **nested** proposals — a room,
and the door inside that room. The untiled `propose()` path does no merging at all, so those nested
proposals all survive; the tiled path deletes them. **The proposals being deleted are structurally
the small nested ones, which on this dataset are the doors.**

This is a real defect in transplanting SAHI's postprocess into an everything-mode segmenter, and it
is a *different* claim from the one the module docstring already makes. `proposals.py`'s step-0
comment correctly reasons that the single-tile short circuit is needed because "the merge is a
CROSS-TILE deduplicator, and with one pass there is nothing cross-tile to deduplicate." The measured
finding is stronger: the merge is applied to the **union of all tiles including within-tile
proposals**, so it suppresses nested proposals *within* a single tile too — which is over-segmentation
collapsing, explicitly the post-retrieval NMS's job, not the merge's. SAHI's own setting does not hit
this because SAHI merges *class detections*, where nesting is rare and meaningless.

### Finding 2 — the pre-registered null control wins the aggregate

R0 pre-registered this test: *"1024 is kept in the sweep precisely as the near-null control — if it
'wins', tiling is not what is helping."* On the aggregate mean recall it **does win**: `1024/0.2/FI=off`
scores **0.508**, the best of all twelve, from only 2.5 passes/plan with **55/84 plans left entirely
untiled**. The R0 test fires.

The nuance the aggregate hides, and which the crowding read was specified to expose: the crowded
(11+) bucket does have a genuine best at a real tiling geometry, `768/0.3/FI=on` at **0.247 vs 0.171**
(**+0.076**). But it is bought, not free — **every tiled geometry regresses the sparse and mid
buckets**:

| geometry | 1–3 | 4–10 | 11+ | net (all, mean) |
|---|---|---|---|---|
| baseline | 0.905 | 0.595 | 0.171 | 0.498 |
| 768 / 0.3 / FI=on | **−0.095** | **−0.015** | **+0.076** | **+0.004** |

A +0.076 crowded-bucket gain against a −0.095 sparse-bucket loss nets **+0.004** overall. Tiling as
currently merged is, on this dataset, **a redistribution of recall across crowding buckets, not an
increase.**

### Finding 3 — SAHI + FI is the one unambiguously positive knob

`tile_include_full_image=True` beats `False` at matched side/overlap in **every** 512 and 768 pair
(11+ bucket: 0.180→0.214, 0.186→0.237, 0.225→0.239, 0.225→0.247), for exactly one extra forward
pass. It is only inverted at 1024, where tiling is near-null anyway. This is consistent with the
merge diagnosis: the full-image pass contributes the *large-context* proposals that survive the merge
as the "kept first" boxes, and it re-supplies whole-object boxes for symbols truncated at tile edges.

### Latency — reported, but explicitly NOT a clean measurement

`mean_proposal_ms` for these runs ranges 70–217 s/plan, versus B0's 1.44 s/plan. **Do not read that
as a ~100× tiling cost.** These twelve processes were each pinned to **4 of 72 cores**
(`OMP_NUM_THREADS=4`) and run concurrently, whereas B0 ran unpinned. The pinning alone is ~16× of
it. The honest cost proxy from this entry is the **forward-pass multiplier** (`mean tiles`: 2.5–10.8×)
plus the merged-proposal multiplier that drives the dominant embedding stage (1.14–1.33×). A clean
wall-clock number belongs to T1c, run unpinned on an idle box, and is not claimed here.

### What this entry does NOT yet settle

Whether tiling is genuinely a dead lever, or whether it is a **live lever behind a broken merge**.
Findings 1 and 2 are consistent with both readings. That is a single measurable question — does
proposal recall recover when the merge stops deleting nested boxes? — and T1e answers it before any
decision is taken on step 1.

---

## T1e — does loosening the IoS merge threshold rescue the deleted nested boxes? **Yes, partly**

- **Local SHA:** `82344c7` (the box's `/root/repo` is an rsync of the source tree, not a clone, so
  the harness records `git_sha: "unknown"`; the local SHA is stated here instead, as B0 and T1a do).
- **Commands:** `… ptrial --name t1e-s{512,768}-o{02,03}-fi1-ios{080,095,099} --splits val
  --tile-side S --tile-overlap O --tile-merge-ios X`, five `nohup`-detached processes, each
  `taskset -c N-M`-pinned to 4 cores with `OMP_NUM_THREADS=4`.
- **Artifacts:** `runs/t1e-s512-o02-fi1-ios{080,095,099}.json`,
  `runs/t1e-s768-o03-fi1-ios{095,099}.json`
- **Scope:** floorplans-door **VAL** (56 plans), proposal stage only. The `merge_ios = 0.5` rows are
  the already-recorded T1b runs at the identical geometry, restated here so the comparison is
  like-for-like rather than cross-entry.

T1b's Finding 1 was that the IoS merge is a budget clamp which deletes FastSAM's naturally nested
proposals (a room, and the door inside it). The default threshold is SAHI's 0.5. This entry sweeps it.

| geometry | `merge_ios` | mean tiles | pre-merge | merged n_prop | 1–3 | 4–10 | **11+** | all (mean) | all (pooled) | small | medium | large |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **B0 baseline (untiled)** | — | 1.0 | — | 46.5 | 0.905 | 0.595 | **0.171** | 0.498 | 0.374 | 0.220 | 0.537 | 0.750 |
| 512 / 0.2 / FI=on | 0.50 (T1b) | 8.8 | 187.4 | 60.6 | 0.857 | 0.556 | **0.214** | 0.484 | 0.378 | 0.238 | 0.520 | 0.812 |
| 512 / 0.2 / FI=on | 0.80 | 8.8 | 187.4 | 62.5 | 0.857 | 0.574 | **0.218** | 0.495 | 0.387 | 0.248 | 0.528 | 0.812 |
| 512 / 0.2 / FI=on | 0.95 | 8.8 | 187.4 | 66.0 | 0.857 | 0.606 | **0.228** | 0.516 | 0.410 | 0.280 | 0.541 | 0.812 |
| 512 / 0.2 / FI=on | 0.99 | 8.8 | 187.4 | 76.5 | 0.857 | 0.622 | **0.226** | 0.524 | 0.416 | 0.277 | 0.559 | 0.812 |
| 768 / 0.3 / FI=on | 0.50 (T1b) | 5.0 | 164.5 | 53.6 | 0.810 | 0.580 | **0.247** | 0.502 | 0.402 | 0.259 | 0.559 | 0.688 |
| 768 / 0.3 / FI=on | 0.95 | 5.0 | 164.5 | 57.6 | 0.857 | 0.605 | **0.268** | 0.528 | 0.431 | 0.298 | 0.572 | 0.750 |
| **768 / 0.3 / FI=on** | **0.99** | 5.0 | 164.5 | 66.8 | 0.857 | 0.615 | **0.280** ← best | **0.538** ← best | **0.440** | **0.312** | 0.576 | 0.750 |

### Finding 1 — loosening the threshold helps, monotonically, on every metric that matters

Answering T1b's open question plainly: **yes, proposal recall recovers when the merge stops deleting
nested boxes — and it recovers in exactly the buckets the diagnosis said were leaking.** At
`768/0.3/FI=on`, moving `merge_ios` 0.5 → 0.99:

| bucket | untiled B0 | tiled @ ios 0.5 | tiled @ ios 0.99 | vs untiled |
|---|---|---|---|---|
| 11+ (crowded) | 0.171 | 0.247 | **0.280** | **+0.109** |
| 4–10 | 0.595 | 0.580 | 0.615 | +0.020 |
| 1–3 (sparse) | 0.905 | 0.810 | 0.857 | −0.048 |
| **small symbols** | 0.220 | 0.259 | **0.312** | **+0.092** |
| all (mean) | 0.498 | 0.502 | **0.538** | **+0.040** |
| all (pooled) | 0.374 | 0.402 | **0.440** | **+0.066** |

Two things changed relative to T1b's reading:

1. **The crowded-bucket gain grew by half again** (+0.076 → +0.109), and the small-symbol bucket —
   where doors actually live — gained **+0.092 (a 42 % relative lift)**.
2. **The sparse-bucket regression halved** (−0.095 → −0.048). T1b's headline objection was that
   tiling *redistributes* recall rather than increasing it (+0.076 crowded bought with −0.095
   sparse, netting +0.004). At the loosened threshold that trade is no longer a wash: **+0.109
   against −0.048, netting +0.040 mean / +0.066 pooled.**

### Finding 2 — the T1b null control no longer wins, which reverses T1b's verdict

R0 pre-registered the test: *"1024 is kept in the sweep precisely as the near-null control — if it
'wins', tiling is not what is helping."* In T1b the near-null `1024/0.2/FI=off` **did** win the
aggregate at 0.508, and that is why T1b's provisional read was "tiling is not the fix".

| config | all (mean) | 11+ |
|---|---|---|
| untiled B0 baseline | 0.498 | 0.171 |
| T1b winner — near-null control `1024/0.2/FI=off` @ ios 0.5 | 0.508 | 0.224 |
| **T1e winner — real tiling `768/0.3/FI=on` @ ios 0.99** | **0.538** | **0.280** |

A genuinely tiled geometry now beats the near-null control on the aggregate **and** on the crowded
bucket. The R0 test no longer fires. **The correct reading of T1b was therefore the second one it
offered — tiling is a live lever that was sitting behind a broken merge, not a dead lever** — and
the earlier "SAHI-style tiling does not help" conclusion is superseded by this entry. It is left
standing above, per the append-only rule, precisely because the sequence of readings is the record.

### Finding 3 — the clamp is still binding at 0.99, and the mechanism says exactly why

The merge is still deleting the majority of proposals even at the loosest threshold tested:

| geometry | `merge_ios` | pre-merge total | post-merge total | kill rate | merged budget vs untiled |
|---|---|---|---|---|---|
| 512 / 0.2 / FI=on | 0.50 | 10 497 | 3 395 | 67.7 % | 1.30× |
| 512 / 0.2 / FI=on | 0.95 | 10 497 | 3 698 | 64.8 % | 1.42× |
| 512 / 0.2 / FI=on | 0.99 | 10 497 | 4 284 | **59.2 %** | 1.65× |
| 768 / 0.3 / FI=on | 0.50 | 9 214 | 3 001 | 67.4 % | 1.15× |
| 768 / 0.3 / FI=on | 0.99 | 9 214 | 3 740 | **59.4 %** | 1.44× |

`_ios` is `intersection / min(area)`, so a box **fully contained** in a kept box scores IoS =
**exactly 1.0**. `_merge_tiled_proposals` matches on a strict `>` (SAHI's convention, verified in R0
and implemented that way), so a fully nested proposal is suppressed at **every threshold below
1.0** — 0.99 included. Loosening from 0.5 to 0.99 only rescues the *partially* overlapping boxes;
the strictly nested ones, which is what an everything-mode segmenter emits most of, are deleted
identically at 0.5 and at 0.99.

That predicts a specific, cheap, decisive endpoint: at **`merge_ios = 1.0`** the strict `>` can never
fire, the merge is disabled outright, and the full pre-merge budget (3.5× untiled) survives. Every
metric above is still climbing monotonically at 0.99 with no sign of a plateau, so the sweep is not
finished. **T1f measures that endpoint**, plus a geometry re-check at the loosened threshold (the
T1b geometry ranking was established under the clamp and does not automatically survive it).

### Finding 4 — what the threshold cannot fix, and the identity property re-confirmed

Per-plan, on the five plans T1a singled out:

| plan | size | n_gt | B0 | 512 @ 0.5 | 512 @ 0.99 | 768 @ 0.5 | **768 @ 0.99** |
|---|---|---|---|---|---|---|---|
| `65_png` | 4000×1685 | 19 | 0.000 | 0.000 | 0.053 | 0.053 | **0.105** |
| `4061_png` | 1170×742 | 11 | 0.000 | 0.273 | 0.273 | 0.273 | 0.273 |
| `155_png` | 818×647 | 7 | 0.000 | 0.000 | 0.000 | 0.143 | 0.143 |
| `4_png` | 513×436 | 7 | 0.000 | 0.000 | 0.000 | 0.000 | **0.000** |
| `22_png` | 482×507 | 17 | 0.059 | 0.059 | 0.059 | 0.059 | 0.059 |

- `65_png`, the worst plan in the dataset, finally moves off 0.000 — **the T1a tracer gate that
  fired now passes** — but only to 0.105. Two of 19 doors.
- **`4_png` is 0.000 at every geometry and every threshold**, with 20–30 proposals over a 513×436
  plan. No merge threshold, and no tile size, changes a backend that does not consider a CAD door
  symbol an object. This remains the single strongest argument for the step-3 contour backend, and
  it is now measured across seven configurations rather than one.
- **`22_png` returns byte-identical numbers (0.059 / 61 proposals) in all seven runs** — both its
  dimensions are ≤ 512, so `_tile_origins` yields the single whole-image tile and the step-0 short
  circuit bypasses the merge entirely. The identity property that makes the chipset/textured/
  synthetic guardrail a no-op **by construction** is re-confirmed here across the whole `merge_ios`
  sweep, not just at the default.

### Consequence for step 1

Step 1 is **not** reverted. It is carried forward to T1f for the endpoint measurement, and the
step-1 finalist will be the argmax over `{geometry × merge_ios}` on VAL — with the sparse-bucket
regression reported alongside the crowded-bucket gain in every case, never netted away.

---

## H1 — harness fix: the `b2` guardrail entry point could not be re-run without clobbering B2

`main()`'s `b2` branch called `regime_check()` with no `--name` passthrough, so it always wrote the
fixed stems `b2--regimes.json` / `b2--regimes-raw.json` — unlike `ptrial`/`trial`, which both take
`--name`. Re-running the guardrail for T1d would therefore have **overwritten the B2 baseline
artifact it is meant to be compared against**, which an append-only notebook cannot tolerate. Fixed
to `regime_check(name=args.name or "b2")`, matching the `ptrial`/`trial` pattern; the default is
unchanged so the original B2 command still reproduces byte-for-byte. Verified before running T1d
that the box's `b2--regimes.json` was still the pristine original (`md5 18ddc9e6…`, mtime Aug 12,
per-regime F1 identical to the B2 entry above) — no prior run had clobbered it.

---

## RUNTIME NOTE — the original box was lost; entries below ran on a new one

Entries **B0 through H1 above** ran on vast.ai contract `47510440`. That box **disappeared from the
account entirely** — not a permission error, not a stopped container: gone, with its working state.
Everything already committed to this notebook survives because the JSON artifacts had been pulled
back to `runs/` and committed; nothing above is re-derived or re-estimated here.

**Every entry from T1f onward ran on vast.ai contract `48124756`** (`ssh -p 14756 root@ssh6.vast.ai`,
RTX 3090 host but **ONNX Runtime is again the CPU build**, 56 cores, ~$0.14/hr). One material
improvement: `/root/repo` on the new box is a real **git clone** (restored from a bundle at
`79fd33e`), not the previous rsync-of-a-source-tree. So `current_git_sha()` now resolves, and every
artifact below carries a true `git_sha: 79fd33e0c1b8af1deec494a074d74f83f890e7ce` rather than the
`"unknown"` that forced T1a/T1b/T1e to state a local SHA in prose.

---

## T1f — the `merge_ios = 1.0` endpoint, and the geometry re-check at a loosened merge

- **SHA:** `79fd33e` (recorded by the harness, verified in every artifact)
- **Commands:** `… ptrial --name t1f-s{512,768,1024}-o{02,03}-fi1-ios{099,100} --splits val
  --tile-side S --tile-overlap O --tile-merge-ios X`, five `nohup`-detached processes,
  `taskset`-pinned to 4 cores each with `OMP_NUM_THREADS=4`.
- **Artifacts:** `runs/t1f-s768-o03-fi1-ios100.json`, `runs/t1f-s512-o02-fi1-ios100.json`,
  `runs/t1f-s768-o02-fi1-ios{099,100}.json`, `runs/t1f-s1024-o02-fi1-ios099.json`
- **Scope:** floorplans-door **VAL** (56 plans), proposal stage only. T1e rows restated for
  like-for-like comparison.

T1e Finding 3 made a falsifiable mechanistic prediction: because `_merge_tiled_proposals` matches on
a **strict `>`**, a fully-nested box (IoS exactly 1.0) is suppressed at every threshold *below* 1.0,
so at `merge_ios = 1.0` the merge should be **disabled outright** and the full pre-merge budget
should survive. That is testable in one line of arithmetic:

| geometry | `merge_ios` | pre-merge total | post-merge total | **kill rate** |
|---|---|---|---|---|
| 768 / 0.3 / FI=on | 0.99 | 9 214 | 3 740 | 59.4 % |
| **768 / 0.3 / FI=on** | **1.00** | 9 214 | **9 214** | **0.0 %** |
| 512 / 0.2 / FI=on | 1.00 | 10 497 | 10 497 | **0.0 %** |
| 768 / 0.2 / FI=on | 1.00 | 9 135 | 9 135 | **0.0 %** |

**The prediction holds exactly.** At 1.0 the kill rate is 0.0 % — not "low", zero — on all three
geometries. The mechanism T1b diagnosed and T1e half-fixed is now fully characterised.

### Recall at the endpoint

| geometry | `merge_ios` | mean tiles | pre-merge | merged n_prop | 1–3 | 4–10 | **11+** | all (mean) | all (pooled) | small | medium | large |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **B0 baseline (untiled)** | — | 1.0 | — | 46.5 | 0.905 | 0.595 | **0.171** | 0.498 | 0.374 | 0.220 | 0.537 | 0.750 |
| 768 / 0.3 / FI=on | 0.99 (T1e) | 5.0 | 164.5 | 66.8 | 0.857 | 0.615 | **0.280** | 0.538 | 0.440 | 0.312 | 0.576 | 0.750 |
| **768 / 0.3 / FI=on** | **1.00** | 5.0 | 164.5 | **164.5** | 0.952 | 0.677 | **0.292** | **0.588** | 0.476 | 0.330 | 0.633 | 0.812 |
| 512 / 0.2 / FI=on | 0.99 (T1e) | 8.8 | 187.4 | 76.5 | 0.857 | 0.622 | **0.226** | 0.524 | 0.416 | 0.277 | 0.559 | 0.812 |
| 512 / 0.2 / FI=on | **1.00** | 8.8 | 187.4 | **187.4** | 0.905 | 0.685 | **0.290** | 0.586 | **0.482** | **0.351** | 0.620 | 0.812 |
| 768 / 0.2 / FI=on | 0.99 | 4.9 | 163.1 | 66.2 | 0.857 | 0.616 | **0.272** | 0.535 | 0.436 | 0.305 | 0.576 | 0.750 |
| 768 / 0.2 / FI=on | **1.00** | 4.9 | 163.1 | **163.1** | 0.952 | 0.678 | **0.284** | 0.585 | 0.472 | 0.323 | 0.633 | 0.812 |
| 1024 / 0.2 / FI=on | 0.99 | 2.9 | 114.4 | 62.6 | 0.905 | 0.627 | **0.235** | 0.536 | 0.421 | 0.284 | 0.568 | 0.750 |

**The climb continues but flattens.** 768/0.3 moves mean recall 0.538 → 0.588 and the crowded bucket
0.280 → 0.292 — the crowded bucket, the one the whole diagnosis is about, gains only **+0.012** for
a **2.5× increase in surviving proposals** (66.8 → 164.5). The sparse-bucket regression that T1b
complained about is fully repaired (0.905 → 0.952, now *above* baseline), but that bucket was never
the problem.

### Finding 1 — magnification is measured to do NOTHING, which kills SAHI's stated premise here

With the merge disabled, two geometries differ in magnification by exactly 2× and can be compared
directly, because at `ios=1.0` nothing is deleted and the only difference is tile geometry:

| geometry | magnification (`1024 / S`) | tiles/plan | n_prop | mean recall | small | 11+ |
|---|---|---|---|---|---|---|
| 512 / 0.2 / FI=on | **2.00×** | 8.8 | 187.4 | **0.586** | 0.351 | 0.290 |
| 768 / 0.2 / FI=on | **1.33×** | 4.9 | 163.1 | **0.585** | 0.323 | 0.284 |

> **A 2× difference in pixels-per-symbol produces a 0.001 difference in mean proposal recall.**

R0's caveat said the two levers (magnification and budget) "happen to point the same way, so
SAHI-style tiling remains the right instrument". At the endpoint they can finally be separated, and
**only budget is real**. The residual 512-vs-768 gap in the `small` bucket (0.351 vs 0.323) tracks
the 15 % higher proposal count (187.4 vs 163.1), not the 50 % higher magnification. SAHI's premise —
*small objects lost to downscaling* — is **measured false for this dataset**. What tiling buys here
is proposals, nothing else. That reframes step 1 entirely and sets up the step-2 question: if budget
is the only lever, is tiling even the cheapest way to buy it?

### Finding 2 — the geometry ranking collapses once the merge is disabled

At `ios=0.5` (T1b) the twelve geometries spread over 0.459–0.508 mean recall and the choice of tile
side looked consequential. At `ios=1.0` the three surviving geometries land at **0.585, 0.586,
0.588** — a spread of 0.003, i.e. nothing. **The T1b geometry ranking was an artifact of the merge
bug**: different tile counts fed the clamp different amounts of nested overlap, so the clamp deleted
different fractions. With the clamp gone, tile geometry stops mattering. This retro-justifies not
re-running the full 12-config geometry sweep at the loosened threshold — the axis is inert.

`768 / 0.3 / FI=on @ ios 1.00` is carried forward as the step-1 geometry on the tie-break of fewest
forward passes among the top three (5.0 tiles vs 8.8 for 512/0.2, at the same recall).

### Finding 3 — per-plan, the plans of record

| plan | size | n_gt | B0 | 768/.3 @0.99 | **768/.3 @1.00** | 512/.2 @1.00 |
|---|---|---|---|---|---|---|
| `65_png` | 4000×1685 | 19 | 0.000 | 0.105 (262p) | **0.105** (549p) | 0.105 (422p) |
| `4061_png` | 1170×742 | 11 | 0.000 | 0.273 (71p) | **0.273** (138p) | 0.273 (197p) |
| `155_png` | 818×647 | 7 | 0.000 | 0.143 (79p) | **0.143** (230p) | 0.000 (260p) |
| `4_png` | 513×436 | 7 | 0.000 | 0.000 (30p) | **0.000** (30p) | 0.000 (88p) |
| `22_png` | 482×507 | 17 | 0.059 | 0.059 (61p) | **0.059** (61p) | 0.059 (61p) |

Note `65_png`: **549 proposals for 19 doors and still recall 0.105.** And `155_png` at 512/0.2 gets
260 proposals and scores **0.000** while the same plan at 768/0.3 with *fewer* proposals (230) scores
0.143 — proposals alone are clearly not sufficient either. `4_png` and `22_png` remain immovable
under every tiling configuration tested, which at this point in the session was still the strongest
argument for the step-3 contour backend. **T2 overturns that reading** — see below.

---

## T1c — step 1 END-TO-END on VAL (does the proposal-stage gain convert to F1?)

- **SHA:** `79fd33e`
- **Commands:** `… trial --dataset floorplans-door --split val --name
  t1c-s768-o03-fi1-ios{099,100}-val --config '{"proposal_tiling": true, "tile_side": 768,
  "tile_overlap": 0.3, "tile_include_full_image": true, "tile_merge_ios": X}'`
- **Artifacts:** `runs/t1c-s768-o03-fi1-ios{099,100}-val.json`
- **Scope:** floorplans-door **VAL**, 56/56 scored, 0 errors, 1 exemplar, `similarity_floor` 0.7 /
  `nms_iou` 0.3 (the B1-final val argmax, held fixed).

Proposal-stage recall is a **ceiling**, not a result. This is the first entry that runs the full
pipeline with tiling on.

| config | n_prop | P | R | **F1** | small | medium | large | abstentions |
|---|---|---|---|---|---|---|---|---|
| **B1-final baseline (untiled, conf 0.4)** | 46.5 | **0.591** | 0.307 | **0.404** | 0.160 | 0.459 | 0.750 | — |
| tiled 768/0.3/FI @ `ios 0.99` | 66.8 | 0.563 | 0.340 | **0.424** | 0.213 | 0.467 | 0.750 | 5 |
| tiled 768/0.3/FI @ `ios 1.00` | 164.5 | 0.487 | **0.380** | **0.426** | 0.241 | 0.520 | 0.812 | 5 |

### Finding 1 — the gain is real, small, and bought with precision

Recall moves **+0.073** (0.307 → 0.380) and every symbol-size bucket improves. But precision falls
**−0.104** (0.591 → 0.487), so F1 moves only **+0.022** (0.404 → 0.426). Reported as P and R
separately per the plan's rule, precisely because F1 alone hides that this is a trade, not a
free win.

Note the transfer ratio: proposal-stage mean recall rose +0.090 (0.498 → 0.588) and final recall rose
+0.073 — consistent with B0's measured ~0.82 proposal-to-final transfer. The retrieval stage is
converting the extra proposals at the expected rate; it is not the bottleneck, exactly as B1-final
concluded.

### Finding 2 — the crowded bucket barely moves end-to-end

The benchmark's own crowding slices (cuts `2-5` / `6-15` / `16+`, which differ from the
proposal-stage `1-3` / `4-10` / `11+` cuts — stated so the two are not confused):

| slice | @ ios 0.99 | @ ios 1.00 |
|---|---|---|
| `2-5` | R 0.704 / P 0.594 / F1 0.644 | R 0.778 / P 0.467 / F1 0.583 |
| `6-15` | R 0.367 / P 0.667 / F1 0.474 | R 0.405 / P 0.574 / F1 0.475 |
| **`16+`** | R 0.115 / P 0.231 / F1 **0.154** | R 0.146 / P 0.241 / F1 **0.182** |

The most crowded slice ends at **F1 0.182**. Step 1 does not fix crowded plans; it nudges them.

### Latency (EVAL-11) — the honest cost of step 1

| config | tiles/plan | median plan | mean plan | worst plan | val wall clock |
|---|---|---|---|---|---|
| baseline (B3-CORRECTION) | 1.0 | — | ~57.6 s | — | ~54 min |
| tiled @ `ios 0.99` | 5.0 | 111.7 s | 203.0 s | 1 203 s | **190.7 min** |
| tiled @ `ios 1.00` | 5.0 | **30.1 s** | **224.5 s** | **2 754 s (46 min!)** | **209.9 min** |

**~3.9× the baseline wall clock for +0.022 F1.** The distribution is violently skewed: at `ios 1.00`
the *median* plan is faster (30.1 s) than at `ios 0.99` (111.7 s) while the *mean* is higher — one
plan (`65_png`, 4000×1685, 549 proposals each needing its own DINOv2 forward pass) takes **46
minutes by itself**. This is the B3 finding compounding: the embedding stage dominates and scales
linearly with proposal count, so disabling the merge moves cost straight onto the embedding stage.

These runs were 4-core-pinned and contended (10+ concurrent processes), so absolute seconds are not
clean wall-clocks; the **ratios** and the tile/proposal multipliers are the honest cost proxies.

---

## T1d — guardrail re-check: did anything leak into the default path?

- **SHA:** `79fd33e`
- **Commands:** `… b2 --name t1d` (regimes); `… trial --dataset floorplans-window --split val
  --name t1d-window-val --config '{}'`; same for `--split test --name t1d-window-test`.
- **Artifacts:** `runs/t1d--regimes.json`, `runs/t1d--regimes-raw.json`, `runs/t1d-window-val.json`,
  `runs/t1d-window-test.json`

### Regimes — byte-identical to B2, all six

| regime | n | B2 F1 | **T1d F1** | delta |
|---|---|---|---|---|
| EASY (chipset) | 10 | 0.9274 | 0.9274 | **+0.0000** |
| TEXTURED (plain) | 16 | 0.9591 | 0.9591 | **+0.0000** |
| VARIED (scale/rotation) | 16 | 0.9385 | 0.9385 | **+0.0000** |
| CLUTTERED | 16 | 0.8209 | 0.8209 | **+0.0000** |
| synthetic | 2 | 0.9091 | 0.9091 | **+0.0000** |
| real-objects | 30 | 0.8683 | 0.8683 | **+0.0000** |
| **overall** | 90 | 0.8987 | **0.8987** | **+0.0000** |

90/90 scored, 0 errors. Not "within rounding" — **identical to four decimal places on every regime**.

### floorplans-window — identical at the default config

| | P | R | F1 |
|---|---|---|---|
| B1-final window VAL (`floor 0.7 / nms 0.3` = the shipped default) | 0.119 | 0.054 | **0.074** |
| **T1d window VAL (default config)** | 0.119 | 0.054 | **0.074** |

The config dumped into the artifact confirms why: `proposal_tiling: false`, `tile_merge_ios: 0.5`,
`tile_side: 1024` — every field added by commit `41b8431` sits at its no-op default, so `search()`
takes the untiled branch and the result is identity **by construction**, which is what the
`tests/test_propose_retrieve.py` byte-identity test asserts in CI without weights.

**Process note, recorded rather than smoothed over:** the first `t1d-window-test` process died
silently at 19:49 box time with no traceback and no artifact — the log simply stops mid-run. Cause
undetermined (the box was running 18 concurrent processes at the time; most likely a resource kill).
It was **re-launched** rather than reported from the val run alone, because a guardrail that "would
have passed" is not a guardrail. Its result is recorded in T2-final below.

---

## T2 — `proposal_conf`, run as a CONTROLLED experiment against tiling (the confound check)

- **SHA:** `79fd33e`
- **Commands:** eight `nohup`-detached, 4-core-pinned processes —
  - **T2a (tiled arm):** `… ptrial --name t2a-scout-conf0{10,20,30,50} --splits val --tile-side 768
    --tile-overlap 0.3 --tile-merge-ios 1.0 --proposal-conf C`
  - **T2b (untiled control arm):** `… ptrial --name t2b-untiled-conf0{10,20,30,50} --splits val
    --proposal-conf C` (no `--tile-side`, so the untiled path)
- **Artifacts:** `runs/t2a-scout-conf0{10,20,30,50}.json`, `runs/t2b-untiled-conf0{10,20,30,50}.json`
- **Scope:** floorplans-door **VAL** (56 plans), proposal stage only.

### Why this entry is a controlled experiment and not the sweep the plan specified

The plan specifies step 2 as *"sweep `proposal_conf` on top of step 1's winner"* — i.e. tiling is
assumed helpful and held fixed underneath. A single-plan observation during T1f made that assumption
unsafe. On val plan `109_png` (603×451, 12 doors — **below the 768 tile, so tiling at 768 is a
provable identity there**):

| `109_png` config | tiles | n_prop | proposal recall |
|---|---|---|---|
| B0, untiled, `conf 0.40` | 1 | 27 | 0.083 |
| **tiled 512/0.2, `conf 0.40`** | 3 | **79 (2.9×)** | **0.083 (+0.000)** |
| **untiled, `conf 0.30`** | 1 | **46 (1.7×)** | **0.333 (+0.250)** |
| untiled, `conf 0.10` | 1 | 103 (3.8×) | 0.417 |
| tiled 768/0.3, `conf 0.10` | 1 | 103 | 0.417 *(identity confirmed)* |

Tiling bought 2.9× the proposals for **exactly zero** recall; the objectness gate bought 1.7× the
proposals for **+0.250**. If that generalises, the plan's "step 1 = tiling, step 2 = conf" ordering
is backwards and step 1's measured gain is substantially a `proposal_conf` confound wearing a tiling
costume — both levers ultimately act by putting more proposals on the table.

So step 2 was run as **two matched arms at identical `proposal_conf` values**, tiled and untiled,
which makes the attribution answerable from data instead of assumed. This is the deviation from the
plan's literal instruction, taken deliberately and recorded here.

### The two arms

| geometry | conf | tiles | n_prop | 1–3 | 4–10 | **11+** | all (mean) | all (pooled) | small | medium | large | proposal s/plan |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **B0 baseline untiled** | 0.40 | 1.0 | 46.5 | 0.905 | 0.595 | **0.171** | 0.498 | 0.374 | 0.220 | 0.537 | 0.750 | 1.4 |
| untiled | 0.50 | 1.0 | 33.7 | 0.905 | 0.488 | 0.108 | 0.418 | 0.290 | 0.138 | 0.445 | 0.750 | 42.6 |
| **tiled 768/0.3** | 0.50 | 5.0 | 123.7 | 0.905 | 0.580 | 0.185 | 0.494 | 0.376 | 0.248 | 0.507 | 0.750 | 161.3 |
| untiled | 0.30 | 1.0 | 65.9 | 1.000 | 0.691 | 0.277 | 0.597 | 0.476 | 0.348 | 0.607 | 0.875 | 37.9 |
| **tiled 768/0.3** | 0.30 | 5.0 | 223.8 | 1.000 | 0.761 | 0.423 | 0.682 | 0.583 | 0.450 | 0.725 | 0.875 | 172.1 |
| untiled | 0.20 | 1.0 | 95.3 | 1.000 | 0.777 | 0.433 | 0.694 | 0.592 | 0.465 | 0.729 | 0.875 | 48.9 |
| **tiled 768/0.3** | 0.20 | 5.0 | 317.7 | 1.000 | 0.821 | 0.556 | 0.758 | 0.676 | 0.546 | 0.821 | 0.875 | 172.5 |
| **untiled** | **0.10** | 1.0 | 161.2 | 1.000 | 0.886 | **0.639** | **0.821** | 0.751 | 0.638 | 0.882 | 0.875 | 36.1 |
| **tiled 768/0.3** | **0.10** | 5.0 | 520.8 | 1.000 | 0.898 | **0.695** | **0.846** | 0.787 | 0.684 | 0.908 | 0.875 | 188.1 |

### Finding 1 — tiling's independent contribution is real, positive, and small

At matched `proposal_conf`, tiling is **not** a pure confound. It adds recall at every conf tested:

| conf | untiled n_prop | tiled n_prop | proposal ratio | Δ mean | Δ small | Δ 11+ | latency ratio |
|---|---|---|---|---|---|---|---|
| 0.50 | 33.7 | 123.7 | 3.67× | **+0.076** | +0.110 | +0.077 | 3.78× |
| 0.30 | 65.9 | 223.8 | 3.40× | **+0.085** | +0.103 | +0.145 | 4.54× |
| 0.20 | 95.3 | 317.7 | 3.33× | **+0.064** | +0.082 | +0.122 | 3.53× |
| 0.10 | 161.2 | 520.8 | 3.23× | **+0.024** | +0.046 | +0.056 | 5.22× |

So the honest statement is *not* "tiling did nothing". It is that tiling contributes **+0.024 to
+0.085 mean proposal recall for 3.2–3.7× the proposals and 3.5–5.2× the proposal latency**, and its
contribution **shrinks as the gate opens** (+0.085 at conf 0.30 → +0.024 at conf 0.10). The two
levers are substitutes competing for the same resource, not complements.

### Finding 2 — at MATCHED PROPOSAL BUDGET, the objectness gate strictly dominates tiling

Because both levers act by spending proposal budget, the fair comparison holds that budget fixed.
Sorting every measured config by proposals-per-plan:

| n_prop | config | mechanism | mean | pooled | small | **11+** | proposal s/plan |
|---|---|---|---|---|---|---|---|
| 33.7 | untiled `conf 0.50` | conf | 0.418 | 0.290 | 0.138 | 0.108 | 42.6 |
| 46.5 | untiled `conf 0.40` (B0) | baseline | 0.498 | 0.374 | 0.220 | 0.171 | 1.4 |
| 65.9 | untiled `conf 0.30` | conf | 0.597 | 0.476 | 0.348 | 0.277 | 37.9 |
| 95.3 | untiled `conf 0.20` | conf | 0.694 | 0.592 | 0.465 | 0.433 | 48.9 |
| 123.7 | tiled `conf 0.50` | tile+conf | 0.494 | 0.376 | 0.248 | 0.185 | 161.3 |
| **161.2** | **untiled `conf 0.10`** | **conf** | **0.821** | **0.751** | **0.638** | **0.639** | **36.1** |
| **164.5** | **tiled `conf 0.40`** (T1f) | **tile** | **0.588** | 0.476 | 0.330 | **0.292** | **103.3** |
| 223.8 | tiled `conf 0.30` | tile+conf | 0.682 | 0.583 | 0.450 | 0.423 | 172.1 |
| 317.7 | tiled `conf 0.20` | tile+conf | 0.758 | 0.676 | 0.546 | 0.556 | 172.5 |
| 520.8 | tiled `conf 0.10` | tile+conf | 0.846 | 0.787 | 0.684 | 0.695 | 188.1 |

The two bolded rows are the decisive pair — **161.2 vs 164.5 proposals per plan, a 2 % difference in
budget**, one bought by opening the gate and one bought by tiling:

> **At an equal proposal budget, opening the objectness gate beats tiling by +0.233 mean recall
> (0.821 vs 0.588), +0.308 on small symbols (0.638 vs 0.330), and +0.347 in the crowded bucket
> (0.639 vs 0.292) — while costing about a THIRD of the proposal-stage latency (36.1 s vs 103.3 s
> per plan).**

The domination is not marginal and it is not confined to that pair. `untiled conf 0.20` (95.3
proposals, mean 0.694) beats `tiled conf 0.50` (123.7 proposals, mean 0.494) with **23 % fewer**
proposals. `untiled conf 0.10` (161.2, 0.821) beats `tiled conf 0.30` (223.8, 0.682) with **28 %
fewer**. Every untiled point lies above the tiled frontier. The only tiled config that beats the best
untiled one is `tiled conf 0.10` — **+0.025 mean for 3.2× the proposals and 5.2× the latency**.

### Finding 3 — the plain attribution, stated as the data has it

**The objectness gate is the primary lever; tiling is a strictly less efficient way to buy the same
thing.** Step 1's headline result — T1f's mean recall 0.498 → 0.588 — is reproduced *and beaten* by
changing one existing config field from `0.4` to `0.3` (0.498 → 0.597), with **no tiling, no new
code, 2.5× fewer proposals and a third of the latency**.

That is a direct correction to this notebook's own step-1 framing. B0's "Consequence for step 1"
concluded that tiling's live lever is budget — correct — but did not ask the follow-up question:
*is tiling the cheapest source of budget?* It is not, and `proposal_conf` was sitting in the shipped
config the entire time. The T1b/T1e/T1f arc (three entries, ~30 h of CPU, a merge-threshold bug hunt
and a 273-line `propose_tiled` implementation) chased a mechanism that a pre-existing scalar
outperforms. Recorded plainly because the repo's discipline is iterate/measure/**revert**, and the
measurement here points at the code written in this very quick task.

The mechanism is consistent with T1f Finding 1: magnification does nothing, so a tile is just a
smaller area over which FastSAM's area-proportional budget is re-spent — while `proposal_conf` lifts
the budget on the *whole* plan without paying for 5 extra forward passes or re-detecting the same
room boundary in every tile. Tiling's residual +0.024…+0.085 is the genuinely new information a tile
adds (a symbol seen without its surrounding context), and that is worth exactly what it measures.

### Finding 4 — this REFUTES T1e Finding 4, the strongest argument for step 3

T1e Finding 4 stated, of the four plans B0 measured at proposal recall 0.000:

> *"`4_png` is 0.000 at every geometry and every threshold… No merge threshold, and no tile size,
> changes a backend that does not consider a CAD door symbol an object. This remains the single
> strongest argument for the step-3 contour backend."*

That inference was sound on tiling evidence and is **wrong**. Those symbols were never invisible to
FastSAM — they were scoring **below the 0.4 objectness gate**:

| plan | size | n_gt | B0 `conf 0.40` | best TILED @ `conf 0.40` | **untiled `conf 0.20`** | **untiled `conf 0.10`** | tiled `conf 0.10` |
|---|---|---|---|---|---|---|---|
| `4_png` | 513×436 | 7 | 0.000 | **0.000** (every geometry) | 0.429 | **0.857** | 0.857 |
| `155_png` | 818×647 | 7 | 0.000 | 0.143 | 0.571 | **0.714** | 0.857 |
| `4061_png` | 1170×742 | 11 | 0.000 | 0.273 | 0.455 | **0.909** | 1.000 |
| `65_png` | 4000×1685 | 19 | 0.000 | 0.105 | 0.053 | **0.053** | **0.263** |
| `22_png` | 482×507 | 17 | 0.059 | 0.059 (identity) | 0.059 | **0.176** | 0.176 |

**Three of the four zero-recall plans are essentially solved by lowering the gate alone.** `4_png`
goes 0.000 → 0.857 (6 of 7 doors) — a plan that was immovable across *seven* tiling configurations —
with 118 proposals from a single untiled pass. "FastSAM does not consider a CAD door symbol an
object" was a conclusion drawn from a knob that was never turned.

Two plans resist and they resist differently, which matters for step 3:

- **`65_png`** (4000×1685, the resolution outlier) is the one plan where **tiling is the lever that
  works**: 0.053 untiled vs **0.263** tiled at conf 0.10, a 5× gain, because at 4000 px wide the
  1024 letterbox genuinely is destroying the symbol. Tiling's real niche on this dataset is a single
  extreme-resolution plan — which is exactly what B0 said when it called failure mode (2) "a
  single-plan anecdote".
- **`22_png`** (482×507, 17 doors) is immovable under *both* levers (0.059 → 0.176 at conf 0.10, with
  274 proposals for 17 doors). Sub-tile-sized, extremely crowded, and neither budget nor
  magnification touches it. This is the residual case that still argues for a different proposal
  mechanism.

### Finding 5 — do the crowded and sparse buckets prefer different conf values? No

The plan asks this explicitly. They do not: recall is **monotone increasing as conf falls in every
bucket**, with no crossover, on both arms. The sparse (1–3) bucket saturates at 1.000 by conf 0.30
and stops changing; the crowded (11+) bucket is still climbing steeply at 0.10 (0.277 → 0.433 →
0.639). So the gate is not a bucket-dependent trade at the **proposal** stage — lower is
monotonically better there for every bucket.

That is precisely why this entry cannot select a finalist. Proposal recall is a **ceiling**, and it
is monotone in a knob whose whole cost is false positives that the *retrieval* stage must then
reject. The selection has to be made on end-to-end F1, where precision finally pushes back — which
is T3.

### Latency (EVAL-11)

Untiled conf 0.10 costs **36.1 s/plan** of proposal stage vs 42.6 s at conf 0.50 and 48.9 s at conf
0.20 — i.e. **flat**, because a single FastSAM forward pass costs the same regardless of the
threshold applied to its output; only the cheap post-filter changes. (These are contended,
4-core-pinned numbers; B0's clean single-process figure is 1.44 s/plan.) **Opening the gate is
free at the proposal stage.** The real cost lands downstream: every surviving proposal gets its own
DINOv2 forward pass, so 161.2 vs 46.5 proposals is a ~3.5× multiplier on the *dominant* embedding
stage (B3's finding). Tiling by contrast pays on **both** stages — 5× the FastSAM passes *and* 3.2×
the embeddings.

---

## T3 — the end-to-end selection grid: `proposal_conf` × `similarity_floor`, UNTILED, on VAL

- **SHA:** `79fd33e` (recorded by the harness in all ten artifacts)
- **Commands:** ten `nohup`-detached, 4-core-pinned processes —
  `… trial --dataset floorplans-door --split val --name t3-untiled-c0CC-f0FF-val
  --config '{"proposal_conf": C, "similarity_floor": F}'`
- **Artifacts:** `runs/t3-untiled-c0{10,20,30}-f0{70,75,80,85}-val.json` (10 files)
- **Scope:** floorplans-door **VAL** (56 plans), **56/56 scored, 0 errors** in every trial, 1
  exemplar, `nms_iou` 0.3, `proposal_tiling: false` (verified in each artifact's dumped config).

T2 closed with the reason this grid exists: proposal recall is a **ceiling** and is monotone in
`proposal_conf`, so the gate cannot be selected at the proposal stage — only end-to-end F1, where
precision finally pushes back, can pick it. T2 also established that tiling is the *more expensive*
way to buy the same proposals, so this grid runs **untiled**.

`similarity_floor` is swept jointly rather than held at its default because B1-final's inertness
finding was measured on the **untiled, `conf 0.4`** proposal distribution. Opening the gate changes
that distribution (46.5 → 161.2 proposals/plan), and a floor's job is to reject the moderate-cosine
false positives that extra proposals bring — so its optimum genuinely might move. Sweeping it is how
that is established rather than assumed.

| `proposal_conf` | `similarity_floor` | P | R | **F1** | abstentions | val wall clock |
|---|---|---|---|---|---|---|
| **0.10** | **0.70** | 0.5258 | **0.5598** | **0.5423** ← val argmax | 0 | 289.4 min |
| 0.10 | 0.75 | 0.6407 | 0.4061 | 0.4971 | 1 | 411.2 min |
| 0.10 | 0.80 | 0.8093 | 0.2979 | 0.4355 | 6 | 410.7 min |
| 0.10 | 0.85 | 0.9259 | 0.1423 | 0.2467 | 20 | 412.0 min |
| 0.20 | 0.70 | 0.5501 | 0.4478 | 0.4937 | 1 | 276.0 min |
| 0.20 | 0.75 | 0.6565 | 0.3264 | 0.4360 | 2 | 278.4 min |
| 0.20 | 0.80 | 0.8188 | 0.2315 | 0.3609 | 8 | 242.5 min |
| 0.30 | 0.70 | 0.5752 | 0.3700 | 0.4503 | 2 | 196.2 min |
| 0.30 | 0.75 | 0.6948 | 0.2808 | 0.4000 | 5 | 243.9 min |
| 0.30 | 0.80 | 0.8387 | 0.1973 | 0.3195 | 11 | 238.2 min |
| *B1-final baseline (`conf 0.4`, floor 0.70)* | — | *0.591* | *0.307* | *0.404* | — | *~54 min* |

Wall clocks are contended and 4-core-pinned across ten concurrent processes; they are not clean
single-process measurements and are recorded for scheduling only.

### Finding 1 — the surface is monotone on both axes, so the argmax is not a knife-edge

Two clean monotonicities, each holding at **every** level of the other axis:

- **F1 falls as `similarity_floor` rises**, at all three conf values (0.542 → 0.497 → 0.436 → 0.247;
  0.494 → 0.436 → 0.361; 0.450 → 0.400 → 0.320). Precision climbs to 0.93 and recall collapses to
  0.14; the floor is buying precision the pooled metric does not want at this operating point.
- **`conf 0.10` beats 0.20 beats 0.30**, at every floor (0.542 > 0.494 > 0.450 at floor 0.70;
  0.497 > 0.436 > 0.400 at 0.75; 0.436 > 0.361 > 0.320 at 0.80).

The argmax therefore sits in a **corner** of the swept region, not on a ridge between competing
trials — the neighbouring cells are 0.494 (conf 0.20) and 0.497 (floor 0.75), both clearly below.
This is the opposite of the `owlv2` doors experience recorded in
`docs/reports/owlv2-floorplans-improvement.md`, where a val-argmax over 56 plans was unstable across
adjacent grid values and failed to generalise. A monotone surface is the well-behaved case.

A corner argmax does raise the obvious question — is the true optimum outside the grid, at
`conf < 0.10` or `floor < 0.70`? Recorded honestly as **not measured**: `similarity_floor` below 0.70
was already swept at `conf 0.4` in B1-final and lost badly there (floor 0.60 → val F1 0.326, floor
0.40 → 0.226), and `conf < 0.10` was not run. The selected point may be a boundary rather than an
interior optimum; the shipped grid entry is the best **measured** configuration, not a claimed global
one.

### Finding 2 — the finalist, and why `similarity_floor` stays at its shipped default

**Finalist: `proposal_conf = 0.10`, `similarity_floor = 0.70`.**

`0.70` is the **existing shipped `ProposeRetrieveConfig` default**. The floor sweep re-confirms, on
the *new* proposal distribution, what B1-final measured on the old one: the shipped floor is already
this domain's optimum. So exactly **one** field moves from its default, and it is a field that
already exists and is already documented — this ships with zero new config fields.

### Finding 3 — the bottleneck has MOVED from the proposal stage to retrieval/calibration

This is the finding the next improvement pass should start from. At the finalist config the
proposal stage is no longer the constraint, and the retrieval/calibration stage now is:

| crowding | proposal-stage recall (T2, untiled `conf 0.10`) | end-to-end recall (T3 finalist) |
|---|---|---|
| sparse | `1–3`: **1.000** | `2-5`: 0.8519 |
| middle | `4–10`: **0.8863** | `6-15`: 0.6268 |
| **crowded** | `11+`: **0.6395** | `16+`: **0.2615** |
| all | mean 0.8212 / pooled 0.7514 | 0.5598 |

**The two bucketings use different cuts** (`1–3`/`4–10`/`11+` at the proposal stage vs the
benchmark's `2-5`/`6-15`/`16+`), so the rows are not exactly matched — stated so the comparison is
not over-read. The gap is far too large to be a cut artifact: in the crowded bucket the proposal
stage puts **0.639** of the doors on the table and the pipeline returns **0.262** — a ~41 % transfer,
against the **~0.82** proposal-to-final transfer B0 measured overall and T1c re-confirmed. Pooled,
transfer is 0.5598 / 0.7514 = **0.745**, already below B0's 0.82 and dragged down almost entirely by
the crowded bucket.

B1-final's conclusion — *"the retrieval stage is not leaving recall on the table; there is nothing
left for it to retrieve"* — was true **at `conf 0.4`**, where final recall 0.399 sat essentially at
the 0.405 proposal ceiling. It is no longer true. Opening the gate raised the ceiling to 0.751 pooled
and the pipeline now delivers 0.560 of it. **DINOv2 embedding + gmm calibration is the new binding
constraint on crowded plans**, and it is where the next floor-plan lever should be aimed — not at the
proposal stage, which is why Task 4's contour backend is not attempted (see the go/no-go entry
below).

---

## T3-final — the ONE test read for the finalist

- **SHA:** `79fd33e`
- **Command:** `… trial --dataset floorplans-door --split test --name t3-final-c010-f070-test
  --config '{"proposal_conf": 0.1, "similarity_floor": 0.7}'`
- **Artifact:** `runs/t3-final-c010-f070-test.json`
- **Scope:** floorplans-door **TEST**, **28/28 scored, 0 errors, 0 abstentions**, 1 exemplar,
  `proposal_tiling: false`, `nms_iou` 0.3. Wall clock **2 137.9 s = 35.6 min** on 8 pinned cores.

The test split is read **exactly once** for this finalist, per the plan's rule. Nothing below was
used to choose anything.

| | P | R | **F1** | abstentions | coverage |
|---|---|---|---|---|---|
| committed `docs/eval/floorplans-findings.md` (GPU-era) | 0.55 | 0.39 | **0.459** | — | 13/14 |
| **B1-final session-local baseline** (`conf 0.4`) | 0.6039 | 0.3991 | **0.4806** | 3 | 28/28 |
| **T3-final finalist** (`conf 0.10`, floor 0.70) | 0.5358 | **0.6738** | **0.5970** | **0** | 28/28 |
| **delta vs B1-final** | **−0.0681** | **+0.2747** | **+0.1163** | −3 | — |

**+0.116 F1 absolute (+24 % relative) over the session-local baseline**, and **+0.138 (+30 %)** over
the committed 0.459 row. Both baselines are stated; neither is silently replaced.

The win is a **recall** win: recall +0.275 (0.399 → 0.674) against a precision cost of −0.068
(0.604 → 0.536). Reported separately per the plan's rule, because a pooled F1 alone would hide that
this is a trade. It is a favourable one — recall gains four times what precision gives up — but it is
a trade, and a workflow that cares more about precision than recall should not take it.

**Abstentions fall 3 → 0.** Three test plans previously returned nothing at all; every plan now
returns something. That is a product-visible change the F1 delta does not express.

### Recall by symbol size — every bucket improves, and the profile flattens

| symbol size | n GT | B1-final baseline | **T3-final** | delta |
|---|---|---|---|---|
| small | 84 | 0.393 | **0.6310** | **+0.238** |
| medium | 135 | 0.415 | **0.7111** | **+0.296** |
| large | 14 | 0.286 | **0.5714** | **+0.286** |

All three buckets improve and the **large** bucket doubles — there is no bucket that pays for the
others. The baseline profile was flat-and-low (0.39/0.42/0.29); the new one is higher everywhere and
peaks in the medium bucket, where 58 % of test doors live. Note `large` is **n = 14 boxes** on this
split, so its +0.286 is four extra matches and should not be over-read.

### By crowding (benchmark cuts) — the crowded bucket is still the weak one

| slice | n plans | P | R | F1 |
|---|---|---|---|---|
| `2-5` | 8 | 0.4615 | 0.6923 | 0.5538 |
| `6-15` | 18 | 0.5467 | 0.7069 | **0.6165** |
| `16+` | **2** | 0.5517 | 0.4848 | 0.5161 |

The `16+` test slice is **two plans** — directional only. It is nonetheless a large move on the
matching val slice (T3 val `16+`: F1 0.2822 at the finalist), and T1c's tiled runs left the same
slice at F1 0.154–0.182.

### Latency (EVAL-11) — the honest cost, and what is NOT comparable

| | mean/plan | p50/plan | worst plan | total wall clock |
|---|---|---|---|---|
| B1-final baseline (`conf 0.4`, 28 test plans) | 91.5 s | 23.6 s | 1 737.5 s | — (12 h 40 m incl. 12 val trials) |
| **T3-final (`conf 0.10`)** | **76.3 s** | **13.2 s** | **1 558.5 s** | **35.6 min** |
| T1c tiled @ ios 1.00 (val, for contrast) | 224.5 s | 30.1 s | 2 754 s | 209.9 min |

**Do not read the finalist as faster than the baseline.** Both are contended multi-process numbers
taken under different loads, and a lower gate puts **3.5× more proposals** (46.5 → 161.2/plan)
through the DINOv2 embedding stage, which B3 measured as the dominant cost. The structurally honest
statement is the one the mechanism supports:

> **`proposal_conf` is free at the FastSAM stage and pays only at the embedding stage** — one forward
> pass costs the same regardless of the threshold applied to its output (T2 measured proposal-stage
> time flat at 36–49 s/plan across `conf` 0.10–0.50), so the cost is ~3.5× on embeddings alone.
> **Tiling pays on both stages** — ~5× the FastSAM passes *and* ~3.2× the embeddings — which is why
> T1c measured it at 3.9× baseline wall clock for +0.022 F1.

---

## T1d-window-final — the guardrail test read, relaunched and clean

- **SHA:** `79fd33e`
- **Command:** `… trial --dataset floorplans-window --split test --name t1d-window-test
  --config '{}'` (DEFAULT config)
- **Artifact:** `runs/t1d-window-test.json`
- **Scope:** floorplans-window **TEST**, 28/28 scored, 0 errors, 9 abstentions.

This is the re-launch of the process that died silently mid-run during T1d (recorded there rather
than smoothed over — a guardrail that "would have passed" is not a guardrail).

| | P | R | F1 | abstentions |
|---|---|---|---|---|
| B1-final window TEST (baseline) | 0.1194 | 0.1026 | **0.1103** | 9 |
| **T1d window TEST (this run)** | 0.1194 | 0.1026 | **0.1103** | 9 |
| delta | **+0.0000** | **+0.0000** | **+0.0000** | 0 |

**Identical to four decimal places**, including the abstention count and the per-size recall profile
(small 0.0619, medium 0.1852, large 0.0000). Combined with T1d's six regimes (also +0.0000 on every
one), the guardrail is complete and clean: **nothing from this quick task leaked into the default
path.**

It is identity **by construction** and the artifact's dumped config says why — `proposal_tiling:
false`, `proposal_conf: 0.4`, `similarity_floor: 0.7`, every tiling field at its no-op default. The
finalist ships as an additive **tuning-grid** entry, not as a changed default, so this property
survives the shipped change: `floorplans-window` and the four chipset/textured/synthetic regimes are
untouched unless a domain-tuning run deliberately selects the new grid cell for them.

---

## STEP-3 GO/NO-GO VERDICT — Task 4 (contour/blob backend) NOT attempted

Recorded per the plan's requirement that the criterion be evaluated on VAL and the verdict written
into this notebook **before** Task 4 begins. The plan's criterion: attempt step 3 only if **(a)**
crowded-bucket proposal recall is still < 0.50 after steps 1+2, **or** **(b)** the combined val F1
gain is < +0.05 absolute.

| criterion | threshold | measured | fired? |
|---|---|---|---|
| **(a)** crowded-bucket recall < 0.50 | < 0.50 | end-to-end `16+` val recall **0.131** at the mid-sweep operating point (`conf 0.10`, floor 0.75); **0.2615** at the final finalist (`conf 0.10`, floor 0.70) | **YES** |
| **(b)** val F1 gain < +0.05 | < +0.05 | **+0.093** at the mid-sweep point; **+0.138** at the finalist (0.404 → 0.542) | **NO** |

Criterion (a) fired. Criterion (b) did not, and is not close — the finalist's val gain is nearly
**3×** the threshold that would have triggered step 3 on those grounds.

**Verdict: Task 4 is deliberately SKIPPED, on evidence, despite criterion (a) firing.**

The criterion is a trigger, not an instruction to build. The plan's own text conditions it on the
lever still being the right one (*"AND the step-1/2 latency cost has not already become the binding
constraint … if [that is so], adding a second backend on top is not the next lever — record that and
stop"*), and the plan's overriding instruction is to let the numbers decide. They decide against it,
for one specific reason:

> **A contour backend supplies MORE PROPOSALS, and the proposal stage is no longer the binding
> stage.** At the finalist operating point the proposal stage delivers crowded-bucket recall
> **0.639** (T2, untiled `conf 0.10`) while the pipeline end-to-end returns only **0.2615** (T3).
> The doors are already being proposed; they are being lost in retrieval/calibration. Adding a
> second source of proposals attacks a stage with ~0.38 of headroom already sitting unconsumed.

Three further pieces of evidence, all measured in this notebook, point the same way:

1. **The premise for a contour backend was refuted.** T1e Finding 4 argued from `4_png` (proposal
   recall 0.000 across seven tiling configurations) that *"FastSAM does not consider a CAD door
   symbol an object"* — the single strongest argument for a non-FastSAM proposer. **T2 Finding 4
   refuted it**: `4_png` reaches **0.857** on the conf gate alone (6 of 7 doors, 118 proposals, one
   untiled pass). Those symbols were never invisible to FastSAM; they were scoring below a 0.4 gate
   that had never been turned. Three of the four zero-recall plans are essentially solved this way.
2. **What genuinely resists is not a proposal-supply problem either.** Two plans still resist, and
   differently: `65_png` (4000×1685) responds to **tiling**, not to a new backend, and `22_png`
   (482×507, 17 doors) is immovable under both levers at **274 proposals for 17 doors** — a plan
   drowning in proposals is not short of them.
3. **Cost.** Steps 1+2 already multiply the dominant embedding stage ~3.5×. Layering a second
   proposal source on top compounds exactly the cost the plan named as a stop condition.

**This is a measured, evidence-based skip under the plan's "let the numbers decide" instruction, not
an omission.** The plan states the outcome must be a recorded measurement in every branch, and it is:
the criterion, the numbers on both sides of it, and the mechanism that overrides it are all above.

The lead it replaces is more valuable than the backend would have been, and is carried into the
report's *Measured and deferred* section: **retrieval/calibration on crowded plans is the next
lever** (T3 Finding 3) — the DINOv2 region embedding and the gmm cut, not the proposer.

---

## CORRECTION — artifact custody: `runs/` is GITIGNORED, not committed

The "RUNTIME NOTE" entry above states that the pre-`T1f` artifacts survived the loss of contract
`47510440` "because the JSON artifacts had been pulled back to `runs/` **and committed**". The second
half of that is **factually wrong** and is corrected here rather than edited above, per the
append-only rule.

`.gitignore:95` carries the rule `.planning/quick/*/runs/`, and `git ls-files` over that directory
returns **zero tracked files** — none of this quick task's 55 run artifacts is, or ever was, in git.
They survived because they were rsynced into this worktree's working tree, which is not the same
thing.

This is the repo's deliberate convention, not a defect: raw per-run JSON is environment-dependent
and regenerable, and is ignored for the same reason `docs/benchmark/`'s tuning dumps are. The
committed, reviewable record is **this notebook** plus
`scripts/propose_retrieve_floorplans_experiment.py`, which is what "every number traces to committed
code" means in this task — the *code and the command* are committed and re-runnable, not the output
bytes. The artifacts were **not** force-added past the ignore rule to make the earlier sentence true.

Practical consequence, stated plainly so a future reader is not misled: **the artifacts are local to
this worktree and are not recoverable from git.** Every number in this notebook and in
`docs/reports/propose-retrieve-floorplans-improvement.md` is reproducible by re-running the recorded
command against the committed split, but the specific JSON files are not archived.

---
