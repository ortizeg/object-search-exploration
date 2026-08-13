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
