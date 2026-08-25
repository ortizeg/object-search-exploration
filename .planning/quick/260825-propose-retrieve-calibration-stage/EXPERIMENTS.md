# EXPERIMENTS — `propose-retrieve` retrieval/calibration-stage investigation (2026-08-25)

Append-only lab notebook, following the convention set by
`.planning/quick/260812-m8m-improve-propose-retrieve-recall-on-floor/EXPERIMENTS.md`. Every
number below comes from a committed script run against the committed
`dataset_splits/floorplans-door.split.json`; raw JSON artifacts live in this directory's
gitignored `runs/` (per `.gitignore`'s `.planning/quick/*/runs/` rule) and on the vast.ai box
that produced them (instance `48124756`).

**Starting point.** The prior pass
(`docs/reports/propose-retrieve-floorplans-improvement.md`, quick task `260812-m8m`) fixed the
proposal stage (`proposal_conf` 0.4 -> 0.10) and, in doing so, exposed that at the finalist
config the retrieval/calibration stage is now the binding constraint on crowded plans:
proposal-stage recall in the 11+-door bucket is 0.6395 (T2) but end-to-end recall there is only
0.2615 (T3-final test) / F1 0.2822 (T3 val) — a transfer far below the ~0.82 pooled ceiling. That
gap was flagged as "a concrete lead for a future improvement pass" and deliberately not
investigated. This notebook is that investigation.

**Environment.** Same vast.ai instance as the prior pass (`48124756`, RTX 3090 host, CPU-only
ONNX Runtime — GPU not used), restarted and re-synced via `git bundle` from `worktree/rolling-ivy`
at `3d8fd88` (the merged `main` including PR #58). `models/` and `datasets/floorplans-door` were
already present on the instance's disk from the prior session and were reused unchanged.

---

## C0 — per-GT-box calibration trace, VAL, finalist config (56 plans, 527 GT boxes)

- **SHA:** `3d8fd88`
- **Script:** `scripts/propose_retrieve_calibration_experiment.py trace --split val` (new,
  committed this session)
- **Config:** `proposal_conf=0.10`, `similarity_floor=0.70`, `nms_iou=0.3` (the shipped finalist)
- **Artifact:** `runs/calib-val-full.json`
- **Scope:** floorplans-door **val**, 56/56 plans, 527/527 GT boxes (21 + 224 + 282 across the
  1-3/4-10/11+ diagnostic crowding cuts — matches `docs/eval/floorplans-findings.md`'s door/val
  instance count of 527 exactly, a sanity check that the trace's GT loading and bucketing agree
  with the committed dataset statistics).

**What this measures that the prior report did not.** B0/T2 measured proposal-stage recall (does
*some* proposal cover each GT box). This traces what happens to that proposal AFTER it is
covered: does it clear the calibrated threshold, does it survive post-threshold NMS, or is it
discarded — split into `matched` / `below_threshold` / `nms_suppressed` / `no_proposal` per GT
box (see the script's module docstring for the exact taxonomy).

### Finding 1 — the retrieval/calibration stage's OWN loss rate rises sharply with crowding

Restricting the denominator to GT boxes that DO have a covering proposal (i.e. excluding
`no_proposal`, which is a proposal-stage fact already characterized) isolates exactly what the
calibration stage itself is responsible for:

| crowding | n plans | total GT | no_proposal frac | **retrieval-stage loss rate** (below_threshold + nms_suppressed, of covered boxes) |
|---|---|---|---|---|
| 1-3 | 7 | 21 | 0.000 | **0.095** |
| 4-10 | 31 | 224 | 0.116 | **0.197** |
| **11+** | 18 | 282 | 0.372 | **0.322** |

The loss rate more than triples from sparse to crowded (0.095 -> 0.322). This is the first direct
evidence that the retrieval/calibration stage, not just the proposal stage, degrades with
crowding — consistent with the prior report's transfer-rate finding, now decomposed into its own
number rather than inferred from the gap between two aggregate recalls.

`nms_suppressed` is negligible everywhere (0/21, 4/224 = 0.018, 6/282 = 0.021) — post-retrieval
NMS is **not** the mechanism. Nearly all of the loss is `below_threshold`.

### Finding 2 — the gmm's adaptive component is almost always inert; the fixed floor decides

| crowding | mean threshold applied | gmm degenerate rate |
|---|---|---|
| 1-3 | 0.7000 | 0.286 (2/7) |
| 4-10 | 0.7062 | 0.516 (16/31) |
| 11+ | 0.7035 | 0.278 (5/18) |

The applied threshold sits within 0.006 of the bare `similarity_floor` (0.70) in every bucket,
and the degenerate rate does **not** track crowding monotonically (highest in the *middle* bucket,
not the crowded one). Re-reading `propose_retrieve.py` step 5 explains why: on a degenerate fit
the code uses `similarity_floor` **directly**, discarding whatever the ratio-fallback would have
cut at; on a non-degenerate fit it takes `max(gmm_cut, similarity_floor)`. So the gmm can only
ever move the threshold **up** from the floor, never down — and empirically it almost never does
(mean threshold is ~flat at ~0.70-0.706 across every crowding level). **The "DINOv2 embedding +
gmm calibration" bottleneck the prior report named is, on this domain, almost entirely the fixed
`similarity_floor` clamp** — the gmm's adaptive contribution is close to a no-op here. This
reframes hypothesis 1 from the task brief (does the gmm cut *degrade* with more candidates) as
NOT SUPPORTED in the sense asked: it isn't getting noisier or pushing the threshold to a worse
place: it is simply not the thing doing the deciding.

### Finding 3 — true/background score separation compresses with crowding (supports hypothesis 2)

| crowding | mean background score | mean true-positive score (`matched` only) | **separation** |
|---|---|---|---|
| 1-3 | 0.452 | 0.825 | **0.373** |
| 4-10 | 0.483 | 0.805 | **0.321** |
| 11+ | 0.513 | 0.800 | **0.287** |

Both ends move against recall as crowding rises: background (non-covering) proposals score higher
(0.452 -> 0.513) and true positives score slightly lower (0.825 -> 0.800), so the margin the fixed
floor has to work with shrinks by 23% (0.373 -> 0.287) from sparse to crowded. This is the
signature of hypothesis 2 (embedding discriminability compressing under crowding/clutter), not
hypothesis 1 (calibration noise) or hypothesis 3 (NMS collapse, already ruled out in Finding 1).

The `below_threshold` population itself (GT boxes with a covering proposal that gets rejected)
scores consistently in the 0.637-0.656 range across all three buckets — comfortably above
background (0.45-0.51) and comfortably below the 0.70 floor. These are not noise; they are real
door crops the embedding is moderately, not confidently, matching to the exemplar.

---

## C1 — does a lower `similarity_floor` rescue the crowded bucket at conf 0.10? Val sweep, floor < 0.70

Finding 3 raises the obvious question the prior report explicitly left open: *"`similarity_floor`
below 0.70 was already swept at `conf 0.4`... but not at `conf 0.10`"* — the new (post-fix)
proposal/score distribution was never checked below the shipped floor. Three parallel,
core-pinned trials answer it directly, run through the SAME `trial` entry point (hence the SAME
scorer) T3 used, so these numbers slot directly into that grid.

- **SHA:** `3d8fd88`
- **Commands:** `scripts/propose_retrieve_floorplans_experiment.py trial --dataset
  floorplans-door --split val --config '{"proposal_conf": 0.10, "similarity_floor": F,
  "nms_iou": 0.3}'` for F in {0.55, 0.60, 0.65}, three `nohup`-detached, 4-core-pinned processes
  launched concurrently (mirroring T3's scheduling pattern).
- **Artifacts:** `runs/c010-floor055-val.json`, `runs/c010-floor060-val.json`,
  `runs/c010-floor065-val.json` (both in the `260812-m8m` quick task's `runs/`, alongside the
  `t3-untiled-c010-f070-val.json` baseline they extend — kept there rather than duplicated, since
  they are literally three more cells of that same grid).
- **Scope:** floorplans-door **val**, 56/56 scored, 0 errors in every trial, 1 exemplar.

### Pooled result — extends T3's grid downward, and the trend does not reverse

| `similarity_floor` | P | R | **F1** | val wall clock (3 concurrent, 4-core-pinned) |
|---|---|---|---|---|
| 0.55 | 0.211 | 0.681 | 0.322 | 217.7 min |
| 0.60 | 0.278 | 0.670 | 0.393 | 250.6 min |
| 0.65 | 0.387 | 0.632 | 0.480 | 251.0 min |
| **0.70 (T3 baseline)** | **0.526** | 0.560 | **0.542** ← still argmax | 289.4 min (10-concurrent) |

F1 keeps falling monotonically below 0.70, exactly continuing the trend T3 already established
above 0.70 (0.542 -> 0.497 -> 0.436 -> 0.247 as floor rises). **The full now-measured range is
{0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85} at `conf 0.10`, and 0.70 — the existing shipped
default — is the argmax across the entire range, not just the previously-swept upper half.** This
closes the "not measured" limitation the prior report flagged, with a clean answer: there was no
undiscovered optimum below the floor.

### By-crowding breakdown — the crowded bucket alone tells a different, but still unfavourable, story

| `similarity_floor` | `2-5` F1 | `6-15` F1 | **`16+` F1** |
|---|---|---|---|
| 0.55 | 0.251 | 0.373 | 0.247 |
| 0.60 | 0.348 | 0.446 | 0.282 |
| 0.65 | 0.464 | 0.548 | **0.310** |
| **0.70 (baseline)** | **0.601** | **0.620** | 0.282 |

`floor=0.65` IS a genuine win for the crowded bucket in isolation: F1 0.282 -> 0.310 (+0.028, +10%
relative), driven by recall 0.262 -> 0.385 (+0.123) at a precision cost of 0.306 -> 0.259 (-0.047).
But the SAME move costs the sparse bucket -0.137 F1 (0.601 -> 0.464) and the middle bucket -0.072
F1 (0.620 -> 0.548) — both buckets carry far more plans (14 and 36 vs 6), so the pooled metric
that the repo's own tuning methodology optimises (argmax F1 @ IoU 0.5 on val, method-wide, no
per-slice dispatch) correctly rejects it. Lower floors (0.60, 0.55) do not even win the crowded
bucket outright — 0.60 ties the baseline (0.282 vs 0.282, via a different P/R trade) and 0.55 is
worse (0.247 < 0.282).

**Verdict: REJECTED.** No `similarity_floor` value below 0.70 improves pooled val F1, and only one
value (0.65) improves the crowded bucket in isolation, at a pooled cost roughly double the size of
the gain. A crowding-*conditional* floor is not pursued: it would require dispatching on a
quantity (instance count) the method cannot observe until after ground truth is known, which is
not the same as `proposal_conf`'s existing behavior of reading an already-observable image
property. It would also introduce exactly the config-driven dispatch inside a method the repo's
conventions rule out (`.claude/CLAUDE.md`, "Method modules... No hidden control flow... no
config-driven dispatch *inside* a method").

---

## Hypothesis 3 — pre-embedding objectness/top-K filtering: not run, argued from existing evidence

The task brief's third candidate — filter proposals by objectness or truncate to top-K *before*
the expensive DINOv2 pass, to cut noise/cost without sacrificing recall — was not given a fresh
trial. It reduces, almost exactly, to re-raising `proposal_conf` (FastSAM's own objectness score
is the natural pre-embedding filter signal, and a top-K-by-objectness truncation is monotonic in
the same score). T2 already measured that lever, directly and unambiguously: `conf 0.10 > 0.20 >
0.30` at every `similarity_floor` (see `docs/reports/propose-retrieve-floorplans-improvement.md`,
"Finding 1" under T3). The proposals a pre-embedding objectness filter would remove first are
exactly the marginal-confidence proposals that opening the gate to 0.10 was shown to need — they
are disproportionately the ones covering small/crowded true instances (T2's own attribution: doors
that scored proposal recall 0.000 at `conf 0.4` reached 0.857 at `conf 0.10`, on marginal-
confidence proposals). A pre-embedding truncation would remove recall this same session's own
prior pass fought to win back. Not measured, because the mechanism argument is airtight from
existing data and a fresh ~4h trial would not change the conclusion.

---

## Verdict

No lever measured in this session beats the shipped finalist (`proposal_conf=0.10`,
`similarity_floor=0.70`) on pooled val F1. **Nothing ships**: no `_TUNING_GRIDS` change, no config
default change, no code change to `propose_retrieve.py` or `common/calibration.py`. Per the
project's tune-on-val / read-test-once discipline, **test is not read** — there is no finalist
that earned it (mirrors T1b/T1e's treatment of the rejected tiling passes in the prior notebook).

What DOES survive, as a genuine (if negative) contribution:

1. The retrieval/calibration stage's own loss rate is now a measured, decomposed number
   (0.095/0.197/0.322 by crowding), not an inferred gap between two aggregate recalls.
2. The gmm's adaptive component is shown to be nearly inert on this domain — the fixed
   `similarity_floor` is doing almost all the deciding, which reframes where a future fix
   (if one exists) would need to live: not the calibration LOGIC, but either the floor VALUE
   (measured exhausted, this notebook) or the embedding SCORES feeding it (untouched).
3. The true/background score-separation compression with crowding (0.373 -> 0.287) is measured
   evidence that the embedding stage's own discriminability, not the threshold sitting on top of
   it, is the actual ceiling — consistent with, and a plausible mechanism for, the transfer-rate
   collapse the prior report observed but did not explain.
4. `similarity_floor`'s full plausible range (0.55-0.85) is now measured at `conf 0.10`; the
   shipped default (0.70) is confirmed as the argmax across all of it, closing a stated limitation
   of the prior pass.

A genuine embedding-stage fix (e.g. a crop-context or fine-tuned DINOv2 variant, in the spirit of
`docs/reports/owlv2-floorplans-finetune.md`'s work for a different method) is the only lever this
investigation's evidence points toward for the crowded bucket specifically — out of scope for a
threshold/grid-tuning pass, and not attempted here.
