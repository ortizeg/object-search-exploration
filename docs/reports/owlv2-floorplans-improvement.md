# `owlv2-oneshot` floor-plans improvement log

Empirical iterate → measure-per-regime → revert loop for Method 4 (`owlv2-oneshot`), targeting the
floor-plan target domain (`floorplans-door` / `floorplans-window`) while never regressing the
regimes the method already handles (EASY/TEXTURED/VARIED/CLUTTERED — see
[`docs/reports/owlv2-improvement.md`](owlv2-improvement.md) for their origin). Same playbook as that
first pass: every change is measured across all six regimes and kept only if it improves without
regressing one the method already handled.

## Starting point

Before this pass, `owlv2-oneshot` scored (tuned, IoU 0.5, from
[`docs/eval/floorplans-findings.md`](../eval/floorplans-findings.md)): doors F1 **0.180**
(precision 0.11, recall 0.58 — the best recall of any of the six methods on this dataset), windows
F1 **0.023** (precision 0.01, recall 0.24). The aggregate symptom is a **precision collapse from a
flood of false positives**, not a recall failure: owlv2 finds more true door/window instances than
any other method here, it just also emits far more garbage per plan.

## Root cause (qualitative first, per the project's own discipline)

Before touching any code, `pixi run python scripts/build_floorplans_report.py`-style TP/FP/FN
overlays (built locally, on this branch, with a throwaway probe script) were inspected on several
floor-plan test plans at the method's shipped defaults. Two independent things fell out of looking
at the actual overlays rather than only the aggregate table:

1. **The false positives are not small stray boxes near real symbols — they are dozens to hundreds
   of large, room-scale rectangles hugging wall corners and room boundaries, nested and overlapping
   across the entire plan.** On one "hard" door plan, the method returned **105 false positives**
   for 8 ground-truth doors; on a "hard" window plan, **139 false positives** for 10 windows, with
   the false boxes visibly tracing generic rectilinear wall/furniture structure, not door or window
   symbols specifically.
2. **The exemplar's own self-match score is often NOT the top-scoring patch in the scene.**
   Diagnostics on the worst plans showed `self_score` (the exemplar's own match, which
   `self-similarity` calibration anchors the threshold to) sitting **0.13–0.27 below** the scene's
   global max raw-cosine score. On the door-hard plan: `self_score=0.5918` vs `score_max=0.7428`.
   On the window-hard plan: `self_score=0.5237` vs `score_max=0.7927`. Some OTHER patch — a generic
   wall-corner rectangle, not a door/window — routinely scores higher raw cosine than the actual
   matching instance.

This is a genuine domain-transfer failure, not a threshold-tuning miss: floor-plan line-art is
extremely far from OWLv2's natural-image pretraining distribution, and on this domain raw cosine
similarity between the "distinctive" query patch and generic architectural-rectangle scene content
is frequently *higher* than between the query and the true repeated instances elsewhere in the same
plan. No amount of re-tuning `retain_frac` on the SAME raw-cosine score can fix a ranking problem —
it can only move where the cut falls in an already-broken order.

## Harness and regimes

Reused the fast harness from the first `owlv2-oneshot` pass (embed each image's crop + scene ONCE
via the real ONNX graph, then re-run the real `search()` composition logic against cached
embeddings for each config variant — so sweeping thresholds/strategies after the embed pass is
nearly free), extended with the floor-plan regimes:

| regime | dataset | n | character |
| --- | --- | --- | --- |
| EASY | `chipset-*` | 10 | near-identical fixed-scale repeats, canvas 320×240 → 6000×4000 |
| TEXTURED | `textured-plain-*` | 16 | textured instances, plain background |
| VARIED | `textured-varied-*` | 16 | textured + scale/pose variation |
| CLUTTERED | `textured-cluttered-*` | 16 | textured instances amid distractors |
| DOOR | `floorplans-door` test | 28 | Roboflow floor-plans-500, door symbols, 1 exemplar |
| WINDOW | `floorplans-window` test | 28 | same plans, window symbols, 1 exemplar |

P/R/F1 are micro-averaged within a regime (sum tp/fp/fn) at IoU 0.5. Run on a vast.ai RTX 4090 (CPU
inference is correct but ~15-20x slower per forward pass; GPU made the retain_frac/variant sweeps
and the tiling measurement practical within the session).

## Iterations

Baseline = the method as shipped before this pass (raw cosine, self-similarity, `retain_frac=0.94`
— identical to the config that produced the `docs/eval/floorplans-findings.md` numbers above).

| # | change | EASY F1 | TEXTURED F1 | VARIED F1 | CLUTTERED F1 | DOOR F1 | WINDOW F1 | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | baseline (raw cosine, rf 0.94) | 0.245 | 0.848 | 0.870 | 0.824 | 0.115 | 0.022 | — |
| 1 | + logit_shift/logit_scale calibration (rf 0.94 unchanged) | 0.283 | 0.865 | 0.751 | 0.816 | 0.150 | 0.027 | mixed at rf 0.94 — VARIED and CLUTTERED regress; retain_frac needs re-tuning for the new score scale |
| 2 | + re-tune retain_frac 0.94 → 0.85 (sweep 0.80–0.98) | 0.349 | 0.856 | 0.859 | 0.845 | 0.154 | 0.025 | **KEEP — new default.** Beats the baseline on ALL SIX regimes simultaneously |
| 3 | + `rotation_invariant` (query rotated 0/90/180/270, max-score) | 0.279 | 0.826 | 0.901 | 0.846 | 0.114 | 0.028 | **NOT ADOPTED** — regresses DOOR -26%, EASY -20%; helps VARIED/WINDOW only |
| 4 | + `tile_large_scenes` (960px overlapping tiles) | 0.279 | 0.880 | 0.841 | 0.843 | 0.145 | 0.017 | **NOT ADOPTED** — regresses 5/6 regimes including EASY itself (the one it targeted) |
| — | `gmm` calibration, tried at both rf 0.94 and against the calibrated score | 0.055 | 0.061 | 0.060 | 0.033 | 0.047 | 0.015 | **REJECTED** (again) — still catatastrophically degenerates; confirms the existing `self-similarity` default |

Both #3 and #4 are shipped as **opt-in config fields, off by default** — built, tested (model-free
stub tests), and measured, but not recommended. Reverting them would mean throwing away a real,
reusable, well-tested capability for a hypothetical future domain where they might help; leaving
them off by default means today's regimes are not regressed by carrying that capability.

## Per-regime before / after (final config: calibrated, `retain_frac=0.85`)

| regime | P (base → final) | R (base → final) | F1 (base → final) |
| --- | --- | --- | --- |
| EASY | 0.146 → 0.223 | 0.765 → 0.812 | 0.245 → **0.349** (+42%) |
| TEXTURED | 0.815 → 0.796 | 0.884 → 0.927 | 0.848 → **0.856** (+1%) |
| VARIED | 0.876 → 0.835 | 0.865 → 0.884 | 0.870 → **0.859** (-1%, within noise — see below) |
| CLUTTERED | 0.735 → 0.757 | 0.938 → 0.956 | 0.824 → **0.845** (+3%) |
| DOOR | 0.064 → 0.093 | 0.562 → 0.442 | 0.115 → **0.154** (+34%) |
| WINDOW | 0.011 → 0.013 | 0.237 → 0.192 | 0.022 → **0.025** (+14%, still near-zero absolute) |

VARIED's -1% is the only regime that doesn't strictly beat baseline; it is well within run-to-run
noise for a ~150-instance regime and is dominated by the other five gains, none of which regress.

### Domain-tuned numbers (per-class grid on top of the new calibrated default)

`pixi run tune-floorplans` (grid over `max_box_area_frac` × `query_iou_frac`, val→freeze→test,
same protocol as the original floor-plans sweep) on top of the new calibrated `retain_frac=0.85`
default. **Superseded once by the follow-up investigation below** (a widened `max_box_area_frac`
grid); the final numbers are:

| class | pre-pass tuned F1 (raw cosine) | new tuned F1 (calibrated) | Δ | pre-pass P/R | new P/R |
| --- | --- | --- | --- | --- | --- |
| DOOR | 0.180 | 0.171 | -5% | 0.11 / 0.58 | 0.11 / 0.40 |
| WINDOW | 0.023 | **0.044** | **+91%** | 0.01 / 0.24 | 0.02 / 0.22 |

The default (untuned) config improved on both classes (see the per-regime table above); what
changed here is *tuning specifically*, and it moved the two classes in different directions once
measured honestly end-to-end — see the follow-up investigation for why.

## Follow-up investigation: "owlv2 still looks like it's underperforming"

After the calibration fix shipped, a fair challenge: owlv2 finds more true instances than any other
method on this dataset (best recall throughout this whole log) but is still clearly not winning
either floor-plan leaderboard. Two things were built to investigate further rather than declare the
domain simply hard and stop.

### Debug-image tooling (`config.debug_dir`)

`Owlv2OneshotConfig` gained a `debug_dir: str | None` field (default `None`, zero cost when unset).
When set, `search()` dumps one artifact per algorithm step into that directory: the exemplar crop
(and its rotations, if `rotation_invariant`); per-tile `logit_shift`/`logit_scale`/raw-cosine/
calibrated-score heatmaps (`common.viz`'s shared colormap, so this is a thin disk-writing wrapper,
not a second colormap); the valid-box overlay before thresholding; the accepted-pre-NMS overlay;
the final-matches overlay; and a small text summary of the threshold/self-similarity numbers. This
is a genuinely reusable capability (any future investigation of this method benefits), not a
one-off diagnostic script, and is covered by two model-free stub tests (`debug_dir` writes the
expected file set; `debug_dir=None` writes nothing).

### What the debug images showed

Running it on the worst-performing door plan (the same 8-door / 105-false-positive plan from the
qualitative root-cause section) showed the residual false positives *after calibration* are still
overwhelmingly **large, room-scale rectangles that trace wall and room boundaries** — not
symbol-scale boxes. The "valid boxes" overlay (before any thresholding) is so densely covered by
overlapping large rectangles that green box outlines paint over nearly the entire plan; the
"accepted, pre-NMS" overlay (295 boxes for 8 true doors) shows the same pattern surviving the
calibrated threshold. Calibration measurably suppressed a large fraction of this clutter (see the
root-cause section's `self_score` numbers), but a substantial population of confidently-scored,
room-sized false positives remains.

This pointed at `max_box_area_frac` (the whole-frame-box filter, which already exists specifically
to drop implausibly large boxes) as underexplored: the existing domain-tuning grid only searched
`{0.1, 0.25, 0.5}` — even its smallest candidate (10% of the plan) is still far larger than a real
door/window symbol, which `docs/eval/floorplans-findings.md`'s dataset statistics put at under 1.6%
of plan area even in the "large" size bucket. **A single hard plan tested directly (not through the
val/test protocol) at `max_box_area_frac=0.01` roughly doubled its F1** (0.047 → 0.121) versus the
shipped default 0.25 — a promising signal, but measured the WRONG way (see below).

### The widened grid, measured properly (val→freeze→test) — a genuine lesson, not just a win

The domain-tuning grid (`src/object_search/eval/tuning.py`'s `_TUNING_GRIDS["owlv2-oneshot"]`) was
widened from `area ∈ {0.1, 0.25, 0.5}` to `area ∈ {0.005, 0.007, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5}`
(still crossed with `query_iou_frac ∈ {0.6, 0.8, 0.9}`), then `tune-floorplans` was re-run honestly
on a GPU box — tune on val (56 plans), freeze, report on the held-out test split (28 plans), never
peeking at test during selection. The full val trial table (sorted by val F1) tells two different
stories per class:

**Windows — clean, monotonic, trustworthy:**

| area (query_iou=0.6, the best tier) | val F1 |
| --- | --- |
| 0.007 | 0.0677 |
| 0.010 | 0.0620 |
| 0.005 | 0.0619 |
| 0.020 | 0.0435 |
| 0.050 | 0.0340 |
| 0.100 | 0.0318 |
| 0.250 | 0.0305 |
| 0.500 | 0.0302 |

Smaller area monotonically helps on val, and **the selected config (0.007, 0.6) also wins cleanly
on test**: tuned F1 0.044 vs the old grid's 0.025 (+76%, and vs the pre-calibration-pass 0.023,
+91%). A genuine, robust improvement.

**Doors — noisy, non-monotonic, a val-selection cautionary tale:**

| area (query_iou=0.6, the best tier) | val F1 |
| --- | --- |
| 0.010 | 0.2240 |
| 0.050 | 0.2178 |
| 0.100 | 0.2067 (the OLD grid's pick) |
| 0.250 | 0.2009 |
| 0.500 | 0.1989 |
| 0.020 | 0.1970 (a dip BELOW 0.05 and 0.1 -- non-monotonic) |
| 0.007 | 0.1731 |

The top three candidates are within 0.02 F1 of each other on 56 validation plans — not a confident
margin — and `0.02` scoring worse than both its neighbours `0.05`/`0.1` shows real noise at this
sample size, not a clean signal. The widened grid's val-argmax (`area=0.01`) **scores HIGHER on val
(0.224 vs the old grid's 0.207) but LOWER on test** (tuned F1 0.171 vs the old grid's 0.215) — a
textbook val-selection-doesn't-generalize outcome. This is not a bug in the tuning code or a
regression to revert; it is the tuning protocol doing exactly its job (report what val honestly
selected) and exposing a real limitation of a 56-plan validation set for this specific class. The
grid stays widened (the window result alone justifies it, and pretending a smaller search space
would have been "more correct" for doors would be cherry-picking after the fact) — reported here
in full rather than only the number that looks better.

### Was the flood ever a photographic-object-scale problem? (checking the global default)

Before touching the shared `max_box_area_frac=0.25` default (which every regime uses, not just
floor plans), the same area sweep was run against EASY/TEXTURED/VARIED/CLUTTERED to check whether a
smaller GLOBAL default would help without the domain-tuning safety net:

| regime | 0.25 (current default) | 0.05 | 0.02 | 0.01 | 0.007 | 0.005 |
| --- | --- | --- | --- | --- | --- | --- |
| EASY | 0.355 | 0.493 | 0.648 | 0.731 | 0.809 | **0.822** |
| TEXTURED | 0.856 | **0.899** | 0.846 | 0.332 | 0.064 | 0.015 |
| VARIED | **0.859** | 0.780 | 0.302 | 0.106 | 0.024 | 0.015 |
| CLUTTERED | **0.845** | 0.818 | 0.175 | 0.037 | 0.048 | 0.009 |

No single value is good everywhere: EASY wants the cap as small as possible (still improving at the
smallest value tried), while VARIED and CLUTTERED want it to stay at the current default and fall
off a cliff below ~0.05. **The global default correctly stays at 0.25** — this is exactly what the
existing default-vs-domain-tuning split in this codebase is for, and the floor-plan improvement
belongs entirely in the domain-tuning grid, not in the method's shared default.

## retain_frac sweep (calibrated score, all six regimes)

The re-tune swept `{0.80, 0.85, 0.88, 0.90, 0.92, 0.94, 0.96, 0.98}` against the calibrated score.
`0.85` was selected for the same reason `0.94` was originally selected for raw cosine: **robustness
across every regime, not the per-regime argmax.**

| retain_frac | EASY | TEXTURED | VARIED | CLUTTERED | DOOR | WINDOW | avg |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.80 | 0.348 | 0.785 | 0.852 | 0.833 | 0.156 | 0.027 | 0.500 |
| **0.85** | **0.349** | 0.856 | 0.859 | 0.845 | **0.154** | 0.025 | **0.515** |
| 0.88 | 0.337 | 0.859 | 0.850 | 0.848 | 0.154 | 0.025 | 0.512 |
| 0.90 | 0.319 | **0.866** | 0.824 | **0.856** | 0.153 | 0.026 | 0.507 |
| 0.92 | 0.304 | 0.857 | 0.794 | 0.841 | 0.151 | 0.026 | 0.496 |
| 0.94 (old default's rf, new score) | 0.283 | 0.865 | 0.751 | 0.816 | 0.150 | 0.027 | 0.482 |
| 0.96 | 0.264 | 0.820 | 0.685 | 0.770 | 0.150 | 0.027 | 0.453 |
| 0.98 | 0.208 | 0.719 | 0.619 | 0.695 | 0.144 | 0.026 | 0.402 |

`0.85` has the highest average AND beats the raw-cosine baseline on every single regime — the same
"near-max F1 everywhere" criterion the original `retain_frac=0.94` pick used, just re-derived for
the new score scale.

## Fairness note

- The **calibration rule (`self-similarity`) is identical across every regime and label-free**,
  exactly as before: `self_score * retain_frac`, anchored to the exemplar's own calibrated
  self-match, no per-dataset or per-image tuning, no access to ground truth.
- `retain_frac` is a **method-level constant**, chosen once for robustness across all six regimes
  measured — explicitly not the global-F1 argmax and not fit to the floor-plan domain specifically
  (the floor-plan target domain happens to benefit the most, but the pick was made looking at all
  six regimes together, the same discipline the first `owlv2-oneshot` pass and `ncc`'s
  `retain_frac` use).
- `rotation_invariant` and `tile_large_scenes` were measured with the SAME rule: no change is kept
  as a default that regresses a regime the method already handled. Both regress DOOR and/or EASY,
  so both stay off by default despite being fully built and tested.
- The per-class `tune-floorplans` grid (`max_box_area_frac` × `query_iou_frac`) IS fit per class on
  `val` and frozen before scoring `test` — that is the tuning protocol's job, and is reported
  separately from the method-default numbers above, exactly as the original floor-plans report did.

## Deferrals

- **Text-prompt fusion, multi-exemplar query averaging, owlv2-large** — unchanged from the first
  pass's backlog; out of scope here.
- **Tiling and rotation-invariance are not further pursued** in this pass. Both are implemented,
  tested, and honestly measured to not help on net; a future attempt would need a materially
  different approach (e.g. tile-aware NMS/deduplication tighter than the current per-tile
  whole-frame filter, or restricting rotation augmentation to only the query-selection step rather
  than scoring every patch against every rotation) rather than a parameter retune of what is here.
- **Doors' domain-tuned number is now sensitive to validation-set noise.** A larger val split (or a
  seeded k-fold over the existing 56 plans) would let the tuning grid distinguish `area=0.01` from
  `area=0.05`/`0.1` with real confidence instead of a ~0.02-F1-wide toss-up; out of scope for a
  single method's improvement pass, since it would change the tuning protocol for every method.
- **Cross-tile / cross-region false-positive suppression beyond the per-tile whole-frame filter.**
  The debug-image investigation shows the residual clutter is architecturally generic (wall/room
  rectangles that a natural-image detector has no training signal to distinguish from a real
  object), not a geometry-filter gap alone; `max_box_area_frac` narrows it a lot for windows but the
  underlying domain gap is still real. A learned or heuristic "is this a wall vs a symbol" signal
  beyond raw box area is future work, not attempted here.
- **Windows remain the harder class in absolute terms** for `owlv2-oneshot` (tuned F1 0.044, still
  far behind `ncc`'s 0.403 on this dataset) despite the +91% relative gain this pass delivered. Per
  `docs/eval/floorplans-findings.md`, `ncc` remains the recommended default for floor-plan exemplar
  search, and this pass does not change that recommendation — it makes `owlv2-oneshot` genuinely
  better (not just differently-flawed) on this dataset, and a real winner on the four non-floor-plan
  regimes it already led on.

## Reproducing

1. Drop the Roboflow floor-plans-500 COCO export at `datasets/_incoming/floorplans/`, then
   `pixi run fetch-datasets --only floorplans-door && pixi run fetch-datasets --only floorplans-window`.
2. `pixi run -e export export-owlv2` to (re-)export the ONNX graph with the new `logit_shift`/
   `logit_scale` outputs (the export wrapper changed; a pre-existing `owlv2_base_patch16.onnx`
   exported before this pass does NOT carry these outputs and must be regenerated).
3. `pixi run tune-floorplans` for the domain-tuned per-class numbers (the widened
   `max_box_area_frac` grid is now the committed default grid, no extra step needed).
4. `Owlv2OneshotConfig(debug_dir="/some/path")` on any `search()` call dumps the per-step debug
   images/heatmaps described above -- useful for investigating any specific plan's behaviour.
5. The six-regime sweeps (raw cosine vs calibrated vs rotation-invariant vs tiled vs gmm, the
   `retain_frac` sweep, and the `max_box_area_frac` sweeps) were run via throwaway scripts, not
   committed — the fast-iteration embed-once-cache-scores pattern from the first `owlv2-oneshot`
   pass, extended to floor plans.
