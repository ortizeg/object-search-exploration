# EXPERIMENTS — `ncc` on the floor-plan door/window domain (quick task 260730-vx4)

Append-only lab notebook. **One entry per experiment, never overwritten**, in the order they were
run. Every number here came out of a run recorded in `runs/` on this branch; nothing is estimated.

**Machine / environment.** macOS (darwin 25.5.0), 14 CPUs, pixi `default` env, OpenCV 4.13
(conda-forge headless), single-process CPU correlation. Several experiments were run concurrently
in separate processes, so **absolute latencies here are contended and only comparable within one
entry**, never across entries.

**Protocol (unchanged from the committed one, so the numbers are comparable to
`docs/eval/floorplans-findings.md`).** Tune on `val` (56 plans, argmax F1 @ IoU 0.5) → freeze →
score the frozen config AND the method's shipped defaults once on `test` (28 plans), 1 exemplar,
seed 0, `seeded-random` exemplar draw. `test` is never read to pick between trials.

**Datasets.** Roboflow floor-plans-500 converted to `floorplans-door` / `floorplans-window`
(committed split manifests, 56 val / 28 test plans per class; both classes come from the same
plans).

---

## E0 — Baselines (2026-07-31)

The two numbers every later experiment is diffed against.

### E0a — synthetic-regime regression guard: `pixi run bench-ci`

Model-free chipset subset (`ncc` + classical `sparse-geo`, no ONNX weights — `models/` is empty in
this worktree, so the full `pixi run bench` learned methods cannot run here). 6 chipset images
spanning the canvas ramp. Raw: `runs/bench-ci-baseline-results.json`.

| method | P | R | F1 | mean AP | abstentions | p50 ms |
|---|---|---|---|---|---|---|
| `ncc` | 1.000 | 1.000 | **1.000** | 1.000 | 0 | 2126 |
| `sparse-geo` (classical) | — | 0.000 | — | 0.033 | 6 | 62 |

`ncc` is perfect on the chipset (the EASY regime of `docs/reports/ncc-improvement.md`), so **any
regression is immediately visible**: F1 can only go down from 1.000.

### E0b — wider synthetic guard: `pixi run bench methods=[ncc]`

`bench-ci` only covers the chipset. The regimes that `docs/reports/ncc-improvement.md` actually
reports (EASY / TEXTURED / VARIED / CLUTTERED) need the textured + synthetic scenes too, and `ncc`
alone needs no weights — so this ran as an additional, stricter guard. Raw:
`runs/bench-ncc-synthetic-baseline.json`.

60 images (chipset ramp + the EVAL-20 textured set + the two scale/pose synthetic scenes),
`ncc` at its shipped defaults, IoU 0.5:

| slice | images | P | R | F1 | mean AP | p50 ms |
|---|---|---|---|---|---|---|
| **overall** | 60 | 0.855 | 0.730 | **0.788** | 0.779 | 1724 |
| `fixed` scale bucket | 35 | 0.942 | 0.959 | **0.950** | 0.972 | 1723 |
| `varied` scale bucket | 25 | 0.658 | 0.412 | **0.506** | 0.510 | 1725 |

These three F1s are the synthetic-regime regression guard used by every later entry. (The
EASY/TEXTURED/VARIED/CLUTTERED table in `docs/reports/ncc-improvement.md` came from a bespoke
harness that no longer exists as a task; `by_scale_bucket` is the committed sweep's equivalent
split and is what is diffed here.)

### E0c — floor-plan baseline: committed `_TUNING_GRIDS["ncc"]`, both classes

The committed 20-entry grid (`scales` ∈ {(1.0,), (0.9,1.0,1.1)} × `retain_frac` ∈
{0.25,0.35,0.45,0.55,0.65} × `nms_iou` ∈ {0.3,0.5}), i.e. today's numbers, re-measured locally on
this machine so later experiments diff against a same-machine baseline rather than against the
vast.ai run in `docs/eval/floorplans-findings.md`. Raw: `runs/baseline--floorplans-{door,window}.json`.

| dataset | tuned overrides | val F1 | test P | test R | test F1 | default test F1 |
|---|---|---|---|---|---|---|
| floorplans-door | `{nms_iou:0.3, retain_frac:0.65, scales:[1.0]}` | 0.1797 | 0.569 | 0.159 | 0.248 | 0.164 |
| floorplans-window | `{nms_iou:0.3, retain_frac:0.65, scales:[1.0]}` | 0.3290 | 0.428 | 0.378 | 0.401 | 0.222 |

(door test F1 0.248 exactly matches `docs/eval/floorplans-findings.md`'s vast.ai number — the
local re-measurement is an honest same-protocol cross-check, not a re-derivation.)

## E1 — Experiment A: rotation-bank sweep (2026-07-31)

Four `angles_deg` banks (shipped ±35° / cardinal 0·90·180·270 / cardinal × shipped ±35° sub-bank
[28 angles] / uniform 30° [12 angles]) × `retain_frac` ∈ {0.35,0.45,0.55}, `scales=(1.0,)` fixed,
both classes. Raw: `runs/rotation-bank--floorplans-{door,window}.json`.

Best val F1 per bank (across the 3 `retain_frac` values):

| n_angles | bank | door best val F1 | window best val F1 |
|---|---|---|---|
| 4 | cardinal | **0.1917** | **0.2985** |
| 12 | uniform30 | 0.1727 | 0.2736 |
| 7 | shipped ±35° | 0.1498 | 0.2518 |
| 28 | cardinal×fine | 0.1610 | 0.2409 |

Cardinal wins clearly for both classes — confirms floor-plan orientation is discretely orthogonal,
not continuously distributed; the widest bank (28 angles) is worse than even the shipped default,
consistent with the "more search flexibility -> more structured-background false peaks" pattern
that recurs throughout this investigation (see E4).

## E2 — Experiment B: mirror, measured separately (2026-07-31)

Cardinal bank (Experiment A's winner) × `retain_frac=0.55` × `mirror ∈ {False, True}`. Raw:
`runs/mirror--floorplans-{door,window}.json`.

| dataset | mirror=False val F1 | mirror=True val F1 | frozen pick |
|---|---|---|---|
| floorplans-door | 0.192 | **0.197** | mirror=True (statistical tie, within val noise) |
| floorplans-window | **0.299** | 0.276 | mirror=False (mirror net-negative) |

## E3 — Final: cardinal + mirror folded into the committed grid (2026-08-01)

`_TUNING_GRIDS["ncc"]` extended additively (`_ncc_grid()`): the original 20-entry base block
unchanged, plus 10 more entries (cardinal bank × `retain_frac` ∈ {0.25,0.35,0.45,0.55,0.65} ×
`mirror` ∈ {False,True}, `scales=(1.0,)`, `nms_iou=0.3` fixed). This is what `pixi run
tune-floorplans` now sweeps by default. Raw: `runs/baseline--floorplans-{door,window}.json`
(overwritten from E0c — the pre-cardinal numbers above are preserved in this file since they were
transcribed before the overwrite).

| dataset | tuned overrides | val F1 | test P | test R | test F1 | default test F1 | delta vs E0c tuned |
|---|---|---|---|---|---|---|---|
| floorplans-door | cardinal, mirror=True, retain=0.65, nms=0.3 | 0.2506 | 0.340 | 0.378 | **0.358** | 0.164 | **+0.110** |
| floorplans-window | cardinal, mirror=False, retain=0.65, nms=0.3 | 0.3588 | 0.287 | 0.449 | **0.350** | 0.222 | **-0.051** |

Doors: robust win, reproduced across E1/E2/E3 (test F1 0.355-0.358 every time). Windows: cardinal
wins on val (0.359 vs E0c's 0.329) but generalizes slightly worse on THIS test split (0.350 vs
0.401) — an honestly-disclosed val/test gap, not reverted (see the report's Fairness section for
why: reverting because test looks worse would itself be conditioning the grid on a test peek).

## E4 — Recall investigation: two more levers, both net negative (2026-08-01)

Per-instance debugging (`scripts/ncc_debug_visualize.py`) found 27/49 missed doors (55%, 10-image
sample) have a correctly-localized (IoU>0.5) candidate scoring just under the 0.65 cutoff (scores
mostly 0.53-0.65). Two candidate fixes tested directly:

**Lower `retain_frac`** (0.35-0.65, cardinal+mirror door config, val). Raw:
`runs/retain-sweep--floorplans-door.json`.

| retain_frac | val P | val R | val F1 |
|---|---|---|---|
| 0.35 | 0.076 | 0.586 | 0.135 |
| 0.45 | 0.099 | 0.548 | 0.168 |
| 0.55 | 0.125 | 0.469 | 0.197 |
| 0.65 | 0.187 | 0.380 | **0.251** |

Monotonically worse going down -- no lower sweet spot within this range; 0.65 already wins.

**Wider scale pyramid** (1/3/5 levels × retain ∈ {0.45,0.55,0.65}, cardinal+mirror door config,
val). Raw: `runs/scale-sweep--floorplans-door.json`.

| scales | retain=0.65 val P | retain=0.65 val R | retain=0.65 val F1 |
|---|---|---|---|
| `[1.0]` (current) | 0.187 | 0.380 | **0.251** |
| `[0.85,1.0,1.15]` | 0.149 | 0.393 | 0.216 |
| `[0.7,0.85,1.0,1.15,1.3]` | 0.155 | 0.416 | 0.226 |

Wider pyramid recovers a little recall but costs more precision every time -- F1 always worse.
Measured within-image door-size coefficient of variation (~19% on box area) is real but is not the
binding constraint; the extra scale levels just add more false-peak opportunities, exactly
mirroring E1's rotation-bank finding. Conclusion (full reasoning in the report): both true-positive
and false-positive scores occupy the same 0.5-0.65 band on this domain, unlike synthetic data where
genuine instances cluster near the ~1.0 self-match -- no threshold or search-bank change can
separate two overlapping distributions. Neither lever shipped.
