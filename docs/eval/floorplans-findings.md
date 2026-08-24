# Floor-plan findings — which method wins on the target domain (DOC-08)

This records the outcome of a real evaluation of the four+two search methods on the **Roboflow
floor-plans-500** target-domain dataset (`floorplans-door` / `floorplans-window`; see
[research-datasets.md](research-datasets.md)). Only **measured metrics** are recorded here — no
dataset images and no raw per-image data (those stay gitignored, regenerable via the run below).

**How it was produced.** Six methods (`ncc`, `mosse`, `sparse-geo`, `dino-dense`,
`propose-retrieve`, `owlv2-oneshot`) were run one-per-GPU on vast.ai, each doing its own sweep +
per-method domain threshold tuning (`pixi run tune-floorplans`, broadened multi-knob grids). Protocol:
tune on **val** (56 plans), report on the frozen **test** split (28 plans). Headline metric: **F1 @
IoU 0.5** at **1 exemplar**, tuned per method per class. `dino-dense` ran with the opt-in fixed-size
letterbox (`fixed_input_side=1120`).

> **Caveat — small test set (28 plans).** Trust the large gaps, not the small ones. `dino-dense`
> still loses 1 of 14 test plans per class to GPU-OOM (coverage 13/14); the letterbox cut its
> failures from 28/28 to 1. `propose-retrieve`'s rows were re-measured at full 28/28 coverage by its
> own later pass (see the note below the tables). Numbers are one run at one operating point; they
> are directional evidence, not a leaderboard to over-fit.

## Dataset statistics

Both classes are converted from the same 84 plans (56 val + 28 test) — every plan has ≥1 door and
≥1 window, so the image sets are identical and only the kept boxes differ. Size buckets are
box-area ÷ plan-area: **small <0.4%, medium <1.6%, large ≥1.6%** (the same cuts the per-slice eval
uses).

| class / split | plans | instances | small | medium | large | instances/plan (min/med/max) |
|---|---|---|---|---|---|---|
| door / val | 56 | 527 | 54% | 43% | 3% | 3 / 8 / 29 |
| door / test | 28 | 233 | 36% | 58% | 6% | 2 / 8 / 17 |
| window / val | 56 | 463 | 75% | 25% | 1% | 1 / 6 / 40 |
| window / test | 28 | 156 | 62% | 35% | 3% | 1 / 4 / 14 |

Two facts shape the results: **almost every symbol is small/medium** (large is <6% everywhere), and
**windows skew smaller than doors** (62% small vs 36% on test). That is why small-symbol robustness —
where `dino-dense` fails outright and `ncc` holds up — is the decisive property on this domain.

![symbol area as a fraction of the plan](img/floorplans/size-distribution.png)

![instances by size bucket, and crowding (instances per plan)](img/floorplans/counts.png)

## Result — winners differ by class

**Doors** (tuned F1 @ IoU 0.5, test):

| # | method | tuned F1 | default F1 | precision | recall | coverage |
|---|---|---|---|---|---|---|
| 1 | `propose-retrieve` | **0.597** | 0.481 | 0.54 | 0.67 | 28/28 |
| 2 | `ncc` | 0.248 | 0.164 | 0.57 | 0.16 | 28/28 |
| 3 | `sparse-geo` | 0.219 | 0.217 | 0.44 | 0.15 | 28/28 |
| 4 | `mosse` | 0.213 | 0.201 | 0.74 | 0.12 | 28/28 |
| 5 | `owlv2-oneshot` | 0.171 | 0.154 | 0.11 | 0.40 | 28/28 |
| 6 | `dino-dense` | 0.147 | 0.092 | 0.13 | 0.17 | 13/14 |

**Windows** (tuned F1 @ IoU 0.5, test):

| # | method | tuned F1 | default F1 | precision | recall | coverage |
|---|---|---|---|---|---|---|
| 1 | `ncc` | **0.403** | 0.226 | 0.43 | 0.38 | 28/28 |
| 2 | `sparse-geo` | 0.309 | 0.290 | 0.63 | 0.21 | 28/28 |
| 3 | `mosse` | 0.148 | 0.077 | 0.23 | 0.11 | 28/28 |
| 4 | `propose-retrieve` | 0.110 | 0.110 | 0.12 | 0.10 | 28/28 |
| 5 | `owlv2-oneshot` | 0.044 | 0.025 | 0.02 | 0.22 | 28/28 |
| 6 | `dino-dense` | 0.047 | 0.048 | 0.03 | 0.11 | 13/14 |

**`propose-retrieve` wins doors; `ncc` wins windows** — and `propose-retrieve` *collapses* on windows
(0.048), so the best method is class-dependent. **`ncc` is the most reliable all-rounder**: 2nd on
doors, 1st on windows, full 28/28 coverage, and (below) uniformly robust to symbol size.

> **`owlv2-oneshot`'s rows above reflect two later, dedicated improvement passes**
> ([`docs/reports/owlv2-floorplans-improvement.md`](../reports/owlv2-floorplans-improvement.md)):
> exporting and applying OWLv2's own learned score-calibration terms (lifted the DEFAULT, untuned
> F1 on both classes: doors 0.115 → 0.154, windows 0.022 → 0.025), then widening the domain-tuning
> grid's `max_box_area_frac` search down to CAD-symbol scale (windows 0.025 → **0.044**, a real,
> monotonic-on-val +76% win; doors landed at 0.171 -- a small, non-monotonic-on-val improvement
> over the default, but *below* an earlier, narrower-grid run's 0.215, because the wider grid's
> val-argmax overfit 56 validation plans in a way that didn't generalize to test -- see the report
> for the full trial table). Net: owlv2 stays #5 on doors (same rank as before ANY of this work,
> though the untuned default is now real ground gained) and moves up to #4 on windows. The other
> five methods' rows are unchanged from the original sweep, not re-run in these passes.

> **`propose-retrieve`'s rows above reflect a later, dedicated improvement pass**
> ([`docs/reports/propose-retrieve-floorplans-improvement.md`](../reports/propose-retrieve-floorplans-improvement.md)).
> That pass diagnosed the door gap to the **proposal stage**, not retrieval: FastSAM's everything-mode
> proposal count scales with image AREA (r = +0.59 over 84 plans) rather than instance count
> (r = +0.22), so a crowded plan got ~46 proposals for ~15 doors and the proposal stage capped recall
> at 0.405 before retrieval ever ran — with final recall (0.399) sitting essentially *at* that
> ceiling. The fix is **one existing config field**: `proposal_conf` 0.4 → 0.10, shipped as an
> additive `_TUNING_GRIDS` entry (no default changed, zero new config fields). **Doors 0.459 →
> 0.597** (+30%), almost entirely recall (0.39 → 0.67) against a precision cost (0.55 → 0.54), with
> abstentions 3 → 0 and every symbol-size bucket improving (the recall-by-size row below is updated
> accordingly). **`propose-retrieve` keeps #1 on doors and widens the gap over `ncc` from 0.21 to
> 0.35 F1.**
>
> Two row changes are **coverage corrections, not improvements**, and are called out so they are not
> mistaken for gains: both `propose-retrieve` rows now report **28/28** instead of 13/14, because
> this pass ran on CPU and lost no plans to GPU-OOM. The **windows** row moves 0.048 → 0.110 (rank
> #5 → #4) purely from that fuller coverage at the method's *default* config — this pass did **not**
> improve windows and did not try to; `propose-retrieve` still collapses there and `ncc` still wins
> the class. The doors row's `default F1` column likewise moves 0.459 → 0.481 for the same coverage
> reason, which is why tuned-vs-default is now a real gap (0.481 → 0.597) where it used to be zero.
>
> **A note on what did NOT ship, because it is the more useful finding.** That pass's primary lever
> was SAHI-style tiled FastSAM proposals. It was fully implemented, measured across seven geometries
> and five merge thresholds over ~30 h of CPU, and **rejected**: at a *matched* proposal budget the
> objectness gate beat tiling by +0.233 mean proposal recall at a third of the latency, and SAHI's
> magnification premise measured inert here (a 2× difference in pixels-per-symbol moved recall by
> 0.001). The tiling code stays in the repo behind default-off fields but is **not recommended for
> this domain**. This independently converges with
> [`dino-dense-floorplans-improvement.md`](../reports/dino-dense-floorplans-improvement.md)'s Pass 4,
> which built, measured and fully reverted tiled inference for `dino-dense` on this same dataset —
> two methods, two mechanisms, same verdict on tiling for CAD floor plans. The other four methods'
> rows are unchanged from the original sweep, not re-run in this pass.

> **`sparse-geo`'s rows above are unchanged, and a later dedicated pass confirms they should be**
> ([`docs/reports/sparse-geo-improvement.md`](../reports/sparse-geo-improvement.md)). That pass
> reproduced this table's `sparse-geo` numbers to three decimals (doors 0.2194 / 0.4416 / 0.1459,
> windows 0.3092 / 0.6275 / 0.2051), then tested two structural hypotheses against the flat door
> recall-by-size below — **mirror acceptance** (door symbols drawn with the opposite hinge hand) and
> a **learned SuperPoint backend** (line-art carries little DoG texture for SIFT). **Both were
> measured and both were disproven**: mirror lost F1 in 4/4 class × voting-mode cells and was fully
> reverted (commit `8ab99a2`); SuperPoint lost F1 in 4/4 cells, dropped window coverage to 26/28 on
> an ONNX/CoreML crash when zero keypoints are detected, and cost 5.3–6.9× latency, so it was never
> committed. `sparse-geo`'s config, defaults and recommended floor-plan tuning
> (`min_inliers=3, nms_iou=0.3`) are therefore **unchanged** — no improvement is being claimed here.
> The flat door recall-by-size remains an **open question**; the report records what is now ruled
> out and where the funnel actually collapses (55 Hough peaks hypothesized for 157 ground-truth
> doors, from 2 664 correspondences).

## What the tuning + fixes bought

- **Domain tuning matters most for `ncc`/`mosse`:** `ncc` window F1 nearly doubled vs default
  (0.226 → 0.403); door 0.164 → 0.248. `owlv2` window 0.025 → 0.044 (post-calibration; the tuning
  grid here is `max_box_area_frac` × `query_iou_frac`, layered on top of the score-calibration fix
  described next, not a substitute for it) — door tuning gained much less (0.154 → 0.171), see below.
- **`ncc`'s selected scale set is a single scale `[1.0]`.** Floor-plan symbols are drawn at a fixed
  size, so the default multi-scale search (0.75–1.3) manufactured false positives; tightening it is
  the single biggest domain-specific lever. `mosse` reached **0.74 precision** on doors the same way.
- **`owlv2` over-detects, but genuinely less than before a dedicated calibration pass.** Exporting
  and applying OWLv2's own learned `logit_shift`/`logit_scale` (query-independent, per-patch score
  recalibration) roughly halved false-positive counts on the worst plans and improved the DEFAULT
  (untuned) F1 on both classes — capping `max_box_area_frac=0.1` alone, tried earlier, did not fix
  this; the root cause was a scoring/ranking problem, not a box-geometry filter gap. A follow-up
  debug-image inspection then found the RESIDUAL false positives (after calibration) are large
  room/wall-sized rectangles, not symbol-sized boxes — CAD-symbol scale is a few percent of the
  plan at most, so the domain-tuning grid's area cap was widened down from {0.1, 0.25, 0.5} to
  include CAD-symbol-scale values (0.005–0.5). This helps windows a lot and cleanly (val-monotonic,
  +76% tuned F1) but doors only a little and noisily (the val-argmax is unstable across nearby area
  values with only 56 validation plans — full trial table in
  [`docs/reports/owlv2-floorplans-improvement.md`](../reports/owlv2-floorplans-improvement.md)).
  Windows are still the harder class in absolute terms — but no longer barely moving. Fine-tuning
  the underlying OWLv2 weights was also tried, independently of this calibration pass — see
  [`owlv2` floor-plan fine-tune](../reports/owlv2-floorplans-finetune.md): four experiments,
  the first two (classification loss, plain contrastive loss) regress doors and are negative
  results, but a crop-context-supervised contrastive objective (with rotation/mirror augmentation)
  beats the pretrained baseline on both classes (door F1 up to 0.253 tuned, window 0.204–0.216) —
  still short of `propose-retrieve`/`ncc`, so the recommendation below is unaffected.
- **The `dino-dense` letterbox works:** coverage 0/28 → 13/14 (OOM failures 28 → 1). Still not
  competitive, but no longer broken.

## Where each method fails — recall by symbol size (doors, test, 1 exemplar)

> `owlv2-oneshot`'s row below predates its calibration pass (see the note above) and has not been
> recomputed at the new default — treat it as directional for the other five methods, not current
> for `owlv2-oneshot`.

| method | small | medium | large |
|---|---|---|---|
| `propose-retrieve` | **0.63** | **0.71** | **0.57** |
| `ncc` | 0.31 | 0.31 | 0.29 |
| `owlv2-oneshot` | 0.61 | 0.56 | 0.36 |
| `mosse` | 0.19 | 0.25 | 0.36 |
| `dino-dense` | **0.00** | 0.25 | 0.18 |
| `sparse-geo` | 0.14 | 0.13 | 0.14 |

- **`propose-retrieve` is now both the strongest and the most size-robust on doors** (0.57–0.71
  across buckets, roughly double every other method in every bucket) — the result of its dedicated
  pass, which lifted the proposal-stage ceiling that had been capping all three buckets at once. Its
  row above is the only one measured after that pass; the other five predate it.
- **`dino-dense` cannot find small symbols** (recall 0.00 on small doors, 0.09 on small windows): the
  fixed letterbox shrinks a small door below the DINOv2 patch grid.
- **`ncc` is size-robust** (≈0.30 across door sizes) — a strong reason it is the dependable default,
  though it is no longer the *most* size-robust method on doors.
- **`sparse-geo` needs size:** on windows its recall climbs from 0.13 (small) to 0.40 (large) —
  keypoint matching needs enough texture, which small stamped symbols lack.
- **`owlv2` biases toward small** (it over-detects everywhere), which is why its recall is inverted
  vs the others.

## Qualitative overlays

**Licence / attribution:** [Roboflow floor-plans-500](https://universe.roboflow.com/university-y9nbi/floor-plans-500) by `university-y9nbi`, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) — confirmed 2026-08-10 against the creator's own listing. These are derivative overlays (boxes drawn over the licensed plan images), committed under that licence's attribution term.

One easy and one hard plan per class, all 6 methods, generated by `pixi run python scripts/build_floorplans_report.py` (also builds a richer self-contained `docs/benchmark/floorplans-report.html` locally, gitignored/regenerate-only — the committed deliverable is this gallery). **green** = matched (TP), **yellow** = missed (FN), **red** = spurious (FP), **orange** = the query exemplar. The pattern is visible at a glance: `dino-dense` leaves the small doors yellow (missed), while `ncc` and `propose-retrieve` fill them green.

### Doors

**Easy plan:**

`ncc`

![ncc on the easy door plan](../benchmark/floorplans-overlays/floorplans-door-easy-ncc.png)

`mosse`

![mosse on the easy door plan](../benchmark/floorplans-overlays/floorplans-door-easy-mosse.png)

`sparse-geo`

![sparse-geo on the easy door plan](../benchmark/floorplans-overlays/floorplans-door-easy-sparse-geo.png)

`dino-dense`

![dino-dense on the easy door plan](../benchmark/floorplans-overlays/floorplans-door-easy-dino-dense.png)

`owlv2-oneshot`

![owlv2-oneshot on the easy door plan](../benchmark/floorplans-overlays/floorplans-door-easy-owlv2-oneshot.png)

`propose-retrieve`

![propose-retrieve on the easy door plan](../benchmark/floorplans-overlays/floorplans-door-easy-propose-retrieve.png)


**Hard plan:**

`ncc`

![ncc on the hard door plan](../benchmark/floorplans-overlays/floorplans-door-hard-ncc.png)

`mosse`

![mosse on the hard door plan](../benchmark/floorplans-overlays/floorplans-door-hard-mosse.png)

`sparse-geo`

![sparse-geo on the hard door plan](../benchmark/floorplans-overlays/floorplans-door-hard-sparse-geo.png)

`dino-dense`

![dino-dense on the hard door plan](../benchmark/floorplans-overlays/floorplans-door-hard-dino-dense.png)

`owlv2-oneshot`

![owlv2-oneshot on the hard door plan](../benchmark/floorplans-overlays/floorplans-door-hard-owlv2-oneshot.png)

`propose-retrieve`

![propose-retrieve on the hard door plan](../benchmark/floorplans-overlays/floorplans-door-hard-propose-retrieve.png)


### Windows

**Easy plan:**

`ncc`

![ncc on the easy window plan](../benchmark/floorplans-overlays/floorplans-window-easy-ncc.png)

`mosse`

![mosse on the easy window plan](../benchmark/floorplans-overlays/floorplans-window-easy-mosse.png)

`sparse-geo`

![sparse-geo on the easy window plan](../benchmark/floorplans-overlays/floorplans-window-easy-sparse-geo.png)

`dino-dense`

![dino-dense on the easy window plan](../benchmark/floorplans-overlays/floorplans-window-easy-dino-dense.png)

`owlv2-oneshot`

![owlv2-oneshot on the easy window plan](../benchmark/floorplans-overlays/floorplans-window-easy-owlv2-oneshot.png)

`propose-retrieve`

![propose-retrieve on the easy window plan](../benchmark/floorplans-overlays/floorplans-window-easy-propose-retrieve.png)


**Hard plan:**

`ncc`

![ncc on the hard window plan](../benchmark/floorplans-overlays/floorplans-window-hard-ncc.png)

`mosse`

![mosse on the hard window plan](../benchmark/floorplans-overlays/floorplans-window-hard-mosse.png)

`sparse-geo`

![sparse-geo on the hard window plan](../benchmark/floorplans-overlays/floorplans-window-hard-sparse-geo.png)

`dino-dense`

![dino-dense on the hard window plan](../benchmark/floorplans-overlays/floorplans-window-hard-dino-dense.png)

`owlv2-oneshot`

![owlv2-oneshot on the hard window plan](../benchmark/floorplans-overlays/floorplans-window-hard-owlv2-oneshot.png)

`propose-retrieve`

![propose-retrieve on the hard window plan](../benchmark/floorplans-overlays/floorplans-window-hard-propose-retrieve.png)


## Recommendation

- **Ship `ncc` as the dependable default** for floor-plan exemplar search — robust across both
  classes and all symbol sizes, full coverage, and it benefits most from the (cheap) domain tuning.
- **Prefer `propose-retrieve` specifically for doors** if a door-only workflow justifies loading the
  proposal + retrieval models — and the case is now much stronger than it was. Its dedicated pass
  ([`propose-retrieve-floorplans-improvement.md`](../reports/propose-retrieve-floorplans-improvement.md))
  took doors to F1 **0.597** with recall 0.67, more than double `ncc`'s door F1 and uniformly better
  across every symbol-size bucket. Tune it with `proposal_conf` in the grid (`pixi run
  tune-floorplans`); the win does not appear at the shipped default, by design. It still collapses
  on windows, so the class-dependent split stands.
- **`owlv2-oneshot` improved but does not change the recommendation.** Post-calibration it stays
  #5 on doors (still well behind `propose-retrieve`/`ncc`/`sparse-geo`/`mosse`) and moves up to #4
  on windows (still well behind `ncc`/`sparse-geo`). Not recommended over `ncc` here, but no longer
  the clearest "avoid" — it is the best all-rounder on the four non-floor-plan regimes measured
  elsewhere (see `docs/reports/owlv2-floorplans-improvement.md`), so it is a real option outside
  this specific target domain.
- **Deprioritise `dino-dense` here** — blind to small symbols (recall 0.00 on small doors).

## Reproducing

`pixi run fetch-datasets` (after dropping the Roboflow COCO export at
`datasets/_incoming/floorplans/`), then `scripts/gpu_bench.sh` on a CUDA box (it runs the per-method
sweep + `tune-floorplans`). Raw/environment-dependent outputs (tuning dumps, the raw results
dump, the local HTML report) land in the gitignored `docs/benchmark/` tree; the qualitative
overlay gallery above (`docs/benchmark/floorplans-overlays/`) is regenerated by
`scripts/build_floorplans_report.py` and is committed (see the licence note above).
