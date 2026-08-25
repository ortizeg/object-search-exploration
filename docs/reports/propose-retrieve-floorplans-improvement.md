# Engineering log — `propose-retrieve` on floor-plan doors (2026-08-24)

A record of the empirical iterate/measure/revert loop that improved `propose-retrieve` (Method ⑤)
on the **floorplans-door** target domain — the method's best floor-plan class but at a low absolute
score (tuned F1 0.459 in [`docs/eval/floorplans-findings.md`](../eval/floorplans-findings.md), with
recall 0.39 against precision 0.55). Follows the
[`propose-retrieve-improvement.md`](propose-retrieve-improvement.md) template and the project's
method-improvement playbook: own session, own branch, iterate → measure → **revert anything that
does not genuinely help**.

**Result: floorplans-door test F1 0.481 → 0.597** (+24 % relative over this session's baseline;
+30 % over the committed 0.459 row), by changing **one existing config field**.

Cross-references: the method is documented in
[`docs/methods/propose-retrieve.md`](../methods/propose-retrieve.md); the general-case improvement
pass is [`propose-retrieve-improvement.md`](propose-retrieve-improvement.md); deferred work is in
[`docs/ROBUSTNESS-BACKLOG.md`](../ROBUSTNESS-BACKLOG.md). The full append-only lab notebook —
every trial, including the ones that failed — is `EXPERIMENTS.md` in this session's quick-task
directory (`.planning/quick/260812-m8m-improve-propose-retrieve-recall-on-floor/`).

---

## Lead finding — this session's own tiling implementation was measured and rejected

The session was scoped around **SAHI-style tiled FastSAM proposals** as the primary lever. That
lever was designed from a research note, implemented in full (273 lines: `propose_tiled`,
`_tile_origins`, `_merge_tiled_proposals` in `proposals.py`, five config fields, model-free unit
tests, commit `41b8431`), measured across seven geometries and five merge thresholds over roughly
30 hours of CPU — **and then beaten decisively by a scalar that was already in the shipped config.**

Three layers, in the order they were measured:

**1. Tiling is not the mechanism, at a matched proposal budget.** Both tiling and the objectness
gate `proposal_conf` act by putting more proposals on the table, so the fair comparison holds that
budget fixed. Run as two matched arms (notebook entry T2):

| n_proposals/plan | config | mechanism | mean proposal recall | small | crowded (11+) | proposal s/plan |
|---|---|---|---|---|---|---|
| **161.2** | **untiled `conf 0.10`** | **gate** | **0.821** | **0.638** | **0.639** | **36.1** |
| **164.5** | **tiled 768/0.3 `conf 0.40`** | **tiling** | 0.588 | 0.330 | 0.292 | 103.3 |

> At an equal proposal budget (a 2 % difference), opening the objectness gate beats tiling by
> **+0.233 mean proposal recall**, **+0.308 on small symbols**, and **+0.347 in the crowded
> bucket** — while costing about a **third** of the proposal-stage latency.

Every untiled point lies above the tiled frontier. `untiled conf 0.20` beats `tiled conf 0.50` with
23 % *fewer* proposals; `untiled conf 0.10` beats `tiled conf 0.30` with 28 % fewer.

**2. SAHI's stated premise is measured INERT for this domain.** SAHI exists to rescue *small objects
lost to downscaling*; its lever is magnification. With the cross-tile merge disabled (`merge_ios =
1.0`, where nothing is deleted and only geometry differs), two configurations differing by exactly
2× in pixels-per-symbol can be compared directly:

| geometry | magnification (`1024 / S`) | n_proposals | mean proposal recall |
|---|---|---|---|
| 512-tile | **2.00×** | 187.4 | **0.586** |
| 768-tile | **1.33×** | 163.1 | **0.585** |

A 2× difference in magnification produces a **0.001** difference in recall. What tiling buys on
floor plans is proposals, and nothing else — and it buys them worse than the gate does.

**3. The plan of record that "proved" FastSAM was blind was a threshold artifact.** Plan `4_png`
(513×436, 7 doors) scored proposal recall **0.000** across *seven* tiling configurations and every
merge threshold. The notebook concluded (entry T1e) that *"no tiling geometry fixes a backend that
does not consider a CAD door symbol an object"* — the single strongest argument for abandoning
FastSAM entirely. **That inference was wrong.** At `conf 0.10`, untiled, `4_png` reaches proposal
recall **0.857** (6 of 7 doors, 118 proposals, one forward pass). Three of the four zero-recall
plans are essentially solved by the gate alone. Those symbols were never invisible; they were
scoring below a gate nobody had turned.

**This is the repo's iterate/measure/revert discipline working correctly, not a failure.** The
tiling code was built to a specification, measured honestly against a control it could lose to, and
lost. It stays in the repo behind default-off config fields (it is genuinely the right lever for one
extreme-resolution plan — see *Measured and rejected*), but it is **explicitly not recommended for
this domain**, and the numbers above are why.

### An independent convergence on the same verdict

[`dino-dense-floorplans-improvement.md`](dino-dense-floorplans-improvement.md)'s **Pass 4**
independently implemented, measured, and **fully reverted** tiled inference for the `dino-dense`
method on this *same* floorplans-door dataset — every tile size regressed, two of three below even
the untiled baseline. Two methods, two independent implementations, two independent measurement
passes, **same negative result for tiling on this domain.**

The mechanisms differ, and that is worth stating rather than glossing: `dino-dense` lost to
**per-tile context starvation** (a ViT tile cannot attend to the surrounding walls and rooms that
give a CAD symbol its meaning), whereas `propose-retrieve` lost because **magnification is inert and
tiling is simply an expensive way to buy proposal budget**. Two different failure modes converging on
one practical conclusion is stronger evidence than either alone: **on CAD floor plans, cutting the
scene into tiles is not the lever it looks like.**

---

## Symptom

`propose-retrieve` is the **best** method on floorplans-door and still only scores tuned F1 0.459
(committed) / 0.481 (this session) — against F1 0.90+ on the four chipset/textured/synthetic regimes
the general-case pass tuned it for. The shape of the failure is lopsided: **precision 0.60, recall
0.399.** The method is not wrong about what it finds; it finds too little.

## Root cause — the PROPOSAL stage, and its budget scales with the wrong quantity

Re-derived from committed code (notebook entry B0) rather than carried over from a scratch
diagnostic: for each of 84 plans, run the proposal stage **alone** and count GT boxes matched by
some proposal at IoU ≥ 0.5. This is an honest **ceiling** on final recall.

**Proposal-stage recall by crowding bucket** (doors per plan):

| doors/plan | n plans | mean n_gt | mean n_proposals | proposal recall |
|---|---|---|---|---|
| 1–3 (sparse) | 11 | 2.7 | 48.6 | **0.864** |
| 4–10 | 47 | 7.3 | 44.9 | **0.551** |
| 11+ (crowded) | 26 | 15.0 | 57.0 | **0.268** |
| all | 84 | 9.0 | 49.1 | 0.504 (pooled 0.405) |

**Proposal-stage recall by symbol-size bucket** (small < 0.4 %, medium < 1.6 %, large ≥ 1.6 % of
plan area — the same cuts the findings table uses):

| symbol size | n GT | proposal recall |
|---|---|---|
| small | 366 | **0.279** |
| medium | 364 | 0.516 |
| large | 30 | 0.600 |

Final test recall (0.399) sits essentially **at** the pooled proposal ceiling (0.405). The retrieval
stage was not leaving recall on the table; there was nothing left for it to retrieve.

**The mechanism, stated precisely.** The session's plan named two candidate failure modes — a fixed
proposal budget, and FastSAM's fixed 1024 letterbox shrinking symbols on large plans. They make
opposite predictions about plan size, so they separate cleanly:

| Pearson correlation (n = 84 plans) | value |
|---|---|
| proposal recall vs **n_gt** | **−0.537** |
| proposal recall vs plan long side | +0.190 |
| **n_proposals** vs n_gt | +0.216 |
| **n_proposals** vs **plan area** | **+0.588** |

> **FastSAM's proposal budget scales with image AREA (r = +0.59), not with instance count
> (r = +0.22). Crowding is what destroys recall (r = −0.54).**

The letterbox hypothesis is **not supported at the aggregate level** — large plans do *better*, not
worse, and better *within* every crowding bucket, because they receive roughly twice the proposals.
It survives as a single-plan anecdote (`65_png`, 4000×1685, ~4× the long side of the next-largest
plan), which turns out to matter later.

## The fix, iterated empirically

Every pass measured on floorplans-door before moving on; **val (56 plans) selects, test (28 plans) is
read exactly once per finalist.** Precision and recall are reported separately throughout, because
F1 alone would hide that the shipped win is a trade.

| pass | mechanism | test P | test R | **test F1** | latency vs baseline | verdict |
|---|---|---|---|---|---|---|
| baseline (`conf 0.4`, floor 0.70) | — | 0.604 | 0.399 | **0.481** | 1× | reference |
| + SAHI tiling (`41b8431`, 768/0.3/FI, ios 0.5) | tile budget | — | — | *val 0.424* | ~3.9× | **rejected** |
| + merge-threshold rescue (T1e/T1f, ios 1.0) | un-clamp merge | — | — | *val 0.426* | ~3.9× | **rejected** |
| **+ `proposal_conf` 0.4 → 0.10** (untiled) | objectness gate | **0.536** | **0.674** | **0.597** | ~3.5× embed only | **SHIPPED** |

The two rejected passes are reported on **val**, because neither earned a test read — the plan's
rule is that test is read only for a finalist, and spending a test read on a lever that lost its
val control would have been exactly the fitting-to-test the protocol exists to prevent.

**Why the middle row exists.** The first tiling measurement showed the cross-tile IoS merge acting
as a **budget clamp**: across a 4.3× range in forward passes (2.5 → 10.8 tiles/plan), the merged
proposal count stayed pinned at 1.14–1.33× baseline — the harder you tile, the harder the merge
deletes. The cause is that IoS is `intersection / min(area)`, so a box **fully contained** in a kept
box scores exactly 1.0, and FastSAM everything-mode routinely emits nested proposals (a room, and the
door inside it). The proposals being deleted were structurally the small nested ones — which on this
dataset are the doors. Loosening and then disabling the merge (`merge_ios = 1.0`, kill rate measured
at exactly 0.0 %) repaired that, and lifted proposal-stage recall 0.498 → 0.588. It was a real bug
fix in the tiling path. It still lost to the gate.

### The shipped change

**One existing field moves: `proposal_conf` 0.4 → 0.10.** `similarity_floor` stays at its shipped
default of **0.70** — the joint sweep re-confirms, on the *new* proposal distribution, that the
shipped floor is already this domain's optimum. **Zero new config fields ship**, and both fields
were already documented in the method doc's Config reference.

The val selection grid is monotone on both axes, which is what makes the argmax trustworthy:

| `proposal_conf` | floor 0.70 | 0.75 | 0.80 | 0.85 |
|---|---|---|---|---|
| **0.10** | **0.542** ← argmax | 0.497 | 0.436 | 0.247 |
| 0.20 | 0.494 | 0.436 | 0.361 | — |
| 0.30 | 0.450 | 0.400 | 0.320 | — |

F1 falls as the floor rises at every conf, and `0.10 > 0.20 > 0.30` at every floor. The argmax is a
**corner** of the swept region, not a ridge between near-equal neighbours — unlike the `owlv2` doors
experience recorded in [`owlv2-floorplans-improvement.md`](owlv2-floorplans-improvement.md), where a
val-argmax over these same 56 plans proved unstable and failed to generalise to test.

**Stated as a limitation:** a corner argmax may be a boundary rather than an interior optimum.
`conf < 0.10` was **not measured**. Floors below 0.70 were swept at `conf 0.4` (and lost badly:
0.60 → val F1 0.326, 0.40 → 0.226) but not at `conf 0.10`. The shipped cell is the best **measured**
configuration, not a claimed global one.

**It ships as an additive tuning-grid entry, not as a changed default.** `_TUNING_GRIDS`
["propose-retrieve"] gains a 3 × 3 `proposal_conf` × `similarity_floor` block alongside the
untouched original `similarity_floor` × `nms_iou` block. A grid entry is opted into by a
domain-tuning run; a changed default would touch every regime. The general-case pass already
measured and declined a global `conf` drop for exactly this reason — its closing note records that
lowering `conf` globally "trades away far more textured precision than it buys". That judgement
stands; this is a domain lever, not a method-wide one.

## Result vs baseline

**floorplans-door, TEST (28 plans, 1 exemplar, tuned-on-val, 28/28 scored, 0 errors):**

| | P | R | **F1** | abstentions | coverage |
|---|---|---|---|---|---|
| committed findings row (GPU-era) | 0.55 | 0.39 | **0.459** | — | 13/14 |
| **session-local baseline** | 0.604 | 0.399 | **0.481** | 3 | 28/28 |
| **shipped (`conf 0.10`)** | 0.536 | **0.674** | **0.597** | **0** | 28/28 |
| **delta vs session baseline** | −0.068 | **+0.275** | **+0.116** | −3 | — |

Both baselines are stated; neither is silently replaced. The session-local baseline is **higher**
than the committed row (0.481 vs 0.459) because it scores all 28 test plans rather than 13/14 — a
coverage difference, pre-registered in the notebook before the runs landed, and not a runtime
difference (the guardrail regimes reproduce the committed GPU-era numbers to within rounding on
this CPU box).

**Recall by symbol size (doors, test) — every bucket improves:**

| symbol size | n GT | baseline | **shipped** | delta |
|---|---|---|---|---|
| small | 84 | 0.393 | **0.631** | **+0.238** |
| medium | 135 | 0.415 | **0.711** | **+0.296** |
| large | 14 | 0.286 | **0.571** | **+0.286** |

No bucket pays for the others, and the profile stops being flat-and-low. The `large` bucket is
**14 boxes**, so its delta is four extra matches — directional, not a strong signal.

**Abstentions fall 3 → 0.** Three test plans previously returned nothing at all. That is a
product-visible change the F1 delta does not express.

**Guardrails — unchanged, and unchanged by construction:**

| guardrail | baseline F1 | after | delta |
|---|---|---|---|
| EASY (chipset) | 0.9274 | 0.9274 | **+0.0000** |
| TEXTURED (plain) | 0.9591 | 0.9591 | **+0.0000** |
| VARIED (scale/rotation) | 0.9385 | 0.9385 | **+0.0000** |
| CLUTTERED | 0.8209 | 0.8209 | **+0.0000** |
| synthetic | 0.9091 | 0.9091 | **+0.0000** |
| real-objects | 0.8683 | 0.8683 | **+0.0000** |
| **floorplans-window (test)** | **0.1103** | **0.1103** | **+0.0000** |

Identical to four decimal places on all six regimes (90/90 scored, 0 errors) and on the window test
read (P 0.1194 / R 0.1026 / F1 0.1103, 9 abstentions, matching per-size recall). This is identity
**by construction**: every tiling field defaults off, no shipped default changed, and the win lives
in a tuning grid that only a floor-plan tuning run selects.

## Latency (EVAL-11)

The cost is stated as a cost, not buried. Two structural facts, both measured:

1. **The embedding stage dominates on floor plans, not the proposal stage.** Measured at 5.9 s of a
   7.5 s mean search, and 14.2 s of 15.6 s on a 1478×958 plan. This **contradicts** the general-case
   claim in `propose_retrieve.py`'s own latency note ("the proposal stage dominates"), which was
   established on the chipset/textured regimes. Every proposal gets its own DINOv2 forward pass, so
   embedding cost scales linearly with proposal count.
2. **Opening the gate is free at the proposal stage.** One FastSAM forward pass costs the same
   regardless of the threshold applied to its output — measured flat at 36–49 s/plan across `conf`
   0.10–0.50 (contended, 4-core-pinned).

| lever | FastSAM passes | proposals (→ embeddings) | net |
|---|---|---|---|
| **`proposal_conf` 0.4 → 0.10 (shipped)** | **1× (unchanged)** | 46.5 → 161.2 = **~3.5×** | pays on **one** stage |
| tiling 768/0.3 @ ios 1.0 (rejected) | **~5×** | 46.5 → 164.5 = ~3.5× | pays on **both** stages |

Measured end-to-end, tiling cost **~3.9× the baseline wall clock for +0.022 F1** — and its
distribution is violently skewed: one plan (`65_png`, 549 proposals) took **46 minutes by itself**.
The shipped path's single test read completed in **35.6 min for 28 plans** (mean 76.3 s/plan, median
13.2 s).

**Do not read the shipped path as faster than the baseline** despite its lower measured mean (76.3 s
vs the baseline's 91.5 s/plan). Those two numbers come from differently-contended multi-process runs
and are not comparable. The defensible statement is the structural one in the table: **~3.5× the
embedding work, and nothing extra at the proposal stage.**

## Measured and deferred / rejected

Kept honest per the iterate/measure/revert discipline. Every item below has numbers.

- **SAHI-style proposal tiling — BUILT AND MEASURED, off by default, NOT RECOMMENDED for this
  domain.** The full record is the lead finding above: beaten by `proposal_conf` by +0.233 mean
  proposal recall at a matched budget and a third of the latency; SAHI's magnification premise
  measured inert (0.586 vs 0.585 across a 2× magnification difference); the plan that "proved"
  FastSAM blind was a threshold artifact. **The code stays in the repo** (`propose_tiled`,
  `_tile_origins`, `_merge_tiled_proposals`, five default-off config fields) rather than being
  reverted, for two reasons: it is an exact identity on any scene that fits in one tile, so it costs
  the other regimes nothing; and it is genuinely the right lever for **one** measured case —
  `65_png` (4000×1685, the resolution outlier) reaches proposal recall 0.263 tiled at `conf 0.10`
  versus 0.053 untiled, a 5× gain, because at 4000 px wide the 1024 letterbox really is destroying
  the symbol. That is a narrow niche, and it is not a recommendation.
- **The cross-tile IoS merge is a budget clamp** (found while measuring the above). SAHI's default
  `postprocess_match_threshold = 0.5` on IoS suppresses *fully nested* proposals at every threshold
  below 1.0, and an everything-mode segmenter emits nested proposals constantly. This is a real
  defect in transplanting SAHI's postprocess — which merges *class detections*, where nesting is
  rare — into a class-agnostic segmenter. Documented in the config field's description; the
  `tile_merge_ios` default is left at SAHI's 0.5 because the whole tiling path is not recommended
  here anyway.
- **A contour/blob proposal backend — NOT ATTEMPTED, by evidence.** The session's pre-stated go/no-go
  criterion fired on one of its two arms (crowded-bucket recall still < 0.50; the val F1 gain arm
  did *not* fire, at +0.138 against a +0.05 threshold). It was deliberately skipped anyway, because
  **a contour backend supplies more proposals and the proposal stage is no longer the binding
  stage** (below). Its original motivation — that FastSAM cannot see CAD door symbols — was refuted
  outright by the `4_png` result (0.000 → 0.857 on the gate alone). The full verdict and its numbers
  are recorded in the notebook before the decision was taken.
- **The bottleneck has MOVED to retrieval/calibration — this is the lead for the next pass.** With
  proposal supply no longer the constraint, the transfer from proposal-stage recall to end-to-end
  recall has collapsed in the crowded bucket specifically:

  | crowding | proposal-stage recall | end-to-end recall |
  |---|---|---|
  | sparse | `1–3`: 1.000 | `2-5`: 0.852 |
  | middle | `4–10`: 0.886 | `6-15`: 0.627 |
  | **crowded** | `11+`: **0.639** | `16+`: **0.262** |
  | all | pooled 0.751 | 0.560 |

  (The two bucketings use different cuts, so rows are not exactly matched — the gap is far too large
  to be a cut artifact.) In the crowded bucket the proposal stage puts **0.639** of the doors on the
  table and the pipeline returns **0.262** — a ~41 % transfer, against the ~0.82 measured overall
  before this change. **DINOv2 region embedding + gmm calibration is now the binding constraint on
  crowded plans**, and it — not the proposer — is where the next floor-plan lever should be aimed.
- **`22_png` remains immovable under every lever tried** (482×507, 17 doors): proposal recall 0.059 →
  0.176 at `conf 0.10`, with **274 proposals for 17 doors**. Sub-tile-sized, extremely crowded, and
  responsive to neither budget nor magnification. A plan drowning in proposals is not short of them —
  this is a retrieval/discrimination case, consistent with the bottleneck shift above.

## Verification

All four gates green on the final commit:

- `pixi run lint` — Ruff clean (line-length 100) on `src/` and `tests/`.
- `pixi run typecheck` — MyPy **strict** clean, 81 source files.
- `pixi run test` — full suite green with the **≥ 80 % coverage floor** enforced.
- `pixi run docs-build` — `mkdocs build --strict`.

The shipped change is grid-only and model-free, so it is covered in CI without ONNX weights: a
dedicated test asserts the block is **additive** (the original 12-entry block survives entire, the
shipped `proposal_conf` default stays reachable), that all nine additive cells validate through the
method's own frozen `config_model`, that the measured finalist is present, and that no tiled cell
leaked into the grid.

Measurement runtime: vast.ai contracts `47510440` (lost mid-session, with the entries that preceded
it already pulled back) and `48124756`, both **ONNX Runtime CPU builds** on RTX 3090 hosts. The
guardrail regimes reproduce the committed GPU-era numbers to within rounding, which is what licenses
comparing this session's CPU numbers against the committed table.

---

## Follow-on (2026-08-25) — the retrieval/calibration lead, investigated and closed as a negative result {: #follow-on-260825-calibration }

This pass's own "Measured and deferred" section above named the next lead: with the proposal
stage fixed, the transfer from proposal-stage recall to end-to-end recall had collapsed in the
crowded bucket specifically (11+ doors: proposal recall **0.639** vs end-to-end recall **0.262**,
a ~41% transfer against ~0.82 pooled), attributed to "DINOv2 embedding + gmm calibration" but not
investigated further. A follow-on session did that investigation. Full record:
`.planning/quick/260825-propose-retrieve-calibration-stage/EXPERIMENTS.md`; diagnostic harness:
`scripts/propose_retrieve_calibration_experiment.py` (new, committed).

**Method.** Trace every ground-truth box in every val plan through the actual pipeline
(`propose` → `embed_regions` → `calibration.calibrate` → threshold → `nms.nms`) and classify each
one as `matched` (survives to a true positive), `below_threshold` (a covering proposal exists but
scores below the calibrated cut), `nms_suppressed` (clears the cut but loses post-retrieval NMS),
or `no_proposal` (no covering proposal at all — a proposal-stage fact, out of scope here).

**Finding 1 — the retrieval/calibration stage's own loss rate, isolated from proposal supply,
triples with crowding:** 0.095 (sparse, 1–3 doors) → 0.197 (medium, 4–10) → **0.322** (crowded,
11+), of GT boxes that DO have a covering proposal. `nms_suppressed` is negligible everywhere
(≤2.1%) — NMS is not the mechanism.

**Finding 2 — the gmm's adaptive component is nearly inert on this domain.** The applied threshold
sits within 0.006 of the bare `similarity_floor` (0.70) in every crowding bucket. Re-reading the
method's own threshold logic explains why: a degenerate fit uses the floor directly (discarding
the ratio fallback), and a non-degenerate fit takes `max(gmm_cut, floor)` — the gmm can only ever
raise the threshold above the floor, and empirically it almost never does here. **The fixed floor,
not the gmm cut, is doing nearly all of the deciding** — which reframes "gmm calibration" in this
pass's earlier framing as more precisely "the fixed `similarity_floor` clamp."

**Finding 3 — true/background cosine-score separation compresses with crowding**, the DINOv2
embedding-discriminability signature: background-proposal scores rise (0.452 → 0.513) while
true-positive scores drift down slightly (0.825 → 0.800) from sparse to crowded, shrinking the
margin the floor has to work with by 23% (0.373 → 0.287). The `below_threshold` population itself
scores 0.637–0.656 across every bucket — real door crops the embedding matches only moderately,
not noise.

**The one candidate lever this motivated — `similarity_floor` below its shipped 0.70 at
`proposal_conf=0.10` — was measured and closes the prior report's stated "not measured" gap.**
Three val trials (floor 0.55 / 0.60 / 0.65, same `trial` scorer T3 used) extend that grid
downward:

| `similarity_floor` | val P | val R | val F1 | `16+`-bucket val F1 |
|---|---|---|---|---|
| 0.55 | 0.211 | 0.681 | 0.322 | 0.247 |
| 0.60 | 0.278 | 0.670 | 0.393 | 0.282 |
| 0.65 | 0.387 | 0.632 | 0.480 | **0.310** |
| **0.70 (shipped, T3 argmax)** | **0.526** | 0.560 | **0.542** | 0.282 |

Pooled F1 falls monotonically below 0.70, continuing the same trend T3 already measured above it
— **0.70 is the argmax across the full now-measured {0.55–0.85} range**, closing that limitation
cleanly. `floor=0.65` *does* win the crowded bucket in isolation (F1 0.282 → 0.310, recall
+0.123 at a precision cost of −0.047), but the same move costs the sparse bucket −0.137 F1 and
the medium bucket −0.072 F1 — both far larger plan populations (14 and 36 plans vs 6) — so the
pooled argmax the repo's tuning methodology optimises correctly rejects it. A crowding-conditional
floor was not pursued: it would need to dispatch on ground truth the method cannot observe at
inference time, and would introduce exactly the config-driven dispatch inside a method this
repo's conventions rule out.

**Hypothesis 3 (pre-embedding objectness/top-K filtering) was argued from existing evidence, not
re-measured**: it reduces to re-raising `proposal_conf`, which this pass's own T2 already measured
to hurt recall monotonically (`conf 0.10 > 0.20 > 0.30` at every floor) — a fresh trial would
remove exactly the marginal-confidence proposals `proposal_conf=0.10` was shipped to rescue.

**Verdict: nothing ships.** No `_TUNING_GRIDS` change, no config default change, no code change.
Per the tune-on-val / read-test-once discipline, **test is not read** — no finalist earned it.
The evidence points to DINOv2 embedding discriminability compressing under crowding as the actual
crowded-bucket ceiling, not a miscalibrated threshold — a genuine fix would need an embedding-stage
change (e.g. a crop-context or fine-tuned backbone, in the spirit of
[`owlv2-floorplans-finetune.md`](owlv2-floorplans-finetune.md)'s work for a different method), out
of scope for a threshold/grid-tuning pass and not attempted here.

Measurement runtime: the same vast.ai instance as this report's original pass (`48124756`,
restarted, disk state — models and datasets — reused unchanged), CPU-only ONNX Runtime.
