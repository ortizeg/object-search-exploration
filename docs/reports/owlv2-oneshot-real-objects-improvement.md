# Investigation — `owlv2-oneshot`'s real-objects precision collapse (2026-08-06)

A record of the investigation into `owlv2-oneshot`'s catastrophic real-objects drop measured in
[`real-objects-findings.md`](real-objects-findings.md) (pooled F1 0.637 → 0.103, precision 0.502 →
0.057 — by far the largest synthetic-to-real gap of any of the six methods). This is a
**plan, not a fix**: the root cause is pinned down with measured diagnostics below, remediation
levers are proposed, but none is implemented or validated here.

Cross-reference: the method is documented in
[`../methods/owlv2-oneshot.md`](../methods/owlv2-oneshot.md); the implementation is
`src/object_search/search/owlv2_oneshot.py`; the same precision-collapse *symptom* (recall high,
precision ~0.01–0.11) is independently recorded on a different domain in
[`../eval/floorplans-findings.md`](../eval/floorplans-findings.md), which is corroborating evidence
that this is a property of the method, not an artifact of the `real-objects` set.

## Symptom, precisely

Per-image on `real-objects` (`PLAIN` regime, clean background — rules out clutter as the cause):

| image | gt | pre-NMS accepted | final matches | self-anchored threshold |
|---|---|---|---|---|
| `real-plain-apple` | 6 | 7 | 6 (clean) | **0.713** |
| `real-plain-claw-hammer` | 7 | 7 | 7 (clean) | **0.730** |
| `real-plain-c-clamp` | 5 | 289 | **45** | 0.478 |
| `real-plain-chess-pawn` | 8 | 432 | **101** | 0.574 |
| `real-plain-screwdriver` | 8 | 827 | **252** | 0.523 |

(Measured by re-running `owlv2-oneshot`'s `search()` directly and reading
`SearchResult.diagnostics.metrics` — not in the committed `results.json`, which only carries
tp/fp/fn.)

## Root cause, measured

The default `calibration="self-similarity"` strategy sets `threshold = self_score * retain_frac`
(`retain_frac=0.94`), where `self_score` is the top cosine score among boxes overlapping the
exemplar. **The failing objects have a much lower `self_score`** (~0.51–0.61) than the clean ones
(~0.76–0.78) — the model's own best patch-box match to its own exemplar is a mediocre cosine score
for these three objects. At that lower absolute threshold, hundreds of unrelated scene patches
(background, not the object) clear the bar too:

- Matched box sizes on `screwdriver` range from **9px to 1017px wide** (image width ~1024px) — from
  slivers to nearly the whole frame — versus `apple`'s consistent 145–222px and `claw-hammer`'s
  consistent 171–174px. The accepted set is not "several tight boxes on the object plus a few
  duplicates"; it is boxes scattered at every scale across the whole scene.
- Mean pairwise IoU among the first 15 matches is **0.03–0.07** for the three failing objects
  (`nms_iou=0.3` never fires — nothing overlaps enough to suppress) versus **0.0 with tightly
  consistent sizes** for the clean cases (nothing to suppress because there is exactly one box per
  instance already). NMS is not failing at its job (collapsing duplicate detections of the *same*
  instance); the problem is upstream — hundreds of *distinct, spurious* patches are clearing a
  threshold that was anchored too low to begin with.

**Hypothesis:** `c-clamp`, `chess-pawn`, and `screwdriver` are all objects whose true silhouette
fills a comparatively small fraction of their own axis-aligned bounding box (a C-shape, a narrow
stem-and-base profile, a long thin diagonal tool) — unlike `apple`/`claw-hammer`'s more
box-filling silhouettes. `select_query_embedding` (owlv2_oneshot.py:297) picks the single exemplar
patch box that best covers the *drawn* crop and is most distinctive from the mean; for a
low-fill-ratio object, no single OWLv2 patch prediction tightly matches the drawn box's full
extent, so both the query embedding *and* the `self_score` read-off (also IoU-gated against the
exemplar box) are measurably weaker matches than they are for a box-filling object. A weaker
self-match drags the *entire scene's* threshold down with it (the calibration is anchored to one
number), and at that lower bar, ordinary background patches — which would never pass a threshold
tuned to a strong self-match — start clearing it in bulk.

This is a **single point of failure**: one object-dependent scalar (`self_score`) sets the
threshold for the whole image with no sanity check against how many patches it ends up admitting.

## Proposed remediation levers (not implemented)

1. **A floor tied to the score distribution, not just `self_score`.** Reject a `self-similarity`
   threshold that is anomalously low relative to the scene's own score spread (e.g. require
   `threshold >= scores.mean() + k * scores.std()`, falling back to `gmm`/`ratio` when the
   self-anchor would admit an implausible fraction of patches). Cheap, and directly targets the
   measured mechanism (threshold too low relative to the background score distribution).
2. **A post-NMS box-size-consistency filter.** The accepted set for a genuine multi-instance scene
   should cluster around one box size (same object, same rough scale); `screwdriver`'s 9–1017px
   spread is itself a detectable pathology. A cheap outlier filter (e.g. drop boxes whose area is
   more than Nx off the *median* accepted box area) would have caught this class of failure without
   touching the threshold logic at all — worth measuring as an independent, more surgical fix.
3. **A more robust self-match read-off.** Instead of the single top-scoring box overlapping the
   exemplar, use e.g. the 75th-percentile score among several patches with high IoU to the exemplar
   box, so one favorably-placed patch (or one poorly-placed one) does not single-handedly set the
   anchor.
4. **Revisit query-embedding selection for low-fill-ratio objects** (`select_query_embedding`,
   `query_iou_frac=0.8`): widening or narrowing the IoU-gate for "covering" patches may change which
   patch gets chosen as the query for a C-clamp/screwdriver-shaped object; this needs the same
   measured-before-and-after discipline the existing `owlv2-improvement.md` log used for the
   original query-selection fix.

Any of these should be validated on the **same measurement setup** `real-objects-findings.md`
already used (P/R/F1/AP per regime, before/after, plus the specific `screwdriver`/`chess-pawn`/
`c-clamp`/`ping-pong-ball` per-image rows) so the fix is provably targeted and not a regression
elsewhere — the `claw-hammer`/`apple` clean cases above are the regression check.

## What this is not

Not a claim that `owlv2-oneshot` is broken in general — it is **perfect or near-perfect on 7 of 10
`PLAIN` objects** and competitive through `VARIED`/`CLUTTERED` for round, box-filling shapes. The
finding is narrower and more actionable: the self-similarity calibration has an unguarded failure
mode for a specific, identifiable class of object geometry, and it is severe enough (pooled
precision 0.06) to make the method unusable as shipped against unconstrained real-photo input.
