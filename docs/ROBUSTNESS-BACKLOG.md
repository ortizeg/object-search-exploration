# Robustness backlog

Deferred robustness work, one section per search method (DOC-05). Each entry is a deliberate
"not now" — a known way to make a method stronger that was scoped out to keep the method
readable and the phase small. Each method's section is mirrored verbatim from that method's
module docstring `ROBUSTNESS BACKLOG` block, so the two cannot drift.

This file grows as methods land; it completes in Phase 8 when every registered method has a
section here.

## `ncc` (Method 1 — normalized cross-correlation)

None of the following is built in Phase 2; all are captured here and in the `ncc.py` docstring.

- **FFT-based correlation for large templates.** The spatial `matchTemplate` is O(H·W·h·w); a
  single full-scene FFT cross-correlation is O(H·W·log(H·W)) and wins decisively once the
  template is large.
- **Log-polar / Fourier-Mellin registration** for joint rotation+scale invariance in one
  correlation, replacing the brute-force rotated-template × pyramid bank.
- **Discriminative correlation filters (MOSSE/KCF)** trained on the single exemplar crop, so
  the filter learns to suppress background instead of correlating raw pixels.

> **Realized by the [`mosse`](methods/mosse.md) method** (see
> [`reports/mosse-improvement.md`](reports/mosse-improvement.md)): the first two items — **FFT-based
> correlation** and **discriminative correlation filters (MOSSE/ASEF)** — are now a separate,
> registered method rather than a change to `ncc` (so the spatial-NCC crossover baseline stays
> intact for a fair head-to-head). The **log-polar / Fourier-Mellin** item remains deferred and is
> carried forward in `mosse`'s own section below.

## `mosse` (MOSSE/ASEF correlation-filter matching via FFT)

None of the following is built in this spike; all are captured here and in the `mosse.py` docstring
and [`methods/mosse.md`](methods/mosse.md) (mirrored so the three cannot drift).

- **Log-polar / Fourier-Mellin front end** so one correlation spans rotation **and** scale,
  retiring the scale pyramid entirely (the rotation bank is already folded into the filter).
- **A dedicated DSST-style scale filter** — a separate 1-D correlation filter over a scale pyramid
  of the peak patch — for continuous scale estimation instead of the discrete pyramid.
- **OTSDF / UMACE variants** exposing an explicit sharpness-vs-noise trade-off parameter, for scenes
  where the MOSSE default is either too sharp (misses poses) or too broad (clutter FPs).
- **Kernelized correlation filters (KCF)** — a non-linear kernel in the closed-form solve, more
  discriminative against structured background than the linear MOSSE filter here.

## `sparse-geo` (Method 2 — sparse keypoint matching + geometric verification)

None of the following is built in Phase 5; all are captured here and in the `sparse_geo.py`
docstring (mirrored verbatim so the two cannot drift).

- **Multi-model fitting (J-linkage / T-linkage)** as a third decomposition strategy alongside
  Hough voting and sequential RANSAC.
- **DISK / ALIKED backends** — additional learned detectors and the permissive-licence escape
  from SuperPoint's non-commercial terms.
- **Post-hoc orientation/scale assignment for frameless keypoints** via local gradient
  histograms, which would unlock `single-4dof` voting for a learned backend.
- **LoFTR / RoMa dense matching** with correspondence-field clustering for low-texture objects
  (a research spike — the ONNX export is awkward).

## `dino-dense` (Method 3 — DINOv2 dense-token best-part matching)

Captured here and in the `dino_dense.py` docstring. The mean-pooled-prototype and thresholding
weaknesses that shipped a single full-frame box were fixed in 2026-07 (`max-token` scoring +
`contrast` calibration + threshold-level extraction — see
[`reports/dino-dense-improvement.md`](reports/dino-dense-improvement.md)); what remains deferred:

- **Sliding-window backbone inference — tried on floorplans-door, REGRESSED at every tile size
  tested** (784px → F1 0.053, 1120px → 0.078, 1568px → 0.072, all below the adaptive+letterbox
  winner's 0.144). Hypothesis: DINOv2 needs the whole scene in one forward pass for its self-
  attention to work; independently-processed tiles lose that global context. Reverted — see
  [`reports/dino-dense-floorplans-improvement.md`](reports/dino-dense-floorplans-improvement.md).
- **Adaptive input resolution — LANDED (opt-in) for floorplans-door, still deferred for chipset.**
  `adaptive_min_exemplar_tokens` + `adaptive_max_side` size the scene so the exemplar spans ≥ N
  stride-14 tokens, instead of a fixed `scene_max_side`. On chipset it measured ~6× recall
  (0.077 → 0.554 on a small-chip subset) but remains deferred there: it fixes recall but not the
  flat-chip precision, costs inference latency, and chipset is NCC's regime. On floorplans-door,
  paired with a matching `fixed_input_side` letterbox, it lifted test F1 0.117 → 0.144 and shipped
  as an opt-in domain override — see
  [`reports/dino-dense-floorplans-improvement.md`](reports/dino-dense-floorplans-improvement.md).
- **Learned feature upsampling (FeatUp)** to recover sub-patch localisation from the stride-14
  grid without a full high-res forward pass.
- **SAM-based box refinement** — snap each coarse component box to the nearest segment mask. A
  cheaper peak/centroid-centred exemplar-shaped-box variant (no SAM) was tried on floorplans-door
  and regressed F1; a real segmentation-based refinement remains open.
- **Spatially-structured (not order-free) part matching.** `max-token` already does many-to-many
  token similarity (DONE — it replaced the mean-pooled prototype and lifted textured F1 from ≈ 0.03
  to ≈ 0.70), but it pools the top-k cosines with no geometric constraint on *where* the matching
  parts sit. A spatial-consistency term would cut clutter false positives further — and is the most
  promising lever for the flat-chip precision the resolution fix leaves untouched.
- **DINOv3 backbone swap** once a clean ONNX export exists.

## `propose-retrieve` (Method 5 — class-agnostic proposals + DINOv2 region embeddings)

None of the following is built in Phase 7; all are captured here and in the `propose_retrieve.py`
docstring (mirrored verbatim so the two cannot drift).

- **FAISS index for corpus-scale retrieval** — unnecessary for a few hundred proposals in one image;
  the `(N, D)` embedding matrix is shaped so it slots in when corpus search arrives.
- **Background-masked region embedding** — embed the FastSAM mask interior rather than the raw box
  crop; the mask is already produced, so this is cheap and likely a real accuracy win.
- **Proposal filtering by an exemplar size/aspect prior** — drop proposals whose shape cannot match
  the exemplar before embedding, cutting both cost and false positives.
- **Multi-crop / test-time augmentation embeddings** for pose-robust region descriptors.
- **Alternative proposal sources (RPN, selective search)** for images where SAM over-segments.
- **MobileSAM everything-mode** with a ported `SamAutomaticMaskGenerator` as a second backend.

## `owlv2-oneshot` (Method 4 — OWLv2 image-conditioned one-shot detection)

Captured here and in the `owlv2_oneshot.py` docstring (mirrored verbatim so the two cannot drift).
This realizes the previously-deferred source-research Method 4 (exemplar-conditioned detectors) with
a permissive, ONNX-exportable model after T-Rex2 / Rex-Omni were rejected on licensing.

- **Tiled / multi-scale inference** to lift small-object recall on large canvases past the fixed
  960 input. **This is the primary remaining weakness**: the EASY (chipset) regime downscales chips
  in a 6000×4000 scene below OWLv2's effective resolution, so precision stays ~0.08 there while the
  textured regimes reach F1 0.78–0.83.
- **Export OWLv2's learned `logit_scale` / `logit_shift`** and apply them before thresholding, so
  scene scores are the model's calibrated logits rather than raw (compressed) cosine — may make the
  distribution genuinely bimodal and remove the need for self-similarity anchoring.
- **Text-prompt fusion** — OWLv2 also takes text queries; combining the drawn exemplar with an
  optional label would use both modalities (the exploration's Milestone 2 seam).
- **Query embedding from multiple exemplars** — average several drawn boxes for a more robust query.
- **owlv2-large** for accuracy at higher latency, gated behind the same export path.
- **Runtime-verify the OWLv2 contract in `.planning/research/MODELS.md`** (the sha256 is now pinned
  from the first verified export).

> Implemented in the improvement pass (`docs/reports/owlv2-improvement.md`): the HF
> `embed_image_query` distinctiveness selection (was a correctness bug), the whole-frame-box filter,
> and self-similarity calibration.

## `marker-conditioned` (Exploration 2 — marker → pointed-at object, Milestone 2)

None of the following is built in Milestone 2; all are captured here and in the
`marker_conditioned.py` docstring (mirrored so the two cannot drift).

- **A learned marker / keypoint tip detector**, replacing the arrowhead-mass heuristic, so the tip is
  found directly rather than inferred from foreground-mass asymmetry — the heuristic is 180°-ambiguous
  on short or near-symmetric arrowheads and abstains there rather than guess.
- **Per-proposal appearance matching** via the already-shipped `embed_regions()`, so the chosen
  proposal must also *look* like the pointed-at object, not merely sit near the pointing ray.
- **Multi-marker disambiguation when arrows cross** — a global assignment (each proposal to at most
  one marker) instead of scoring every proposal against every marker independently, which lets two
  close markers double-claim one object.
- **A dedicated marker detector** trained on marker gestures, instead of reusing a Milestone 1 method
  whose invariances were designed for whole objects — the demo shows the cost of the mismatch:
  `ncc` is not rotation-invariant, and classical `sparse-geo` abstains on the low-texture synthetic
  arrows, so no shipped finder resolves all randomly-rotated arrows.
- **Ray-to-box distance and a per-marker size prior** in the scorer, replacing proposal-centre
  distance and a single global `size_prior_frac`, so an elongated object off to one side of the
  pointing ray is not unfairly penalised and over-segmented proposals are handled better.

## Cross-cutting — applies to every method

These are not any single method's backlog; they are structural robustness work and deferred
whole methods, recorded here so the reasoning survives (IDEA.md §12).

### Lattice fitting as post-detection verification (highest leverage)

**Not built. Likely the single highest-leverage robustness item for the shelf / PCB / tile /
lattice images**, and the reason those scenes are in the demo set.

Every method above treats each instance independently. But the objects being hunted are
overwhelmingly arranged on a **regular grid** — chips on a PCB, products on a shelf, tiles on a
floor. A grid is a strong, cheap prior that none of the detectors currently uses:

- **Fit a 2-D lattice** (two basis vectors + an origin) to the *set* of accepted detections by
  robust voting on the pairwise offset vectors between matches. A handful of confident detections
  is enough to recover the lattice.
- **Recover misses:** every lattice cell with no detection is a predicted location. Re-score just
  those cells (a cheap local NCC or embedding check) to recover instances the detector dropped —
  directly attacking the recall collapse `sparse-geo` and `dino-dense` show on the chipset.
- **Kill false positives:** a "detection" that sits off-lattice, where no grid cell predicts an
  instance, is almost certainly spurious and can be dropped.

This recovers misses and removes false positives *more effectively than tuning any single
detector's threshold*, because it adds information (the arrangement) rather than trading precision
against recall on the same per-instance score. It is method-agnostic — it consumes the accepted
match set and would sit in `search/common/` as an optional post-verification offering, wired into
whichever method opts in. Deferred from Milestone 1 to keep each method readable and self-contained.

### Deferred whole methods (IDEA.md §12)

**Method 4 — exemplar-conditioned detectors and counters.** _Partially realized_ by
`owlv2-oneshot` (see its section above): OWLv2 is the permissive (Apache-2.0), ONNX-exportable
exemplar detector that fills this bucket. The originally-named candidates remain **deferred/rejected
on the same grounds that motivated OWLv2**: **T-Rex2** is API-only (no downloadable weights) and
IDEA License 1.0 non-commercial; **Rex-Omni** is a 3B PyTorch MLLM (no ONNX path) under IDEA +
Qwen research licences; both conflict with the local-first, ONNX-first, no-cloud, permissive-licence
constraints (full comparison in `docs/library-reviews/owlv2.md`). Still open in this bucket:
few-shot **counters** (FamNet, BMNet+, SAFECount, CounTR, LOCA, CACViT) and **CountGD**. This corner
of the field moves fast; re-check what has landed and what has a clean ONNX export before adding
another.

**Method 6 — one-shot personalized segmentation.** PerSAM / PerSAM-F, Matcher, SegGPT / Painter,
SAM 2 memory-bank propagation. **Deferred because** Milestone 1's output contract is boxes, not
masks. It becomes cheap once Method 5's FastSAM proposal stage exists (the masks are already being
produced), which makes it a natural Milestone 3 rather than a Milestone 1 method.
