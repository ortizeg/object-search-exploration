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

## `dino-dense` (Method 3 — DINOv2 dense-token prototype matching)

None of the following is built in Phase 6; all are captured here and in the `dino_dense.py`
docstring.

- **Sliding-window backbone inference** for very large scenes, so localisation no longer
  degrades at the resolution cap.
- **Learned feature upsampling (FeatUp)** to recover sub-patch localisation from the stride-14
  grid without a full high-res forward pass.
- **SAM-based box refinement** — snap each coarse component box to the nearest segment mask.
- **Many-to-many token similarity with spatial aggregation** instead of a single mean-pooled
  prototype — measurably better for articulated objects like the basketball frames.
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

**Method 4 — exemplar-conditioned detectors and counters.** Open-vocabulary / exemplar
detectors (T-Rex2, CountGD) and few-shot counters (FamNet, BMNet+, SAFECount, CounTR, LOCA,
CACViT). T-Rex2 is arguably the closest off-the-shelf fit to the "draw one box, get the rest"
workflow. **Deferred because** the weights are heavy and licence-encumbered, ONNX export is not a
solved path, and several are API-gated — all three conflict with the local-first, ONNX-first,
no-cloud constraints. This corner of the field moves fast; re-check what has landed and what has a
clean ONNX export before committing to any one.

**Method 6 — one-shot personalized segmentation.** PerSAM / PerSAM-F, Matcher, SegGPT / Painter,
SAM 2 memory-bank propagation. **Deferred because** Milestone 1's output contract is boxes, not
masks. It becomes cheap once Method 5's FastSAM proposal stage exists (the masks are already being
produced), which makes it a natural Milestone 3 rather than a Milestone 1 method.
