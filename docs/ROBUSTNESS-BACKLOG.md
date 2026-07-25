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
