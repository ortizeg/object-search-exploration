# Method 2 — `sparse-geo` (sparse keypoint matching + geometric verification)

Detect local keypoints on the exemplar crop and on the whole scene, match them **many-to-many**,
then recover **many** geometric models — one per instance — by clustering the matches in pose
space (generalized Hough) and verifying each cluster with its own RANSAC-fitted 4-DoF similarity.
This is Lowe's original multi-object recognition pipeline (IJCV 2004 §7.3).

The module `src/object_search/search/sparse_geo.py` is meant to be read top to bottom; the
numbered steps below match the `# 1.` … `# 9.` comments in `search()` one-for-one (METHOD-11).

## What it is and when it wins

`sparse-geo` targets the case NCC struggles with: instances that are the same object but sit at
**different rotations or scales**. A keypoint frame plus geometric verification is rotation- and
scale-tolerant in a way raw correlation is not.

## The two things easiest to get wrong

1. **The standard Lowe ratio test is DISABLED.** Lowe's ratio test (keep a match only if the best
   neighbour is much closer than the *second* best) exists to reject descriptors with several
   equally-good matches — which is exactly the signature of a repeated instance. Every crop
   keypoint on a repeated object matches one scene keypoint per instance at near-equal distance,
   so the standard ratio test would discard precisely the correspondences we need. We take the
   top-`k` neighbours **unconditionally**; the only optional test is the **k+1 ratio** (compare
   the k-th neighbour to the (k+1)-th), which still rejects a descriptor non-discriminative
   against the whole image while keeping up to `k` repeats. `k` is an explicit **ceiling** on
   findable instances; when the k-th neighbour is still strong, `k_ceiling_hit` records the
   likely truncation.
2. **SuperPoint keypoints carry no scale or orientation**, so single-correspondence 4-DoF voting
   is invalid for that backend. `single-4dof` **raises** on a frameless backend rather than
   silently degrading. Three voting modes exist: `single-4dof` (classical only), `translation-2dof`
   (any backend), `pairwise-4dof` (any backend, O(n²) pairs sampled to a cap).

## Pre-processing (exact)

- The BGR scene is converted **once** to grayscale (`cv2.COLOR_BGR2GRAY`); all three classical
  detectors operate on intensity. Kept **uint8**, C-contiguous. No mean/std normalization —
  each detector normalizes its own descriptors, so a second normalization would only
  desynchronise the crop and scene descriptors.
- The exemplar crop is `gray[y:y2, x:x2]`; its keypoints are shifted by the crop origin `(x, y)`
  so every coordinate lives in **scene** pixels.
- Backend defaults to **SIFT**, not ORB (research measured ORB yielding ~1 keypoint where SIFT
  yields 83 on a 64×64 crop). The descriptor **distance metric is a property of the backend**,
  never a config field: SIFT and AKAZE (configured for its float KAZE descriptor) are L2; ORB is
  binary Hamming.

## Post-processing (exact)

- Votes are cast in 4-DoF pose space `(centre_x, centre_y, log_scale, theta)` with **soft
  binning** (2 nearest bins per dimension, 16 in 4-DoF) and **circular theta** (a vote near 0/360°
  reaches the adjacent bin, not the opposite end). Bin widths are Lowe's §7.3 values: **30°**
  orientation, **factor 2** scale, **0.25 × max projected crop dimension** location. The location
  bin width is **scale-dependent**, so votes live in a **dict keyed by the bin tuple**, not a
  dense array.
- Each peak (≥ `min_votes` weight, de-duplicated against its 3⁴ neighbourhood) is verified by a
  **NumPy** RANSAC seeded from `np.random.default_rng(config.seed)` — **not** `cv2.setRNGSeed`,
  which has no effect on OpenCV RANSAC. A peak is accepted at ≥ `min_inliers` inliers.
- Degeneracy rejection uses **scale plausibility** and **mirror rejection** (negative determinant
  of the fitted linear part). Shear and aspect distortion are deliberately NOT tested — a 4-DoF
  similarity has neither by construction, so those tests are vacuous.
- **METHOD-12**: multiple distinct models are returned; there is no single-best short-circuit.

## Algorithm

### 1. Grayscale and build the backend

Convert once to grayscale and construct the requested detector (SIFT default). The backend fixes
the descriptor distance metric.

### 2. Detect and describe on the crop and scene

Detect keypoints on the crop and on the full scene with the same backend; crop coordinates are
shifted into scene pixels.

### 3. Low-keypoint guard (METHOD-04c)

If the crop yields fewer than `min_exemplar_keypoints`, return `outcome=EMPTY` **with a
diagnostic note** — the cause is insufficient texture, not "found nothing", and abstaining
legibly beats a silent empty result.

### 4. Many-to-many top-k matching — ratio test disabled

For each crop keypoint take its top-`k` scene neighbours unconditionally. Optionally apply the
k+1 ratio only. Record `k_ceiling_hit` when the ceiling likely truncated instances.

### 5. Cast pose votes

Turn each correspondence into a 4-DoF pose vote under the selected mode. `single-4dof` uses the
keypoint frame and **raises** on a frameless backend; `translation-2dof` votes in `(Δx, Δy)`;
`pairwise-4dof` fits a similarity per correspondence pair, sampling up to `pairwise_cap` (recorded
in diagnostics).

### 6. Decompose into instance hypotheses

`hough` accumulates the votes (soft binning, circular theta, scale-dependent location bins in a
hash table) and enumerates peaks; `sequential-ransac` (METHOD-04b) is the pluggable alternative —
fit the dominant model, remove its inliers, repeat — behind the same interface.

### 7. Per-peak RANSAC and degeneracy rejection

Fit a 4-DoF similarity per peak with the seeded NumPy RANSAC (2-point minimal samples, proper and
reflected candidates). Reject on mirror (negative determinant) or implausible scale; accept at
≥ `min_inliers` inliers.

### 8. Box mapping, exemplar self-match, diagnostics

Transform the exemplar box corners by each fitted model to an axis-aligned scene box. The
best-overlapping instance above the IoU floor is labelled `is_exemplar=True` — a genuine instance,
neither dropped nor double-counted. Sub-threshold peaks are kept as candidates (vote weight as
score) for EVAL-08. Diagnostics carry the correspondences and Hough peaks the overlay renders.

### 9. Assemble the result

Return every verified instance as a `Match` (with its fitted 2×3 transform). METHOD-12: multiple
distinct models, no single-best short-circuit.

## Known failure modes

- **Textureless / low-keypoint crop.** Handled by the step-3 guard: `outcome=EMPTY` with a note.
- **Small near-identical instances (the NCC crossover).** When instances are small and nearly
  identical, almost every tentative match is wrong and Hough's discriminative power is
  insufficient — and that is exactly the regime where **Method 1 (NCC) is strongest**. This is an
  **expected finding, not a bug**; it is a large part of why four methods exist, and the Phase 8
  benchmark should *demonstrate* the crossover rather than hide it.
- **ORB on tiny crops.** ORB's keypoint yield collapses on small crops; SIFT is the default for
  this reason.

## ROBUSTNESS BACKLOG

Deferred deliberately (mirrored from the module docstring); none is built in this phase:

- **Multi-model fitting (J-linkage / T-linkage)** as a third decomposition strategy alongside
  Hough voting and sequential RANSAC.
- **DISK / ALIKED backends** — additional learned detectors and the permissive-licence escape
  from SuperPoint's non-commercial terms.
- **Post-hoc orientation/scale assignment for frameless keypoints** via local gradient histograms,
  which would unlock `single-4dof` voting for a learned backend.
- **LoFTR / RoMa dense matching** with correspondence-field clustering for low-texture objects
  (a research spike — the ONNX export is awkward).
