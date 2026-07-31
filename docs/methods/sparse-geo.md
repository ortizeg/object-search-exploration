# Method 2 — `sparse-geo` (sparse keypoint matching + geometric verification)

Detect local keypoints on the exemplar crop and on the whole scene, match them **many-to-many**,
then recover **many** geometric models — one per instance — by clustering the matches in pose
space (generalized Hough) and verifying each cluster with its own RANSAC-fitted 4-DoF similarity.
This is Lowe's original multi-object recognition pipeline (IJCV 2004 §7.3).

The module `src/object_search/search/sparse_geo.py` is meant to be read top to bottom; the
numbered steps below match the `# 1.` … `# 9.` comments in `search()` one-for-one (METHOD-11).

> How the current defaults were arrived at — the experiments run, kept, and reverted, with
> before/after benchmark numbers — is recorded in the [tuning log](sparse-geo-tuning.md).

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

## Backends — two behind one interface, switched by config alone

`sparse-geo` runs one of two backend families through the **same** code path; the only two
things the rest of the module needs from a backend are its descriptor **distance metric** and
whether its keypoints carry a **geometric frame** (scale + orientation). Switching backend is
config alone — no separate code path.

| `backend` | Kind | Metric | Frame? | Default voting mode |
|---|---|---|---|---|
| `sift` (default) | classical (no weights) | L2 | yes | `single-4dof` |
| `akaze` | classical (float KAZE descriptor) | L2 | yes | `single-4dof` |
| `orb` | classical (binary descriptor) | Hamming | yes | `single-4dof` |
| `superpoint` | learned (ONNX) | L2 (descriptors pre-normalized) | **no** | `translation-2dof` |

The descriptor **distance metric is a property of the backend, never a config field** — getting
it wrong yields garbage matches that still *look* like matches. Because SuperPoint keypoints are
**frameless**, selecting `voting_mode="single-4dof"` with `backend="superpoint"` is rejected at
config-construction time (METHOD-04a) rather than silently degraded; its working default is
`translation-2dof`.

## Pre-processing (exact) — classical backends

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

## Pre-processing (exact) — SuperPoint (learned) backend

Full contract in `src/object_search/inference/superpoint.py` and
`docs/library-reviews/superpoint.md` (verdict **Trial**). The exact steps (stated per the
project's explicit-preprocessing constraint, exact numbers not "standard normalization"):

- **Input** `image`, f32, NCHW, `[1, 1, H, W]` — batch fixed at 1, a **single grayscale
  channel**, H/W dynamic.
- **Colour**: BT.601 luma `gray = 0.299·R + 0.587·G + 0.114·B`, which the shared
  `cv2.COLOR_BGR2GRAY` reproduces (METHOD-11: the equivalence is written down because the two
  paths differ in rounding and can shift a borderline keypoint). The scene is grayscaled once and
  the same single-channel crop/scene are handed to SuperPoint.
- **Range [0, 1]** via `/255`, and **NO mean subtraction, NO std division** — SuperPoint wants
  raw luma; applying ImageNet normalization here silently wrecks detection.
- **Pad to a multiple of 8, do not resize.** Non-multiple sides are silently floored (trailing
  rows/columns dropped — a coordinate truncation), so the inferencer zero-pads bottom/right to
  the next multiple of the stride 8. Padding on the far edges preserves the top-left origin, so
  keypoint coordinates need no remapping.
- Keypoints are shifted by the crop origin into **scene** pixels, exactly as the classical path.

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
- **Duplicate suppression.** Hough de-duplicates in *pose* space (the 3⁴ neighbourhood), but two
  peaks in genuinely different pose bins can still map the exemplar to nearly the **same scene
  box** — a duplicate the benchmark scores as 1 TP + 1 FP (EVAL-16). A final IoU NMS
  (`nms_iou`, keyed on inlier support) drops the weaker of each overlapping pair. The shared
  deterministic `nms` offering is used for its `(-score, y, x)` tie-break, so the symmetric-lattice
  case (all scores tie exactly) stays byte-identical across runs.
- **METHOD-12**: multiple distinct models are returned; there is no single-best short-circuit.

### SuperPoint output decoding (exact)

- **Outputs** `keypoints` **int64** `[1, N, 2]` `(x, y)` in input pixels, `scores` f32 `[1, N]`,
  `descriptors` f32 `[1, N, 256]`. All three share one symbolic `N`, so lengths always agree; the
  batch dimension (fixed 1) is dropped.
- **Descriptors are already L2-normalized** (measured ‖d‖ = 1.0000) — **do not re-normalize**.
  kNN is therefore a plain matmul: cosine `= D_crop @ D_scene.T`, squared-L2 `= 2 − 2·cos`. The
  backend metric is `l2` because on unit vectors L2 and cosine agree monotonically.
- **Keypoints carry no scale/orientation** (frameless), so `scale`/`angle` are `None` — which is
  exactly what makes `single-4dof` raise and `translation-2dof` the default for this backend.
- **Effective border is 8 px**, not the configured `remove_borders=4` (the border mask is applied
  on the 8×-upsampled score grid), so no correspondence lands within 8 px of the scene edge —
  relevant when an instance is clipped by the frame.
- **`N` is genuinely variable** (unbounded on textured scenes), which is what lets the
  low-keypoint guard (METHOD-04c) read `keypoints.shape[1]` directly; `pairwise-4dof` caps its
  O(N²) pair sampling via `pairwise_cap`.

## Config reference

The frozen `SparseGeoConfig` (its JSON Schema drives the UI form — one source of truth for
defaults, ranges, and help text):

| field | type / range | default | purpose |
|---|---|---|---|
| `backend` | `sift`/`akaze`/`orb`/`superpoint` | `sift` | detector/descriptor; metric + frame fixed by it |
| `k` | int ≥ 1 | 8 | top-k scene neighbours kept per crop keypoint (ratio test disabled) |
| `use_kplus1_ratio` | bool | `false` | enable the only ratio test available (k-vs-(k+1)) |
| `kplus1_ratio` | 0 < f ≤ 1 | 0.9 | drop a crop keypoint when dist(k) ≥ ratio·dist(k+1) |
| `voting_mode` | `single-4dof`/`translation-2dof`/`pairwise-4dof` | `single-4dof` (classical), `translation-2dof` (superpoint) | how a correspondence becomes a pose vote |
| `decomposition` | `hough`/`sequential-ransac` | `hough` | cluster-into-instances strategy |
| `min_votes` | int ≥ 1 | 2 | min accumulated bin weight to hypothesize a cluster (a cheap pre-filter before RANSAC) |
| `min_inliers` | int ≥ 2 | 5 | min RANSAC inliers to accept a verified instance |
| `pairwise_cap` | int ≥ 1 | 20000 | cap on sampled pairs for `pairwise-4dof` (O(n²)) |
| `min_exemplar_keypoints` | int ≥ 1 | 8 | below this the crop abstains WITH a note (METHOD-04c) |
| `ransac_iters` | int ≥ 1 | 200 | RANSAC iterations per peak (2-point samples) |
| `ransac_thresh_px` | f > 0 | 3.0 | inlier reprojection-error threshold in scene pixels |
| `nms_iou` | 0 < f ≤ 1 | 0.4 | duplicate-instance suppression: drop a box overlapping a stronger one by more than this IoU |
| `allow_mirror` | bool | `false` | accept MIRRORED instances: voting also casts a reflected pose hypothesis AND the `det < 0` gate stops rejecting. Measured **inert under `single-4dof`** (SIFT orientations are not mirror-consistent); effective only with `pairwise-4dof` |
| `min_scale` / `max_scale` | f > 0 | 0.2 / 5.0 | scale-plausibility bounds for degeneracy rejection |
| `seed` | int ≥ 0 | 0 | the REAL seed for `np.random.default_rng` (NOT `cv2.setRNGSeed`) |

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

## Pseudocode

**Method ② sparse-geo** — second of the four *implemented* methods (implementation numbering
①–④: `ncc`, `sparse-geo`, `dino-dense`, `propose-retrieve`; source-research numbering 1, 2, 3, 5,
with research Methods 4 and 6 deferred). The steps below mirror the `# 1.` … `# 9.` comments in
`search()` (METHOD-11); read `src/object_search/search/sparse_geo.py` for the ground truth.

```
1. gray <- BGR2GRAY(scene) once; build backend (SIFT default)  # backend fixes metric + frame?

2. detect+describe on the crop and on the FULL scene with the same backend
   shift crop keypoints by the crop origin (x, y) into scene pixels

3. if #crop_keypoints < min_exemplar_keypoints:      # low-keypoint guard (METHOD-04c)
       return EMPTY with a diagnostic note

4. for each crop keypoint: take its top-k scene neighbours UNCONDITIONALLY  # Lowe ratio DISABLED
       optionally apply only the k-vs-(k+1) ratio
   record k_ceiling_hit when the k-ceiling likely truncated instances

5. cast each correspondence -> a 4-DoF pose vote (cx, cy, log_scale, theta) under voting_mode:
       single-4dof       -> uses the keypoint frame; RAISES on a frameless (SuperPoint) backend
       translation-2dof  -> votes in (dx, dy)
       pairwise-4dof     -> similarity per correspondence pair, sampling up to pairwise_cap

6. decompose into instance hypotheses:
       hough: soft-bin votes (2 bins/dim = 16 in 4-DoF, circular theta, 30 deg / factor-2 /
              0.25*max-crop-dim scale-dependent location bins) in a dict keyed by bin tuple;
              enumerate peaks with weight >= min_votes, de-duped against the 3^4 neighbourhood
       (or sequential-ransac: fit dominant model, remove inliers, repeat)

7. for each peak: NumPy RANSAC seeded by np.random.default_rng(seed)  # NOT cv2.setRNGSeed
       2-point minimal samples (proper + reflected candidates), ransac_iters iterations
       reject on mirror (det < 0) or implausible scale (< min_scale or > max_scale)
       accept at >= min_inliers inliers

8. transform the exemplar box corners by each fitted model -> axis-aligned scene box
   IoU NMS over the verified boxes (nms_iou, keyed on inliers) -> drop duplicate detections
   label the best-overlapping instance is_exemplar=True; keep sub-threshold peaks as candidates

9. return every verified instance as a Match (with its fitted 2x3 transform)  # METHOD-12: no single-best
```

## References

- Lowe, "Distinctive Image Features from Scale-Invariant Keypoints", IJCV 2004 (§7.3 Hough clustering): https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf
- DeTone et al., "SuperPoint: Self-Supervised Interest Point Detection and Description", CVPRW 2018: https://arxiv.org/abs/1712.07629
- Fischler & Bolles, "Random Sample Consensus (RANSAC)", 1981: https://doi.org/10.1145/358669.358692
- Ballard, "Generalizing the Hough Transform to Detect Arbitrary Shapes", 1981: https://doi.org/10.1016/0031-3203(81)90009-1
- Rublee et al., "ORB: an efficient alternative to SIFT or SURF", ICCV 2011: https://doi.org/10.1109/ICCV.2011.6126544
- LightGlue-ONNX (standalone SuperPoint export used here): https://github.com/fabio-sim/LightGlue-ONNX
