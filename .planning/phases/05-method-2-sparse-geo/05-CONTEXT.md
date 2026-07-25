# Phase 5 Context — Method 2 (`sparse-geo`)

**Source:** `.planning/IDEA.md` §5 Method 2 in full — including §2a, §2b, §2c, §2d, which the
brief marks as **requirements, not notes** — plus `.planning/research/PITFALLS.md` (Hough binning,
RANSAC determinism, ORB keypoint yield) and `.planning/research/MODELS.md` (SuperPoint contract).

## Domain

Keypoints on the crop, matched into the scene, then **many** geometric models recovered rather
than one. This is Lowe's original multi-object recognition pipeline (IJCV 2004 §7.3): Hough
clustering in pose space, then per-cluster affine verification. It is proven for exactly this
task, and four details decide whether it works — each is a place a naive implementation silently
fails.

## Locked Decisions

### The two things that are easiest to get wrong

1. **The standard Lowe ratio test is DISABLED.** It exists specifically to reject matches that
   have multiple good candidates, and the literature credits it with reducing wrong registrations
   *caused by repetitive structures*. Repeated instances produce exactly that signature: every
   crop keypoint has N near-equal scene matches, one per instance. Applying it would discard every
   correspondence we need.
   - Take the top-k scene neighbours **unconditionally** (k ≈ 5–10).
   - The **only** ratio test available is the optional **k+1 ratio**: compare the k-th neighbour's
     distance to the (k+1)-th. This keeps up to k repeated instances while still rejecting
     descriptors that are non-discriminative against the whole image.
   - `k` is therefore an explicit **ceiling on findable instances**. When the k-th neighbour is
     still a strong match, instances were probably truncated — surface that in diagnostics as a
     `k_ceiling_hit` metric.
   - A test must prove the standard ratio test *would have* suppressed the repeated instances that
     the k+1 variant keeps. That is a Phase 5 success criterion, not a nice-to-have.

2. **SuperPoint keypoints carry no scale or orientation, so single-correspondence 4-DoF voting is
   invalid for that backend.** Single-correspondence voting works only because a SIFT keypoint
   carries a full geometric frame `(x, y, scale, orientation)`, so one match determines a
   similarity transform. SuperPoint produces pixel-located detections at a fixed 8× stride with
   256-D descriptors and nothing else. Three voting modes, selected by config:

   | Mode | Valid backends | How a vote is formed |
   |------|----------------|----------------------|
   | `single-4dof` | SIFT / AKAZE / ORB only | One correspondence → full similarity transform → votes for the object centre. Lowe's original, free. |
   | `translation-2dof` | any (SuperPoint default) | Vote in `(Δx, Δy)` only, assuming instances share the exemplar's scale and rotation. Correct and fast for the near-identical case. |
   | `pairwise-4dof` | any (SuperPoint, full) | Each **pair** of correspondences determines a 4-DoF similarity. Sample pairs up to a cap. Recovers scale/rotation without keypoint frames, at O(n²) sampled cost. |

   Selecting `single-4dof` with the SuperPoint backend must **raise a clear error**, not silently
   degrade. A config that is accepted and then quietly does something else is worse than a refusal.

### Everything else

3. **Two backends behind one interface.** Classical (SIFT / AKAZE / ORB — no weights, no ONNX)
   ships first; SuperPoint ONNX second. Both run through the same code path and are switched by
   config alone. The backend boundary returns `(keypoints, descriptors)` where a keypoint carries
   optional `scale`/`angle` — `None` for SuperPoint, which is exactly what makes mode validation
   possible.
4. **SIFT is the default classical detector, not ORB.** Research measured ORB yielding **1
   keypoint where SIFT yields 83** on the same 64×64 crop. On the small crops this project deals
   in, ORB routinely falls below the vote floor, which would make the method look broken when the
   detector is the problem.
5. **Soft binning is required.** Votes near a bin boundary otherwise split across bins and no peak
   clears the floor. Vote into the 2 nearest bins per dimension — 16 bins per vote in 4-DoF
   (Lowe's fix).
6. **Verified Lowe bin widths** (from IJCV 2004 **§7.3**, not §7): **30°** for orientation, a
   **factor of 2** for scale, and **0.25 × the max projected training-image dimension** for
   location, with **16** hash entries per vote and **≥3** votes per bin to hypothesize a cluster.
   The usually-dropped detail: **the location bin width is scale-dependent**, which is why Lowe
   uses a hash table rather than a dense array. Implement it as a dict keyed by the bin tuple.
7. **θ wraps circularly.** A vote near 0°/360° must land in adjacent bins, not opposite ends of
   the histogram. This is a classic silent bug — bin θ modulo 360 and make the soft-binning
   neighbour wrap.
8. **Per-peak RANSAC, implemented in NumPy — not via `cv2.setRNGSeed`.**
   `cv2.setRNGSeed` does **not** affect RANSAC: OpenCV hardcodes `RNG rng((uint64)-1)` in
   `ptsetreg.cpp` (lines 171/284), deliberately per a maintainer statement, and `theRNG()` is
   thread-local. OpenCV's RANSAC is therefore already deterministic but its seed is **not
   user-controllable**.
   Consequence: **do not add a `ransac_seed` config field wired to OpenCV** — an advertised,
   inert control is worse than none. Implement the 4-DoF similarity RANSAC in NumPy with
   `np.random.default_rng(config.seed)`. From 2-point samples this is ~30 readable lines, it makes
   the sampling visible in the one file the reader is meant to read, and the seed is real.
   `cv2.estimateAffinePartial2D` may still be used as an optional cross-check, documented as
   having a fixed, non-configurable internal seed.
9. **Minimum evidence:** ≥3 votes to hypothesize a cluster, ≥4–6 RANSAC inliers to accept.
10. **Degeneracy rejection** before a fitted transform becomes a box: discard extreme shear,
    extreme aspect distortion, or near-zero determinant. Compute these from the 2×3 affine.
11. **Exemplar self-match is labelled, not discarded or double-counted.** If the crop comes from
    the scene, its keypoints match themselves and produce an identity-transform peak. That is a
    true instance — set `is_exemplar=True`.
12. **Low-keypoint guard (METHOD-04c).** Below ~20 exemplar keypoints, emit an explicit diagnostic
    that the method is unreliable on this crop. **Never silently return an empty result** when the
    real cause is insufficient texture. SuperPoint's variable keypoint count makes this easy —
    read `keypoints.shape[1]` directly, which is why the variable-length export is preferred over
    a fixed top-K one.
13. **Sequential RANSAC is a pluggable alternative to Hough voting** (METHOD-04b), behind the same
    interface: fit the dominant model, remove its inliers, repeat until the inlier count falls
    below threshold. Mirrors the `calibration.py` / `peaks.py` selectable-strategy pattern.

## Canonical References

- `.planning/research/MODELS.md` — SuperPoint: **v1.0.0 release asset** `superpoint.onnx` from
  `fabio-sim/LightGlue-ONNX` (the repo's `main` no longer exports it standalone — IDEA.md §14 is
  stale). Input `image` f32 `[1,1,H,W]` grayscale BT.601, **no mean/std**. Outputs `keypoints`
  **int64** `(x,y)`, `scores`, `descriptors` **already L2-normalized** (so kNN is a plain matmul).
  Effective border is **8 px**, not the configured 4. Licence: MagicLeap **non-commercial
  research-only**, DERIVATIVES clause covers the ONNX file — never redistribute.
- `.planning/research/PITFALLS.md` §Hough, §RANSAC, §reproducibility
- `.planning/IDEA.md` §5 Method 2 (all sub-sections)

## Specifics — known limitation to record, not fix

The literature is explicit that when instances are small and nearly identical, almost all
tentative matches are wrong and Hough's discriminative power is insufficient. **That is precisely
the regime where Method 1 (NCC) is strongest. This is an expected finding, not a bug** — it is a
large part of why four methods exist, and the Phase 8 benchmark should *demonstrate* the crossover
rather than hide it. Record it in `docs/methods/sparse-geo.md` under Known Failure Modes.

## Deferred (robustness backlog)

Multi-model fitting (J-linkage, T-linkage) as a third decomposition strategy; DISK / ALIKED as
additional backends (also the permissive-licence escape from SuperPoint's non-commercial terms);
post-hoc orientation/scale assignment for SuperPoint keypoints via local gradient histograms,
which would unlock `single-4dof` for the learned backend; **LoFTR or RoMa dense matching with
correspondence-field clustering** for low-texture objects — with the caveat that LoFTR's ONNX
export is awkward (variable keypoint counts defeat static shapes; only partial community exports
exist), so it is a research spike, not a scheduled task.

## Scope Fence

**In:** `search/sparse_geo.py` (one self-contained module), the backend abstraction, all three
voting modes, Hough voting with soft binning and circular θ, NumPy per-peak RANSAC, degeneracy
rejection, sequential-RANSAC alternative, low-keypoint diagnostic, `SuperPointInferencer`,
`fetch-models` entry, `docs/methods/sparse-geo.md`, sample runs.

**Out:** LightGlue/SuperGlue (deliberately avoided — assignment-based matchers assume one-to-one,
exactly wrong here). Any change to the API or UI. Method 3 or 5.

## Risk Summary

- **The k+1-vs-standard ratio test proof is a success criterion.** Design the fixture so the
  standard test demonstrably kills the repeats: an image with ≥6 near-identical textured
  instances, where each crop keypoint has several near-equal distances.
- **`pairwise-4dof` is O(n²)** in correspondences. Cap the sampled pairs from config and record
  the cap in diagnostics, so a slow run is explained rather than mysterious.
- **Descriptor distance metric differs by backend** — SIFT/AKAZE are float L2, ORB is binary
  Hamming, SuperPoint is L2-normalized float (so cosine ≡ dot product). Getting this wrong yields
  garbage matches that still *look* like matches. Make the metric a property of the backend, not a
  config field.
- **Coverage.** This is the largest method module; budget real unit tests for the voting and RANSAC
  helpers independently of the end-to-end path, or the 80% floor will be missed.
