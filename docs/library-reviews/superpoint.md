# Library Review — SuperPoint (ONNX) — **Trial**

**Subject:** the frozen **v1.0.0 release asset `superpoint.onnx`** from
`fabio-sim/LightGlue-ONNX`, the standalone SuperPoint keypoint detector + descriptor this
project runs as the learned backend for Method 2 (`sparse-geo`, Phase 5).

**Verdict:** **Trial** (download-only; no code dependency on the primary path). An immutable
5.03 MiB release artifact, runtime-verified I/O contract, third-party-validated via Kornia,
but a single-maintainer repo and — critically — **MagicLeap non-commercial research-only
weights**. Adopted for local research use only, with the weights gitignored so the derivatives
clause is satisfied at zero cost.

| | |
|---|---|
| Artifact | `fabio-sim/LightGlue-ONNX` release **v1.0.0** → `superpoint.onnx` |
| Download URL | `https://github.com/fabio-sim/LightGlue-ONNX/releases/download/v1.0.0/superpoint.onnx` |
| Size | 5,272,808 bytes (5.03 MiB), released 2023-10-03, immutable release asset |
| Registry key | `superpoint` in `src/object_search/inference/models.py` (`source="github-release"`) |
| Code licence | Apache-2.0 (`fabio-sim/LightGlue-ONNX`) |
| **Weights licence** | **MagicLeap non-commercial research-only** ⚠️ (see *Licence* below) |
| Full analysis | `.planning/research/MODELS.md` § SuperPoint (runtime-verified) |
| Evaluated | 2026-07-24 |

## Why Trial (and why the exposure is near zero)

- **The primary path imports nothing.** Method 2 downloads one immutable release asset via
  plain `urllib` (no third-party package enters the dependency graph) and loads it under ONNX
  Runtime. The single-maintainer repo (Fabio Milentiansen Sim) is the only structural risk, and
  the primary path never runs its code.
- **Runtime-verified contract, not inferred.** Input `image`, f32, NCHW, `[1, 1, H, W]` with
  batch fixed at 1 and H/W dynamic; **single-channel grayscale**; range `[0, 1]` (divide by
  255); **no mean subtraction, no std division**; colour is BT.601 luma from RGB
  (`0.299·R + 0.587·G + 0.114·B`), which `cv2.COLOR_BGR2GRAY` reproduces. Outputs `keypoints`
  **int64** `[1, N, 2]` `(x, y)` in input pixels, `scores` f32 `[1, N]`, `descriptors` f32
  `[1, N, 256]` **already L2-normalized** (measured ‖d‖ = 1.0000). All three share one symbolic
  `N`, so lengths always agree.
- **Variable keypoint count is exactly what Method 2 needs.** The v1.0.0 asset was exported with
  `max_num_keypoints=None`, so `N` is genuinely data-dependent (19 keypoints on a 120×160 input,
  ~1851 on 1024×1024). METHOD-04c's low-keypoint diagnostic reads `keypoints.shape[1]` directly;
  a fixed top-K export (what current `lightglue_dynamo.models.SuperPoint` produces) would return
  K=1024 noise keypoints on a small crop and defeat that guard.
- **Downstream validation.** Integrated into **Kornia** as `kornia.feature.OnnxLightGlue` —
  meaningful third-party vetting for a niche export repo (666 stars, 73 forks, last push
  2026-07-24, Apache-2.0, typed with tests).

## Adoption constraints carried into the inferencer

Recorded so the `Trial` verdict travels with its caveats (mirrored in the
`SuperPointInferencer` docstring and `docs/methods/sparse-geo.md`):

- **Grayscale, no mean/std.** Scale to `[0, 1]` only. Do **not** apply ImageNet normalization;
  SuperPoint expects raw luma.
- **Descriptors are already L2-normalized — do not re-normalize.** kNN is therefore a plain
  matmul: cosine `= D_crop @ D_scene.T`, squared-L2 `= 2 − 2·cos`.
- **Keypoints are `int64` `(x, y)`** — no sub-pixel refinement, and **no scale or orientation**.
  A SuperPoint keypoint carries no geometric frame, so single-correspondence 4-DoF (`single-4dof`)
  Hough voting is genuinely invalid for this backend and **raises**; `translation-2dof` is the
  SuperPoint default, `pairwise-4dof` recovers scale/rotation from correspondence pairs.
- **Effective border is 8 px, not the configured `remove_borders=4`** (the border mask is
  applied on the 8×-upsampled score grid). Method 2 never gets a correspondence within 8 px of
  the scene edge, which matters when an instance is clipped by the frame.
- **Pad to a multiple of 8, do not resize the crop.** Non-multiple sides are silently floored
  (trailing rows/columns dropped), a coordinate-range truncation rather than an error. The
  inferencer pads bottom/right with zeros (origin-preserving) so no coordinate remapping is
  needed.
- **Unbounded `N`** on textured scenes → O(N²) blow-up in `pairwise-4dof`; the `pairwise_cap`
  config field bounds it.

## Licence — MagicLeap non-commercial research-only (the load-bearing flag)

The `fabio-sim/LightGlue-ONNX` **code** is Apache-2.0, but the SuperPoint **weights** baked into
this ONNX file are MagicLeap's original SuperPoint weights, released **for non-commercial
research use only**, and the licence's **DERIVATIVES clause covers this exported `.onnx`
file**. Consequences carried into the project:

1. **Never redistribute the weights.** The file must not be committed or published. INFRA-11
   already gitignores everything under `models/`, so this is satisfied at zero additional cost
   (`.gitignore` line `models/`, fetched only by `pixi run fetch-models`).
2. **Non-commercial research use only.** This is an exploration harness for local research, which
   is within terms. Any commercial deployment would require different weights.
3. **The permissive-licence escape hatch is on the robustness backlog.** DISK / ALIKED are
   learned detectors under permissive licences that would remove the non-commercial constraint;
   they are recorded in `docs/methods/sparse-geo.md` § ROBUSTNESS BACKLOG as the sanctioned way
   out if this method ever needs to ship.

## The frozen-asset caveat

The v1.0.0 asset is a **frozen artifact whose exporter no longer exists on `main`**:
`lightglue_dynamo/cli.py::export` now unconditionally builds `Pipeline(extractor, matcher)` and
emits the fused `keypoints`/`matches`/`mscores` graph — there is no `--extractor-only` flag any
more (verified against `main` at `pushed_at = 2026-07-24`; IDEA.md §14's "exports SuperPoint
standalone" is **stale**). The standalone export survives only as this release asset. It cannot
be regenerated from current HEAD, so:

- The download URL is pinned to the immutable `v1.0.0` tag.
- `sha256` is `None` on first adoption and should be pinned on first successful fetch to become a
  hard integrity gate (EVAL-09).
- The reproducible-from-HEAD fallback (`torch.onnx.export` of
  `lightglue_dynamo.models.SuperPoint`, export-time-only, in a separate pixi env) produces a
  **different** contract (fixed `N`) and is documented in `.planning/research/MODELS.md` as the
  break-glass path only.

## Rejected candidates

- **The v2.0 asset `superpoint_lightglue_pipeline.onnx` (51 MB)** — the fused extractor+matcher
  pipeline the project deliberately avoids (assignment-based matchers assume one-to-one, exactly
  wrong for repeated-instance search). Do not download it.
- **HF Optimum path** — does not exist; there is no `SuperPointOnnxConfig`, and
  `transformers`' `SuperPointForKeypointDetection` produces a ragged per-image list
  (data-dependent-shape export problem).
- **`magic-leap-community/superpoint` (HF)** — usable as weights but `license: other`, no ONNX
  path.
- **`AXERA-TECH/superpoint`, `thomasonzhou/superpoint-lightglue`, `shadow-cann/...`** — NPU /
  vendor-specific or pipeline exports, 0–7 downloads. **Hold.**
