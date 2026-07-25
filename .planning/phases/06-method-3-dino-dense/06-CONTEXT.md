# Phase 6 Context — Method 3 (`dino-dense`)

**Source:** `.planning/IDEA.md` §5 Method 3, plus `.planning/research/MODELS.md` (verified DINOv2
contract) and `.planning/research/PITFALLS.md` (§DINOv2, §connected components).

## Domain

The general-purpose default for "same object, moderate appearance variation". DINOv2 dense patch
tokens for the scene and the crop, mean-pool the crop into a prototype, cosine-similarity the
prototype against every spatial location, threshold, connected components → boxes.

This phase also produces the `DINOv2Inferencer` that **Phase 7 reuses rather than duplicating** —
one model, one preprocessing contract, one download.

## Locked Decisions

1. **Model: `onnx-community/dinov2-small-ONNX`, revision pinned.** Apache-2.0 (clean, verified
   three ways), ~84.4 MiB. Verdict: **Adopt**. This supersedes IDEA.md §14's `sefaburak*`
   references.
2. **Verified input contract** — put these exact numbers in the inferencer docstring:
   - input `pixel_values`, f32, NCHW, **fully dynamic** H/W
   - RGB, scale `1/255`, mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`
   - **bicubic** interpolation
   - **do NOT centre-crop** — HF's default preprocessor would, and that silently discards the
     image border
3. **Verified output contract:** `last_hidden_state`, shape
   `[B, floor(H/14) * floor(W/14) + 1, 384]`. Index 0 is the CLS token. Verified at six
   resolutions.
4. **Snap input sides to multiples of 14 — this is mandatory.** DINOv2 does **not** validate the
   multiple-of-14 requirement; HF's patch embedding silently floor-divides, so a 225 px side
   yields 16 patches (224 px of content) and the trailing pixels vanish. The result is a
   **systematic spatial offset** in the similarity map, not an error. Snap each side to the
   nearest multiple of 14 and **return the exact scale factors**, because token coordinates cannot
   be mapped back to image pixels without them.
5. **Strip the CLS token — and any register tokens — before reshaping to a spatial grid.** The
   correct slice is `[1 + n_register:]`. `dinov2-small` has no register tokens, so `n_register=0`
   and the slice is `[1:]` — but write it as `1 + n_register` with `n_register` derived from the
   token count rather than hardcoding `1`, because HF itself shipped a bug here and a
   with-registers variant would silently shift the whole feature map by a few patches.
   **Verify the arithmetic at load:** assert
   `tokens == floor(H/14) * floor(W/14) + 1 + n_register`, and raise if not. This turns a silent
   spatial shift into a load-time error, which is the same discipline as INFRA-09.
6. **Patch-grid orientation must be verified empirically, not assumed.** Research flags a
   transposed similarity map as a plausible-looking bug — it produces a map that looks like a
   heatmap and is wrong. Reshape as `(H//14, W//14, 384)` (row-major, height first) and **prove it
   with a test**: run a scene with a distinctive object in a known, non-centred, non-square
   position and assert the similarity peak lands at the corresponding token coordinate. A square
   test image cannot catch a transpose — use a deliberately non-square one.
7. **High-resolution scene inference with bilinear upsampling of the similarity map** is the
   shipped v1 mitigation for stride-14 coarseness. Run the scene at a configurable high input
   resolution, compute the similarity at token resolution, then bilinearly upsample the *map*
   (not the tokens) to image resolution.
8. **`connectedComponentsWithStats` label 0 is the background.** Emitting it produces a
   full-image false positive. Skip label 0 explicitly, with a comment.
9. **Cosine similarity: L2-normalize both the prototype and the token grid before the dot
   product.** Normalizing once, in the right order, is the difference between cosine and an
   unnormalized dot product that is dominated by token magnitude.
10. **Reuse the `search/common/calibration.py` strategies** — Phase 6 success criterion 3 requires
    all three producing different inspectable thresholds on the same image. This is where
    calibration earns its keep, because absolute similarity thresholds genuinely do not transfer
    across images for deep features.

## Canonical References

- `.planning/research/MODELS.md` § DINOv2 — the verified contract, `fetch-models` sketch, licence
- `.planning/research/PITFALLS.md` § DINOv2 (multiple-of-14, register slice, pos-embed
  interpolation) and § connected components
- `src/object_search/inference/onnx_inferencer.py` — the Phase 1 base with `snap-to-multiple`

Note on positional-embedding interpolation: FB's reference implementation uses `offset=0.1` with
antialiasing, HF uses `align_corners=False`, and the two **differ**. The ONNX export bakes in
whichever the exporter used, so do not attempt to re-derive or "correct" it — just record which
export is in use and that the interpolation is fixed at export time.

## Scope Fence

**In:** `DINOv2Inferencer`, its `fetch-models` entry, `search/dino_dense.py` (one self-contained
module), high-res dense similarity with upsampling, threshold + connected components → boxes,
diagnostics with the similarity heatmap, `docs/methods/dino-dense.md`, sample runs.

**Out:** Method 5's proposal/embedding stages (Phase 7, though it reuses this inferencer).
FeatUp, SAM refinement, DINOv3, many-to-many token similarity — all backlog.

## Deferred (robustness backlog)

Sliding-window backbone inference for very large images; learned feature upsampling (FeatUp);
SAM-based box refinement; **many-to-many token similarity with spatial aggregation instead of a
single mean-pooled prototype** — a single prototype loses part structure and is measurably worse
for articulated or non-compact objects, which is directly relevant to the basketball demo frames;
DINOv3 backbone swap.

## Risk Summary

- **The transpose bug is the highest-risk item** and it is invisible without a deliberately
  non-square, off-centre fixture. Write that test first.
- **Phase 6 success criterion 1 requires finding instances where `ncc` fails** — pose or lighting
  variation. The `scatter-scaled` synthetic spec plus a basketball frame are the candidates; verify
  the two methods actually disagree rather than asserting it.
- **Memory at high resolution.** A 1568×1568 scene is 112×112 = 12,544 tokens × 384 floats. Fine,
  but a 4K scene at full resolution is not — cap the input resolution in config and log when the
  cap engages rather than silently OOMing.
- **Model download in CI.** Weights are gitignored, so CI cannot run any test that needs the real
  model. Mark those tests to skip when the model file is absent, and keep the contract/arithmetic
  tests model-free so CI still gates the risky logic.
