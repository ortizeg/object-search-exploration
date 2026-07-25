# Method 3 — `dino-dense` (DINOv2 dense-token prototype matching)

The general-purpose default for "same object, moderate appearance variation". It embeds the
exemplar crop and the whole scene into DINOv2 dense patch tokens, **mean-pools** the crop tokens
into one prototype, scores every scene location by the **cosine similarity** of its token to
that prototype, upsamples the similarity map to pixels, thresholds it, and turns the connected
components into boxes. Where `ncc` (Method 1) correlates raw intensities and so misses instances
under pose or lighting change, DINOv2 features are appearance-robust — this is the method that
finds "the same object, differently posed".

The module `src/object_search/search/dino_dense.py` is meant to be read top to bottom; the
numbered steps below match the `# 1.` … `# 9.` comments in `search()` one-for-one (METHOD-11).
The backbone is the shared `DINOv2Inferencer` built in Phase 6 (1/2) and reused here — one
download, one preprocessing contract, reused again by Method 5 in Phase 7.

## What it is and when it wins

`dino-dense` wins exactly where `ncc` loses: instances that are the same object but differ in
**pose, rotation, or lighting**. DINOv2's self-supervised features are largely invariant to
those, so a rotated or relit copy still scores high against the prototype. It costs one ONNX
forward pass over the scene (tens to a few hundred ms on CPU) and is coarser than NCC — boxes
are quantised to ~14 px patches — so for near-identical, pixel-aligned repeats at the exemplar's
scale, NCC is still the cheaper, sharper tool. That crossover is the whole reason both exist.

## Algorithm

### 1. Get the one shared DINOv2 inferencer

The inferencer is built once, lazily, from the gitignored weight and cached at module level, so
every query reuses the **same** loaded model. When the weight is absent the method returns
`outcome=error` with a `model_unavailable` note rather than raising, so the sample renderer and
the API degrade honestly. (The `SearchFn` contract shares nothing but `(image, exemplar,
config)`, so the method loads its own cached backbone rather than reading `app.state` — see the
deviation note in the phase summary.)

### 2. Embed the crop → mean-pool → **L2-normalize** the prototype

The exemplar crop is embedded on its own; its patch tokens are mean-pooled into a single vector,
which is then **L2-normalized once**. Pooling first and normalizing second is deliberate: the
prototype is the mean embedding whose self-cosine is `1.0`, not the mean of per-token unit
vectors. Mean-pooling loses part structure (a known limitation for articulated objects — see the
backlog), but it is the readable v1.

### 3. Run the scene at high resolution (capped) → **L2-normalize** the grid

DINOv2's stride-14 patches are coarse, so a higher input resolution buys a finer similarity map.
The scene is run at native resolution up to a cap (`scene_max_side`, default 1568); a scene whose
long side exceeds the cap is **downscaled and the cap is logged** — never silently truncated,
which is how a 6000 px image would otherwise become a handful of meaningless tokens. The
inferencer snaps each side to a multiple of 14 and returns the scale factors that invert the
snap. Every returned token is then L2-normalized.

### 4. Cosine similarity map

`sim = normalized_prototype · normalized_grid`, a `(gh, gw)` map in `[-1, 1]`. Both sides are
L2-normalized **before** the dot product, so this is genuine cosine similarity rather than an
unnormalized dot dominated by token **magnitude** — and DINOv2-small ships high-norm background
"artifact" tokens whose magnitude would otherwise dominate a raw dot. Normalize, then dot, in
that order.

### 5. Bilinearly upsample the **map** (not the tokens)

The `(gh, gw)` map is bilinearly upsampled to the model-input pixel resolution using the scale
factors the inferencer returned. The token grid covers a snapped input of `gh·14 × gw·14` px, so
the true input size is `gw·14 / scale_x` by `gh·14 / scale_y`; `cv2.resize` maps source cell
centre `(gx+0.5)` to destination fraction `(gx+0.5)/gw`, which is exactly each patch centre, so a
token peak lands on its pixel. Upsampling the **map** is cheap; upsampling the 384-d tokens first
would be 384× the work for the same result. A test pins that a known token peak upsamples to the
right pixel.

### 6. Calibrate the threshold

`common.calibration.calibrate` with the configured strategy (default `gmm`), fed the
token-resolution similarity distribution. Absolute cosine thresholds **do not transfer across
images** for deep features, which is exactly what the calibration layer is for. `gmm` fits two
modes (foreground/background) and cuts between them; `self-similarity` anchors on the prototype's
`self_score = 1.0`; `ratio` cuts at the largest gap in the top scores; `fixed` passes through
`config.threshold`. The calibrator returns its **reasoning**, which becomes an inspectable
diagnostics note (Phase 6 success criterion 3: the three strategies produce different, inspectable
thresholds on the same map).

### 7. Threshold → connected components (**skip label 0**)

The upsampled map is thresholded and `cv2.connectedComponentsWithStats` labels the blobs.
**Label 0 is the background and is skipped explicitly** — emitting it would be one image-sized
false positive. Each remaining component above `min_component_area` becomes one box (scaled back
from the capped inference resolution to original scene pixels), with its score set to the peak
similarity inside it. Components are found at a floor `candidate_margin` **below** the accept
threshold so that sub-threshold near-misses are captured for step 8.

### 8. Split into matches and sub-threshold candidates

The top `max_candidates` components (ranked by raw similarity) are kept as `Candidate`s **with
raw scores** regardless of the threshold — that is what makes an offline threshold sweep possible
later (EVAL-08). Components whose score clears the threshold become `Match`es. **METHOD-12: every
clearing component survives — there is no single-best / argmax-only short-circuit; connected
components returns as many instances as the image contains.** The component overlapping the
exemplar box is labelled `is_exemplar=True` rather than dropped or double-counted (METHOD-04c).

### 9. Assemble diagnostics and the result

`Diagnostics` carries the upsampled similarity map as a base64 PNG heatmap (the UI's debug
overlay) plus `metrics` (chosen threshold, grid size, `cap_engaged`/`cap_scale`, component and
match counts, similarity max/mean). `LatencyBreakdown` attributes the ONNX forward pass to
`inference_ms` and the rest to `postprocess_ms` (EVAL-11). A run that clears nothing returns
`outcome=EMPTY` with a note, never a silent empty (METHOD-04c).

## Pre-processing (exact)

The ONNX numbers live in the `DINOv2Inferencer` docstring and the Phase 6 (1/2) doc and are not
re-derived here; they are, verbatim: input `pixel_values`, f32, NCHW, RGB; scale `1/255`, mean
`[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`; **bicubic** resize with **snap-to-14** and
**NO centre-crop**. Three points matter specifically for this method:

- **Snap-to-14 is mandatory, and the scale factors travel with the tokens.** DINOv2 does not
  validate the multiple-of-14 requirement; the stride-14 patch conv silently floor-divides, so an
  un-snapped side drops its trailing pixels and produces a *systematic spatial offset* in the
  similarity map. Snapping removes it, and the returned `scale_x`/`scale_y` are what let step 5
  map token coordinates back to real pixels.
- **No centre-crop.** HF's default preprocessor shortest-edge-resizes then centre-crops to
  224×224, silently discarding the image border. This method needs the **whole** scene, so the
  inferencer disables the crop.
- **Register tokens are stripped by a *derived* count, not a hardcoded `1`.** The correct slice
  is `[1 + n_register:]` (one CLS plus any register tokens). For this model,
  `onnx-community/dinov2-small-ONNX`, **`n_register = 0`** — it ships no register tokens, so the
  slice is effectively `[1:]`. The count is nonetheless *derived from the token count at load*
  (`n_register = tokens − 1 − gh·gw`) rather than hardcoded, because HF itself shipped a bug here
  and a with-registers variant would report `n_register = 4` and silently shift the whole feature
  map by four patches if the slice were pinned to `1`. Deriving it turns that silent shift into a
  load-time check.

**Positional-embedding interpolation is fixed at export time.** FB's reference implementation
interpolates the position embeddings with `offset=0.1` and antialiasing; HF uses
`align_corners=False`; the two **differ**. Whichever the exporter used is baked into the `.onnx`
graph. This method uses the pinned `onnx-community` fp32 export and does **not** attempt to
re-derive or "correct" the interpolation — the export in use is recorded and treated as fixed.

## Post-processing (exact)

- **L2-normalize both sides before the dot product** (cosine, not a magnitude-dominated dot) —
  step 4.
- **Upsample the MAP, not the tokens**, using the inferencer's scale factors so token centres
  land on their pixels — step 5.
- **`connectedComponentsWithStats` label 0 (background) is skipped explicitly** — step 7.
- **Calibrate the threshold** (default `gmm`); absolute cosine thresholds do not transfer across
  images — step 6.
- **Retain sub-threshold candidates** for an offline PR sweep (EVAL-08); return every clearing
  component (METHOD-12) — step 8.

## Config reference

Generated from `DinoDenseConfig`'s JSON Schema — the same schema that drives the UI form — so it
cannot drift from the code.

| field | default | effect |
| --- | --- | --- |
| `scene_max_side` | `1568` | Cap on the scene's longest side (pixels) before DINOv2 inference. Higher = a finer similarity map but more tokens and memory; a scene above this is downscaled and the cap is logged. 1568 = 112 patches. |
| `calibration` | `"gmm"` | How the accept threshold is chosen when `threshold` is `null`. gmm fits two modes (foreground/background) on the similarity map; absolute cosine thresholds do not transfer across images for deep features. |
| `threshold` | `null` | Fixed accept threshold on the raw cosine similarity. `null` ⇒ use the calibrator. |
| `candidate_margin` | `0.1` | How far below the accept threshold a component is still logged as a sub-threshold candidate for offline PR sweeps (EVAL-08). |
| `min_component_area` | `4` | Minimum connected-component area in pixels; smaller blobs are dropped as noise. |
| `max_candidates` | `50` | How many top components (with raw scores) to keep for the EVAL-08 candidate log. |
| `seed` | `0` | `random_state` for the gmm calibrator (its only genuinely stochastic step). |
| `retain_frac` | `0.7` | self-similarity accepts scores above `self_score × retain_frac` (`self_score = 1.0`). |

## Known failure modes

- **Stride-14 coarseness.** Even with high-res inference and upsampling, boxes are quantised to
  ~14 px patches; a small object a few pixels across is a single fuzzy token. This is the headline
  limitation and the reason for the sliding-window / FeatUp backlog items.
- **Very large scenes hit the resolution cap.** Above `scene_max_side` the scene is downscaled
  (and it is logged), so effective localisation on a 6000 px image is coarser than the cap
  suggests.
- **A single mean-pooled prototype loses part structure.** For articulated or non-compact objects
  (the basketball-player frames) a many-to-many token similarity would do better; this is a
  deliberate v1 simplification, not a bug.
- **Weights absent.** With no weight present the method returns `outcome=error` with a
  `model_unavailable` note rather than raising.

## ROBUSTNESS BACKLOG

Deferred deliberately (mirrored verbatim from the module docstring and
`docs/ROBUSTNESS-BACKLOG.md`); none is built in this phase:

- **Sliding-window backbone inference** for very large scenes, so localisation no longer degrades
  at the resolution cap.
- **Learned feature upsampling (FeatUp)** to recover sub-patch localisation from the stride-14
  grid without a full high-res forward pass.
- **SAM-based box refinement** — snap each coarse component box to the nearest segment mask.
- **Many-to-many token similarity with spatial aggregation** instead of a single mean-pooled
  prototype — measurably better for articulated objects like the basketball frames.
- **DINOv3 backbone swap** once a clean ONNX export exists.

## Sample runs

Regenerated by `pixi run samples` and committed under
[`docs/samples/dino-dense/`](../samples/dino-dense/) (see its
[`index.md`](../samples/dino-dense/index.md) for the per-image outcome table). Each panel shows
the query, the matches overlay, and the DINOv2 similarity heatmap.

| image | panel |
| --- | --- |
| `cluttered-distractors` — appearance-robust matches amid clutter | ![cluttered-distractors](../samples/dino-dense/cluttered-distractors.png) |
| `lattice-plain` — repeated instances on a plain lattice | ![lattice-plain](../samples/dino-dense/lattice-plain.png) |
| `lattice-touching` — touching instances (heatmap shows the coarseness) | ![lattice-touching](../samples/dino-dense/lattice-touching.png) |
| `scatter-scaled` — scale + pose variation, where `dino-dense` beats `ncc` | ![scatter-scaled](../samples/dino-dense/scatter-scaled.png) |

## Pseudocode

**Method ③ dino-dense** — third of the four *implemented* methods (implementation numbering
①–④: `ncc`, `sparse-geo`, `dino-dense`, `propose-retrieve`; source-research numbering 1, 2, 3, 5,
with research Methods 4 and 6 deferred). The steps below mirror the `# 1.` … `# 9.` comments in
`search()` (METHOD-11); read `src/object_search/search/dino_dense.py` for the ground truth.

```
1. inferencer <- the ONE shared, module-cached DINOv2Inferencer
   if weight absent: return ERROR with a model_unavailable note  # never raise

2. embed the exemplar crop -> mean-pool its patch tokens -> L2-normalize ONCE -> prototype
   (pool first, normalize second: self-cosine of the prototype is 1.0)

3. run the SCENE at native resolution capped at scene_max_side  # downscale + LOG the cap if above
   snap each side to a multiple of 14; strip CLS + n_register tokens (n_register DERIVED, not hardcoded)
   L2-normalize every returned token -> grid

4. sim = prototype . grid  -> a (gh, gw) cosine map in [-1, 1]   # normalize BOTH sides, THEN dot

5. bilinearly upsample the MAP (not the tokens) to pixel resolution using the inferencer scale factors
   (token centre (gx+0.5) maps to its pixel; upsampling tokens would be 384x the work)

6. calibrate the accept threshold (gmm default) on the token-resolution similarity distribution
   (absolute cosine thresholds do not transfer across images)

7. threshold -> connectedComponentsWithStats; SKIP label 0 (background) explicitly
   components found at candidate_margin BELOW the threshold; area >= min_component_area -> box
   (scaled back to original scene pixels; score = peak similarity inside the component)

8. keep top max_candidates components as Candidates WITH raw scores (for the EVAL-08 sweep)
   components whose score >= threshold -> Matches  # METHOD-12: every clearing component survives
   label the component overlapping the exemplar box is_exemplar=True

9. assemble Diagnostics (upsampled heatmap + metrics) and LatencyBreakdown
   a run that clears nothing returns EMPTY with a note (METHOD-04c)
```

## References

- Oquab et al., "DINOv2: Learning Robust Visual Features without Supervision", 2023: https://arxiv.org/abs/2304.07193
- DINOv2 code: https://github.com/facebookresearch/dinov2
- ONNX model used here (`dinov2-small-ONNX`): https://huggingface.co/onnx-community/dinov2-small-ONNX
