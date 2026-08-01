# Method 4 — `owlv2-oneshot` (OWLv2 image-conditioned one-shot detection)

The **permissive learned detector** on the scoreboard, and the closest Apache-2.0 analogue to the
visual-prompt detectors (T-Rex2, Rex-Omni) that were **rejected on licensing** — both are IDEA
License 1.0, non-commercial research-only, and Rex-Omni additionally inherits the Qwen Research
License. OWLv2 (Google, **Apache-2.0**) gives the same "draw a box, find every matching instance"
capability with no copyleft or non-commercial constraint, and — unlike a 3B PyTorch MLLM — a real
ONNX-Runtime path that fits this repo's "ONNX Runtime for every learned model" rule.

It runs OWLv2 in its **image-conditioned (one-shot) detection** mode: the exemplar crop is encoded
as a *query image*, a single query embedding is selected from it, and every patch of the scene is
scored by the **cosine similarity** of its OWLv2 class embedding to that query embedding. The
accepted patches, mapped through OWLv2's own trained detection boxes, are the matches. Its selling
point over the appearance-similarity methods (Methods 3 and 5) is that the boxes come from a
**supervised detection head**, and one model does both localization and matching in a single
forward pass per image.

The module `src/object_search/search/owlv2_oneshot.py` is meant to be read top to bottom; the
numbered steps below match the `# 1.` … `# 8.` comments in `search()` one-for-one (METHOD-11).

## One vision graph, run twice — there is no second ONNX input

OWLv2 image-guided detection is a "two image" task (query crop + scene), but the exported ONNX
graph has a **single** `pixel_values` input. `search()` encodes the exemplar crop and the scene
through the **same** `OWLv2Inferencer.embed_image` — one call each — and does the two-image logic
(query-embedding selection, cosine scoring) in NumPy in this one file. That keeps the ONNX layer a
plain single-input inferencer and the "image-guided" cleverness legible.

## Algorithm

### 1. Encode the exemplar crop as a query image

The crop `image[exemplar.box]` is encoded through `OWLv2Inferencer.embed_image`, yielding per-patch
`class_embeds` `(P, 512)` and normalized `pred_boxes` `(P, 4)`. Same graph, same preprocessing as
the scene encode in step 3. If `config.rotation_invariant`, the crop rotated 90/180/270 degrees is
ALSO encoded (OWLv2 patch embeddings are not rotation-equivariant, so a mirrored/rotated scene
instance may not match the exemplar's own orientation) — measured not to net-help (see the
ROBUSTNESS BACKLOG below), so it stays off by default.

### 2. Select one query embedding per rotation (most-distinctive covering patch)

The user's box tightly frames the object, so the object fills the crop. Among the crop patches whose
predicted box covers it (IoU with the full `[0, 1]` box within `query_iou_frac` of the maximum),
`select_query_embedding` takes the single patch whose embedding is **least similar to the mean patch
embedding** — the most *distinctive* one, which is the object rather than the generic whole-frame
direction. This is HuggingFace's `embed_image_query` heuristic, and it is a **correctness
requirement**: mean-pooling the covering patches instead returns the generic embedding, which then
matches scene patches predicting whole-image boxes — the method scored ~0 F1 until this was fixed
(see [`docs/reports/owlv2-improvement.md`](../reports/owlv2-improvement.md)).

### 3. Encode the scene (optionally tiled)

`embed_image(image)` → the scene's per-patch `class_embeds`, `pred_boxes`, `logit_shift`,
`logit_scale`. If `config.tile_large_scenes` and the scene is wider or taller than OWLv2's native
960px, it is instead split into overlapping 960px tiles (`tile_boxes`), each encoded separately and
mapped back to scene pixels, then merged into one flat candidate set before step 4 — a scene that
already fits within 960×960 is a single "tile" == the whole image, so this is byte-identical to the
untiled path below that size. Measured not to net-help (see the ROBUSTNESS BACKLOG below), so it
stays off by default.

### 4. Cosine similarity, then recalibrate with OWLv2's own logit_shift/logit_scale

`cosine = l2_normalize(scene.class_embeds) @ query_embedding`. Because both sides are
L2-normalized, the dot product **is** cosine similarity in `[-1, 1]`. One image, so a plain NumPy
matmul — no FAISS. The score used everywhere downstream is then **recalibrated**:
`score = (cosine + logit_shift) * logit_scale`, HF's own `Owlv2ClassPredictionHead` formula, using
the SCENE's own per-patch `logit_shift`/`logit_scale` (exported alongside `class_embeds`/
`pred_boxes`; computed with no query, so query-independent). Unlike a single global scale/shift
this is **per patch**, so it can (and does) re-rank patches — measured on floor-plan line-art, raw
cosine routinely ranked generic wall/room-corner rectangles above the true door/window instances;
calibration suppresses that (see
[`docs/reports/owlv2-floorplans-improvement.md`](../reports/owlv2-floorplans-improvement.md)).

### 5. Map predicted boxes to scene pixels, drop the whole-frame box

OWLv2 pads the input bottom-right to a square of side `max(tile_w, tile_h)` before resizing (per
tile, so `= max(H, W)` in the untiled case), so a normalized `pred_box` maps to scene pixels by a
plain multiply by that side, a clip, and (when tiled) an offset by the tile's origin — **no pad
offset otherwise**. A box that clips to sub-pixel size is dropped (a 0-area detection is not a
box). Boxes whose area exceeds `max_box_area_frac` of their **source tile's** area are **also
dropped**: OWLv2 emits one generic whole-frame box per forward pass (so one per tile) that scores
highest but is never a valid instance, and left in it anchors the threshold (step 6) and dominates
NMS. Per-patch score alignment is kept by index throughout.

### 6. Calibrate the threshold (self-similarity, against the calibrated score)

A fixed `score_threshold` passes straight through; otherwise the default **`self-similarity`**
strategy cuts at `self_score * retain_frac`, where `self_score` is the exemplar's own self-match
score (the top CALIBRATED score among boxes overlapping the exemplar box, falling back to the
global max). `gmm` still degenerates badly even against the calibrated score (measured, not just
theorized — see the improvement report); anchoring to the exemplar's own score is stable and
**label-free**. The calibrator returns its reasoning, which becomes an inspectable diagnostics
note.

### 7. Split into matches and candidates, then NMS

Patches clearing the threshold are accepted; the top `max_candidates` by raw score are retained as
sub-threshold `Candidate`s regardless of the threshold (EVAL-08). OWLv2 emits one box per patch, so
several neighbouring patches fire on one object; **post-detection NMS at `nms_iou`** collapses them
into a single detection. Ties sort `(-score, y, x)`, never score alone (the reproducibility rule).
**METHOD-12: every accepted box survives NMS — there is no single-best / argmax short-circuit.** The
kept box overlapping the exemplar is labelled `is_exemplar=True` rather than dropped or
double-counted (METHOD-04c).

### 8. Diagnostics and the latency split

`Diagnostics` carries the top candidate boxes as `proposals` (the UI's debug overlay) plus `metrics`
(threshold, patch/accepted/match counts, `collapsed_by_nms`, score max/mean).
`LatencyBreakdown.inference_ms` carries the summed model time, but the metrics report **`query_ms`
and `target_ms` as distinct numbers** and a note states which dominates — the scene encode does. A
run that clears nothing returns `outcome=EMPTY` with a note; an absent weight returns
`outcome=ERROR` with a `model_unavailable` note — never a silent empty and never a raise
(METHOD-04c).

## Pre-processing (exact)

OWLv2's preprocessing is **not** re-derived here; it is written once in the `OWLv2Inferencer`
docstring and reused. Input `pixel_values`, f32, NCHW, **RGB**; in order: rescale `1/255`; **pad
bottom-right to a square** of side `max(H, W)` with grey `0.5` (in the rescaled `[0, 1]` space —
OWLv2 pads bottom-right, not centred, so the content origin stays top-left and normalized boxes
need no pad offset); resize the square to **960×960 bilinear**; normalize with the CLIP mean
`[0.48145466, 0.4578275, 0.40821073]` and std `[0.26862954, 0.26130258, 0.27577711]`. The input is
**static 960×960** because OWLv2's learned position embeddings fix the resolution.

> These are the documented HuggingFace processor/architecture constants, **verified at export**
> (`scripts/export_owlv2.py` asserts the graph I/O) and exercised end-to-end by the improvement
> pass below. The `sha256` is pinned in the registry from the first verified export (EVAL-09); a
> byte-different re-export refuses to install.

## Post-processing (exact)

- **L2-normalize both sides**, then cosine = a plain NumPy matmul (`(P, D)` scene matrix against the
  `(D,)` query embedding) in `[-1, 1]`, per rotation if `rotation_invariant`, taking the per-patch
  MAX across rotations — step 4.
- **Recalibrate**: `score = (cosine + logit_shift) * logit_scale`, OWLv2's own learned,
  query-independent per-patch calibration — step 4.
- **Map normalized `pred_boxes` to pixels** per tile, by multiplying by `max(tile_w, tile_h)`,
  clipping, and offsetting by the tile origin; drop degenerate boxes **and the generic whole-frame
  box** (area > `max_box_area_frac` of the SOURCE TILE's area) — step 5.
- **Calibrate the threshold** with `self-similarity` (`self_score * retain_frac`, against the
  calibrated score) by default, or a fixed `score_threshold`; `gmm` still degenerates even against
  the calibrated score — step 6.
- **Post-detection NMS at `nms_iou`** collapses the several patches OWLv2 fires on one object; the
  candidate count is in diagnostics — step 7.
- **Retain sub-threshold candidates** (EVAL-08); return every accepted box after NMS (METHOD-12) —
  step 7.

## Config reference

Generated from `Owlv2OneshotConfig`'s JSON Schema — the same schema that drives the UI form — so it
cannot drift from the code.

| field | default | effect |
| --- | --- | --- |
| `score_threshold` | `null` | Fixed accept threshold on the CALIBRATED scene-patch↔query score. `null` ⇒ calibrate with the `calibration` strategy (absolute cuts do not transfer across images for deep features). |
| `calibration` | `"self-similarity"` | How the threshold is chosen when `score_threshold` is null. `self-similarity` cuts at `self_score * retain_frac`, anchored to the exemplar's own self-match. `gmm` degenerates badly even against the calibrated score (measured). |
| `retain_frac` | `0.85` | `self-similarity` accepts scene patches above `self_score * retain_frac`. Higher is stricter. `0.85` is the robust sweet spot against the CALIBRATED score — beat the prior `0.94`/raw-cosine baseline's F1 on every regime measured. |
| `query_iou_frac` | `0.8` | Query-embedding selection: among the exemplar-crop patches whose predicted box IoU with the full crop is at least this fraction of the maximum, pick the single most distinctive (least similar to the mean). Lower widens the candidate set. |
| `rotation_invariant` | `false` | Also encode the crop rotated 90/180/270 degrees and score on the per-patch MAX across all four. Measured: helps VARIED/WINDOW, regresses DOOR (-26%) and EASY (-20%) — not recommended. |
| `max_box_area_frac` | `0.25` | Drop any predicted box whose area exceeds this fraction of its SOURCE TILE's area — OWLv2's generic whole-frame box, which scores highest but is never a valid instance. |
| `tile_large_scenes` | `false` | Split a scene wider/taller than 960px into overlapping 960px tiles instead of downscaling into one pass. Measured: regressed 5 of 6 regimes, including EASY (the one it targeted) — not recommended. |
| `nms_iou` | `0.3` | Post-detection NMS IoU. OWLv2 fires several overlapping patches on one object, so a tight `0.3` collapses those duplicates (a big precision win) while distinct instances, which rarely overlap that much, survive. |
| `max_candidates` | `50` | How many top-scoring patches (with raw scores) to keep as sub-threshold candidates for an offline PR sweep (EVAL-08), regardless of the threshold. |
| `seed` | `0` | `random_state` for the gmm calibrator (only used if `calibration="gmm"`). |
| `debug_dir` | `null` | Local debugging aid, not a search parameter. When set, dump one PNG/txt per algorithm step (exemplar crop, per-tile `logit_shift`/`logit_scale`/raw-cosine/calibrated-score heatmaps, valid/pre-NMS/final box overlays, threshold summary) into this directory. `null` (the default) costs nothing. |

## Licence — OWLv2 is Apache-2.0 (the whole point of choosing it)

OWLv2 is **Apache-2.0** (Google), the same permissive tier as DINOv2 and the SuperPoint *code*, with
**no** AGPL §13 clause and **no** non-commercial restriction. This is precisely why it was adopted
over T-Rex2 and Rex-Omni: both are **IDEA License 1.0** (non-commercial, research-only), and
Rex-Omni additionally inherits the **Qwen Research License** — either would put the first
non-commercial encumbrance into a repo whose other learned models are all permissive. Adopting
OWLv2 does not constrain how this repo may be shared. Recorded in three places: this doc, the
`owlv2-base-patch16` `ModelSpec.license_note`, and `docs/library-reviews/owlv2.md`.

## Known failure modes

- **The exemplar self-match.** The exemplar is part of the scene, so one accepted patch overlaps it;
  that patch is labelled `is_exemplar=True` rather than dropped or silently counted (METHOD-04c).
- **Fixed 960 input caps small-object recall on large canvases.** OWLv2's position embeddings pin
  the input to 960; a small instance in a 6000×4000 scene occupies few patches. `tile_large_scenes`
  exists for this but MEASURED not to help — see the ROBUSTNESS BACKLOG below and the improvement
  report. The EASY (chipset) regime stays the weakest even after calibration (F1 ~0.35).
- **Query selection assumes a tightly-drawn box.** The distinctiveness selection works because the
  object fills the crop; a loose box that includes background can make the "distinctive" patch the
  background instead.
- **Weights absent.** With the weight absent the method returns `outcome=error` with a
  `model_unavailable` note rather than raising.

## ROBUSTNESS BACKLOG

Mirrored verbatim from the module docstring and `docs/ROBUSTNESS-BACKLOG.md`.

- **`tile_large_scenes` — BUILT AND MEASURED, off by default, not recommended.** Splitting a large
  scene into overlapping 960px tiles was hypothesized to lift small-object recall on large canvases
  (the EASY/chipset regime's known weakness). Measured across six regimes (see
  [`docs/reports/owlv2-floorplans-improvement.md`](../reports/owlv2-floorplans-improvement.md)): it
  regressed 5 of 6, INCLUDING EASY itself (F1 -20%, recall completely unchanged — every extra tile
  added false positives, not one new true positive). Kept as a documented opt-in for further
  investigation, not as a recommendation.
- **`rotation_invariant` — BUILT AND MEASURED, off by default, not recommended.** Scoring on the max
  cosine across 0/90/180/270-degree query rotations was hypothesized to help mirrored/rotated
  floor-plan symbols. Measured: helped VARIED (+5%) and WINDOW (+12%, still near-zero absolute) but
  regressed DOOR badly (-26%, one of the two target-domain regimes this exploration cares about) and
  EASY (-20%). Kept as a documented opt-in, not as a recommendation.
- **Text-prompt fusion** — OWLv2 also takes text queries; combining the drawn exemplar with an
  optional label would use both modalities (the exploration's Milestone 2 seam).
- **Query embedding from multiple exemplars** — average several drawn boxes for a more robust query.
- **owlv2-large** for accuracy at higher latency, gated behind the same export path.

## References

- Minderer et al., "Scaling Open-Vocabulary Object Detection" (OWLv2), 2023: https://arxiv.org/abs/2306.09683
- Minderer et al., "Simple Open-Vocabulary Object Detection with Vision Transformers" (OWL-ViT), 2022: https://arxiv.org/abs/2205.06230
- HuggingFace OWLv2 model card (Apache-2.0): https://huggingface.co/google/owlv2-base-patch16-ensemble
- HuggingFace `Owlv2ForObjectDetection.image_guided_detection` (the reference for the query-embedding selection): https://huggingface.co/docs/transformers/model_doc/owlv2
