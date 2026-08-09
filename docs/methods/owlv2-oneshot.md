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
the scene encode in step 3.

### 2. Select one query embedding (most-distinctive covering patch)

The user's box tightly frames the object, so the object fills the crop. Among the crop patches whose
predicted box covers it (IoU with the full `[0, 1]` box within `query_iou_frac` of the maximum),
`select_query_embedding` takes the single patch whose embedding is **least similar to the mean patch
embedding** — the most *distinctive* one, which is the object rather than the generic whole-frame
direction. This is HuggingFace's `embed_image_query` heuristic, and it is a **correctness
requirement**: mean-pooling the covering patches instead returns the generic embedding, which then
matches scene patches predicting whole-image boxes — the method scored ~0 F1 until this was fixed
(see [`docs/reports/owlv2-improvement.md`](../reports/owlv2-improvement.md)).

### 3. Encode the scene

The second and only other forward pass: `embed_image(image)` → the scene's per-patch `class_embeds`
and `pred_boxes`.

### 4. Cosine similarity — a plain NumPy matmul

`scores = l2_normalize(scene.class_embeds) @ query_embedding`. Because both sides are L2-normalized,
the dot product **is** cosine similarity in `[-1, 1]`. One image, so a plain NumPy matmul — no
FAISS.

### 5. Map predicted boxes to scene pixels, drop the whole-frame box

OWLv2 pads the input bottom-right to a square of side `max(H, W)` before resizing, so a normalized
`pred_box` maps to scene pixels by a plain multiply by that side and a clip — **no pad offset**. A
box that clips to sub-pixel size is dropped (a 0-area detection is not a box). Boxes whose area
exceeds `max_box_area_frac` of the image are **also dropped**: OWLv2 emits a generic whole-frame box
that scores highest but is never a valid instance, and left in it anchors the threshold (step 6) and
dominates NMS. Per-patch score alignment is kept by index throughout.

### 6. Calibrate the threshold (self-similarity)

A fixed `score_threshold` passes straight through; otherwise the default **`self-similarity`**
strategy cuts at `self_score * retain_frac`, where `self_score` is the exemplar's own self-match
score (the top score among boxes overlapping the exemplar box, falling back to the global max).
OWLv2's cosine scores are **compressed near 1.0 and not cleanly bimodal**, so the `gmm` strategy
degenerates to an unstable `ratio` cut that floods some scenes and starves others (measured: it gave
~0 precision); anchoring to the exemplar's own score is stable and **label-free**. The calibrator
returns its reasoning, which becomes an inspectable diagnostics note.

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
  `(D,)` query embedding) in `[-1, 1]` — step 4.
- **Map normalized `pred_boxes` to pixels** by multiplying by `max(H, W)` and clipping; drop
  degenerate boxes **and the generic whole-frame box** (area > `max_box_area_frac`) — step 5.
- **Calibrate the threshold** with `self-similarity` (`self_score * retain_frac`) by default, or a
  fixed `score_threshold`; absolute cosine cuts do not transfer across images, and OWLv2's are too
  compressed for `gmm` — step 6.
- **Post-detection NMS at `nms_iou`** collapses the several patches OWLv2 fires on one object; the
  candidate count is in diagnostics — step 7.
- **Retain sub-threshold candidates** (EVAL-08); return every accepted box after NMS (METHOD-12) —
  step 7.

## Config reference

Generated from `Owlv2OneshotConfig`'s JSON Schema — the same schema that drives the UI form — so it
cannot drift from the code.

| field | default | effect |
| --- | --- | --- |
| `score_threshold` | `null` | Fixed accept threshold on the scene-patch↔query cosine. `null` ⇒ calibrate with the `calibration` strategy (absolute cosine cuts do not transfer across images for deep features). |
| `calibration` | `"self-similarity"` | How the threshold is chosen when `score_threshold` is null. `self-similarity` cuts at `self_score * retain_frac`, anchored to the exemplar's own self-match. OWLv2 cosine is compressed near 1.0 and not bimodal, so `gmm` degenerates to `ratio` and thresholds unstably. |
| `retain_frac` | `0.94` | `self-similarity` accepts scene patches above `self_score * retain_frac`. Higher is stricter. `0.94` is the robust sweet spot across regimes (near-max F1 everywhere, recall ~0.9). |
| `query_iou_frac` | `0.8` | Query-embedding selection: among the exemplar-crop patches whose predicted box IoU with the full crop is at least this fraction of the maximum, pick the single most distinctive (least similar to the mean). Lower widens the candidate set. |
| `crop_context_margin_frac` | `0.0` | Grow the exemplar box by this fraction of its own width/height on each side (clamped to the scene) before cropping the query image. `0.0` crops the exemplar box exactly, unchanged from prior behavior. A margin pulls in real neighboring pixels instead of the synthetic pad color a tight crop gets blown up on. |
| `max_box_area_frac` | `0.25` | Drop any predicted box whose area exceeds this fraction of the image — OWLv2's generic whole-frame box, which scores highest but is never a valid instance. |
| `nms_iou` | `0.3` | Post-detection NMS IoU. OWLv2 fires several overlapping patches on one object, so a tight `0.3` collapses those duplicates (a big precision win) while distinct instances, which rarely overlap that much, survive. |
| `max_candidates` | `50` | How many top-scoring patches (with raw scores) to keep as sub-threshold candidates for an offline PR sweep (EVAL-08), regardless of the threshold. |
| `seed` | `0` | `random_state` for the gmm calibrator (only used if `calibration="gmm"`). |

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
  the input to 960; a small instance in a 6000×4000 scene occupies few patches. This is the primary
  remaining weakness — the EASY (chipset) regime stays at ~0.24 F1 while textured regimes reach
  0.82–0.87. Tiling is the fix, deferred to the backlog.
- **Query selection assumes a tightly-drawn box.** The distinctiveness selection works because the
  object fills the crop; a loose box that includes background can make the "distinctive" patch the
  background instead.
- **Weights absent.** With the weight absent the method returns `outcome=error` with a
  `model_unavailable` note rather than raising.

## ROBUSTNESS BACKLOG

Deferred deliberately (mirrored verbatim from the module docstring and
`docs/ROBUSTNESS-BACKLOG.md`); none is built in this phase:

- **Tiled / multi-scale inference** to lift small-object recall on large canvases past the fixed 960
  input — the primary remaining weakness (EASY regime).
- **Export OWLv2's learned `logit_scale` / `logit_shift`** and apply them before thresholding, so
  scene scores are calibrated logits rather than raw (compressed) cosine — may make the distribution
  bimodal and remove the need for self-similarity anchoring.
- **Text-prompt fusion** — OWLv2 also takes text queries; combining the drawn exemplar with an
  optional label would use both modalities (the exploration's Milestone 2 seam).
- **Query embedding from multiple exemplars** — average several drawn boxes for a more robust query.
- **owlv2-large** for accuracy at higher latency, gated behind the same export path.

## References

- Minderer et al., "Scaling Open-Vocabulary Object Detection" (OWLv2), 2023: https://arxiv.org/abs/2306.09683
- Minderer et al., "Simple Open-Vocabulary Object Detection with Vision Transformers" (OWL-ViT), 2022: https://arxiv.org/abs/2205.06230
- HuggingFace OWLv2 model card (Apache-2.0): https://huggingface.co/google/owlv2-base-patch16-ensemble
- HuggingFace `Owlv2ForObjectDetection.image_guided_detection` (the reference for the query-embedding selection): https://huggingface.co/docs/transformers/model_doc/owlv2
