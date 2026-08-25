# Method 5 — `propose-retrieve` (class-agnostic proposals + DINOv2 region embeddings)

The method for "give me tight boxes around every instance". Instead of scoring a dense similarity
map (Method 3, `dino-dense`), it first asks a **class-agnostic segmenter — FastSAM in "everything
mode"** — for a few hundred region proposals, then embeds each proposed region with the **same
DINOv2 backbone as Method 3**, and ranks those regions by the **cosine similarity** of their
embedding to the exemplar's embedding. The accepted regions are the matches.

Its selling point over `dino-dense` is **boundary alignment**: FastSAM proposals hug object edges,
so the returned boxes are tight rectangles around real objects instead of the blobby connected
components a stride-14 similarity map produces. That is not an impression — it is a **measured
number**: a test asserts the mean IoU of the returned boxes against the exact chipset ground truth
is above `0.70` (it measures ~`0.99` on the 1600×1200 chipset), which is Phase 7 success criterion 1.

The module `src/object_search/search/propose_retrieve.py` is meant to be read top to bottom; the
numbered steps below match the `# 1.` … `# 7.` comments in `search()` one-for-one (METHOD-11).

## The two independently callable units — the Milestone 2 seam

This is the defining constraint of Phase 7, and the reason Milestone 2 (marker-conditioned
proposals) can **add an exploration rather than fork the app**:

- **`propose(image, config) -> list[Proposal]`** — the proposal stage (built in plan 07-01, in
  `search/proposals.py`).
- **`embed_regions(image, boxes, config) -> NDArray`** — the embedding stage, built here: one
  L2-normalized embedding per box. It knows **nothing** about proposals, exemplars, or retrieval.

`search()` **composes** the two and does nothing they cannot do alone. Neither unit reaches into the
other's internals, and a **seam test calls each directly**, not through `search()` (Phase 7 success
criterion 2).

## One DINOv2 model, shared with Method 3

The embedding stage deliberately reuses Method 3's module-level `DINOv2Inferencer` singleton
(`dino_dense._get_inferencer`) rather than constructing a second one — **one model download, one
preprocessing contract** (CONTEXT locked decision 6). There is structurally no second DINOv2 loader
in the codebase, and a test asserts `embed_regions` routes through that one singleton and that the
registry holds exactly one `dinov2-*` key (Phase 7 success criterion 3).

## Algorithm

### 1. Propose regions (FastSAM everything-mode)

`propose(image, FastSAMConfig(conf_thres=config.proposal_conf))` returns class-agnostic region
proposals. FastSAM's internal box NMS is deliberately **loose** (`iou_thres=0.9`): "everything
mode" *wants* overlapping proposals, because a missed object cannot be recovered later while a
duplicate can. Over-segmentation is collapsed by a **second** NMS after retrieval (step 6), not
here.

**Optional SAHI-style tiling (`proposal_tiling`, default off).** When on, `propose_tiled` runs the
same backend over overlapping `tile_side` × `tile_side` tiles in **native image pixels** — step
`round(tile_side × (1 − tile_overlap))`, the final tile **clamped to the image edge rather than
padded** — offsets each tile's boxes into full-image coordinates, unions in the untiled whole-image
pass when `tile_include_full_image` is set ("SAHI + FI"), and merges the union by
**intersection-over-SMALLER (IoS) at `tile_merge_ios`**. IoS, not IoU: a symbol truncated by a tile
edge is nearly *contained* in the whole-object box found by an overlapping tile, so it has high IoS
and **low** IoU, and an IoU merge would keep both. `max_proposals` is applied **after** the merge, so
the budget stays global. On a scene that already fits inside one tile this is an **exact identity**,
which is why the chipset/textured/synthetic regimes are unaffected by construction.

The lever is **proposal budget**, not magnification: FastSAM's everything-mode proposal count scales
with image *area* (r = +0.59 on 84 floor plans) and barely with instance count (r = +0.22), so a
crowded plan gets ~46 proposals for ~15 symbols and the proposal stage caps recall at 0.27 before
retrieval runs.

> **Measured, and NOT recommended for CAD floor plans** — see
> [the floor-plan report](../reports/propose-retrieve-floorplans-improvement.md). Three findings, in
> the order they were measured:
>
> - **Magnification does nothing here.** At a disabled merge, 512-tiles (2.00× magnification) and
>   768-tiles (1.33×) scored **0.586 vs 0.585** mean proposal recall — a 2× difference in
>   pixels-per-symbol worth 0.001. SAHI's stated premise (small objects lost to downscaling) does not
>   apply to this domain.
> - **N tiles do NOT buy N× the budget** at the default `tile_merge_ios` of 0.5. Across a 4.3× range
>   in forward passes (2.5 → 10.8 tiles/plan) the *merged* proposal count stayed pinned at
>   **1.14–1.33×** baseline: IoS is `intersection / min(area)`, so a proposal fully **contained** in
>   a kept one scores exactly 1.0 and is deleted — and the contained ones are precisely the small
>   nested symbols being hunted. The merge is a budget clamp on an everything-mode segmenter.
> - **`proposal_conf` buys the same budget more cheaply.** At a *matched* proposal budget the
>   objectness gate beat tiling by **+0.233** mean proposal recall, **+0.347** in the crowded bucket,
>   for about a **third** of the proposal-stage latency — and tiling pays on both the FastSAM and the
>   embedding stage, where the gate pays only on embeddings.
>
> Tiling is kept as an opt-in because it is an exact identity on single-tile scenes and *is* the
> right lever for one measured extreme-resolution case (a 4000×1685 plan: proposal recall 0.053
> untiled → 0.263 tiled). For floor plans generally, tune `proposal_conf` instead.

### 2. Embed the proposal regions

`embed_regions(image, [p.box for p in proposals], config)` → an `(N, D)` matrix, one **L2-normalized**
DINOv2 embedding per proposal. Each box is cropped from the scene, embedded on its own, its patch
tokens **mean-pooled**, then the whole matrix is L2-normalized row-wise.

### 3. Embed the exemplar

`embed_regions(image, [exemplar.box], config)[0]` → the `(D,)` exemplar embedding, through the
**same unit and the same backbone** as step 2. The exemplar is part of the scene, so its own region
usually appears among the proposals and scores ~`1.0` against itself — which anchors the calibrator
in step 5.

### 4. Cosine nearest-neighbour — a plain NumPy matmul (no FAISS)

`scores = proposal_embeddings @ exemplar_embedding`. Because both sides are already L2-normalized,
the dot product **is** cosine similarity in `[-1, 1]`. This is a **plain NumPy matmul — FAISS is
deliberately not adopted** (CONTEXT decision 7): for a few hundred proposals in one image a FAISS
index is pure dependency cost. The embedding matrix is shaped `(N, D)` precisely so a FAISS index
slots in unchanged when corpus-scale search arrives (backlog).

### 5. Calibrate the threshold

A fixed `retrieval_threshold` passes straight through; otherwise `common.calibration.calibrate`
fits a two-mode `gmm` and cuts between the "matches" mode (the exemplar's own region and its
duplicates near `1.0`) and the background mode. The per-image `gmm` cut is then **clamped to an
absolute `similarity_floor`** — it may rise above the floor but never sink below it. The floor is a
cosine to the exemplar (whose self-cosine is `1.0`), applied identically on every image, so it is a
distribution-independent *anchor*, not a label-fit cut, and AP stays threshold-free.

The floor fixes two failure modes of a bare two-mode fit:

- **A low `gmm` cut admitting background.** On cluttered scenes the `gmm` sometimes cuts down in the
  background shoulder; the floor holds the line and is the dominant precision win.
- **The degenerate single-mode catastrophe.** A uniform lattice of *identical* instances scores
  `~1.0` with no second mode; the `gmm` degeneracy guard falls back to `ratio`, which lands the cut
  *at* the max score, and the strict `> threshold` then rejects **every true match** (recall 0 — the
  worst possible failure for a repeated-instance finder). With the floor, the degenerate case is
  decided by the floor alone: the near-`1.0` regions are all accepted, while an image with no other
  instances scores below the floor and is correctly rejected.

The calibrator returns its **reasoning**, and the applied floor is added to it, so both are
inspectable diagnostics notes.

### 6. Split into matches and candidates, then post-retrieval NMS

Proposals clearing the threshold are accepted; the top `max_candidates` by raw score are retained as
sub-threshold `Candidate`s regardless of the threshold (EVAL-08), so an offline sweep can rebuild a
PR curve. The accepted set is then run through **post-retrieval NMS at `nms_iou`** to **collapse SAM
over-segmentation** — one object that FastSAM split into several overlapping proposals, each of which
embeds well, would otherwise surface as duplicate detections. The pre-NMS proposal count and the
number collapsed are recorded in diagnostics so the over-segmentation is **visible, not hidden**.
**METHOD-12: every accepted region survives NMS — there is no single-best / argmax short-circuit.**
The kept proposal overlapping the exemplar box is labelled `is_exemplar=True` rather than dropped or
double-counted (METHOD-04c).

### 7. Diagnostics and the latency split

`Diagnostics` carries the **full proposal set** as `proposals` (the UI's debug overlay, Phase 7
success criterion 4) plus `metrics` (threshold, proposal/accepted/match counts, `collapsed_by_nms`,
score max/mean, and the tiling cost: `n_tiles`, `n_proposals_pre_merge`, `merged_by_tiling` — `1`,
`n`, `0` when tiling is off). `LatencyBreakdown.inference_ms` carries the summed model time, but the
metrics report **`proposal_ms` and `embedding_ms` as distinct numbers** and a note states which
dominates and over how many tiles. **Which stage dominates is domain-dependent** — that is the whole
point of the EVAL-11 breakdown. On the chipset the **proposal** stage dominates (~200 ms vs ~45 ms
embedding); on CAD floor plans the **embedding** stage dominates by a lot (measured 1.4 s proposal vs
5.9 s embedding mean, 1.4 s vs 14.2 s on a 1478×958 plan), because every proposal gets its own
DINOv2 forward pass and a plan yields ~50 of them. That asymmetry is why tiling's cost must be read
as *both* extra FastSAM passes *and* a larger merged proposal count feeding the dominant stage. A run that clears nothing
returns `outcome=EMPTY` with a note; an absent weight returns `outcome=ERROR` with a
`model_unavailable` note — never a silent empty and never a raise (METHOD-04c).

## Pre-processing (exact)

Neither backbone's preprocessing is re-derived here; both are written down once in their inferencer
docstrings and reused verbatim.

**FastSAM proposals** (`FastSAMInferencer`): input `images`, f32, NCHW, **RGB**; scale `1/255`,
**NO mean subtraction, NO std division** (YOLO does no normalization); resize **letterbox** to
1024×1024 by `min(1024/W, 1024/H)`, padded with **fill 114** (the YOLO grey), centred — so the pad
is subtracted when mapping boxes back. Output decoding (YOLOv8-seg, verified at 1024²/640²/512×768):
transpose `output0` `[1, 37, anchors]` → `[anchors, 37]`; split into `[0:4]` box `(cx,cy,w,h)`,
`[4]` objectness, `[5:37]` 32 mask coeffs; confidence-filter at `conf_thres`; convert to `xyxy`;
box **NMS** at `iou_thres` with deterministic `(-score, y1, x1)` tie-break; undo the letterbox.
Masks are decoded only when requested (`sigmoid(coeff @ protos)`, reshaped, then **cropped to each
box** — mandatory, or the prototype-combination mask bleeds outside the detection); Milestone 1 uses
boxes only.

**Tiled FastSAM proposals** (`propose_tiled`, only when `proposal_tiling` is on): the scene is cut
into overlapping `tile_side` tiles in native pixels with step `round(tile_side × (1 − tile_overlap))`
and the final tile **clamped to the image edge** (not padded — padding would letterbox grey into the
model's field of view for whichever symbols land there). **Each tile then goes through the exact
FastSAM preprocessing above**, i.e. it is letterboxed to the same fixed 1024 square, which is what
magnifies a symbol by `1024 / tile_side` (a real mechanical effect whose *benefit* measured inert on
floor plans — see the [Algorithm §1 note](#1-propose-regions-fastsam-everything-mode)). Tile boxes
are offset into full-image coordinates and
clipped to the scene; `return_masks=True` is **rejected with a `ValueError`** because FastSAM masks
come back in tile-local coordinates and mapping them back is out of scope — silently returning
tile-local masks would be a correctness bug. Tiling adds **no randomness**: tile order is `(y0, x0)`
and the merge order is the canonical `(−objectness, y, x)`, so there is deliberately **no seed
parameter** for it.

**DINOv2 region embeddings** (`DINOv2Inferencer`, shared with Method 3): input `pixel_values`, f32,
NCHW, **RGB**; scale `1/255`, mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`; **bicubic**
resize with **snap-to-multiple(14)** and **NO centre-crop**; CLS and any register tokens stripped by
a *derived* `[1 + n_register:]` slice (`n_register = 0` for `dinov2-small`, derived not hardcoded).
A proposal crop smaller than one 14 px patch is up-sized to 14×14 first, so a small proposal still
yields a token.

## Post-processing (exact)

- **Mean-pool then L2-normalize**, in that order, per region — the region embedding is the mean of
  its patch tokens, normalized *once* so its self-cosine is `1.0`. Normalizing per token before
  pooling weights every token equally regardless of magnitude — a different, wrong quantity (the
  same DINOv2 high-norm-artifact trap Method 3 documents).
- **Cosine NN is a plain NumPy matmul — no FAISS** (step 4).
- **Calibrate the threshold** (`gmm` default, or fixed `retrieval_threshold`) **clamped to an
  absolute `similarity_floor`**; the floor holds precision on cluttered scenes and rescues the
  degenerate uniform-lattice case (recall 0 without it) — step 5.
- **Post-retrieval NMS at `nms_iou` (0.3)** collapses SAM over-segmentation; the proposal count is in
  diagnostics — step 6.
- **Retain sub-threshold candidates** (EVAL-08); return every accepted region after NMS (METHOD-12)
  — step 6.

## Config reference

Generated from `ProposeRetrieveConfig`'s JSON Schema — the same schema that drives the UI form — so
it cannot drift from the code.

| field | default | effect |
| --- | --- | --- |
| `proposal_backend` | `"fastsam"` | Which class-agnostic proposal backend to use. Only `fastsam` is implemented in Milestone 1; MobileSAM is a documented deviation (its ONNX decoder takes one prompt per call, so everything-mode is ~1024 calls plus a ported mask generator). |
| `proposal_conf` | `0.4` | FastSAM objectness threshold: keep proposals whose class-0 confidence exceeds this. FastSAM's default is 0.4. Lower surfaces more (and smaller) regions. |
| `retrieval_threshold` | `null` | Fixed accept threshold on the cosine similarity between a proposal embedding and the exemplar embedding. `null` ⇒ calibrate with a two-mode gmm clamped to `similarity_floor`. |
| `nms_iou` | `0.3` | Post-retrieval NMS IoU. A later accepted box overlapping a kept one by MORE than this is suppressed — this collapses FastSAM over-segmentation (one object split into several partially-overlapping proposals) into a single detection. Tighter than a classical 0.5 because "everything mode" emits many shifted/partial proposals of the same object. |
| `similarity_floor` | `0.7` | Absolute cosine floor on the calibrated accept threshold (ignored when a fixed `retrieval_threshold` is given). The gmm cut may rise above this but never below it: the floor stops a low gmm cut from admitting background, and rescues the degenerate single-mode case (a uniform lattice of identical instances, where the bare gmm/ratio fallback rejects every true match). Anchored on the exemplar self-cosine (=1.0). |
| `max_candidates` | `50` | How many top-scoring proposals (with raw scores) to keep as sub-threshold candidates for an offline PR sweep (EVAL-08), regardless of the threshold. |
| `seed` | `0` | `random_state` for the gmm calibrator (its only genuinely stochastic step). |
| `proposal_tiling` | `false` | Run the proposal backend over overlapping tiles (SAHI-style) instead of one whole-image pass, merging across tiles by intersection-over-smaller. **Off by default.** Its lever is proposal **budget**: FastSAM's proposal count scales with image area, not instance count, so a crowded scene is starved — N tiles buy ~N× the budget, and each tile is magnified by the fixed 1024 letterbox. A domain knob for large, densely-populated scenes (CAD floor plans); an exact identity on a scene that fits in one tile. |
| `tile_side` | `1024` | Tile edge in **native image pixels** (not model input pixels). FastSAM letterboxes every tile to a fixed 1024 square, so a tile of side S magnifies each symbol by `1024/S`. A scene whose width and height are both ≤ this yields exactly one tile equal to the whole image. |
| `tile_overlap` | `0.2` | Fraction of `tile_side` shared by consecutive tiles (SAHI's verified default). The overlap band is what lets an instance straddling a tile boundary be seen untruncated by at least one tile, so it should comfortably exceed the typical instance size. |
| `tile_merge_ios` | `0.5` | Cross-tile merge threshold on intersection-over-**smaller** (SAHI's default metric and threshold). **Not IoU** — a tile-edge-truncated fragment is nearly contained in the whole-object box, so it has high IoS and low IoU and an IoU merge would keep both. |
| `tile_include_full_image` | `true` | Union the untiled whole-image pass in before merging (SAHI's `perform_standard_pred`, "SAHI + FI"). Recovers instances too large for any tile's overlap band, at exactly one extra forward pass. Ignored when the scene fits in one tile. |

## Licence — FastSAM is AGPL-3.0 (a real sharing constraint, stated plainly)

FastSAM is **AGPL-3.0**, and the exported `.onnx` file **itself embeds that licence string**. The
export-time-only isolation (Ultralytics lives only in the `export` pixi env, so the runtime
dependency graph stays torch-free) protects the *dependency graph* but **not the weights**. Private
local use triggers nothing; **publishing this repo or network-exposing the FastAPI app fires
AGPL §13.** This is recorded in three places — `assets/demo/LICENSES.md`, the `fastsam-s`
`ModelSpec.license_note`, and here — because it is a real constraint on how the repo may later be
shared, not a footnote. Revisit before publishing or exposing the app.

## Deviation — MobileSAM is not implemented as a working second backend

The brief imagined a FastSAM/MobileSAM switch. **MobileSAM is not implemented as a working second
backend in Milestone 1, and this is a documented deviation** (CONTEXT locked decision 5). The reason,
from research: the ONNX SAM decoder accepts **one prompt per call**, so "everything mode" means a
32×32 grid ⇒ **~1024 sequential decoder calls** plus a hand-ported `SamAutomaticMaskGenerator`. That
is a *phase of work, not a backend swap*. What ships instead is a `ProposalBackend` protocol with
FastSAM as the single implementation, so MobileSAM can be added later **without restructuring** — the
seam is open, the work is deferred. (IDEA.md §14's `awarebayes/MobileSamONNX` reference was also
dropped, verdict **HOLD**: 0 stars, a 4-day project untouched for 14 months, single author, while
first-party export scripts and a better-provenanced MIT artifact both exist.)

## Known failure modes

- **SAM over-segmentation.** One object becomes several proposals; each embeds well, so without the
  post-retrieval NMS they surface as duplicate detections. NMS at `nms_iou` collapses them, and the
  pre-NMS proposal count in diagnostics is how the practitioner sees it happening. This is the
  expected failure mode and the tuning signal — **the chipset (exact ground truth) is where you tune
  it.**
- **The raw box crop includes background.** A tight box around a non-convex object still embeds some
  surrounding pixels; the FastSAM mask is available (`return_masks`) to mask that background out.
  **Measured (2026-07-25):** pixel-masking hurt (artificial fill edges crashed synthetic recall),
  token-masking gave only +0.006 macro-F1 at ~2× latency — deferred, see the backlog.
- **Latency is dominated by the proposal stage.** This is a *finding*, not a defect — see the EVAL-11
  latency split; it is why the breakdown attributes proposal and embedding time separately.
- **AGPL / non-commercial licence constraints.** The FastSAM weights carry AGPL-3.0 (above); this
  constrains sharing, not local exploration.
- **Weights absent.** With FastSAM or DINOv2 absent the method returns `outcome=error` with a
  `model_unavailable` note rather than raising.

## ROBUSTNESS BACKLOG

Deferred deliberately (mirrored verbatim from the module docstring and
`docs/ROBUSTNESS-BACKLOG.md`); none is built in this phase:

- **FAISS index for corpus-scale retrieval** — unnecessary for a few hundred proposals in one image;
  the `(N, D)` embedding matrix is shaped so it slots in when corpus search arrives.
- **Background-masked region embedding** — embed the FastSAM mask interior rather than the raw box
  crop. **Measured (2026-07-25), deferred.** Pixel-masking (ImageNet-mean fill) HURT — synthetic
  recall 0.94 → 0.65 from artificial fill edges. Token-masking (pool only tokens whose patch centre
  is inside the mask) gave +0.006 macro-F1 at ~2× latency and erodes the mask-free `embed_regions`
  seam. Revisit if cluttered precision becomes the priority.
- **Proposal filtering by an exemplar size/aspect prior** — drop proposals whose shape cannot match
  the exemplar before embedding. **Measured (2026-07-25), rejected** — an area-ratio gate crashed
  textured recall (varied 0.93 → 0.59) because true instances legitimately span a range of scales;
  a size prior fights the scale-invariance this method exists to provide.
- **Multi-crop / test-time augmentation embeddings** for pose-robust region descriptors.
- **Alternative proposal sources (RPN, selective search)** for images where SAM over-segments.
  **Considered and NOT built (2026-08-24)** — a contour/blob proposer for line-art plans was scoped,
  its go/no-go criterion fired, and it was still skipped on evidence: with `proposal_conf` tuned, the
  proposal stage stopped being the binding stage (crowded-bucket proposal recall 0.639 vs end-to-end
  0.262), and its motivating claim — that FastSAM cannot see CAD door symbols — was refuted (the plan
  of record went 0.000 → 0.857 on the gate alone). See
  [the floor-plan report](../reports/propose-retrieve-floorplans-improvement.md).
- **MobileSAM everything-mode** with a ported `SamAutomaticMaskGenerator` as a second backend.
- **SAHI-style proposal tiling (`proposal_tiling` et al.) — BUILT AND MEASURED, off by default, NOT
  RECOMMENDED for CAD floor plans.** At a matched proposal budget the existing `proposal_conf` gate
  beat it by **+0.233 mean proposal recall at a third of the latency**, and SAHI's magnification
  premise measured **inert** here (a 2× difference in pixels-per-symbol moved recall by 0.001). Kept
  as an opt-in rather than reverted: it is an exact identity on single-tile scenes and *is* the right
  lever for one measured extreme-resolution case (4000×1685 plan, 0.053 → 0.263). See the
  [Algorithm §1 note](#1-propose-regions-fastsam-everything-mode) and
  [the floor-plan report](../reports/propose-retrieve-floorplans-improvement.md).
- **Crowded-bucket retrieval/calibration gap — INVESTIGATED (2026-08-25), no lever ships.** A
  per-GT-box trace through propose/embed/calibrate/NMS found the retrieval-stage loss rate (of GT
  boxes WITH a covering proposal) triples with crowding (0.095 → 0.197 → 0.322), that the gmm's
  adaptive cut is nearly inert on this domain (the fixed `similarity_floor` decides in nearly every
  case), and that true/background cosine-score separation compresses with crowding (0.373 → 0.287)
  — an embedding-discriminability signature, not a calibration-logic defect. Sweeping
  `similarity_floor` below 0.70 at `proposal_conf=0.10` monotonically WORSENS pooled val F1;
  `0.70` is confirmed the argmax across the full measured {0.55–0.85} range. Nothing ships. See
  [the follow-on section](../reports/propose-retrieve-floorplans-improvement.md#follow-on-260825-calibration)
  of the floor-plan report.

## Sample runs

Regenerated by `pixi run samples` and committed under
[`docs/samples/propose-retrieve/`](../samples/propose-retrieve/index.md) (see its
[`index.md`](../samples/propose-retrieve/index.md) for the per-image outcome table). Each panel
shows the query and the matches overlay; the proposal set is the diagnostics overlay in the app.

| image | panel |
| --- | --- |
| `cluttered-distractors` — tight boxes amid clutter | ![cluttered-distractors](../samples/propose-retrieve/cluttered-distractors.png) |
| `lattice-plain` — repeated instances on a plain lattice | ![lattice-plain](../samples/propose-retrieve/lattice-plain.png) |
| `lattice-touching` — a uniform lattice of identical instances (where the `similarity_floor` earns its keep: the bare gmm rejected all of them) | ![lattice-touching](../samples/propose-retrieve/lattice-touching.png) |
| `scatter-scaled` — scale + pose variation | ![scatter-scaled](../samples/propose-retrieve/scatter-scaled.png) |

## Pseudocode

**Method ④ propose-retrieve** — fourth of the four *implemented* methods (implementation
numbering ①–④: `ncc`, `sparse-geo`, `dino-dense`, `propose-retrieve`; source-research numbering
1, 2, 3, 5, with research Methods 4 and 6 deferred — this is research **Method 5**). The steps
below mirror the `# 1.` … `# 7.` comments in `search()` (METHOD-11); read
`src/object_search/search/propose_retrieve.py` for the ground truth.

```
1. proposals <- propose(image, FastSAMConfig(conf_thres=proposal_conf))  # FastSAM everything-mode
   (internal box NMS is deliberately LOOSE, iou_thres=0.9: wants overlapping proposals)

2. proposal_embeddings <- embed_regions(image, [p.box for p in proposals])  # (N, D)
   (crop each box, DINOv2 mean-pool its patch tokens, row-wise L2-normalize)

3. exemplar_embedding <- embed_regions(image, [exemplar.box])[0]  # (D,)
   (SAME unit, SAME shared DINOv2 backbone as step 2)

4. scores = proposal_embeddings @ exemplar_embedding  # cosine NN, a plain NumPy matmul (no FAISS)

5. calibrate the threshold:
       fixed retrieval_threshold passes straight through, OR
       two-mode gmm cut, CLAMPED to similarity_floor: threshold = floor if degenerate else max(gmm, floor)
       (the floor holds precision on clutter AND rescues the degenerate uniform-lattice case)

6. proposals with score >= threshold -> accepted; keep top max_candidates as Candidates (EVAL-08)
   POST-RETRIEVAL NMS at nms_iou over the accepted set  # collapses SAM over-segmentation
   label the kept proposal overlapping the exemplar box is_exemplar=True  # METHOD-12: no single-best

7. Diagnostics carry the FULL proposal set + metrics (proposal_ms vs embedding_ms, collapsed_by_nms)
   EMPTY-with-note if nothing clears; ERROR model_unavailable if a weight is absent (never a raise)
```

## References

- Zhao et al., "Fast Segment Anything", 2023: https://arxiv.org/abs/2306.12156
- FastSAM code: https://github.com/CASIA-IVA-Lab/FastSAM
- Kirillov et al., "Segment Anything (SAM)", 2023: https://arxiv.org/abs/2304.02643
- Oquab et al., "DINOv2", 2023 (region embeddings): https://arxiv.org/abs/2304.07193
