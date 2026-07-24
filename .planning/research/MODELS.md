# ONNX Model Research

Research date: **2026-07-24**. Every claim marked ✅ below was verified by downloading the
artifact and running it under `onnxruntime` on this machine (via `pixi exec`, macOS arm64,
CPUExecutionProvider). Claims marked ⚠️ **UNVERIFIED** are listed again in
[Unverified Items](#unverified-items-must-be-confirmed-at-implementation-time).

Verification harness used (throwaway, not committed): a pure-Python ONNX protobuf probe for
graph I/O plus `ort.InferenceSession(...).run()` at several resolutions. The equivalent
permanent check belongs in `ONNXInferencer.__init__` (INFRA-09).

---

## Summary Table

| Model | Chosen path | Verdict | Dynamic shapes | License | Size |
|---|---|---|---|---|---|
| **DINOv2-small** (default) | `onnx-community/dinov2-small-ONNX` → `onnx/model.onnx`, pinned revision | **Adopt** | ✅ Full dynamic `B,C,H,W` | Apache-2.0 (from `facebook/dinov2-small`) | 84.4 MiB (fp32) / 43.3 MiB (fp16) |
| **DINOv2-base** (opt-in) | `onnx-community/dinov2-base-ONNX` → `onnx/model.onnx`, pinned revision | **Adopt** | ✅ Full dynamic `B,C,H,W` | Apache-2.0 (from `facebook/dinov2-base`) | 330.6 MiB (fp32) / 167 MiB (fp16) |
| — fallback exporter | `optimum-cli export onnx --model facebook/dinov2-small --task feature-extraction` | **Adopt** (export-time only) | ✅ same axes | Apache-2.0 tooling | — |
| **SuperPoint** | `fabio-sim/LightGlue-ONNX` release **v1.0.0** asset `superpoint.onnx` (direct download) | **Trial** (download-only) | ✅ dynamic `H,W`; batch fixed 1; **dynamic keypoint count** | Code Apache-2.0; **weights MagicLeap non-commercial research-only** ⚠️ | 5.03 MiB |
| — fallback exporter | Scripted `torch.onnx.export` of `lightglue_dynamo.models.SuperPoint` | **Trial** (export-time only) | ✅ dynamic `H,W`; fixed top-K | same | ~5 MiB |
| **FastSAM-s** (default proposals) | Scripted `ultralytics` export of `FastSAM-s.pt` → `FastSAM-s.onnx` | **Trial** (export-time only) | ✅ `batch, H, W`, anchors, mask grid | **AGPL-3.0** (code *and* weights) ⚠️ | 45.0 MiB (from 22.7 MB `.pt`) |
| **FastSAM-x** (quality) | same, `FastSAM-x.pt` | **Trial** | ✅ same | **AGPL-3.0** ⚠️ | ~276 MiB ⚠️ est. |
| **MobileSAM** (alternative) | `Acly/MobileSAM` pre-exported encoder + decoder | **Assess** — do not build in M1 | ✅ encoder dynamic H/W; decoder dynamic point count | MIT (weights `dhkim2810/MobileSAM` MIT; code Apache-2.0) | 26.9 MiB encoder + 15.7 MiB decoder |

**Headline recommendations**

1. **DINOv2 is a solved problem.** The pre-exported `onnx-community` graph exposes exactly
   what Methods 3 and 5 need — CLS + dense patch tokens — at *arbitrary* resolution, with
   the token count symbolically encoded as `floor(H/14)*floor(W/14) + 1`. Verified at four
   resolutions. No export script needed; keep `optimum-cli` as a documented fallback.
2. **SuperPoint: the brief's premise is stale and must be corrected.** IDEA.md §5 and §14
   say `fabio-sim/LightGlue-ONNX` "exports SuperPoint standalone". **On `main` as of
   2026-07-24 it does not** — the CLI only exports the fused extractor+matcher pipeline
   (`output_names = ["keypoints", "matches", "mscores"]`). The *right* path is the frozen
   **v1.0.0 release asset**, which is a genuinely standalone SuperPoint with a
   variable-length keypoint output — better for this project than any fixed-top-K export.
3. **FastSAM's AGPL-3.0 is real and it is not confined to the exporter.** The exported
   `.onnx` file itself carries `license: AGPL-3.0` in its embedded metadata. See
   [License Concerns](#license-concerns).
4. **Do not build MobileSAM "everything mode" in Milestone 1.** The ONNX SAM decoder takes
   one prompt per call; a 32×32 automatic-mask grid means ~1024 sequential decoder
   invocations plus a hand-ported `SamAutomaticMaskGenerator`. That is a phase of work, not
   a backend swap.

---

## DINOv2

### Chosen export path

**Primary — pre-exported, reputable org, pinned revision.**

| | |
|---|---|
| Repo (small) | `onnx-community/dinov2-small-ONNX` |
| Pinned revision | `08c606e3123472a388efa59181b677d428f69bbd` (lastModified 2025-05-09) |
| File | `onnx/model.onnx` (84.4 MiB) |
| Repo (base) | `onnx-community/dinov2-base-ONNX` |
| Pinned revision | `9a1732c724ef0847dcedf00d3d5a93f61dea0370` (lastModified 2025-04-14) |
| File | `onnx/model.onnx` (330.6 MiB) |
| Base weights | `facebook/dinov2-small` (3.26 M downloads), `facebook/dinov2-base` (2.18 M downloads) — both `license: apache-2.0` |

Direct URLs (revision-pinned; use these, not `main`):

```
https://huggingface.co/onnx-community/dinov2-small-ONNX/resolve/08c606e3123472a388efa59181b677d428f69bbd/onnx/model.onnx
https://huggingface.co/onnx-community/dinov2-base-ONNX/resolve/9a1732c724ef0847dcedf00d3d5a93f61dea0370/onnx/model.onnx
```

Also present in each repo (all verified to exist by size, **not** functionally tested):
`model_fp16.onnx`, `model_int8.onnx` = `model_quantized.onnx` = `model_uint8.onnx`,
`model_q4.onnx`, `model_q4f16.onnx`, `model_bnb4.onnx`. **Use fp32 for v1** — quantized
ViT feature extractors degrade cosine-similarity geometry, which is the entire signal
Method 3 thresholds on.

**Fallback — scripted first-party export.** Verified in `optimum-onnx` source that
`dinov2` is registered for `feature-extraction` and inherits `ViTOnnxConfig`, whose
`inputs` are `{"pixel_values": {0: "batch_size", 2: "height", 3: "width"}}` and whose
`feature-extraction` output is `{"last_hidden_state": {0: "batch_size", 1: "sequence_length"}}`
— i.e. the same contract.

```bash
pip install "optimum-onnx[onnxruntime]"
optimum-cli export onnx --model facebook/dinov2-small --task feature-extraction models/dinov2_small/
```

**Rejected candidates from IDEA.md §14:**

- `sefaburak/dinov2-small-onnx` (`dinov2_vits14.onnx`, 82.6 MiB) — **Hold**. 0 downloads,
  last modified 2024-01-07, single-author, and the I/O contract is undocumented. Superseded.
- `sefaburakokcu/dinov2_onnx` — not needed; the `onnx-community` artifact is verified and
  the `optimum-cli` fallback is first-party.
- `deepghs/dinov2_onnx` — not evaluated; no reason to prefer it.

### Input contract

| Property | Value |
|---|---|
| Input name | `pixel_values` |
| dtype | `float32` (`tensor(float)`) |
| Shape | `[batch_size, num_channels, height, width]` — **all four dynamic** ✅ |
| Layout | **NCHW** |
| Channel order | **RGB** (convert from the project's BGR scene: `img[..., ::-1]`) |
| Scale | `pixel / 255.0` (`rescale_factor = 0.00392156862745098`) |
| Mean | `[0.485, 0.456, 0.406]` |
| Std | `[0.229, 0.224, 0.225]` |
| Resize interpolation | `resample: 3` = PIL BICUBIC in the HF processor; OpenCV equivalent `cv2.INTER_CUBIC` |
| opset / IR | `ai.onnx` 14 / IR 7, producer `pytorch` |

Constants taken verbatim from `facebook/dinov2-small/preprocessor_config.json` ✅.

**Resize policy — this project deviates from the HF processor on purpose.** The HF default
is `size.shortest_edge = 256` then `center_crop 224×224`. **Do not center-crop.** Methods 3
and 5 need the whole scene. Instead:

```
target_h = round(H * s / 14) * 14      # s chosen so max(target) ≈ long_side_target (e.g. 896)
target_w = round(W * s / 14) * 14
```

Rationale, verified experimentally ✅: the graph accepts non-multiples of 14 but **silently
discards the trailing pixels** via the stride-14 patch conv. `H=225, W=225` returns 257
tokens (= 16×16+1), i.e. the last row and column of pixels contribute nothing. Snapping both
dimensions to a multiple of 14 removes a silent up-to-13-pixel misalignment between the
similarity map and the image — which would otherwise show up as a systematic box offset in
Method 3.

`num_channels` is dynamic in the signature but the patch-embedding conv has 3 input
channels; pass 3.

### Output contract

| Property | Value |
|---|---|
| Output name | `last_hidden_state` (**the only output** — no `pooler_output`) |
| dtype | `float32` |
| Shape | `[batch_size, floor(height/14)*floor(width/14) + 1, D]` — symbolic, read straight off the graph ✅ |
| `D` | 384 (small) / 768 (base) ✅ |

Decoding to the dense patch grid Method 3 needs:

```python
hp, wp = H // 14, W // 14                 # H, W = the *padded/snapped* input size
tokens = out[0]                           # (1 + hp*wp, D)
cls    = tokens[0]                         # CLS — not used by Methods 3/5
patch  = tokens[1:].reshape(hp, wp, D)     # dense grid, row-major
```

Row-major ordering is the standard ViT flattening and is consistent with the
`floor(H/14)*floor(W/14)` symbolic dim ✅ but the row-vs-column-major orientation itself
is ⚠️ **UNVERIFIED** — confirm with the one-line test in
[Unverified Items](#unverified-items-must-be-confirmed-at-implementation-time).

Method 3 then: L2-normalize `patch` along `D`; mean-pool the crop's tokens → prototype;
L2-normalize the prototype; `sim = patch_norm @ proto` → `(hp, wp)`; bilinearly upsample
`sim` to `(H, W)`; calibrate a threshold; connected components → boxes.

### Dynamic axes

**Fully supported and runtime-verified** ✅. Measured token counts exactly match
`floor(H/14)*floor(W/14)+1` at every resolution tried:

| H × W | tokens returned | `floor(H/14)·floor(W/14)+1` |
|---|---|---|
| 224 × 224 | 257 | 16·16+1 = 257 ✅ |
| 518 × 518 | 1370 | 37·37+1 = 1370 ✅ |
| 644 × 896 | 2945 | 46·64+1 = 2945 ✅ |
| 630 × 462 | 1486 | 45·33+1 = 1486 ✅ |
| 640 × 480 | 1531 | 45·34+1 = 1531 ✅ (non-multiple, floors) |
| 225 × 225 | 257 | 16·16+1 = 257 ✅ (non-multiple, floors) |

This settles the single largest risk in the brief: DINOv2's positional-embedding
interpolation is baked into the graph as a genuinely dynamic op, so "run the scene at high
resolution and upsample the similarity map" (IDEA.md §5, Method 3) works with no export
surgery.

### `fetch-models` implementation sketch

```python
# src/object_search/models/fetch.py  — invoked by `pixi run fetch-models`
DINOV2 = {
    "small": ModelSpec(
        repo_id="onnx-community/dinov2-small-ONNX",
        revision="08c606e3123472a388efa59181b677d428f69bbd",
        filename="onnx/model.onnx",
        dest="models/dinov2_small.onnx",
        sha256=None,  # fill in on first run; then it is a hard gate (EVAL-09 model hash)
        embed_dim=384,
        patch_size=14,
    ),
    "base": ModelSpec(
        repo_id="onnx-community/dinov2-base-ONNX",
        revision="9a1732c724ef0847dcedf00d3d5a93f61dea0370",
        filename="onnx/model.onnx",
        dest="models/dinov2_base.onnx",
        sha256=None,
        embed_dim=768,
        patch_size=14,
    ),
}
# huggingface_hub.hf_hub_download(repo_id=..., filename=..., revision=...)
# then copy/symlink into models/ and record sha256 for EVAL-09 provenance.
```

`huggingface_hub` is a small pure-Python runtime-free dependency, or use plain
`urllib`/`httpx` against the pinned `resolve/<sha>/` URLs above and skip the dependency
entirely. **Prefer the plain URL** — `fetch-models` should not need the HF stack.

### Library review verdict — **Adopt**

| Criterion | Assessment |
|---|---|
| Source | `onnx-community` — HF-affiliated org, 1241 models, 2089 followers, 411 members |
| Provenance | Auto-converted from `facebook/dinov2-{small,base}` via HF's own conversion Space |
| One-person project? | No — organizational |
| Last activity | Repos last modified 2025-04/2025-05; upstream `facebookresearch/dinov2` pushed 2026-06-03, 13.2k stars |
| Popularity of the artifact | Low (59 / 26 downloads, 1 like) — the *only* soft signal against |
| Runtime vs build-time | **Neither** — it is a data download; the runtime dep is `onnxruntime` only |
| License | Apache-2.0 inherited |
| Mitigation for low download count | Verified by hand (this document) *and* a first-party `optimum-cli` fallback documented above |

The low download count is why the revision is pinned and a hash gate is specified: if the
repo ever changes or disappears, the fallback exporter reproduces the same contract.

### Risks

1. **The `onnx-community` repos declare no `license` field** in their card metadata (only
   `base_model`). The license is inherited from `facebook/dinov2-*` (Apache-2.0) but it is
   not stated on the derivative. Record the inheritance chain in `LICENSES.md` (DOC-01).
2. **Low-traffic auto-converted artifacts can be re-generated or deleted.** Mitigated by the
   pinned SHA + local hash gate + documented fallback export.
3. **Stride-14 coarseness is not fixable by resolution alone.** At a long side of 896 the
   token grid is only 64 wide; instances smaller than ~14 px are sub-token. This is the
   known limitation IDEA.md §5 already records; the sub-threshold candidate log (EVAL-08)
   will make it measurable.
4. **Memory at high resolution.** Attention is O(tokens²): 644×896 → 2945 tokens. A 1400-px
   long side on `base` will be slow on CPU and is worth a latency guard in the config model.
5. `num_channels` being symbolic means a 1-channel input will pass shape validation and
   then fail deep in the graph. `ONNXInferencer` should assert `C == 3` explicitly rather
   than trusting the declared axes (exactly the INFRA-09 case).

---

## SuperPoint

### Chosen export path

**Primary — direct download of a frozen release asset.** ✅ verified end-to-end.

```
https://github.com/fabio-sim/LightGlue-ONNX/releases/download/v1.0.0/superpoint.onnx
```

- 5,272,808 bytes (5.03 MiB), released 2023-10-03, immutable release asset.
- Exported from `lightglue_onnx/superpoint.py` at tag `v1.0.0` with the defaults
  ✅ read from that tag's source:
  `descriptor_dim=256`, `nms_radius=4`, `detection_threshold=0.0005`,
  `remove_borders=4`, `max_num_keypoints=None`.
- `max_num_keypoints=None` is why this file emits a **variable** number of keypoints —
  which is what Method 2 wants (see below).

**⚠️ Correction to IDEA.md.** §5 (Method 2) and §14 both state that
`fabio-sim/LightGlue-ONNX` "exports SuperPoint standalone". Verified against `main` at
`pushed_at = 2026-07-24`: `lightglue_dynamo/cli.py::export` unconditionally builds
`Pipeline(extractor, matcher)` and writes `output_names = ["keypoints", "matches",
"mscores"]`. There is **no** `--extractor-only` / `--end2end` flag any more; the README
says so explicitly ("The CLI will export the full extractor-matcher pipeline"). The
standalone-export capability existed in the pre-Dynamo `lightglue_onnx` package and now
survives only as release assets. Update the brief.

**Why the older asset beats a fresh export for *this* project.** Method 2 needs a
*variable* keypoint count on both the crop and the scene: METHOD-04c requires an explicit
low-keypoint diagnostic when an exemplar crop yields <~20 keypoints. A fixed top-K export
(what `lightglue_dynamo.models.SuperPoint` produces) would return K=1024 keypoints on a
40×40 crop, almost all of them noise below threshold, and the low-keypoint guard could
never fire from the output shape. The threshold-based v1.0.0 asset returns 19 keypoints on
a 120×160 input ✅ — the guard reads directly off `keypoints.shape[1]`.

**Fallback — scripted export (export-time only).** `lightglue_dynamo/models/superpoint.py`
is a self-contained `nn.Module` (Apache-2.0 repo code, MagicLeap-derived weights) whose
`forward(image: (B,1,H,W)) -> (keypoints (B,N,2) xy, scores (B,N), descriptors (B,N,256))`
is exactly the contract needed, with fixed `N = num_keypoints`. It self-downloads
`superpoint_v1.pth` from `https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/superpoint_v1.pth`.
Two ways to use it:

```bash
# a) add the repo as an export-time-only dependency, then in a tiny script:
python - <<'PY'
import torch
from lightglue_dynamo.models import SuperPoint
m = SuperPoint(num_keypoints=2048).eval()
torch.onnx.export(
    m, (torch.zeros(1, 1, 1024, 1024),), "models/superpoint.onnx",
    input_names=["image"], output_names=["keypoints", "scores", "descriptors"],
    opset_version=17,
    dynamic_axes={"image": {2: "height", 3: "width"}},
)
PY
```

```bash
# b) or vendor superpoint.py (~200 lines) under src/object_search/export/ and drop the dep.
```

**Rejected candidates:**

- `colmap/LightGlue-ONNX` — ⚠️ **not verified to exist**; every live reference resolves to
  `fabio-sim/LightGlue-ONNX`. Treat the IDEA.md §14 mention as unconfirmed. Do not plan
  against it.
- **HF Optimum path for SuperPoint: does not exist.** ✅ Verified — `grep -i superpoint`
  over `optimum-onnx/optimum/exporters/onnx/model_configs.py` returns nothing. There is no
  `SuperPointOnnxConfig`, so `optimum-cli export onnx --model magic-leap-community/superpoint`
  will fail. Additionally, `transformers`' `SuperPointForKeypointDetection` selects
  keypoints by threshold into a *ragged* per-image list, which is a data-dependent-shape
  export problem; the LightGlue-ONNX formulation avoids it.
- `magic-leap-community/superpoint` (HF) — usable as *weights*, but `license: other` and
  no ONNX path. See [License Concerns](#license-concerns).
- `AXERA-TECH/superpoint`, `thomasonzhou/superpoint-lightglue`, `shadow-cann/...` — NPU/
  vendor-specific or pipeline exports, 0–7 downloads. **Hold.**
- The newer `v2.0` asset `superpoint_lightglue_pipeline.onnx` (51 MB) — this is the fused
  matcher pipeline the project deliberately avoids (IDEA.md §13). Do not download it.

### Input contract

| Property | Value (✅ read from the graph and confirmed at runtime) |
|---|---|
| Input name | `image` |
| dtype | `float32` |
| Shape | `[1, 1, 'height', 'width']` — **batch fixed at 1**, H/W dynamic |
| Layout | **NCHW, single channel (grayscale)** |
| Range | `[0, 1]` — divide by 255 |
| Mean / std | **None.** No mean subtraction, no std division. |
| Color conversion | BT.601 luma from **RGB**: `gray = 0.299·R + 0.587·G + 0.114·B` |
| Stride constraint | 8. Non-multiples are accepted but internally floored (see below) |
| opset / IR | `ai.onnx` 17 / IR 8 |

Exact preprocessing, transcribed from `lightglue_dynamo/preprocessors/superpoint.py` on
`main` (identical in intent to `onnx_runner/utils.py` at `v1.0.0`) ✅:

```python
# input `image` is the project's BGR uint8 scene or crop, shape (H, W, 3)
x = image[..., ::-1] / 255.0                      # BGR -> RGB, scale to [0,1]
x = (x * [0.299, 0.587, 0.114]).sum(-1)           # BT.601 luma -> (H, W)
x = x[None, None].astype(np.float32)              # -> (1, 1, H, W)
```

Note this is numerically the same weighting `cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)` uses,
so `cv2.cvtColor(...).astype(np.float32)[None, None] / 255.0` is an acceptable and faster
equivalent — but write down which one you picked (METHOD-11), because the two differ in
rounding and can shift a borderline keypoint.

**Resize policy.** The `v1.0.0` reference runner resized the **longer side** to
`img_size` (default 512) with `cv2.INTER_AREA`, preserving aspect ratio. For this project,
prefer **no resize at all** on the crop, and resize the scene only if you need to bound
latency — SuperPoint is fully convolutional and dynamic H/W is verified. If you do resize,
record the scale factor: keypoint coordinates come back in *input-image* pixels and must be
divided by the scale to land in original-scene coordinates.

**Padding to a multiple of 8 is recommended.** ✅ Measured: `H=483, W=641` returns
keypoints with `x ∈ [8, 631]`, `y ∈ [8, 471]` — identical bounds to `H=480, W=640`. The
trailing 3 rows / 1 column are silently dropped. Snap to a multiple of 8 (pad, don't
resize) to avoid a silent coordinate-range truncation.

### Output contract

✅ All three read from the graph and confirmed at runtime.

| Output | dtype | Shape | Meaning |
|---|---|---|---|
| `keypoints` | **`int64`** | `[1, N, 2]` | `(x, y)` in **input-image pixel** coordinates, integer. Order is `x` then `y`. |
| `scores` | `float32` | `[1, N]` | Detector confidence, post-NMS. Floor is `detection_threshold = 0.0005`. |
| `descriptors` | `float32` | `[1, N, 256]` | **Already L2-normalized** — measured `‖d‖ = 1.0000` exactly ✅ |

`N` is a single shared symbolic dim (`NonZero_362_o0__d1`) — all three outputs always agree
in length. Because descriptors are pre-normalized, Method 2's kNN can use a plain matmul:
cosine similarity `= D_crop @ D_scene.T`, and squared-L2 `= 2 - 2·cos`. **Do not
re-normalize** and do not use `cv2.BFMatcher` with `NORM_L2` expecting unnormalized inputs.

Measured behaviour on random noise (sanity, not accuracy):

| H × W | N keypoints | x range | y range | score range |
|---|---|---|---|---|
| 480 × 640 | 516 | [8, 631] | [8, 471] | [0.0005, 0.0513] |
| 1024 × 1024 | 1851 | [8, 1015] | [8, 1015] | [0.0005, 0.1801] |
| 120 × 160 | 19 | [25, 134] | [20, 105] | [0.0005, 0.0058] |
| 483 × 641 | 547 | [8, 631] | [8, 471] | [0.0005, 0.0763] |

Effective border exclusion is **8 px**, not the configured `remove_borders=4` (the border
mask is applied on the 8×-upsampled score grid). Note this in the inferencer docstring:
Method 2 will never get a correspondence within 8 px of the scene edge, which matters when
an instance is clipped by the frame.

**No `max_num_keypoints` cap.** On a large textured scene `N` can be several thousand. Cap
it yourself by sorting on `scores` and truncating — this keeps the Hough vote count bounded
and should be a config field (`max_keypoints`, default e.g. 4096) rather than a hard-coded
constant. If you would rather have the cap in the graph, the `v0.1.3` asset
`superpoint_1024.onnx` has the same I/O with `N = Min(1024, ...)` ✅ verified.

### Dynamic axes

- `H`, `W`: **dynamic** ✅ (verified 120×160 through 1024×1024, square and non-square).
- Batch: **fixed at 1**. Method 2 runs the crop and the scene as two separate calls at two
  different sizes, so this is a non-issue — but `ONNXInferencer` must not assume a
  batchable session.
- `N` (keypoint count): **dynamic, data-dependent**. This is the important one for
  INFRA-09: the shape validator must accept symbolic output dims and validate *rank* and
  *dtype* plus the trailing constants (`2`, `256`), not concrete lengths. A validator that
  demands fully static output shapes will reject a working model.

### `fetch-models` implementation sketch

```python
SUPERPOINT = ModelSpec(
    url=("https://github.com/fabio-sim/LightGlue-ONNX/releases/download/"
         "v1.0.0/superpoint.onnx"),
    dest="models/superpoint.onnx",
    size_bytes=5_272_808,          # exact, verified
    sha256=None,                   # fill on first run -> hard gate
    descriptor_dim=256,
    stride=8,
    detection_threshold=0.0005,    # baked into the graph; documented, not settable
    nms_radius=4,                  # baked in
    effective_border_px=8,         # baked in
)
```

Plain `urllib`/`httpx` GET — no third-party library enters the dependency graph at all.
This is the lowest-risk possible adoption: a single 5 MiB file from a GitHub release.

### Library review verdict — **Trial** (download-only; no code dependency)

`fabio-sim/LightGlue-ONNX`, evaluated 2026-07-24:

| Criterion | Value | Read |
|---|---|---|
| Stars | 666 | Healthy for a niche export repo |
| Forks | 73 | Real reuse |
| Open issues | 15 | Low ratio vs 137+ commits and 3 years of history |
| Last push | **2026-07-24** (same day) | Actively maintained |
| Created | 2023-06-26 | 3-year track record |
| Archived | No | — |
| License | Apache-2.0 | Compatible |
| One-person project? | **Yes** — single maintainer (Fabio Milentiansen Sim) | The main structural risk |
| Downstream validation | Integrated into **Kornia** as `kornia.feature.OnnxLightGlue` | Meaningful third-party vetting |
| Code quality signals | typed, `tests/` present, `AGENTS.md`, `uv` + `pyproject`, changelog | Above average |
| Runtime vs build-time | **Neither for the primary path** — we download one release asset. Build-time only for the fallback. | Lowest possible exposure |

**Verdict rationale.** Single-maintainer is normally a Trial-not-Adopt signal, and it stays
Trial here — but the exposure is close to zero. The primary path takes an immutable
release artifact and never imports the package; the fallback path imports it only inside
`fetch-models`, in a separate pixi env, and can be replaced by vendoring one ~200-line
file. Kornia's adoption is real external validation. **Trial, with a documented vendoring
escape hatch.**

The v1.0.0 asset itself gets an extra caveat: it is a **frozen artifact whose exporter no
longer exists on `main`**. It cannot be re-generated from current HEAD. That is precisely
why the fallback export (which *is* reproducible from HEAD, at the cost of fixed top-K) is
specified above, and why the hash must be pinned.

### Risks

1. **Weights license is non-commercial research only.** See
   [License Concerns](#license-concerns). This is the single most important flag in this
   document after the AGPL one.
2. **The primary artifact is not reproducible from source HEAD.** If the GitHub release is
   deleted, the fallback export produces a *different* contract (fixed `N`), which would
   change Method 2's low-keypoint diagnostic. Mirror the 5 MiB file somewhere durable and
   record its hash.
3. **`keypoints` is `int64`.** Sub-pixel refinement is not available, so `translation-2dof`
   Hough bins should not be finer than ~1 px, and pairwise-4DoF scale estimates from
   nearby keypoint pairs will be noisy. IDEA.md §5 (2b) already anticipates this.
4. **No scale or orientation on keypoints** — as IDEA.md §5 (2b) states. Confirmed by
   inspecting the module: the output is `(x, y)` only. `single-4dof` voting is genuinely
   invalid for this backend; the three-mode design is correct and necessary.
5. **8-px effective border** silently excludes edge instances.
6. **Unbounded `N`** on textured scenes → O(N²) blow-up in `pairwise-4dof`. Cap in config.

---

## FastSAM / MobileSAM

### Chosen export path

**Primary — FastSAM-s via a scripted Ultralytics export. ✅ Actually performed and verified
on this machine (ultralytics 8.4.104, torch 2.12.1, opset 17).**

```bash
# weights (immutable GitHub release asset)
https://github.com/ultralytics/assets/releases/download/v8.4.0/FastSAM-s.pt   # 23,851,578 B (22.7 MB)
https://github.com/ultralytics/assets/releases/download/v8.4.0/FastSAM-x.pt   # 144,972,346 B (138 MB)
```

```python
from ultralytics import FastSAM
m = FastSAM("FastSAM-s.pt")            # auto-downloads from the URL above
m.export(format="onnx", imgsz=1024, dynamic=True, simplify=False, opset=17)
# -> FastSAM-s.onnx, 47,203,739 bytes (45.0 MiB), 2.1 s on CPU
```

Backbone is plain **YOLOv8s-seg**: 85 layers, 11,779,987 params, 39.9 GFLOPs at 1024². Embedded
ONNX metadata confirms `task: segment`, `head: Segment`, `names: {0: 'object'}` (single
class → "everything mode"), `stride: 32`, `imgsz: [1024, 1024]`, and
`license: AGPL-3.0 License`.

`simplify=False` was used deliberately — `onnxslim` is an extra export-time dependency and
adds nothing needed here. Set `simplify=True` only if you measure a benefit.

**No reputable pre-exported FastSAM ONNX exists.** ✅ Searched HF Hub: `badongtakla/fastsam-onnx`,
`jasonash1/fastsam-onnx`, `EclipseAidge/Fast_SAM-S`, `Tensorabdullah/FastSAM_Quantized`,
`circulus/FastSAM-{s,x}-ov` — all **0 downloads**, no model cards, unverifiable provenance.
`qualcomm/FastSam-S` / `FastSam-X` are `license: other` and target Qualcomm NPUs. **Hold on
all of them.** Scripting the export is both more reproducible and lower-risk.

`CASIA-LMC-Lab/FastSAM` (the original repo, formerly `CASIA-IVA-Lab`): 8.4k stars,
AGPL-3.0, **last pushed 2024-07-30** — effectively dormant, 146 open issues. Use the
Ultralytics path (pushed 2026-07-24), not the original repo.

**Alternative — MobileSAM. Recommend NOT building this in Milestone 1.**

Best available path if it is ever wanted (`Acly/MobileSAM`, MIT):

```
https://huggingface.co/Acly/MobileSAM/resolve/main/mobile_sam_image_encoder.onnx   # 28,157,093 B (26.9 MiB)
https://huggingface.co/Acly/MobileSAM/resolve/main/sam_mask_decoder_multi.onnx     # 16,496,559 B (15.7 MiB)
https://huggingface.co/Acly/MobileSAM/resolve/main/sam_mask_decoder_single.onnx     # 16,501,323 B (15.7 MiB)
```

Upstream weights `dhkim2810/MobileSAM/mobile_sam.pt` (MIT, 40,728,226 B); also mirrored as
`mobile_sam.pt` in the Ultralytics v8.4.0 assets release. Reproducible export scripts:
`export_image_encoder.py` (in the `Acly/MobileSAM` HF repo) and `scripts/export_onnx_model.py`
(in `ChaoningZhang/MobileSAM`, which exports the **decoder only** — the encoder script is
what `Acly` added).

### Input contract

**FastSAM** ✅ verified:

| Property | Value |
|---|---|
| Input name | `images` |
| dtype | `float32` |
| Shape | `['batch', 3, 'height', 'width']` (channel dim is **static 3**) |
| Layout | **NCHW** |
| Channel order | **RGB** (`im[..., ::-1]` from OpenCV BGR — from `predictor.py::preprocess`) |
| Scale | `/ 255.0` |
| Mean / std | **None.** YOLO does no mean/std normalization. |
| Resize policy | **Letterbox** to `imgsz` (1024×1024): scale by `min(1024/H, 1024/W)`, then pad to 1024×1024 |
| Letterbox fill | ⚠️ **UNVERIFIED numerically** — Ultralytics `LetterBox` pads with a constant grey; the long-standing value is `(114, 114, 114)`. Confirm before writing it down. |
| Letterbox centering | `center=True` by default → padding split evenly on both sides; must be subtracted when mapping boxes back |
| opset / IR | `ai.onnx` 17 / IR 8 |

Exact preprocessing transcribed from `ultralytics/engine/predictor.py::preprocess` ✅:
`LetterBox` → `im[..., ::-1]` (BGR→RGB) → `transpose(0,3,1,2)` (BHWC→BCHW) →
`ascontiguousarray` → `float()` → `/= 255`.

**MobileSAM encoder** ✅ verified from the graph and from `onnx_image_encoder.py`:

| Property | Value |
|---|---|
| Input name | `input_image` |
| dtype | `float32` |
| Shape | `['image_height', 'image_width', 3]` — **HWC, no batch dim** (unusual; do not assume NCHW) |
| Channel order | **RGB**, range `[0, 255]` (do **not** divide by 255) |
| Mean | `[123.675, 116.28, 103.53]` (SAM constants, on the 0–255 scale) |
| Std | `[58.395, 57.12, 57.375]` |
| Resize | Caller resizes **longest side to 1024**, preserving aspect ratio, *before* the call |
| Padding | **Done inside the graph** — right/bottom pad to 1024×1024. Do not pre-pad. |

Normalization, permute, and pad are all baked into the exported graph
(`use_preprocess=True`), so the caller only does: BGR→RGB, resize-longest-side-to-1024,
keep `float32` HWC in 0–255.

### Output contract

**FastSAM** ✅ verified at three resolutions:

| Output | dtype | Shape | Meaning |
|---|---|---|---|
| `output0` | `float32` | `[batch, 37, anchors]` | Per-anchor predictions, **channels-first** |
| `output1` | `float32` | `[batch, 32, mask_h, mask_w]` | Mask **prototypes** at stride 4 |

The 37 channels decompose as (single class, `names: {0: 'object'}`):

```
output0[0, 0:4,  a]   -> cx, cy, w, h   in *letterboxed input* pixel coords (xywh, not xyxy)
output0[0, 4,    a]   -> objectness / class-0 confidence
output0[0, 5:37, a]   -> 32 mask coefficients
```

Measured shapes ✅:

| Input H × W | `output0` | `output1` | anchors = (H/8)(W/8)+(H/16)(W/16)+(H/32)(W/32) |
|---|---|---|---|
| 1024 × 1024 | `(1, 37, 21504)` | `(1, 32, 256, 256)` | 16384+4096+1024 = 21504 ✅ |
| 640 × 640 | `(1, 37, 8400)` | `(1, 32, 160, 160)` | 6400+1600+400 = 8400 ✅ |
| 512 × 768 | `(1, 37, 8064)` | `(1, 32, 128, 192)` | 6144+1536+384 = 8064 ✅ |

Decoding to the proposals Method 5 needs:

```python
p = output0[0].T                              # (anchors, 37)
boxes_xywh, conf, coeff = p[:, :4], p[:, 4], p[:, 5:]
keep = conf > conf_thres                      # FastSAM default conf=0.4
boxes = xywh2xyxy(boxes_xywh[keep])
keep2 = nms(boxes, conf[keep], iou_thres)     # FastSAM default iou=0.9 (deliberately loose:
                                              # "everything mode" wants overlapping proposals)
# masks, only if you need them (Milestone 1's contract is boxes):
protos = output1[0].reshape(32, -1)                       # (32, mh*mw)
masks  = sigmoid(coeff[keep][keep2] @ protos)             # (K, mh*mw)
masks  = masks.reshape(-1, mask_h, mask_w)
masks  = crop_to_box(masks, boxes[keep2] / 4)             # stride-4 proto grid
# then undo letterbox: subtract pad, divide by scale, clip to original H, W
```

Milestone 1's output contract is **boxes** (IDEA.md §4), so `output1` is optional for
Method 5's proposal stage. But Method 5's robustness backlog names "region embedding with
background masked out" — that needs `output1`, so decode it behind a config flag rather
than dropping it.

**MobileSAM** ✅ verified from the graph:

| Model | Inputs | Outputs |
|---|---|---|
| encoder | `input_image` f32 `[H, W, 3]` | `image_embeddings` f32 `[1, 256, 64, 64]` |
| decoder (`_multi`) | `image_embeddings` f32 `[1,256,64,64]`; `point_coords` f32 `[1,'num_points',2]`; `point_labels` f32 `[1,'num_points']`; `mask_input` f32 `[1,1,256,256]`; `has_mask_input` f32 `[1]`; `orig_im_size` f32 `[2]` | `masks` f32 (4-D, dynamic); `iou_predictions` f32 `[·, 4]`; `low_res_masks` f32 (4-D, dynamic) |

`sam_mask_decoder_single.onnx` is the same with a single mask returned.

### Dynamic axes

**FastSAM: fully dynamic** ✅ with `dynamic=True` at export. From `exporter.py`:

```python
dynamic = {"images":  {0: "batch", 2: "height", 3: "width"}}
dynamic["output0"] = {0: "batch", 2: "anchors"}
dynamic["output1"] = {0: "batch", 2: "mask_height", 3: "mask_width"}
```

Verified at 1024², 640², and 512×768 ✅. **H and W must be multiples of 32** (`stride: 32`).
Export at `imgsz=1024` (FastSAM's trained resolution) and keep 1024 as the operating point;
dynamic shapes are a convenience for smaller-image speedups, not a substitute for the
letterbox. Note that `dynamic=True` is *not* the Ultralytics default — pass it explicitly.

**MobileSAM:** encoder H/W dynamic (padded to 1024 internally); decoder `image_embeddings`
is fixed `[1, 256, 64, 64]` and `num_points` is dynamic **within a single prompt**. The
decoder is **not batched over prompts** — this is the crux of the recommendation below.

### `fetch-models` implementation sketch

```toml
# pixi.toml — the export env is separate so torch never enters the runtime env
[feature.export.dependencies]
python      = "3.12.*"
ultralytics = "*"        # AGPL-3.0, export-time only
pytorch-cpu = "*"
torchvision = "*"
onnx        = "*"

[environments]
default = { solve-group = "default" }
export  = { features = ["export"] }

[tasks]
fetch-models = "python -m object_search.models.fetch"
```

```python
# src/object_search/models/fetch.py
def fetch_fastsam(variant: str = "s", imgsz: int = 1024) -> Path:
    """Export FastSAM to ONNX. Runs in the `export` pixi env (needs torch).

    AGPL-3.0: ultralytics is an EXPORT-TIME-ONLY dependency and is never imported
    by the runtime package. The produced .onnx is still AGPL-3.0 (see LICENSES.md).
    """
    from ultralytics import FastSAM                       # noqa: PLC0415 (export env only)
    m = FastSAM(f"FastSAM-{variant}.pt")                   # -> ultralytics/assets v8.4.0
    out = m.export(format="onnx", imgsz=imgsz, dynamic=True, simplify=False, opset=17)
    return shutil.move(out, f"models/fastsam_{variant}.onnx")

# MobileSAM (only if/when the alternative backend is actually built):
MOBILESAM = [
    ModelSpec(url="https://huggingface.co/Acly/MobileSAM/resolve/main/mobile_sam_image_encoder.onnx",
              dest="models/mobilesam_encoder.onnx", size_bytes=28_157_093),
    ModelSpec(url="https://huggingface.co/Acly/MobileSAM/resolve/main/sam_mask_decoder_multi.onnx",
              dest="models/mobilesam_decoder.onnx", size_bytes=16_496_559),
]
```

Run as `pixi run -e export fetch-models`. Every downloaded/exported file gets a recorded
`sha256` for EVAL-09 provenance.

### Library review verdict

**`ultralytics` — Trial (export-time only), with an explicit AGPL carve-out.**

| Criterion | Value | Read |
|---|---|---|
| Stars | 59,835 | Dominant in the space |
| Open issues | 169 | Very low for that scale (aggressive triage) |
| Last push | **2026-07-24** | Extremely active |
| One-person project? | No — company-backed | — |
| License | **AGPL-3.0** | **The problem.** See below. |
| Runtime vs build-time | **Export-time only** — `import ultralytics` appears in exactly one function, in a separate pixi env | The mitigation |
| Alternatives | None credible — every community FastSAM ONNX has 0 downloads | Forced choice |

Maintenance is not the concern; the license is. The verdict is Trial rather than Adopt
because the dependency is retained **only** for as long as FastSAM is the proposal backend,
and because AGPL means the decision must be revisited before this repo is ever published or
network-exposed.

**`Acly/MobileSAM` — Assess.** MIT, correct and verified I/O, includes its export scripts,
derived from `ChaoningZhang/MobileSAM` (5.8k stars, Apache-2.0, pushed 2026-05-05). But it
was last modified 2023-08-03, reports 0 downloads, and is a personal repo. Acceptable as a
*data* download for a Milestone-2+ spike. Do not put it on the M1 critical path.

**`awarebayes/MobileSamONNX` — HOLD. Recommend against.** Evaluated 2026-07-24:
**0 stars**, 1 fork, 0 open issues, created 2025-05-12, last pushed 2025-05-16 (a 4-day
project, untouched for 14 months), single author, no external validation. Apache-2.0, which
is the only positive. IDEA.md §14 names it — **drop that reference.** A first-party path
exists (`ChaoningZhang/MobileSAM`'s own export scripts) and a better-provenanced
pre-exported artifact exists (`Acly/MobileSAM`). Adopting a zero-star, 4-day-old,
single-author repo when a first-party path exists is exactly the case the library-review
gate is meant to catch.

### Risks

1. **AGPL-3.0 propagates to the artifact, not just the toolchain.** The exported
   `FastSAM-s.onnx` embeds `license: AGPL-3.0 License` in its own metadata. See
   [License Concerns](#license-concerns).
2. **MobileSAM "everything mode" over ONNX is a phase of work, not a config flag.** The
   decoder signature `point_coords: [1, num_points, 2]` treats all `num_points` as points
   belonging to **one** prompt, so a 32×32 automatic-mask grid requires ~1024 sequential
   decoder calls plus a hand-ported `SamAutomaticMaskGenerator` (multi-crop layers,
   stability-score filtering, per-crop box NMS, mask deduplication). IDEA.md §5 says
   MobileSAM's "automatic-mask path is heavier" — this quantifies it. ⚠️ Per-call latency
   **UNVERIFIED**, but even 10 ms/call is ~10 s per image, versus one FastSAM forward pass.
   **Recommendation: keep MobileSAM as a documented backlog item, not a Phase 7 backend.**
   The AGPL escape hatch it was intended to provide does not come cheap.
3. **FastSAM's loose default `iou=0.9`** is intentional for everything-mode but produces
   heavily overlapping proposals. Method 5's retrieval NMS must be the deduplicating step,
   and the duplicate/fragment convention (EVAL-16, "two boxes on one instance = 1 TP + 1 FP")
   will be exercised hard by this. Surface the raw proposal count in diagnostics.
4. **Letterbox coordinate round-trip is a classic silent bug.** Boxes come back in
   letterboxed 1024×1024 space with centered padding. Getting `(box - pad) / scale` wrong
   yields plausible-looking boxes that are systematically offset. Write a unit test with a
   synthetic non-square image (testing skill), not just an eyeball check.
5. **FastSAM-x is ~276 MiB** ⚠️ (est. from 72M params × 4 B; the `.pt` is 138 MB fp16).
   Confirm the real size if you export it. FastSAM-s is the right default.
6. **Ultralytics pulls torch + torchvision into the export env** — a large solve. Keeping it
   in a separate pixi feature/environment keeps the runtime env `onnxruntime`-only, which is
   what IDEA.md §8 demands.
7. **Ultralytics moves fast and can change export defaults between versions.** Pin
   `ultralytics` to the exact version used (8.4.104 verified here) in `pixi.lock`, and hash
   the resulting `.onnx`, or the model silently changes under you and EVAL-09 provenance
   becomes meaningless.

---

## License Concerns

Three separate flags, in descending order of practical impact. All three should be recorded
in `LICENSES.md` (DOC-01).

### 1. Ultralytics / FastSAM — AGPL-3.0. **Stated plainly: this is a real constraint.**

- `ultralytics` (the exporter) is AGPL-3.0.
- `CASIA-LMC-Lab/FastSAM` (the original weights) is AGPL-3.0.
- The **exported ONNX file itself** carries `license: AGPL-3.0 License
  (https://ultralytics.com/license)` in its embedded metadata ✅ (read directly from
  `FastSAM-s.onnx`).

What this means, concretely:

| Scenario | AGPL obligation |
|---|---|
| Private repo, local single-user demo, never distributed, never network-exposed | **None.** AGPL triggers on conveying the work or on §13 remote network interaction. Private internal use is unrestricted. This is the current Milestone-1 scope (IDEA.md §4: "Local single-user demo"). |
| Repo made public on GitHub | Publishing source *is* conveying. The whole combined work must be AGPL-3.0-licensed. |
| The FastAPI app exposed to any other user over a network | **AGPL §13 fires.** Complete corresponding source must be offered to every user interacting with it remotely. |
| Any commercial use | Requires an Ultralytics Enterprise License. |

The AGPL "export-time only" argument **does not fully save you here**, and it is important
not to pretend otherwise. Isolating `ultralytics` in a separate pixi env protects the
*runtime dependency graph*, but the *weights* are AGPL-derived and ship as part of the
system's behaviour. If this repo is ever published or network-exposed, FastSAM must either
be dropped or the project relicensed AGPL-3.0.

**Recommendation:** proceed with FastSAM for Milestone 1 (private exploration, local demo —
no obligation triggers), and record this as an explicit, dated decision in `LICENSES.md`
with a "revisit before publishing or exposing the API" trigger. Note that MobileSAM's
MIT/Apache licensing is the clean alternative but costs a hand-ported automatic-mask
pipeline (see FastSAM/MobileSAM Risks #2) — the escape hatch exists but is not cheap.

### 2. SuperPoint weights — MagicLeap non-commercial research-only. **Also a real constraint.**

The `LICENSE` in `magic-leap-community/superpoint` ✅ reads, verbatim in its header:

> **ACADEMIC OR NON-PROFIT ORGANIZATION NONCOMMERCIAL RESEARCH USE ONLY**

Notable clauses read in full:

- "personal, non-exclusive, non-transferable license to use the Software for noncommercial
  research purposes, without the right to sublicense"
- "**DERIVATIVES**: You may create derivatives of or make modifications to the Software,
  however, You agree that all and any such derivatives and modifications will be owned by
  Licensor" — **an exported ONNX file is a derivative.**
- "You may not sell, rent, lease, sublicense, lend, time-share or transfer, in whole or in
  part, or provide third parties access to prior or present versions"

This applies to the *weights*, and therefore to `superpoint.onnx`, regardless of the
Apache-2.0 license on `fabio-sim/LightGlue-ONNX`'s code. HF tags both
`magic-leap-community/superpoint` and `ETH-CVG/lightglue_superpoint` as `license: other` ✅
for exactly this reason.

| Scenario | Status |
|---|---|
| Private, non-commercial, internal research/exploration | **Permitted** — this is the licensed use case. Milestone 1 fits. |
| Redistributing `superpoint.onnx` (e.g. committing it, or mirroring it publicly) | **Not permitted.** |
| Any commercial use | **Not permitted.** |

**Recommendation:** SuperPoint is fine for this repo as scoped. Two hard rules: (a) the
weights stay gitignored — which INFRA-11 already mandates, so the constraint costs nothing;
(b) if Method 2's learned backend ever needs to be commercial or redistributable, swap to
**DISK** or **ALIKED** (both already in `lightglue_dynamo`, both from `cvg` under
permissive terms — verify at that time). This is already in Method 2's robustness backlog
as "DISK / ALIKED as additional backends"; note the license motivation there too.

### 3. DINOv2 — Apache-2.0. **Clean. No concerns.**

✅ Verified in three places: `facebookresearch/dinov2` repo license is Apache-2.0;
`facebook/dinov2-small` and `facebook/dinov2-base` both carry `license: apache-2.0` in card
metadata and tags; `facebook/dinov2-base/config.json` states Apache 2.0. (DINOv2 was
originally CC-BY-NC-4.0 and was relicensed to Apache-2.0 — the current release is
unambiguously Apache-2.0.)

Minor bookkeeping only: the `onnx-community/dinov2-*-ONNX` derivative repos declare **no**
license field (their `cardData` has only `library_name` and `base_model`). Record the
inheritance from `facebook/dinov2-*` explicitly in `LICENSES.md`.

### 4. MobileSAM — MIT / Apache-2.0. **Clean.**

`ChaoningZhang/MobileSAM` code is Apache-2.0; `dhkim2810/MobileSAM` weights are MIT;
`Acly/MobileSAM` ONNX exports are MIT. The cleanest licensing of any proposal backend — and
the reason it is worth keeping on the backlog despite the implementation cost.

---

## Unverified Items (must be confirmed at implementation time)

| # | Item | How to confirm |
|---|---|---|
| 1 | **DINOv2 patch-token grid orientation** (row-major `(hp, wp)` vs `(wp, hp)`). Row-major is near-certain from the symbolic dim and standard ViT flattening, but a transposed similarity map is a subtle, plausible-looking bug. | Feed a deliberately non-square, asymmetric image (e.g. a bright 20-px square in the top-left of a 448×896 input). Reshape to `(hp, wp, D)`, cosine-match against that patch's own token, and assert the argmax lands at `(1, 1)`-ish, not `(1, 32)`. One test, permanently. |
| 2 | **Ultralytics letterbox fill value** (assumed `(114, 114, 114)`). | `python -c "from ultralytics.data.augment import LetterBox; import inspect; print(inspect.getsource(LetterBox.__call__))"` and read the `cv2.copyMakeBorder(..., value=...)` argument. Do not hard-code 114 until you have. |
| 3 | **FastSAM-x ONNX file size** (estimated ~276 MiB from 72M params × 4 B). | Export it and `ls -la`, or skip — FastSAM-s is the default. |
| 4 | **DINOv2 fp16 / int8 variant behaviour.** Sizes confirmed from the HF file listing; numerical fidelity **not** tested. | Export/download `model_fp16.onnx`, run both at the same resolution, and report `max|Δcos|` on a real image pair. Only then consider fp16 for latency. |
| 5 | **MobileSAM decoder per-call CPU latency**, which determines whether a 1024-point everything-mode grid is remotely viable. | `ort` session on `sam_mask_decoder_multi.onnx`, one dummy prompt, 100 iterations, report the median. If >5 ms, the grid approach is dead on CPU. |
| 6 | **SHA-256 of every artifact.** Sizes are exact and verified; hashes were not computed. | `shasum -a 256 models/*.onnx` on first `fetch-models` run; commit the hashes into the fetch spec as a hard gate (EVAL-09). |
| 7 | **`colmap/LightGlue-ONNX` existence** (named in IDEA.md §14). Every search resolved to `fabio-sim/LightGlue-ONNX`; no such repo was found. | `gh repo view colmap/LightGlue-ONNX`. If absent, delete the reference from IDEA.md §14. |
| 8 | **Quantized DINOv2 duplicates.** `model_int8.onnx`, `model_quantized.onnx`, and `model_uint8.onnx` are all exactly 24,427,763 bytes (small) / 90,948,899 (base) — almost certainly byte-identical copies. Cosmetic; do not download all three. | `shasum -a 256` on any two. |
| 9 | **`ONNXInferencer` shape-validation policy for symbolic output dims.** SuperPoint's `N` and FastSAM's `anchors` are data-dependent; a validator demanding static output shapes will reject working models. | Design decision, not a lookup: validate rank + dtype + trailing static dims (`2`, `256`, `37`, `32`), treat symbolic dims as wildcards. Write the failing-model test (INFRA-09) for a *wrong* model, not a dynamic one. |

Everything else in this document was verified against a live source or by executing the
model. Items explicitly runtime-verified: DINOv2 small/base I/O and dynamic resolution
(6 resolutions), SuperPoint v1.0.0 and v0.1.3 I/O and dynamic resolution (4 resolutions,
including descriptor L2-norm = 1.0 and the 8-px effective border), FastSAM-s export from
`.pt` and I/O at 3 resolutions including embedded metadata.

---

## Sources

**DINOv2**
- https://huggingface.co/onnx-community/dinov2-small-ONNX — chosen artifact (rev `08c606e3123472a388efa59181b677d428f69bbd`)
- https://huggingface.co/onnx-community/dinov2-base-ONNX — chosen artifact (rev `9a1732c724ef0847dcedf00d3d5a93f61dea0370`)
- https://huggingface.co/facebook/dinov2-small — base weights, Apache-2.0
- https://huggingface.co/facebook/dinov2-small/blob/main/preprocessor_config.json — mean/std/rescale/resample
- https://huggingface.co/facebook/dinov2-base/blob/main/config.json — `hidden_size=768`, `patch_size=14`, `image_size=518`
- https://github.com/facebookresearch/dinov2 — upstream, Apache-2.0
- https://github.com/huggingface/optimum-onnx — `Dinov2OnnxConfig`, `ViTOnnxConfig`, `_TASK_TO_COMMON_OUTPUTS`
- https://huggingface.co/docs/optimum/exporters/onnx/usage_guides/export_a_model — `optimum-cli export onnx`
- https://huggingface.co/sefaburak/dinov2-small-onnx — rejected candidate (0 downloads, 2024-01)

**SuperPoint**
- https://github.com/fabio-sim/LightGlue-ONNX — 666 stars, Apache-2.0, pushed 2026-07-24
- https://github.com/fabio-sim/LightGlue-ONNX/releases/tag/v1.0.0 — the `superpoint.onnx` asset
- https://github.com/fabio-sim/LightGlue-ONNX/blob/main/lightglue_dynamo/cli.py — proves the CLI now exports only the fused pipeline
- https://github.com/fabio-sim/LightGlue-ONNX/blob/main/lightglue_dynamo/models/superpoint.py — standalone module, fallback export
- https://github.com/fabio-sim/LightGlue-ONNX/blob/main/lightglue_dynamo/preprocessors/superpoint.py — BT.601 grayscale preprocessing
- https://github.com/fabio-sim/LightGlue-ONNX/blob/v1.0.0/lightglue_onnx/superpoint.py — the baked-in `detection_threshold=0.0005`, `nms_radius=4`, `remove_borders=4`
- https://github.com/fabio-sim/LightGlue-ONNX/blob/v1.0.0/onnx_runner/utils.py — v1.0.0-era resize/normalize reference
- https://huggingface.co/magic-leap-community/superpoint — weights; `license: other`
- https://huggingface.co/magic-leap-community/superpoint/blob/main/LICENSE — the non-commercial terms quoted above
- https://kornia.readthedocs.io/en/latest/feature.html — `kornia.feature.OnnxLightGlue`, third-party validation

**FastSAM / MobileSAM**
- https://docs.ultralytics.com/models/fast-sam/ — weights URLs, `imgsz=1024`, everything-mode defaults
- https://github.com/ultralytics/ultralytics — 59.8k stars, AGPL-3.0, pushed 2026-07-24
- https://github.com/ultralytics/ultralytics/blob/main/ultralytics/engine/exporter.py — ONNX output names and dynamic-axes map
- https://github.com/ultralytics/ultralytics/blob/main/ultralytics/engine/predictor.py — preprocess: letterbox, BGR→RGB, BCHW, `/255`
- https://github.com/ultralytics/ultralytics/blob/main/ultralytics/data/augment.py — `LetterBox`
- https://github.com/ultralytics/assets/releases/tag/v8.4.0 — `FastSAM-s.pt`, `FastSAM-x.pt`, `mobile_sam.pt` sizes
- https://github.com/CASIA-IVA-Lab/FastSAM → `CASIA-LMC-Lab/FastSAM` — AGPL-3.0, dormant since 2024-07-30
- https://github.com/ChaoningZhang/MobileSAM — 5.8k stars, Apache-2.0, pushed 2026-05-05; `scripts/export_onnx_model.py`
- https://huggingface.co/Acly/MobileSAM — pre-exported encoder + decoders, MIT
- https://huggingface.co/Acly/MobileSAM/blob/main/mobile_sam_encoder_onnx/onnx_image_encoder.py — SAM `pixel_mean`/`pixel_std`, in-graph pad
- https://huggingface.co/dhkim2810/MobileSAM — upstream weights, MIT
- https://github.com/awarebayes/MobileSamONNX — **HOLD**: 0 stars, 4-day project, untouched since 2025-05-16
