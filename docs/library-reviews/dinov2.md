# Library Review — DINOv2 (ONNX) — **Adopt**

**Subject:** `onnx-community/dinov2-small-ONNX` (`onnx/model.onnx`), the pre-exported fp32
DINOv2-small backbone this project runs for dense patch tokens (Method 3 `dino-dense`, Phase 6;
reused by Method 5 `propose-retrieve`, Phase 7).

**Verdict:** **Adopt.** Pinned revision, clean Apache-2.0 licence, runtime-verified I/O contract,
and a first-party `optimum-cli` export path as a fallback. This supersedes IDEA.md §14's
`sefaburak*` references (see *Rejected candidates*).

| | |
|---|---|
| Artifact | `onnx-community/dinov2-small-ONNX` → `onnx/model.onnx` |
| **Pinned revision** | `08c606e3123472a388efa59181b677d428f69bbd` (lastModified 2025-05-09) |
| Size | ~84.4 MiB (fp32) |
| Base weights | `facebook/dinov2-small` (3.26 M downloads), `license: apache-2.0` |
| Licence | **Apache-2.0** (inherited; see *Licence* below) |
| Registry key | `dinov2-small` in `src/object_search/inference/models.py` |
| Full analysis | `.planning/research/MODELS.md` § DINOv2 (runtime-verified) |

## Why Adopt

- **Maintenance / provenance.** Auto-converted from `facebook/dinov2-small` via Hugging Face's
  own conversion Space, hosted under the reputable `onnx-community` org. Upstream
  `facebookresearch/dinov2` is actively maintained (13.2k stars, pushed 2026-06-03). The base
  `facebook/dinov2-small` has 3.26 M downloads.
- **Contract, runtime-verified** (not inferred). Input `pixel_values`, f32, NCHW, all four dims
  dynamic, RGB, `rescale 1/255`, mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`,
  bicubic (`resample: 3`). Output `last_hidden_state` `[B, floor(H/14)*floor(W/14)+1, 384]`,
  index 0 = CLS, the only output (no `pooler_output`). Token counts verified equal to
  `floor(H/14)*floor(W/14)+1` at six resolutions — see MODELS.md.
- **Dynamic resolution works with no export surgery.** The positional-embedding interpolation is
  baked into the graph as a genuinely dynamic op, so "run the scene at high resolution and
  bilinearly upsample the similarity map" is supported directly.
- **Type / dtype support.** Plain ONNX Runtime `session.run`; no custom ops. fp32 is chosen over
  the fp16/int8/q4 siblings because quantized ViT feature extractors degrade the
  cosine-similarity geometry that Method 3 thresholds on.
- **Fallback exists (first-party).** `optimum-cli export onnx --model facebook/dinov2-small
  --task feature-extraction` produces the same `pixel_values` → `last_hidden_state` contract;
  `dinov2` is registered for `feature-extraction` and inherits `ViTOnnxConfig`.

## Licence — Apache-2.0, clean, verified three ways

DINOv2 (originally CC-BY-NC) was **relicensed to Apache-2.0**. Verified in three places:

1. `facebookresearch/dinov2` repository licence is Apache-2.0.
2. `facebook/dinov2-small` and `facebook/dinov2-base` both carry `license: apache-2.0` in card
   metadata and tags.
3. `facebook/dinov2-base/config.json` states Apache-2.0.

Bookkeeping caveat: the `onnx-community/dinov2-*-ONNX` derivative repos declare **no** licence
field of their own. The inheritance from `facebook/dinov2-small` is therefore recorded
explicitly here and in `assets/demo/LICENSES.md`. No publishing or network-exposure constraints
apply (unlike the SuperPoint and FastSAM weights).

## Rejected candidates (from IDEA.md §14)

- `sefaburak/dinov2-small-onnx` (`dinov2_vits14.onnx`, 82.6 MiB) — **Hold**. 0 downloads, last
  modified 2024-01-07, single-author, I/O contract undocumented. Superseded by the verified
  `onnx-community` artifact.
- `sefaburakokcu/dinov2_onnx` — not needed; the `onnx-community` artifact is verified and the
  `optimum-cli` fallback is first-party.
- `deepghs/dinov2_onnx` — not evaluated; no reason to prefer it.

## Adoption constraints carried into the inferencer

Recorded so the `Adopt` verdict travels with its caveats:

- **Snap input sides to a multiple of 14** and return the exact scale factors — the graph
  silently floor-divides trailing pixels on a non-multiple side (a systematic offset, not an
  error).
- **Strip CLS + any register tokens** with `[1 + n_register:]`, `n_register` derived from the
  token count, asserted at load (INFRA-09 discipline).
- **Patch-grid orientation is verified empirically**, not assumed — a transposed similarity map
  is a plausible-looking bug that only a non-square, off-centre fixture can catch.
- **Positional-embedding interpolation is fixed at export time** (HF's `align_corners=False`
  differs from FB's `offset=0.1` + antialiasing). Do not re-derive or "correct" it; the export
  in use is the `onnx-community` fp32 graph at the pinned revision above.
