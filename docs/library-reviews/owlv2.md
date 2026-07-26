# Library Review — OWLv2 (ONNX via transformers export) — **Trial**

**Subject:** a **scripted `transformers` + `torch` export** of `google/owlv2-base-patch16-ensemble`
into a single-input image-guided vision graph (`pixel_values` → `class_embeds`, `pred_boxes`), the
learned detector backing Method 4 (`owlv2-oneshot`) — the source-research exemplar-detector bucket,
realized permissively. The runtime package loads the exported
`.onnx` under ONNX Runtime and imports **nothing** from `torch` or `transformers`; the exporter
lives only in the `export` pixi environment.

**Verdict:** **Trial** — export-time-only dependency, **contract not yet runtime-verified**. OWLv2
is a mature, widely-used Google model on a permissive licence, so the reservation is *not* code
quality or licensing. It is that, unlike the other three models, **no weights were fetched in the
authoring environment**, so the exact preprocessing constants and graph I/O are the documented
HuggingFace values *asserted at export* rather than measured in `.planning/research/MODELS.md`. The
first in-env `pixi run -e export export-owlv2` is the verification step; it pins the `sha256`.

| | |
|---|---|
| Artifact | Scripted export: `google/owlv2-base-patch16-ensemble` → `owlv2_base_patch16.onnx` |
| Exporter | `transformers` + `torch` (conda-forge), **export env only**, Apache-2.0 |
| Export call | `torch.onnx.export` of a `pixel_values → (class_embeds, pred_boxes)` wrapper, opset 17 |
| Registry key | `owlv2-base-patch16` in `src/object_search/inference/models.py` (`source="export"`) |
| Reproduce | `pixi run -e export export-owlv2` (script `scripts/export_owlv2.py`) |
| **Licence** | **Apache-2.0** ✅ — code *and* weights, no copyleft, no non-commercial clause |
| Full analysis | Pending first in-env export (this review + the export-time graph assertions) |
| Evaluated | 2026-07-25 |

## Why Trial (and why OWLv2 at all)

- **The runtime path imports nothing from `torch`/`transformers`.** The dependency is quarantined in
  the `export` pixi feature (`transformers`, `pytorch`, `torchvision`, `onnx`), so the default
  runtime environment stays torch-free — the "ONNX Runtime for every learned model" constraint stays
  **structural**, exactly as for FastSAM.
- **Permissive licence is the deciding factor.** This method exists because the two visual-prompt
  detectors originally proposed — **T-Rex2** and **Rex-Omni** — are both **IDEA License 1.0**
  (non-commercial, research-only); Rex-Omni additionally inherits the **Qwen Research License**, and
  T-Rex2 is furthermore **API-only** (no downloadable weights → violates local-first + ONNX-only
  outright). OWLv2 delivers the same image-conditioned "one example box → all instances" capability
  under **Apache-2.0**, the same permissive tier as the repo's existing DINOv2/SuperPoint/FastSAM
  code, so it introduces no new sharing constraint.
- **Single-input graph keeps the ONNX layer simple.** Image-guided detection is a "two image" task,
  but the export is a single-input vision encoder run twice (query crop, then scene); the
  query-embedding selection and cosine scoring stay in NumPy in the readable method module. No
  custom two-input ONNX contract, no argmax/heuristic baked into the graph.

## Adoption constraints carried into the inferencer

Mirrored in the `OWLv2Inferencer` docstring and `docs/methods/owlv2-oneshot.md`:

- **Static 960×960 input, patch 16 → 3600 patches.** OWLv2's learned position embeddings fix the
  resolution; a different size silently mis-indexes them.
- **Preprocessing is OWLv2's own policy, not the base stretch/letterbox path.** Rescale `1/255`; pad
  **bottom-right** to a square of side `max(H, W)` with grey `0.5`; resize to 960×960 bilinear;
  normalize with CLIP mean `[0.48145466, 0.4578275, 0.40821073]` / std
  `[0.26862954, 0.26130258, 0.27577711]`. Because the pad is bottom-right, a normalized `pred_box`
  maps to scene pixels by a plain multiply by `max(H, W)` — no pad offset. `preprocess` is overridden
  in the inferencer (the base docstring anticipates this for non-standard backends).
- **Outputs are ours, named at export.** `class_embeds` f32 `[batch, num_patches, 512]` (projected
  class-head embeddings, L2-normalized in the method) and `pred_boxes` f32
  `[batch, num_patches, 4]` (`(cx, cy, w, h)` normalized to `[0, 1]`). Only the input is validated at
  load (INFRA-09); the patch dim exports symbolic, so the export script asserts names + last dims.

## Licence — Apache-2.0 (the load-bearing *advantage*)

OWLv2 and its exporter are **Apache-2.0**. There is no AGPL §13 network clause (as with FastSAM) and
no non-commercial restriction (as with SuperPoint's MagicLeap weights, or the rejected T-Rex2 /
Rex-Omni). The exported `.onnx` is still gitignored (INFRA-11) and arrives only via
`pixi run -e export export-owlv2`, for provenance and size reasons, not licence ones — private local
use, publishing the repo, and network-exposing the API are all within terms.

## Rejected candidates (why not the visual-prompt detectors)

- **T-Rex2** (IDEA-Research) — **Hold / rejected.** IDEA License 1.0, **non-commercial research
  only**, *and* **API-only**: the repo is "API code for T-Rex2" against `cloud.deepdataspace.com`
  with a token; no downloadable weights, no ONNX. Violates local-first + ONNX-only + the licence
  bar simultaneously.
- **Rex-Omni** (IDEA-Research, CVPR 2026) — **Hold / rejected.** Open weights (HF), but a **3B
  Qwen-based MLLM** run via `transformers`/`vLLM` (PyTorch, no ONNX path — collides with ONNX-only),
  under **IDEA License 1.0 + Qwen Research License** (doubly non-commercial). SOTA zero-shot
  accuracy (COCO F1 72.0 @ IoU 0.5) but wrong task shape (text/point-prompted, not exemplar-box),
  poor high-IoU localization (F1 15.9 @ 0.95), and seconds-per-image latency. Unshippable here.
- **YOLO-World** — **Reject.** **GPL-v3** (copyleft — would encumber the repo) and text-prompted, not
  exemplar-conditioned.
- **DE-ViT** — **Assess (deferred).** Genuinely few-shot from example images, but a heavy
  Detectron2 stack with no clean ONNX export path; revisit only if OWLv2's fixed-resolution
  small-object ceiling proves limiting.
- **Pre-exported OWLv2 ONNX (e.g. transformers.js / Xenova)** — **Hold.** Those export the full
  text+image model with text inputs, not the custom image-guided `class_embeds`/`pred_boxes` head
  this method needs. The scripted export is both more fit-for-purpose and better-provenanced.
