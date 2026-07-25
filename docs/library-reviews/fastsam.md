# Library Review — FastSAM (ONNX via Ultralytics) — **Trial**

**Subject:** the **scripted Ultralytics export** of `FastSAM-s.pt` → `fastsam_s.onnx`, the
class-agnostic proposal backend for Method 5 (`propose-retrieve`, Phase 7). The runtime package
loads the exported `.onnx` under ONNX Runtime and imports **nothing** from `ultralytics`; the
exporter itself (`ultralytics`, AGPL-3.0) lives only in the `export` pixi environment.

**Verdict:** **Trial** — export-time only, **licence-gated**. The export was actually performed
and verified during research (ultralytics 8.4.104, torch 2.12.1, opset 17); the graph I/O is
runtime-verified at three resolutions. The reservation is not code quality — `ultralytics` is
dominant and well-maintained — it is **AGPL-3.0**, which is a real constraint on how this repo
may later be shared (see *Licence* below).

| | |
|---|---|
| Artifact | Scripted export: `FastSAM-s.pt` (ultralytics/assets **v8.4.0**) → `fastsam_s.onnx` |
| Exporter | `ultralytics` (PyPI), **export env only**, AGPL-3.0 |
| Export call | `FastSAM("FastSAM-s.pt").export(format="onnx", imgsz=1024, dynamic=True, simplify=False, opset=17)` |
| Size | `.pt` 23,851,578 B (22.7 MB) → `.onnx` 47,203,739 B (45.0 MiB), ~2.1 s on CPU |
| Registry key | `fastsam-s` in `src/object_search/inference/models.py` (`source="export"`) |
| Reproduce | `pixi run -e export export-fastsam` (script `scripts/export_fastsam.py`) |
| **Licence** | **AGPL-3.0** ⚠️ — code *and* the exported weights (see *Licence* below) |
| Full analysis | `.planning/research/MODELS.md` § FastSAM / MobileSAM (runtime-verified) |
| Evaluated | 2026-07-24 |

## Why Trial (export-time only)

- **The runtime path imports nothing from `ultralytics`.** The AGPL dependency is quarantined in
  the `export` pixi feature (`[feature.export.dependencies]`: `ultralytics`, `pytorch`,
  `torchvision`, `onnx`). The default/runtime environment stays torch-free, which makes the
  project's "ONNX Runtime for every learned model" constraint **structural** rather than a
  convention reviewers must police.
- **Runtime-verified contract, not inferred.** Input `images` f32 NCHW `[batch, 3, H, W]`
  (channel dim static 3), **RGB**, scale `1/255`, **no mean/std**, letterbox to 1024×1024 with
  fill 114. Outputs `output0` f32 `[batch, 37, anchors]` (4 box + 1 conf + 32 mask coeffs,
  channels-first) and `output1` f32 `[batch, 32, mask_h, mask_w]` (mask prototypes at stride 4).
  Verified at 1024² (`21504` anchors, `256²` protos), 640², and 512×768.
- **Scripting the export beats every pre-exported ONNX on offer.** Every FastSAM ONNX on HF Hub
  (`badongtakla/fastsam-onnx`, `jasonash1/fastsam-onnx`, `EclipseAidge/Fast_SAM-S`,
  `Tensorabdullah/FastSAM_Quantized`, `circulus/FastSAM-*-ov`) has **0 downloads** and
  unverifiable provenance; `qualcomm/FastSam-*` are `license: other` and NPU-targeted. A scripted
  export is both more reproducible and lower-risk. **Hold on all pre-exported artifacts.**

## Adoption constraints carried into the inferencer

Mirrored in the `FastSAMInferencer` docstring and `docs/methods/propose-retrieve.md`:

- **Letterbox to 1024×1024, fill 114, RGB, `/255`, no mean/std** — YOLO does no normalization.
  The letterbox scale/pad must be undone to map boxes back to image pixels.
- **Decode YOLOv8-seg style, and crop each mask to its own box.** Transpose `output0` →
  `[anchors, 37]`; split 4 box + 1 conf + 32 coeffs; confidence-filter; NMS; for survivors
  `masks = sigmoid(coeffs @ protos.reshape(32, -1))` reshaped to `mask_h×mask_w`; **crop each
  mask to its own box** (mandatory — without it a mask bleeds outside its detection); upsample;
  undo the letterbox.
- **NMS is deliberately loose (default iou=0.9).** "Everything mode" *wants* overlapping
  proposals; SAM over-segmentation (one object → several proposals) is handled by NMS *after*
  retrieval, not here. Record the proposal count in diagnostics.
- **Masks are optional in Milestone 1.** The output contract is **boxes**; `output1` is decoded
  behind a config flag because the robustness backlog's "region embedding with background masked
  out" needs it.

## Licence — AGPL-3.0 (the load-bearing flag, one of three records)

FastSAM is **AGPL-3.0**, and — verified from the exported graph's embedded ONNX metadata
(`license: AGPL-3.0 License`) — **the exported `.onnx` file itself embeds that licence string**.
Export-time isolation protects the runtime *dependency graph* but **not the weights**.
Consequences carried into the project (this is **one of the three places** the AGPL constraint is
recorded; the others are `assets/demo/LICENSES.md` and the `fastsam-s` `ModelSpec.license_note`
in `src/object_search/inference/models.py`):

1. **Private local use triggers nothing.** Milestone 1 is a local, single-user, private
   exploration — fully within terms.
2. **Publishing this repo publicly fires the AGPL copyleft**, and **network-exposing the FastAPI
   app fires AGPL §13** (the "remote network interaction" clause): users interacting with the
   service over a network must be offered the corresponding source. Revisit before either.
3. **The weights are gitignored (INFRA-11)** and arrive only via `pixi run -e export
   export-fastsam`, so the `.onnx` never enters git history regardless.

## MobileSAM — **Hold**, and the documented non-implementation deviation

The brief names MobileSAM as the permissively-licensed (MIT) alternative proposal backend.
**It is NOT implemented as a working second backend in Milestone 1. This is a documented
deviation from the brief**, recorded here, in the PR body, and in the `propose-retrieve.md`
ROBUSTNESS BACKLOG.

- **Why deferred (cost, from research):** the ONNX SAM mask decoder accepts **one prompt per
  call**. "Everything mode" therefore means a 32×32 grid ⇒ **~1024 sequential decoder calls**
  plus a hand-ported `SamAutomaticMaskGenerator` (grid sampling, per-mask stability/IoU
  filtering, mask NMS). That is **a phase of work, not a backend swap** — the AGPL escape hatch
  exists but is not cheap.
- **What ships instead:** a `ProposalBackend` protocol (`src/object_search/search/proposals.py`)
  with **FastSAM as the single implementation**, so MobileSAM can be added later without
  restructuring.
- **`awarebayes/MobileSamONNX` — Hold.** IDEA.md §14 references it; verdict **Hold**: 0 stars, a
  4-day project untouched for 14 months, single author. Better-provenanced paths exist — the
  first-party export scripts (`ChaoningZhang/MobileSAM` decoder, `Acly/MobileSAM` encoder) and
  the MIT `Acly/MobileSAM` pre-exported encoder+decoder (`dhkim2810/MobileSAM` MIT weights). If
  MobileSAM is ever built, use those, not `awarebayes/MobileSamONNX`.

## Rejected / held candidates

- **Pre-exported FastSAM ONNX on HF Hub** — all 0-download, unverifiable provenance. Hold.
- **`CASIA-LMC-Lab/FastSAM`** (the original repo) — AGPL-3.0, last pushed 2024-07-30, effectively
  dormant (146 open issues). Use the Ultralytics export path (pushed 2026-07-24) instead.
- **FastSAM-x** (quality variant) — same export path, but ~276 MiB. Not needed for Milestone 1;
  `FastSAM-s` at 45 MiB is the operating point.
- **`awarebayes/MobileSamONNX`** — Hold (see above).
