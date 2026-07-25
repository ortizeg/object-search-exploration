# Phase 7 Context — Method 5 (`propose-retrieve`)

**Source:** `.planning/IDEA.md` §5 Method 5 and §11 (Milestone 2), plus
`.planning/research/MODELS.md` (FastSAM export performed and verified, MobileSAM assessment).

## Domain

Instance retrieval confined to one image: class-agnostic proposals → embed each region with the
**same DINOv2 inferencer as Method 3** → cosine nearest-neighbour against the exemplar embedding,
threshold, NMS.

**This is the method Milestone 2 reuses**, so its proposal and embedding stages are built as
independently callable units from the start — that is a Phase 7 success criterion verified by a
test that calls them directly rather than through `search()`.

## Locked Decisions

1. **FastSAM-s is the implemented proposal backend.** Export verified during research:
   `FastSAM-s.pt` → 45.0 MiB ONNX, `output0 [1, 37, 21504]`, `output1 [1, 32, 256, 256]` at
   1024². Single class `{0: 'object'}`. The 37 channels are **4 box + 1 conf + 32 mask
   coefficients**. Dynamic H/W verified at three resolutions.
2. **Output decoding (YOLOv8-seg style)** — write this sequence out explicitly in the docstring:
   transpose `output0` to `[21504, 37]`, split into boxes / conf / 32 coeffs, filter by confidence,
   NMS on boxes, then for survivors compute `masks = sigmoid(coeffs @ protos.reshape(32, -1))`,
   reshape to 256×256, **crop each mask to its own box**, and upsample to image resolution. The
   crop-to-box step is not optional — without it a mask bleeds outside its detection.
3. **Ultralytics is an export-time-only dependency, isolated in a separate pixi environment.**
   The runtime environment stays torch-free, which makes the project's "ONNX Runtime for every
   learned model" constraint **structural** rather than a convention reviewers must police.
4. **AGPL-3.0 must be stated plainly, in three places** (`LICENSES.md`,
   `docs/methods/propose-retrieve.md`, and the `ModelSpec.license_note`):
   FastSAM is AGPL-3.0 and **the exported `.onnx` file itself embeds that licence string**.
   Export-time-only isolation protects the runtime dependency graph but **not the weights**.
   Private local use triggers nothing; **publishing this repo or network-exposing the FastAPI app
   would fire AGPL §13.** This is a real constraint on how the repo may later be shared, not a
   footnote.
5. **MobileSAM is NOT implemented as a working second backend in Milestone 1. This is a documented
   deviation from the brief.** Reason, from research: the ONNX SAM decoder accepts **one prompt per
   call**, so "everything mode" means a 32×32 grid ⇒ ~1024 sequential decoder calls plus a
   hand-ported `SamAutomaticMaskGenerator`. That is a phase of work, not a backend swap. The AGPL
   escape hatch exists but is not cheap.
   **What ships instead:** a `ProposalBackend` protocol with FastSAM as the single implementation,
   so MobileSAM can be added later without restructuring. Record the deviation in the phase docs,
   the PR body, and the robustness backlog.
   Also drop IDEA.md §14's `awarebayes/MobileSamONNX` reference — verdict **HOLD**: 0 stars, a
   4-day project untouched for 14 months, single author, while first-party export scripts and a
   better-provenanced MIT artifact (`Acly/MobileSAM`) both exist.
6. **The embedding stage reuses Phase 6's `DINOv2Inferencer` instance contract.** One model
   download, one preprocessing contract. A test asserts no second DINOv2 model file is fetched.
7. **Plain NumPy matmul for nearest-neighbour. FAISS is deliberately not adopted in Milestone 1** —
   for a few hundred proposals in one image it is pure dependency cost. Shape the embedding store
   so a FAISS index slots in when corpus search arrives.
8. **Two independently callable units** — this is the Milestone 2 seam and the phase's defining
   constraint:
   - `propose(image, config) -> list[Proposal]` where `Proposal` carries box, mask (optional),
     and objectness
   - `embed_regions(image, boxes, config) -> NDArray` returning one L2-normalized embedding per box
   `search()` composes them and does nothing else that they cannot do alone. Neither may reach
   into the other's internals.

## Canonical References

- `.planning/research/MODELS.md` § FastSAM / MobileSAM — the performed export, verified graph I/O,
  licence analysis, and the MobileSAM cost assessment
- `src/object_search/inference/` — Phase 1 base, Phase 6 `DINOv2Inferencer`
- `.planning/IDEA.md` §11 — what Milestone 2 reuses and why these seams exist

## Scope Fence

**In:** `FastSAMInferencer` with documented output decoding, the `export` pixi environment and the
scripted FastSAM export in `fetch-models`, `propose()` and `embed_regions()` as independent units,
`search/propose_retrieve.py`, diagnostics carrying the proposal set,
`docs/methods/propose-retrieve.md`, sample runs.

**Out:** MobileSAM implementation (deviation, documented). FAISS. Marker-conditioned proposal
(Milestone 2 — specified in Phase 8, not built). Mask output as the primary contract — boxes remain
the output.

## Deferred (robustness backlog)

FAISS index for corpus-scale search; proposal filtering by size/aspect prior derived from the
exemplar; multi-crop / TTA embeddings; **region embedding with the background masked out rather
than the raw box crop** (FastSAM already produces the mask, so this is cheap and likely a real
win); alternative proposal sources (RPN, selective search) for images where SAM over-segments;
MobileSAM everything-mode with a ported automatic mask generator.

## Risk Summary

- **Phase 7 success criterion 1 is "boxes tightly aligned to object boundaries."** That is the
  method's whole selling point over `dino-dense`'s blobby connected components. Verify it against
  synthetic ground truth with an IoU threshold rather than by eye, so the claim is a number.
- **SAM over-segmentation** is the expected failure: one object becomes several proposals, each
  embedding well, producing duplicate detections. NMS after retrieval is what handles it — tune
  and document the IoU, and record the proposal count in diagnostics.
- **Latency will be dominated by the proposal stage.** That is a *finding*, not a defect, and it is
  exactly why EVAL-11 mandates a latency breakdown instead of a single number. Make sure the
  breakdown attributes proposal time to `inference_ms` and embedding time separately, or the
  finding is invisible.
- **Model download in CI.** Same as Phase 6 — weights are gitignored, so real-model tests skip when
  absent. Keep the output-decoding arithmetic testable with synthetic tensors so CI still gates it.
