---
planStatus:
  planId: plan-real-object-insertion-eval-dataset
  title: Real-Object Insertion Evaluation Dataset
  status: draft
  planType: feature
  priority: medium
  owner: ortizeg
  stakeholders: []
  tags: [eval, dataset, fastsam]
  created: "2026-08-06"
  updated: "2026-08-06T00:00:00.000Z"
  progress: 0
---

# Real-Object Insertion Evaluation Dataset

## Objective

Add a fifth benchmark set — real backgrounds with real, segmented objects pasted onto them at
known positions — that closes the standing TODO in `assets/demo/LICENSES.md`:

> **Generic repeated-instance photos — TODO (revisited in Phase 8).** None are included in Phase
> 1, deliberately: no clearly-licensed source could be obtained without a network fetch of
> unverifiable provenance, and downloading images of unknown licence to satisfy a requirement is
> exactly the mistake this file exists to prevent.

Every existing benchmark set (`chipset`, `textured`, `synthetic`) is drawn/rendered by this repo —
they prove the method comparison but say nothing about real photographic texture, lighting, or
backgrounds. The research datasets (RPINE, FSCD-147, floor plans, ...) are real but their
per-instance ground truth comes from someone else's annotations, and the images/raw data are
gitignored (licence-gated or large), so nothing real is browsable in the committed repo. This set
is the missing middle: **real photographic pixels, committed, with exact ground truth by
construction** — because we control where every object is pasted, exactly like chipset/textured.

## Decisions locked in (from user Q&A)

1. **Image sourcing:** auto-fetched by this work, from a single well-documented, per-file-licensed
   source — **Wikimedia Commons**. Every downloaded file gets an individually recorded
   title/author/licence/source-URL, the same honesty discipline already used for the basketball
   frames' "Exact source paths" table, rather than a bulk dataset with one blanket licence.
2. **Segmentation backend:** **FastSAM**, already integrated (`FastSAMInferencer`, Trial-approved,
   AGPL-3.0 export-only, gitignored weights). It already decodes masks
   (`FastSAMConfig(return_masks=True)`); this adds a single-object-selection heuristic on top. No
   new dependency, no new library-review.

## Non-goals

- No SAM2 export/library-review (deferred; FastSAM covers the need).
- No change to any search method (`ncc`, `sparse-geo`, `dino-dense`, `propose-retrieve`,
  `mosse`, `owlv2-oneshot`) — this is a dataset-only addition, same as chipset/textured were.
- Not wired into the research-dataset framework (`eval/datasets.py`, `DatasetSpec`,
  `datasets/_incoming/`) — that machinery exists for large, externally-annotated,
  train/val/test corpora. This set is small, self-composited, and exact-GT-by-construction, so it
  belongs next to `chipset.py`/`textured.py` in `object_search/synthetic/`, not the research layer.

## Design

### 1. Source manifest (Wikimedia Commons)

A `RealObjectManifest`/`RealBackgroundManifest` pair of frozen Pydantic entries — the single
source of truth, exactly like `CHIPSET_SPECS`/`TEXTURED_SPECS` — each recording `title`,
`file_url` (direct Commons original), `source_page`, `author`, `license` (e.g. `CC0-1.0`,
`CC-BY-SA-4.0`), and `category` (the object/scene name).

Candidate categories (to be resolved to specific, licence-verified Commons files during
implementation — this list is a target, not a commitment; any category without a clean CC0/CC-BY
match on Commons is swapped for another common object/scene):

- **Objects (≥10):** tennis ball, rubber duck, coffee mug, claw hammer, screwdriver, C-clamp,
  padlock, pinecone, apple, orange, hockey puck, chess pawn.
- **Stress object: plain white ping-pong ball.** Deliberately textureless and rotationally
  symmetric — a real-photo analogue of what `textured.py` already tests synthetically (a flat
  chip trips NCC's low-variance guard; a low-keypoint patch trips `sparse-geo`'s ≥20-SIFT-keypoint
  floor and it abstains). Expected to make `ncc` and `sparse-geo` abstain or fail outright while
  `dino-dense`/`propose-retrieve`/`mosse`/`owlv2-oneshot` should still find it on shape/colour —
  the real-photo case none of the other four sets exercises. Also stresses the FastSAM cutout step
  itself (specular highlight, low-contrast boundary against some backgrounds), which is a useful
  test of the extraction heuristic's failure mode (log-and-skip), not just the search methods.
- **Backgrounds (~8-10, all mutually distinct real scenes):** wood-plank floor, poured concrete,
  grass lawn, gravel, brick wall, sand, carpet, granite countertop, asphalt, corrugated cardboard.

Each object photo should show the object reasonably isolated (uncluttered surface, plain-ish
background) so FastSAM's automatic mode can find it without a prompt — see §2.

### 2. Object cutout extraction (FastSAM, no prompting)

`FastSAMInferencer.predict(image, FastSAMConfig(return_masks=True, conf_thres=..., iou_thres=...))`
already returns every class-agnostic `Proposal` (box + soft mask + objectness) for a scene in
"everything mode." For a single-object photo, the target is reliably the **largest proposal whose
box is centred in the frame** (a simple centrality-weighted-area heuristic, no new model
capability). `extract_cutout(photo, proposals) -> Cutout` picks that proposal, thresholds its soft
mask to a hard alpha, and returns an RGBA crop tight to the mask's bounding box.

This function takes an already-produced `list[Proposal]` (not the inferencer itself), so it is
pure and testable with synthetic proposals — no ONNX weight needed for its unit tests, mirroring
how `decode_fastsam` is CI-tested with synthetic tensors.

Failure mode: if the heuristic picks a wrong/degenerate proposal for a given photo, that photo is
logged and skipped rather than silently producing a bad cutout (mirrors the achieved-count-honesty
discipline elsewhere in `synthetic/`) — worst case we lose one object candidate, not silently
corrupt the benchmark.

### 3. Compositing (mirrors `textured.py`)

`RealInsertionImageSpec` (frozen Pydantic, mirrors `TexturedImageSpec`): background id, target
object id, `n_instances`, seed, `scale_min`/`scale_max`, `rotation_deg`, optional
`n_distractors` + distractor object id.

`generate_real_insertion_image(spec, cutouts) -> SyntheticImage` (reuses the existing
`SyntheticImage` dataclass, like chipset/textured do):

1. Load the background photo as the canvas (resized/cropped to a fixed working resolution).
2. For `n_instances`, rejection-sample a non-overlapping position (same strict-non-overlap +
   attempt-cap-with-achieved-count-honesty pattern as chipset/textured), scale and rotate the
   cutout's RGBA (`cv2.warpAffine`, same approach as `textured._transform_instance`), alpha-blend
   it onto the canvas, and record the **AABB of the warped alpha mask** as the ground-truth box —
   not the nominal cutout size, for the same reason textured.py computes it from the transformed
   corners.
3. Optionally paste a *different* object's cutout as distractors — pasted, not recorded, exactly
   like the textured/chipset distractor convention.
4. Sort boxes `(y, x)`, build `SliceMetadata` (achieved count, scale/rotation range), return.

`write_real_insertion(out_dir, cutouts_dir, *, force=False) -> list[Path]` orchestrates: ensure raw
Commons photos are present (log + skip missing ones, never crash — the graceful-degradation pattern
`eval/datasets.py` already uses), build/cache one cutout per object photo under `cutouts_dir`
(skipped if already cached, like `--force` elsewhere), then generate and save every spec's
composited PNG + `<image_id>.gt.json` sidecar in the **exact existing sidecar schema** (`image`,
`width`/`height`, `seed`, `requested_n`/`achieved_n`, `exemplar_index`, `slice_metadata`, `boxes`)
— so `object_search.eval.labels` needs no schema change, only a new search root.

### 4. What's committed vs. gitignored

| Artifact | Status | Why |
|---|---|---|
| Raw Commons downloads (`assets/demo/real-objects/_raw/`) | gitignored | Regenerable via `fetch-real-photos`; no need to double-commit source material we also composite from. |
| Cached cutouts (`assets/demo/real-objects/cutouts/`) | gitignored | Regenerable from the raw photos + FastSAM; depends on gitignored ONNX weights. |
| Composited benchmark images + `.gt.json` sidecars (`assets/demo/real-objects/*.png`) | **committed** | Same rule as chipset/textured: small, exact GT, and downstream code (eval harness, report, CI-adjacent tooling) must work with **no weights and no network**, exactly like the other three sets. |

### 5. Integration points

- **`src/object_search/synthetic/real_insertion.py`** (new) — manifest, `extract_cutout`,
  `RealInsertionImageSpec`, `generate_real_insertion_image`, `write_real_insertion`, plus a
  `fetch_real_photos(out_dir, *, force=False)` network step, following the module docstring
  conventions (load-bearing invariants called out, numbered steps) used by `chipset.py`/`textured.py`.
- **`src/object_search/cli.py`** — two new Typer commands: `fetch-real-photos` (download raw
  Commons files) and `real-objects` (generate the composited set from cached/fetched inputs),
  mirroring the existing `chipset`/`textured` commands exactly.
- **`src/object_search/eval/labels.py`** — add `Path("assets") / "demo" / "real-objects"` to
  `_GT_ROOTS`, extend `_source_for` to map it to `"real-objects"`, add a
  `real_objects_image_ids()` helper mirroring `chipset_image_ids()` / `textured_image_ids()`. No
  change to `GroundTruth` itself — same sidecar schema.
- **`src/object_search/eval/benchmark.py`** — include `real_objects_image_ids()` in the **full**
  sweep alongside chipset/textured (the model-free `ci=True` subset stays chipset-only, unchanged,
  per the existing CI-runtime-bound constraint).
- **`pixi.toml`** — `fetch-real-photos` and `real-objects` tasks, next to `chipset`/`textured`.
- **`docs/DATASETS.md`** — new section + table row, following the existing per-set write-up
  format (what it tests / ground truth / source).
- **`assets/demo/LICENSES.md`** — replace the "Generic repeated-instance photos — TODO" section
  with a real per-file manifest table (title/author/licence/source URL for every object and
  background photo), plus a short note on the composited set itself (our own output, like
  chipset/textured).

### 6. Testing (must pass with **no weights, no network**, per repo CI convention)

- **Pure placement/compositing logic** — determinism, strict non-overlap, achieved-count honesty,
  mask-AABB-not-nominal-size — tested by feeding `generate_real_insertion_image` **synthetic RGBA
  cutouts** (small solid-colour patches built in the test), never real photos or FastSAM output.
  Mirrors how `chipset`/`textured` tests need no ONNX weight at all.
- **`extract_cutout` heuristic** — tested against synthetic `Proposal` lists (fabricated boxes/masks
  of known geometry), not a live `FastSAMInferencer` — same "stub the inferencer boundary" pattern
  used elsewhere in this repo for model-free coverage (per project convention: CI has no ONNX
  weights).
- **`eval.labels` round-trip** — a tiny fixture sidecar in the new schema loads correctly and
  cross-checks `achieved_n`, exactly like the existing loader tests.
- **Network/model-dependent steps** (`fetch_real_photos`, the real FastSAM cutout pass) are
  exercised manually/locally when regenerating the committed set, not by the automated suite —
  same posture as `fetch-models`/`fetch-datasets`, which are never invoked by CI.

## Open risks

- **Commons licence/quality verification is real work** — ~20 individual files need a manual (my)
  pass confirming licence tags and that the object is reasonably isolated in-frame; some candidate
  categories may need substitution. Flagged as the first implementation task, not assumed solved.
- **FastSAM automatic-mode heuristic may mis-segment** some object photos (background clutter,
  low contrast). Mitigation: log-and-skip per photo (never silently wrong), pick photos where the
  object is visually dominant, and manually spot-check every generated cutout before committing.
- **Composited realism ceiling** — flat alpha-paste (no shadow/lighting harmonization) will look
  more "cut-and-paste" than real repeated instances; acceptable for Milestone 1 scope, same
  simplicity trade-off chipset/textured already made, and can be flagged in the module's
  `ROBUSTNESS BACKLOG` docstring section rather than solved now.

## Task breakdown

1. Source and licence-verify ≥10 object photos + ≥8 background photos on Wikimedia Commons;
   record the manifest.
2. `real_insertion.py`: manifest, `fetch_real_photos`, `extract_cutout`, `RealInsertionImageSpec`,
   `generate_real_insertion_image`, `write_real_insertion`, `ROBUSTNESS BACKLOG` docstring section.
3. `cli.py` commands + `pixi.toml` tasks.
4. Generate the committed set locally (fetch → cutout → composite); spot-check every image.
5. `eval/labels.py` + `eval/benchmark.py` wiring.
6. `docs/DATASETS.md` + `assets/demo/LICENSES.md` updates.
7. Tests (placement logic, cutout heuristic, GT loader round-trip) ≥80% coverage on new code.
8. `docs/methods/` untouched (no method changes) — confirm via `pixi run bench` that all four/six
   methods still run against the new set end-to-end.
9. Ruff + MyPy strict clean; PR with requirement IDs and how success criteria were verified.
