# Object Search Exploration

## What This Is

An interactive exemplar-based object search demo: the user draws a box around one object
in an image, and the system finds every other instance of that same object **in that same
image**. Four independent search methods sit behind one interface, selectable *before* the
box is drawn, so the same query can be run through different algorithms and compared. A
rating layer records how well each method did on each query, and a statistics layer turns
those ratings — plus objective metrics on ground-truthed images — into a per-method
scoreboard.

This is an **exploration harness, not a product**. Its value is that each method is
readable, editable, and measurable by an ML practitioner, and that adding a fifth method or
a whole new exploration is a small, obvious diff.

## Core Value

Given one hand-drawn exemplar box, return all matching instances in the image — through any
of four interchangeable methods — and accumulate enough evidence (subjective ratings plus
objective precision/recall) to say which method actually works, on which kind of image, and
at what latency.

## Primary User

A machine-learning practitioner (the repo owner) who will **read and edit the method code
directly**. This is not an abstract persona; it is the single hard constraint that shapes
the architecture. It drives two non-negotiable design rules:

1. **Each method is one self-contained, top-to-bottom readable Python module.** The full
   algorithm is visible in one file, with numbered step comments matching the documentation.
   Shared helpers are imported explicitly and are never required — a method may inline its
   own variant if that reads better.
2. **No hidden control flow.** No plugin magic, no deep inheritance chains, no
   config-driven dispatch inside a method. One registry decorator per method is the only
   indirection.

Readability outranks DRY inside method modules.

## Requirements

### Validated

(None yet — ship to validate)

### Active

Full detail with IDs in `.planning/REQUIREMENTS.md`. Summarized:

- [ ] Pixi/Python-3.12 scaffold with Ruff + MyPy strict + pre-commit + ≥80% coverage as
      hard merge gates, GitHub repo with protected `main` and CI (INFRA-01…INFRA-07)
- [ ] Frozen Pydantic schemas for every inter-layer contract; `ONNXInferencer` base with
      init-time dtype/shape validation; `SearchMethod` protocol + decorator registry;
      scripted `fetch-models` (INFRA-08…INFRA-11)
- [ ] Method 1 `ncc` — NCC template matching with pyramid scale search and optional
      rotation bank (METHOD-01)
- [ ] Method 2 `sparse-geo` — classical + SuperPoint backends, many-to-many kNN with the
      standard ratio test **disabled**, Hough pose voting with soft binning, per-peak
      RANSAC, three voting modes, sequential-RANSAC alternative, low-keypoint diagnostic
      (METHOD-02…METHOD-04c)
- [ ] Method 3 `dino-dense` — DINOv2 ONNX dense tokens, prototype cosine similarity,
      threshold, connected components (METHOD-05)
- [ ] Method 5 `propose-retrieve` — FastSAM/MobileSAM ONNX proposals, DINOv2 region
      embeddings, NN retrieval, with proposal and embedding stages independently callable
      (METHOD-06)
- [ ] Shared threshold calibration and peak extraction as optional offerings
      (METHOD-07, METHOD-08)
- [ ] Diagnostics payload, robustness backlog, and explicit pre/post-processing docs on
      every method (METHOD-09…METHOD-12)
- [ ] FastAPI backend: `/methods` `/search` `/images` `/ratings` `/stats`, run persistence,
      lifespan-loaded ONNX sessions, typed errors (API-01…API-08)
- [ ] Canvas web UI: method+config selected before the box is drawn, box drawing with
      zoom/pan, result and diagnostics overlays, tiered rating widget, schema-driven config
      form, stats dashboard (UI-01…UI-08)
- [ ] SQLite run + rating store with raw judgments only, sub-threshold candidate logging,
      provenance, slice metadata, latency breakdown, nullable human counts, Wilson
      intervals, Bradley-Terry paired ranking (EVAL-01…EVAL-18)
- [ ] Demo asset set with licensing provenance, pre-rendered sample runs committed for
      every method, per-method docs, aggregated robustness backlog, Milestone 2 spec
      (DOC-01…DOC-06)

### Out of Scope

- **Cross-image / corpus search** — search is confined to a single image. The Phase 7
  embedding store is shaped so corpus search is a later addition, not a rewrite.
- **Training or fine-tuning any model** — all models are pretrained and frozen.
- **Video / temporal search** — single still images only.
- **Segmentation masks as the primary output** — boxes are the output contract. Method 6
  from the source research (one-shot personalized segmentation) is deferred; it becomes
  cheap once Method 5's SAM proposal stage exists, making it a natural Milestone 3.
- **Method 4 from the source research** (exemplar-conditioned detectors T-Rex2/CountGD and
  counters FamNet/BMNet+/SAFECount/CounTR/LOCA/CACViT) — weights are heavy and
  licence-encumbered, ONNX export is not a solved path, and several are API-gated, which
  conflicts with the local-first and ONNX-first constraints.
- **Multi-user auth, deployment, or scaling** — local single-user demo.
- **Real-time performance guarantees** — latency is *measured*, not *guaranteed*.
- **FAISS** — hundreds of proposals in one image do not need an ANN index. Adopt it when
  corpus search actually arrives.
- **LightGlue / SuperGlue** — assignment-based matchers assume roughly one-to-one
  correspondence, which is exactly wrong for repeated instances. Replaced by many-to-many
  kNN + Hough voting, which also supplies a principled stopping criterion.
- **PyTorch at inference time** — ONNX Runtime for every learned model. If a model has no
  usable ONNX export path, script the export in `fetch-models` or drop the model.
- **Lattice fitting as post-verification** — documented in the robustness backlog, not
  built. Likely the single highest-leverage robustness item for shelf/PCB/tile images.

## Context

**Technical environment.** macOS arm64 primary, Python 3.12, Pixi-managed environment.
Local-first: no cloud inference APIs, no hosted model endpoints.

**Prior work being reused.** The sibling project `basketball-2d-to-3d` at
`/Users/ortizeg/1Projects/⛹️‍♂️ Next Play/code/basketball-2d-to-3d/` already proves the
`ONNXInferencer` pattern — init-time dtype and shape validation plus a post-processor
strategy — in `src/basketball_2d_to_3d/inference/`. That **design** is ported here; a
dependency on that project is **not** added. That project also supplies basketball
broadcast frames for the demo image set, and its pinned `onnxruntime` version is the
starting point here (1.24.1 had macOS platform-tag issues with the pixi solver).

**Source research.** The project was scoped by a six-method survey of exemplar-based object
search. Two of its findings drive the design more than the method choice does: **absolute
similarity thresholds do not transfer across images**, and **plain NMS merges touching
instances**. Both therefore get real, selectable implementations in `search/common/` rather
than being buried inside each method.

**The open question the research left.** It flagged that LightGlue/SuperGlue assume
roughly one-to-one assignment — wrong for repeated instances — and asked whether to run
LightGlue sequentially or switch to LoFTR. Neither is needed. Many-to-many top-k matching
followed by generalized Hough voting in pose space and per-peak RANSAC is Lowe's original
IJCV 2004 multi-object recognition pipeline, is proven for exactly this task, and makes the
stopping criterion fall out of the vote histogram instead of an arbitrary loop count.

**Milestone 2 is already specified** (marker-conditioned region proposal: find every
instance of an arrow/marker, then resolve the best object proposal it points at). It is not
built in Milestone 1, but Milestone 1 is built with the seams it needs — specifically, an
"exploration" is a registry-level concept rather than just a method, and Method 5's
proposal and embedding stages are independently callable units from day one.

## Constraints

- **Environment**: Pixi only — not venv, not uv, not pip, not conda directly. Every command
  runs as `pixi run <task>`. — A single reproducible lockfile is the whole point; mixing
  environment managers destroys it.
- **Python version**: 3.12 — onnxruntime wheel availability.
- **OpenCV source**: conda-forge `opencv`, not PyPI `opencv-python-headless` — consistent
  binary provenance with the rest of the conda-forge stack, and avoids the duplicate-native-
  library conflicts that the two sources produce together.
- **Inference**: ONNX Runtime for every learned model. No quiet PyTorch inference path. — A
  PyTorch fallback would silently defeat the portability and load-time-validation
  guarantees the whole `inference/` layer exists to provide.
- **Pre/post-processing is explicit**: every ONNX model's normalization, resize policy,
  layout, and output decoding is written down in the inferencer docstring *and* the method
  doc. — This is a stated user requirement, not a nicety; undocumented preprocessing is the
  most common source of silently-wrong ONNX inference.
- **Logging**: Loguru only. No `print()`, no stdlib `logging`.
- **Quality gates are hard gates**: Ruff (line-length 100), MyPy strict, and the ≥80%
  coverage floor block the merge. They are not advisory.
- **Readability over cleverness** in method modules — the primary user reads and edits
  these directly. A method may repeat a few lines rather than import a helper if that makes
  it read better standalone.
- **Reproducibility**: same image + box + method + config ⇒ identical results. Any
  stochastic step (RANSAC, SAM prompt sampling) takes an explicit seed from config.
- **Local-first**: no cloud inference APIs, no hosted model endpoints.
- **Third-party ONNX exporters are gated** through an explicit `library-review` verdict
  before adoption — several candidates are community projects of varying maintenance
  quality, and that judgment should be explicit rather than incidental.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| FastAPI backend + canvas web frontend, not Gradio | Box drawing is the core interaction; Gradio would put it behind a third-party community component. Full control over draw/zoom/redraw, and a clean seam for Milestone 2's second mode. | — Pending |
| One self-contained module per method | The primary user reads and edits these. Readability outranks DRY here. | — Pending |
| Shared `calibration.py` / `peaks.py` are optional imports, never mandated | The research says thresholding and peak extraction are the weak links, so they deserve real implementations — but mandating them would fight the self-contained-module rule. | — Pending |
| Many-to-many kNN + Hough voting instead of LightGlue | Assignment-based matchers assume one-to-one, which is exactly wrong for repeated instances. Hough peaks also supply a principled stopping criterion. Avoids a LightGlue ONNX dependency entirely. | — Pending |
| Standard Lowe ratio test **disabled** in Method 2 | It is designed to suppress matches with multiple good candidates, which is the exact signature of the repeated instances we are searching for. Top-k unconditionally, with an optional k+1 ratio instead. | — Pending |
| Three voting modes in Method 2 (`single-4dof` / `translation-2dof` / `pairwise-4dof`) | SuperPoint keypoints carry no scale or orientation, so single-correspondence 4-DoF voting is invalid for the learned backend. | — Pending |
| Same DINOv2 ONNX inferencer for Methods 3 and 5 | One model, one preprocessing contract, one download. Method 5 becomes a thin layer on Phase 6's work. | — Pending |
| No FAISS in Milestone 1 | Hundreds of proposals in one image do not need an ANN index. Adopt it when corpus search actually arrives. | — Pending |
| Pydantic configs for the API, Hydra only for the CLI | Configs arrive as JSON over HTTP; Hydra composition adds nothing there. Method config models double as the UI form schema — one source of truth for defaults, ranges, and docstrings. | — Pending |
| `ONNXInferencer` design ported from `basketball-2d-to-3d`, not depended on | Init-time dtype/shape validation is a proven pattern in a sibling project. Reuse the design, not a dependency. | — Pending |
| UI/API land before methods 2, 3, 5 | Every method after the first is human-testable and ratable the day it lands. The alternative — all four methods, then a UI — would leave three methods untested by a human until the very end. | — Pending |
| Synthetic images with exact ground truth | Makes the per-method comparison a number rather than an impression, without hand-labelling cost. | — Pending |
| Paired comparison mode (same box, all methods) | Ratings across different boxes are confounded; the same box across methods is a clean comparison. | — Pending |
| Thumbs + counts instead of a 1–5 star scale | Star scales drift within a session and are not comparable across methods. Binary judgment plus objective counts is cheaper to give and more honest. | — Pending |
| Expected-instance count inferred, never entered | Counting every instance in a cluttered image is slow and error-prone; "how many did it miss?" is a glance, and `expected = TP + FN` recovers the rest. | — Pending |
| Count fields empty, never prepopulated with 0 | `null` means "not assessed", `0` means "assessed, none". Seeding `0` defaults every unreviewed run to perfect precision and recall — the exact way a scoreboard lies. Cheap zeros come from one-click buttons, not from defaults. | — Pending |
| `wrong_count` as a fast path beside per-match verdicts | Typing "12" beats clicking 12 boxes, but it loses score attribution and so cannot feed the offline threshold sweep. Both paths exist; `rating_completeness` records which was used. | — Pending |
| Log sub-threshold candidates with raw scores | Converts each rating session into a full PR curve via offline threshold sweep, rather than one operating point — aimed squarely at the weak link the research identifies. | — Pending |
| Derived metrics computed in queries, never stored as columns | A stored precision column goes stale the moment a rating is edited, and invites the null-coercion bug the whole evaluation design exists to prevent. | — Pending |
| "Exploration" is a registry-level concept from day one | Milestone 2 adds an exploration rather than forking the app. A mode selector sits above the method selector in the UI shell. | — Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-07-24 after initialization*
