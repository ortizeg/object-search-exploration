<!-- GSD:project-start source:PROJECT.md -->

## Project

**Object Search Exploration**

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

**Core Value:** Given one hand-drawn exemplar box, return all matching instances in the image — through any
of four interchangeable methods — and accumulate enough evidence (subjective ratings plus
objective precision/recall) to say which method actually works, on which kind of image, and
at what latency.

### Constraints

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
<!-- GSD:project-end -->

<!-- GSD:stack-start source:STACK.md -->

## Technology Stack

Technology stack not yet documented. Will populate after codebase mapping or first phase.
<!-- GSD:stack-end -->

<!-- GSD:conventions-start source:CONVENTIONS.md -->

## Conventions

**These are hard rules. A PR violating any of them does not merge.**

### Environment — Pixi only

- Every command is `pixi run <task>`. Never `pip install`, never `python -m venv`, never
  `uv`, never `conda` directly. If a dependency is missing, add it to `pixi.toml` and
  re-lock — do not install it ad hoc.
- Python 3.12. OpenCV comes from conda-forge (`opencv`), never PyPI
  `opencv-python-headless`.
- `onnxruntime` starts pinned at **1.23.2** (1.24.1 had macOS platform-tag issues with the
  pixi solver in the sibling project). Changing that pin requires verifying the solve on
  osx-arm64.

### Logging — Loguru only

`from loguru import logger`. No `print()`. No `import logging`. Ruff is configured to flag
both.

### Code quality

- Ruff, line-length **100**. MyPy **strict**. Both must be clean — not "clean except for
  ignores added to make them clean". A genuine `# type: ignore[code]` needs a comment
  saying why.
- Pytest with a **≥80% coverage floor** that fails the build.
- Pre-commit hooks are installed and run before every commit.

### Method modules are self-contained and top-to-bottom readable

This is the single most important convention in the repo, and it deliberately overrides DRY.

- Each search method is **one file** in `src/object_search/search/`. The full algorithm is
  visible in that one file, read top to bottom.
- Use **numbered step comments** (`# 1. Detect keypoints on the crop`) that match the
  headings in `docs/methods/<name>.md`.
- Shared helpers in `search/common/` are **offerings, not requirements**. A method may
  import them, or may inline its own variant if that reads better standalone. Do not
  refactor a method to use a shared helper purely to remove duplication.
- **No hidden control flow**: no plugin magic, no deep inheritance, no config-driven
  dispatch *inside* a method. The `@register_method` decorator is the only indirection.
- Every method module carries a `ROBUSTNESS BACKLOG` docstring section, mirrored into
  `docs/methods/<name>.md`.
- Every method documents its pre-processing and post-processing explicitly, in the module
  docstring **and** the method doc.

### ONNX

- ONNX Runtime for every learned model. **No PyTorch inference path**, not even as a
  fallback. If a model has no usable ONNX export, script the export inside
  `pixi run fetch-models` or drop the model.
- Every inferencer subclasses `ONNXInferencer` and validates dtype and shape **at
  construction**, so a wrong model fails at load rather than at first frame.
- The docstring of every inferencer states: input name, dtype, shape, layout, colour order
  (RGB vs BGR), resize policy, normalization constants, and how each output is decoded.
  Exact numbers, not "standard ImageNet normalization".
- Weights live in `models/`, are **gitignored**, and arrive only via `pixi run fetch-models`.
- Any third-party ONNX-export repo gets an explicit `library-review` verdict
  (Adopt/Trial/Assess/Hold) recorded in the phase docs before it is adopted.

### Schemas

Frozen Pydantic v2 models for every inter-layer contract. Each method's `config_model`
doubles as the JSON Schema that generates the UI form — one source of truth for defaults,
ranges, and docstrings.

### Reproducibility

Same image + box + method + config ⇒ identical results.

Every stochastic step takes an explicit seed from config — **but only where a seed genuinely
controls anything.** Never add a seed parameter that does nothing; a control that is advertised
and inert is worse than no control.

Verified specifics (see `.planning/research/PITFALLS.md`):

- **`cv2.setRNGSeed` does NOT affect RANSAC.** OpenCV hardcodes `RNG rng((uint64)-1)` in
  `ptsetreg.cpp`, deliberately. OpenCV's RANSAC is therefore already deterministic, but its seed
  is **not user-controllable**. Do not expose a `ransac_seed` config field for
  `cv2.estimateAffinePartial2D` et al. Where explicit seeding matters, implement the sampling in
  NumPy with `np.random.default_rng(config.seed)` inside the method module — which also makes it
  visible in the one file the reader is meant to read.
- **Thread counts do not affect output.** ORT, OpenCV, and BLAS thread counts and argmax tie
  order were all measured bit-identical, and `use_deterministic_compute` is a no-op on the CPU
  EP. Do not describe thread pinning as a determinism measure.
- **What actually threatens reproducibility, and so what gets pinned:** set/dict iteration order,
  NMS tie-breaking (sort by `(-score, y, x)`, never score alone), config-hash key ordering
  (serialize with sorted keys), and library-version drift (guarded by the committed `pixi.lock`
  and the model SHA-256 in provenance).
- **`cv2.matchTemplate` output depends on the search extent** — cropping the search region changes
  ~73% of the returned floats. Always correlate over the full scene.

### Evaluation — two rules that are easy to regress

1. **Human count fields are nullable and stored EMPTY.** `wrong_count` and `missed_count`
   are never prepopulated or defaulted to `0` at *any* layer — not the form, not the API
   default, not the DB column default. `null` means "not assessed"; `0` means "assessed,
   none". Defaulting to `0` makes every unreviewed run claim perfect precision and recall.
2. **Derived metrics are computed in queries, never stored as columns.** Precision, recall,
   F1, and expected-count come from views over `retrieved`, per-match verdicts, and
   `missed_count`, with NULL propagating rather than defaulting.

### Method 2 specifics — easy to get wrong

- The standard Lowe ratio test is **DISABLED**. It exists to suppress matches with multiple
  good candidates, which is exactly the signature of the repeated instances being hunted.
  Take top-k unconditionally; the optional **k+1** ratio test is the only ratio test used.
- SuperPoint keypoints carry **no scale or orientation**, so single-correspondence 4-DoF
  Hough voting is invalid for that backend. Three voting modes exist: `single-4dof`
  (classical only), `translation-2dof`, `pairwise-4dof`.

### Git

- One PR per phase checkpoint (2 PRs per phase). Atomic commits within a branch.
- PR bodies state which requirement IDs the PR satisfies and how the phase success criteria
  were verified.
- **Always confirm `git rev-parse --show-toplevel` ends in `object-search-exploration`
  before committing.** The parent directory is a stray empty git repo containing every
  sibling project — never run git commands there.
- Commit messages end with:
  `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`
<!-- GSD:conventions-end -->

<!-- GSD:architecture-start source:ARCHITECTURE.md -->

## Architecture

```
src/object_search/
├── schemas/          # ExemplarBox, Match, SearchResult, RunRecord, Rating (Pydantic, frozen)
├── inference/        # BaseInferencer, ONNXInferencer, DINOv2Inferencer,
│                     # SuperPointInferencer, FastSAMInferencer
├── search/
│   ├── registry.py   # @register_method — the only indirection
│   ├── ncc.py                # Method 1 — self-contained
│   ├── sparse_geo.py         # Method 2 — self-contained
│   ├── dino_dense.py         # Method 3 — self-contained
│   ├── propose_retrieve.py   # Method 5 — self-contained
│   └── common/       # calibration.py, peaks.py, nms.py, viz.py — optional offerings
├── api/              # FastAPI app, routes, dependencies
├── store/            # SQLite runs + ratings, stats queries
├── eval/             # ground-truth labels, metrics, benchmark runner
└── cli.py            # batch runs, sample-run rendering, model export
frontend/             # static HTML/JS canvas UI served by FastAPI
assets/demo/          # demo images + ground-truth labels
docs/samples/         # pre-rendered sample runs, committed
models/               # ONNX weights (gitignored, fetched by `pixi run fetch-models`)
```

### The one abstraction that matters

```python
class SearchMethod(Protocol):
    """Every search method is a callable with this shape. Nothing else is shared."""

    name: str                      # registry key, e.g. "ncc"
    config_model: type[BaseModel]  # frozen Pydantic model; drives the UI form

    def search(
        self,
        image: npt.NDArray[np.uint8],   # BGR scene
        exemplar: ExemplarBox,          # the user's drawn box
        config: BaseModel,              # instance of config_model
    ) -> SearchResult: ...
```

`SearchResult` carries the matches (box + score), a latency breakdown, and a
method-specific `diagnostics` payload (similarity map, keypoint correspondences, Hough
peaks, proposal set) that the UI renders as a debug overlay. Diagnostics are how a
practitioner sees *why* a method failed, not just *that* it did.

### Rule of Three

The registry, the schemas, and `ONNXInferencer` are shared because three or more methods
need them. `calibration.py` and `peaks.py` are shared **offerings** — imported by choice,
never mandated. Nothing else gets abstracted until a third method demands it.

### Configuration split

Hydra is used **only** for the CLI/batch benchmark entrypoint, where sweeping
method × config × image is the point. The API path uses plain frozen Pydantic config models
per method, because configs arrive as JSON over HTTP and Hydra's composition adds nothing
there.

### Milestone 2 seam

An "exploration" is a registry-level concept, not just a method. The UI shell, store, and
API host more than one exploration from the start (a mode selector above the method
selector), and Method 5's proposal and embedding stages are independently callable units —
so Milestone 2 adds an exploration rather than forking the app.
<!-- GSD:architecture-end -->

<!-- GSD:skills-start source:skills/ -->

## Project Skills

No project skills found. Add skills to any of: `.claude/skills/`, `.agents/skills/`, `.cursor/skills/`, `.github/skills/`, or `.codex/skills/` with a `SKILL.md` index file.
<!-- GSD:skills-end -->

<!-- GSD:workflow-start source:GSD defaults -->

## GSD Workflow Enforcement

Before using Edit, Write, or other file-changing tools, start work through a GSD command so planning artifacts and execution context stay in sync.

Use these entry points:

- `/gsd-quick` for small fixes, doc updates, and ad-hoc tasks
- `/gsd-debug` for investigation and bug fixing
- `/gsd-execute-phase` for planned phase work

Do not make direct repo edits outside a GSD workflow unless the user explicitly asks to bypass it.
<!-- GSD:workflow-end -->

<!-- GSD:profile-start -->

## Developer Profile

> Profile not yet configured. Run `/gsd-profile-user` to generate your developer profile.
> This section is managed by `generate-claude-profile` -- do not edit manually.
<!-- GSD:profile-end -->
