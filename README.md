# Object Search Exploration

Draw a box around one object in an image; find every other instance of that same object
**in that same image**. Four independent search methods sit behind one interface, selectable
*before* the box is drawn, so the same query can be run through different algorithms and
compared side by side. A rating layer records how well each method did on each query, and a
statistics layer turns those ratings — plus objective metrics on ground-truthed images —
into a per-method scoreboard.

This is an **exploration harness, not a product**. Its value is that each method is
readable, editable, and measurable by an ML practitioner, and that adding a fifth method or
a whole new exploration is a small, obvious diff.

> **Status: Phase 1 (foundation) in progress.** The scaffold, quality gates and CI are in
> place. **No search method is implemented yet** — the table below is the plan, not a
> feature list. Anything not explicitly marked done does not exist.

## Quickstart

Pixi is the only supported environment manager. Do not use `pip`, `venv`, `uv`, or `conda`
directly — a single reproducible lockfile is the whole point.

```bash
pixi install            # solve + install from the committed pixi.lock
pixi run fetch-models   # download ONNX weights into models/ (not yet implemented)
pixi run serve          # FastAPI + static canvas UI on :8000 (not yet implemented)
```

## The four methods

| Method | Key | Idea | Status |
| --- | --- | --- | --- |
| 1 | `ncc` | Zero-model baseline: `cv2.matchTemplate` with `TM_CCOEFF_NORMED`, then peak extraction and NMS over the response map. Pyramid scale search, optional rotation bank. | Planned |
| 2 | `sparse-geo` | Keypoints on the crop matched into the scene, then **many** geometric models recovered rather than one. Classical (SIFT/AKAZE/ORB) and learned (SuperPoint ONNX) backends. | Planned |
| 3 | `dino-dense` | Dense deep-feature similarity: DINOv2 patch tokens for scene and exemplar, cosine similarity map, calibrate, peak-pick. The general-purpose default for moderate appearance variation. | Planned |
| 5 | `propose-retrieve` | Propose → embed → retrieve: class-agnostic proposals, embed each, rank against the exemplar. Its proposal and embedding stages are independently callable units. | Planned |

Method numbering follows the project brief, which is why there is no Method 4.

## Sample runs

Pre-rendered sample runs are committed under `docs/samples/` so the behaviour of each method
is reviewable without running anything.

**Not yet populated** — Phase 2 renders the first samples, alongside Method 1.

## Project layout

```
src/object_search/
├── schemas/          # ExemplarBox, Match, SearchResult, RunRecord, Rating (Pydantic, frozen)
├── inference/        # BaseInferencer, ONNXInferencer, per-model inferencers
├── search/
│   ├── registry.py   # @register_method -- the only indirection in the codebase
│   ├── ncc.py                # Method 1 -- self-contained
│   ├── sparse_geo.py         # Method 2 -- self-contained
│   ├── dino_dense.py         # Method 3 -- self-contained
│   ├── propose_retrieve.py   # Method 5 -- self-contained
│   └── common/       # calibration, peaks, nms, viz -- optional offerings, never mandated
├── api/              # FastAPI app, routes, dependencies
├── store/            # SQLite runs + ratings, stats queries
├── eval/             # ground-truth labels, metrics, benchmark runner
├── log.py            # Loguru sink configuration -- the only place sinks are configured
└── cli.py            # batch runs, sample rendering, model export
frontend/             # static HTML/JS canvas UI served by FastAPI
assets/demo/          # demo images + ground-truth labels (committed)
docs/samples/         # pre-rendered sample runs (committed)
models/               # ONNX weights (gitignored; fetched by `pixi run fetch-models`)
```

Of the above, only `log.py` and the package skeleton exist today.

## Development

```bash
pixi run lint          # ruff check (line-length 100)
pixi run format        # ruff format
pixi run format-check  # ruff format --check (what CI runs)
pixi run typecheck     # mypy --strict
pixi run test          # pytest with a >=80% coverage floor
pixi run quality       # all four, in order
```

These are **hard gates, not advisory**. All four run in CI on every pull request and must
pass before a branch can merge into a protected `main`.

Two conventions the tooling enforces mechanically, because they are easy to regress:

- **Loguru only.** `print()` is a lint error (ruff `T201`) and `import logging` is a lint
  error (ruff `TID251`, via a banned-api rule). Use `from loguru import logger`.
- **Coverage floor.** Dropping below 80% fails `pixi run test`, and therefore CI.

Install the local hooks once after cloning:

```bash
pixi run pre-commit install
```

## Conventions worth knowing before editing

- **Method modules are self-contained and top-to-bottom readable.** This deliberately
  overrides DRY. A method may inline its own variant of a shared helper if that reads better
  standalone; do not refactor a method purely to remove duplication.
- **ONNX Runtime for every learned model.** There is no PyTorch inference path, not even as
  a fallback. Every inferencer validates dtype and shape at construction, so a wrong model
  fails at load rather than at first frame.
- **Reproducibility.** Same image + box + method + config ⇒ identical results. Every
  stochastic step takes an explicit seed from config.
- **Frozen Pydantic v2 models** for every inter-layer contract.

## License

MIT
