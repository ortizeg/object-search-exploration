# Stack Research

> Verified **2026-07-24** against live PyPI JSON API, anaconda.org conda-forge API,
> conda-forge feedstock sources, GitHub Releases API, and pixi docs `main`.
> No version number below comes from training data. Anything unverifiable is marked
> **LOW** and listed again under *Open Questions*.

**Headline findings (read these first):**

1. **The `onnxruntime == 1.23.2` pin is obsolete and should be replaced.** The sibling
   project's failure was a real, diagnosable pixi/macOS wheel-tag mismatch, not a bug in
   1.24.1. Root cause and two fixes are in *Known Incompatibilities #1*.
   **Recommendation: take `onnxruntime` from conda-forge, not PyPI.** conda-forge ships
   `onnxruntime 1.26.0` with `py312` builds for both `osx-arm64` and `linux-64`, depends
   only on `__osx >=11.0` (no macOS 14 floor), and — verified in the feedstock — **enables
   the CoreML execution provider on Apple Silicon** with an explicit import-time test.
2. **`opencv = "*"` from conda-forge is now actively dangerous.** conda-forge has
   `opencv 5.0.0`, and **its 5.0.0 builds have no Python bindings at all** (build strings
   are `headless_hXXXXXXX_N` / `qt6_hXXXXXXX_N` — no `py3XX` variant on any subdir). An
   unpinned `opencv` can resolve to 5.0.0 and `import cv2` will fail. **Pin `>=4.13,<5`.**
3. **`[system-requirements]` is deprecated in current pixi.** Per-platform virtual
   packages now go in `workspace.platforms` inline tables. Skeleton below uses the new form.
4. **mypy is on 2.x and ruff just shipped 0.16.0 with a 59 → 413 default-rule change.**
   Both are handled below; an explicit `[tool.ruff.lint] select` neutralizes #2 of those.
5. **`hydra-core` must come from PyPI.** conda-forge is stuck at 1.3.2; PyPI has **1.3.4,
   which is a security patch** (`hydra.utils.instantiate()` `_target_` blocklist).

---

## Recommended Versions

| Package | Version (pin to use) | Latest verified | Source | Confidence | Notes |
|---|---|---|---|---|---|
| python | `3.12.*` | 3.12 line current on conda-forge (3.14.6 is cf latest) | conda-forge | HIGH | Mandated. All packages below have `py312` builds on `osx-arm64` **and** `linux-64` — individually verified. |
| onnxruntime | `>=1.26,<1.27` | cf **1.26.0**; PyPI 1.27.0 (GH tag v1.27.1) | **conda-forge** | HIGH | cf py312 builds exist for 1.24.2 / 1.24.4 / 1.25.1 / 1.26.0 on osx-arm64 + linux-64. **1.23.x was never published to conda-forge.** cf deps: `__osx >=11.0`, `numpy >=1.23,<3`. CoreML EP ON for osx-arm64. 1.27.x not yet on conda-forge. |
| opencv | `>=4.13,<5` build `headless*` | cf 4.13.0 (upstream 4.14.0 released 2026-07-19, not yet on cf) | conda-forge | HIGH | **Never leave unpinned** (see #2 above). `libopencv 4.13.0` deps: `__osx >=11.0`, `numpy >=1.23,<3`. Feedstock `build.sh` sets `-DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib/modules`, so **`cv2.xfeatures2d` / contrib is included** — SIFT, AKAZE, ORB, `matchTemplate`, watershed all present. Both `headless` and `qt6` variants exist for py312 on both platforms; **choose `headless`** (this is a FastAPI/CLI app, no `cv2.imshow`; qt6 drags in all of Qt6). |
| numpy | `>=2.3,<3` | 2.5.1 (PyPI + cf) | conda-forge | HIGH | numpy 2.x is the correct choice — see *numpy 2.x verdict* below. numpy 2.5.1 requires Python `>=3.12`, so 3.12 is the floor, not a problem. |
| scipy | `>=1.17,<2` | 1.18.0 | conda-forge | HIGH | 1.18.0 requires Python `>=3.12`. Needed for `watershed` peak strategy support and Hough/geometry math. |
| scikit-learn | `>=1.8,<2` | 1.9.0 | conda-forge | HIGH | For `gmm` calibration strategy (`sklearn.mixture.GaussianMixture`). 1.9.0 requires `numpy>=1.24.1`, `scipy>=1.10.0` — compatible. |
| fastapi | `>=0.139,<1` | 0.139.2 | conda-forge | HIGH | Requires `starlette>=0.46.0`, `pydantic>=2.9.0`. cf has `starlette 1.3.1` which satisfies it. Do **not** use the `[standard]` extra from conda-forge; install `uvicorn-standard` + `python-multipart` explicitly. |
| uvicorn | `uvicorn-standard >=0.51,<1` | 0.51.0 (cf metapackage exists) | conda-forge | HIGH | `uvicorn-standard` is the conda-forge metapackage that pulls `uvloop`, `httptools`, `watchfiles`, `websockets`. Gives `--reload` for the dev serve task. |
| python-multipart | `>=0.0.32,<0.1` | 0.0.32 | conda-forge | HIGH | Required by API-06 (ad-hoc image upload endpoint). FastAPI errors at import of `UploadFile` routes without it. |
| pydantic | `>=2.13,<3` | 2.13.4 | conda-forge | HIGH | v2 as mandated. `pydantic-core==2.46.4` is pinned exactly by pydantic — let the solver handle it, do not pin `pydantic-core` yourself. |
| loguru | `>=0.7.3,<0.8` | 0.7.3 (PyPI + cf) | conda-forge | HIGH | 0.7.3 is still the latest; declares `requires_python <4.0,>=3.5`. No 0.8 exists. |
| ruff | `>=0.15.22,<0.16` **now** | cf 0.15.22; PyPI **0.16.0** (2026-07-23) | conda-forge | HIGH | conda-forge lags PyPI by ~1 day on ruff. **0.16.0 changed the default rule set from 59 → 413 rules.** Start on 0.15.22 for a reproducible Phase 1, keep `[tool.ruff.lint] select = [...]` **explicit** (an explicit `select` fully overrides the new defaults), then bump to `>=0.16,<0.17` once cf publishes. `line-length = 100` as mandated. |
| mypy | `>=2.3,<3` | 2.3.0 (PyPI + cf) | conda-forge | HIGH | **mypy 2.x is a major release**: `--local-partial-types` and `--strict-bytes` are now default; `--allow-redefinition` changed meaning. Greenfield project ⇒ adopt 2.x now rather than migrate later. cf `mypy 2.3.0 py312 osx-arm64` deps on `python-librt >=0.13.0` + `ast-serialize >=0.6.0`, both present on conda-forge. New: `--num-workers N` parallel checking (up to ~5x). |
| pytest | `>=9.1,<10` | 9.1.1 (PyPI + cf) | conda-forge | HIGH | **pytest 9 adds native TOML config** under `[tool.pytest]` (not `[tool.pytest.ini_options]`) — use it, but the two tables are mutually exclusive. Also adds `strict = true` (turns on `strict_config` + `strict_markers` + `strict_parametrization_ids` + `strict_xfail`) and built-in `subtests`. |
| pytest-cov | `>=7.1,<8` | 7.1.0 (PyPI + cf) | conda-forge | HIGH | Requires `coverage[toml]>=7.10.6`, `pytest>=7`. cf `coverage 7.15.2` satisfies. Drives the ≥80% gate via `--cov-fail-under=80`. |
| pytest-asyncio | `>=1.4,<2` | 1.4.0 | conda-forge | MEDIUM | Only needed if you test `async def` route handlers directly. If you test through `httpx.ASGITransport` / `TestClient`, you can drop it. |
| httpx | `>=0.28,<1` | 0.28.1 | conda-forge | HIGH | For FastAPI `TestClient` / `ASGITransport` API tests. |
| hydra-core | `>=1.3.4,<1.4` | PyPI **1.3.4** (2026-07-04); cf only 1.3.2 | **PyPI** | HIGH | 1.3.4 is a **security patch** (blocklist for security-sensitive `_target_` in `instantiate()`); 1.3.3 fixed source builds against modern setuptools. conda-forge has neither. Pull from PyPI so uv resolves `antlr4-python3-runtime==4.9.*` from PyPI too. Scope: CLI/benchmark entrypoint only, per IDEA §6. |
| matplotlib | `matplotlib-base >=3.10,<4` | 3.11.1 | conda-forge | HIGH | **Use `matplotlib-base`, not `matplotlib`** — the full `matplotlib` metapackage pulls PyQt/Tk GUI stacks the same way `opencv` qt6 does. Charts are written to files (`Agg`), never shown. |
| pillow | `>=12,<13` | 12.3.0 (PyPI + cf) | conda-forge | HIGH | Image I/O for PNG/JPEG sample renders; also a matplotlib dep. |
| pre-commit | `>=4.6,<5` | 4.6.1 (2026-07-21, PyPI + cf) | conda-forge | HIGH | Install into the dev env so the hook runner itself is locked. |
| onnx | `>=1.20,<2` | 1.22.0 (PyPI + cf) | conda-forge | MEDIUM | Only needed by `fetch-models` (graph inspection/validation of exported models). Put in the `export` feature, not the runtime env. |
| onnxslim | `>=0.1.94,<0.2` | 0.1.94 (PyPI); cf 0.1.94 | conda-forge | MEDIUM | Optional graph slimming in `fetch-models`. `export` feature only. |
| huggingface_hub | `>=1.24,<2` | 1.24.0 (PyPI + cf) | conda-forge | MEDIUM | Downloading pre-exported ONNX artefacts (e.g. `sefaburak/dinov2-small-onnx`) in `fetch-models`. Note **hub is on 1.x now**, API differs from 0.x. |
| torch / transformers / optimum / ultralytics | unpinned in `export` feature | torch 2.13.0, transformers **5.14.1**, optimum 2.2.0, ultralytics 8.4.104 | PyPI (export feature only) | LOW | Needed **only** to *produce* ONNX (DINOv2 via Optimum, FastAPI-irrelevant; FastSAM via Ultralytics). Must live in a separate `export` environment so the runtime/CI env stays torch-free — this is what keeps the "ONNX Runtime for every learned model" constraint structurally enforced. **transformers is on 5.x**, a major break from 4.x; version-check the export scripts when Phase 6/7 lands. |

### numpy 2.x verdict (research question 5) — **numpy 2.x, confirmed on both sides**

- **onnxruntime**: PyPI metadata for 1.23.2 / 1.24.4 / 1.25.1 / 1.26.0 / 1.27.0 all declare
  `numpy>=1.21.6` with **no upper bound**. The **conda-forge** `onnxruntime 1.26.0
  py312h753a246_0_cpu` (osx-arm64) declares `numpy >=1.23,<3` — i.e. explicitly built
  against the numpy 2 ABI and explicitly permitting 2.x. **HIGH confidence.**
- **conda-forge opencv**: `libopencv 4.13.0 headless_py312h839ed7b_0` (osx-arm64) declares
  `numpy >=1.23,<3`. Same story. **HIGH confidence.**
- **scipy 1.18 / scikit-learn 1.9 / matplotlib 3.11 / pillow 12** are all numpy-2-native.
- ⇒ Pin `numpy = ">=2.3,<3"`. Do **not** pin `<2`; on Python 3.12 with these package
  versions, numpy 1.x would be the incompatible choice, not numpy 2.x.
- The classic numpy-2 breakage (`opencv-python` wheels built against numpy 1 ABI →
  `_ARRAY_API not found` at `import cv2`) is **structurally impossible here**, because the
  constraint already forbids `opencv-python*` from PyPI. Keeping opencv on conda-forge is
  the numpy-2 safety mechanism, not just a style preference.

---

## Pixi Dependency Split

**Rule applied:** conda-forge for *everything* the solver can supply — not only native
code. A single solver over a single channel produces one coherent lock; every package
moved to `[pypi-dependencies]` is a second resolver whose result the conda solver cannot
see. Only two things go to PyPI: the local editable package (no choice) and `hydra-core`
(conda-forge is 2 patch releases behind, one of them a security fix).

```toml
[dependencies]
python = "3.12.*"

# --- native / binary components: conda-forge, non-negotiable ---
numpy           = ">=2.3,<3"
opencv          = { version = ">=4.13,<5", build = "headless*" }
onnxruntime     = ">=1.26,<1.27"
scipy           = ">=1.17,<2"
scikit-learn    = ">=1.8,<2"
pillow          = ">=12,<13"
matplotlib-base = ">=3.10,<4"

# --- pure-python runtime: still conda-forge, so one solver owns the whole graph ---
pydantic         = ">=2.13,<3"
fastapi          = ">=0.139,<1"
uvicorn-standard = ">=0.51,<1"
python-multipart = ">=0.0.32,<0.1"
loguru           = ">=0.7.3,<0.8"

[pypi-dependencies]
# The project itself — must be PyPI-side; hatchling builds it from pyproject.toml.
object-search = { path = ".", editable = true }
# conda-forge is stuck on 1.3.2; PyPI 1.3.4 is a SECURITY release. PyPI is correct here.
hydra-core    = ">=1.3.4,<1.4"

[feature.dev.dependencies]
ruff        = ">=0.15.22,<0.16"   # bump to >=0.16,<0.17 once conda-forge publishes 0.16
mypy        = ">=2.3,<3"
pytest      = ">=9.1,<10"
pytest-cov  = ">=7.1,<8"
httpx       = ">=0.28,<1"
pre-commit  = ">=4.6,<5"
# pytest-asyncio = ">=1.4,<2"     # only if you test `async def` handlers directly

# Model export/download tooling. Deliberately a SEPARATE environment so torch never
# enters the runtime or CI-test env — this is what makes "ONNX Runtime for every
# learned model" a structural guarantee rather than a code-review convention.
[feature.export.dependencies]
onnx             = ">=1.20,<2"
onnxslim         = ">=0.1.94,<0.2"
huggingface_hub  = ">=1.24,<2"

[feature.export.pypi-dependencies]
torch        = ">=2.9,<3"
transformers = ">=5.0,<6"
optimum      = ">=2.2,<3"
ultralytics  = ">=8.4,<9"
```

**Things deliberately NOT in the split, and why**

| Not added | Why |
|---|---|
| `opencv-python`, `opencv-python-headless`, `opencv-contrib-python` | Forbidden by constraint; also the exact source of numpy-2 ABI breakage. |
| `sqlite` / `pysqlite3` | `sqlite3` ships with CPython from conda-forge. Nothing to add. |
| `faiss` | Explicitly deferred per IDEA §5 (Method 5) — plain NumPy matmul in Milestone 1. |
| `torch` in the default env | Would let a PyTorch inference path in through the back door. `export` feature only. |
| `matplotlib` (metapackage) | Pulls Qt/Tk. Use `matplotlib-base`. |
| `onnxruntime` from `[pypi-dependencies]` | This is the whole macOS wheel-tag problem. See *Known Incompatibilities #1*. |
| `pydantic-settings` | Not needed; configs are per-method frozen models + Hydra for CLI. Add only if env-var config appears. |

---

## pixi.toml Skeleton

Verified against pixi docs `main` (`docs/reference/pixi_manifest.md`), pixi **0.73.0**
(latest release, 2026-07-15).

> ⚠️ The brief says "pixi 0.62.x". **Current pixi is 0.73.0.** Two things changed between
> 0.62 and now that affect this file: `[system-requirements]` is deprecated in favour of
> inline-table `platforms` entries, and `[project]` is fully superseded by `[workspace]`.
> Pin whatever version you install in CI explicitly (see *CI Notes*).

```toml
[workspace]
name        = "object-search-exploration"
version     = "0.1.0"
description = "Exemplar-based object search: draw one box, find every other instance"
authors     = ["Enrique G. Ortiz <ortizeg@gmail.com>"]
license     = "MIT"
readme      = "README.md"
channels    = ["conda-forge"]

# Bare strings are correct here. Only switch an entry to an inline table if you must
# raise a virtual-package floor, e.g. { platform = "osx-arm64", macos = "14.0" } to let
# pixi accept macosx_14_0_arm64 PyPI wheels. Not needed while onnxruntime comes from
# conda-forge. Inline tables REPLACE the deprecated [system-requirements] table.
platforms   = ["osx-arm64", "linux-64"]

# ---------------------------------------------------------------- dependencies
[dependencies]
python = "3.12.*"

numpy           = ">=2.3,<3"
opencv          = { version = ">=4.13,<5", build = "headless*" }
onnxruntime     = ">=1.26,<1.27"
scipy           = ">=1.17,<2"
scikit-learn    = ">=1.8,<2"
pillow          = ">=12,<13"
matplotlib-base = ">=3.10,<4"

pydantic         = ">=2.13,<3"
fastapi          = ">=0.139,<1"
uvicorn-standard = ">=0.51,<1"
python-multipart = ">=0.0.32,<0.1"
loguru           = ">=0.7.3,<0.8"

[pypi-dependencies]
object-search = { path = ".", editable = true }
hydra-core    = ">=1.3.4,<1.4"

# ---------------------------------------------------------------- features
[feature.dev.dependencies]
ruff       = ">=0.15.22,<0.16"
mypy       = ">=2.3,<3"
pytest     = ">=9.1,<10"
pytest-cov = ">=7.1,<8"
httpx      = ">=0.28,<1"
pre-commit = ">=4.6,<5"

[feature.export.dependencies]
onnx            = ">=1.20,<2"
onnxslim        = ">=0.1.94,<0.2"
huggingface_hub = ">=1.24,<2"

[feature.export.pypi-dependencies]
torch        = ">=2.9,<3"
transformers = ">=5.0,<6"
optimum      = ">=2.2,<3"
ultralytics  = ">=8.4,<9"

# ---------------------------------------------------------------- environments
# `default` deliberately includes dev: one env for local work and for CI lint/type/test,
# so a green CI run and a green local run are provably the same solve.
# `export` is a separate solve group — torch must never influence the runtime solve.
[environments]
default = { features = ["dev"], solve-group = "main" }
prod    = { features = [], solve-group = "main" }          # runtime-only, for image builds
export  = { features = ["export"], solve-group = "export" }

# ---------------------------------------------------------------- tasks
[tasks]
# ---- quality gates (hard gates per IDEA §8) ----
lint        = { cmd = "ruff check src tests", description = "Ruff lint (line-length 100)" }
format      = { cmd = "ruff format src tests", description = "Ruff format in place" }
format-check = { cmd = "ruff format --check --output-format github src tests", description = "Ruff format check, CI annotations" }
typecheck   = { cmd = "mypy --strict src tests", description = "MyPy strict" }
test        = { cmd = "pytest", description = "Pytest with coverage gate (see pyproject)" }
test-fast   = { cmd = "pytest -x -q --no-cov -m 'not slow'", description = "Fast inner-loop tests" }
check       = { depends-on = ["lint", "format-check", "typecheck", "test"], description = "Everything CI runs" }
hooks       = { cmd = "pre-commit install --install-hooks", description = "Install git hooks (run once)" }
precommit   = { cmd = "pre-commit run --all-files", description = "Run all hooks over the repo" }

# ---- models (INFRA-11): scripted + reproducible, weights gitignored ----
fetch-models = { cmd = "python -m object_search.cli fetch-models --dest models",
                 description = "Download/export every ONNX model into ./models",
                 default-environment = "export",
                 outputs = ["models/*.onnx", "models/MANIFEST.json"] }
verify-models = { cmd = "python -m object_search.cli verify-models --dest models",
                  description = "Load every ONNX model and assert dtype/shape contracts (INFRA-09)",
                  inputs = ["models/MANIFEST.json"] }

# ---- serve (API + static frontend) ----
serve     = { cmd = "uvicorn object_search.api.app:app --host 127.0.0.1 --port 8000 --reload",
              description = "Dev server with reload",
              depends-on = ["verify-models"] }
serve-prod = { cmd = "uvicorn object_search.api.app:app --host 0.0.0.0 --port 8000 --workers 1",
               description = "Single-worker serve (ONNX sessions are per-process, API-07)",
               default-environment = "prod" }

# ---- benchmark / evaluation (Hydra entrypoint, IDEA §6) ----
bench = { cmd = "python -m object_search.eval.benchmark",
          description = "Hydra benchmark sweep: method x config x image -> P/R/F1/AP/latency",
          depends-on = ["verify-models"],
          outputs = ["outputs/bench"] }
bench-charts = { cmd = "python -m object_search.eval.charts --input outputs/bench --dest docs/bench",
                 description = "Render benchmark tables + charts (EVAL-06)" }

# ---- committed sample runs (DOC-02) ----
samples = { cmd = "python -m object_search.cli render-samples --dest docs/samples",
            description = "Re-render the committed per-method sample runs",
            depends-on = ["verify-models"],
            outputs = ["docs/samples"] }
samples-check = { cmd = "git diff --exit-code -- docs/samples",
                  description = "Fail if committed sample runs drifted",
                  depends-on = ["samples"] }

# ---- data ----
synth = { cmd = "python -m object_search.eval.synthetic --dest assets/demo/synthetic --seed {{ seed }}",
          args = [{ arg = "seed", default = "0" }],
          description = "Generate synthetic images + exact ground truth (EVAL-03)" }
```

**Notes on the skeleton**

- `description` on tasks makes `pixi task list` self-documenting — cheap, and it is the
  only discoverability surface a new contributor gets.
- `default-environment = "export"` on `fetch-models` means `pixi run fetch-models` works
  from a plain shell without the caller having to remember `-e export`.
- `inputs`/`outputs` enable pixi's task caching; skip them on tasks whose output must
  always be recomputed.
- `serve-prod` uses `--workers 1` on purpose: API-07 loads ONNX sessions once at startup
  via `lifespan`, and each uvicorn worker is a separate process with its own sessions.
  More workers = N× the model memory. Make that a deliberate decision, not a default.
- `samples-check` is the CI guard for DOC-02 ("regenerable by one CLI command").
- Task shell is `deno_task_shell`, not bash — it is cross-platform but does **not** support
  every bash-ism. Keep task bodies to a single `python -m ...` invocation and put logic in
  Python. (The sibling project's `pipeline-dag` task, a 700-char inline `python -c`, is the
  anti-pattern to avoid.)

### Companion `pyproject.toml` fragments (the non-obvious bits)

```toml
[tool.ruff]
line-length = 100
target-version = "py312"
src = ["src", "tests"]

[tool.ruff.lint]
# Keep `select` EXPLICIT. ruff 0.16.0 raised the default rule set from 59 to 413 rules;
# an explicit select is immune to that change, an `extend-select` is not.
select = ["E", "F", "W", "I", "N", "UP", "B", "A", "C4", "SIM", "ARG", "PTH", "RUF",
          "ANN", "D", "TID", "T20", "LOG", "G", "NPY", "PL"]
ignore = ["D203", "D213"]

[tool.ruff.lint.per-file-ignores]
"tests/**" = ["D", "ANN", "PLR2004"]

[tool.ruff.lint.flake8-tidy-imports.banned-api]
"logging".msg = "Use loguru (INFRA-05)."

[tool.mypy]
python_version = "3.12"
strict = true
warn_unreachable = true
# mypy 2.x: --local-partial-types and --strict-bytes are ON by default. Do not re-enable
# the legacy behaviour; a greenfield codebase should be written against the new defaults.
num_workers = 4          # mypy 2.0 parallel checking
files = ["src", "tests"]

[[tool.mypy.overrides]]
module = ["cv2.*", "onnxruntime.*", "sklearn.*", "scipy.*"]
ignore_missing_imports = true

# pytest 9 native TOML table. NOTE: [tool.pytest] and [tool.pytest.ini_options] are
# mutually exclusive — pick one. Native TOML is the better choice on pytest >= 9.
[tool.pytest]
minversion = "9.1"
testpaths = ["tests"]
strict = true            # pytest 9: strict_config + strict_markers + strict_xfail + ids
addopts = [
  "-ra",
  "--cov=object_search",
  "--cov-report=term-missing",
  "--cov-report=xml",
  "--cov-fail-under=80",    # INFRA-06 hard gate
]
markers = ["slow: needs ONNX weights or is otherwise slow"]
```

### `.pre-commit-config.yaml` — the version-coupling rule

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.15.22            # MUST equal the ruff pinned in [feature.dev.dependencies]
    hooks: [{ id: ruff-check, args: [--fix] }, { id: ruff-format }]

  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v6.0.0
    hooks:
      - id: check-added-large-files
        args: [--maxkb=512]   # keeps ONNX weights out of git (INFRA-11)
      - id: check-yaml
      - id: check-toml
      - id: end-of-file-fixer
      - id: trailing-whitespace
      - id: detect-private-key

  # Run mypy as a LOCAL hook through pixi, not via pre-commit/mirrors-mypy (v2.3.0 exists,
  # but the mirror builds its own isolated venv that lacks numpy/pydantic/cv2 stubs, so
  # strict mode reports different errors than `pixi run typecheck`. One source of truth.)
  - repo: local
    hooks:
      - id: mypy
        name: mypy (pixi, strict)
        entry: pixi run -e default mypy --strict
        language: system
        types: [python]
        pass_filenames: false
```

Latest verified hook revs: `astral-sh/ruff-pre-commit` **v0.16.0** (v0.15.22 also tagged),
`pre-commit/pre-commit-hooks` **v6.0.0**, `pre-commit/mirrors-mypy` **v2.3.0**,
`pre-commit` itself **v4.6.1** (2026-07-21).

---

## Known Incompatibilities

### 1. onnxruntime PyPI wheels vs pixi's default macOS floor — the sibling project's bug, explained

Verified from the PyPI file listing for every recent onnxruntime release:

| onnxruntime | cp312 macOS wheel tag |
|---|---|
| 1.21.1 – 1.22.1 | `macosx_13_0_universal2` |
| 1.23.0, 1.23.1, **1.23.2** | `macosx_13_0_arm64` + `macosx_13_0_x86_64` |
| **1.24.1** – 1.27.0 | `macosx_14_0_arm64` (arm64 only; x86_64 macOS dropped) |

Verified from pixi docs: the **default `__osx` virtual package for `osx-arm64` is `13.0`**.
pixi resolves PyPI wheels against the platform's virtual packages, so a
`macosx_14_0_arm64` wheel does not satisfy `__osx 13.0` and the solve fails with
"no wheel / no source distribution for the current platform." That is exactly the boundary
between 1.23.2 (last `13_0` tag) and 1.24.1 (first `14_0` tag), which is why the sibling
project's pin landed where it did. **The pin was a correct workaround for a real
constraint, not superstition — but it is now the wrong fix.**

Two fixes, in order of preference:

- **(A, recommended) Move onnxruntime to conda-forge.** conda packages use `__osx >=11.0`
  for onnxruntime on osx-arm64, so the wheel-tag question disappears entirely. Also gets
  you CoreML EP, and one solver over the whole graph. Cost: conda-forge tops out at 1.26.0
  (1.27.x not yet published), and it is a `_cpu` build (no CUDA on osx-arm64 anyway; the
  linux-64 CI job only needs CPU).
- **(B, fallback if you need 1.27+ now)** Keep it on PyPI and raise the floor:
  ```toml
  platforms = [{ platform = "osx-arm64", macos = "14.0" }, "linux-64"]
  ```
  Cost: the environment now refuses to install on macOS 13 machines. Acceptable for a
  single-developer local-first project; note it in the README.

**Do not** re-pin `onnxruntime == 1.23.2`: it is the only version in the range that
conda-forge never published, so pinning it forces you back onto PyPI and back into this
whole problem.

### 2. conda-forge `opencv >= 5.0.0` has no Python bindings — `opencv = "*"` will break `import cv2`

Verified by enumerating conda-forge build strings. On **every** subdir (osx-arm64, osx-64,
linux-64, linux-aarch64, linux-ppc64le, win-64):

- `opencv 4.13.0` → `headless_py312h...`, `qt6_py312h...` (per-Python builds ⇒ bindings)
- `opencv 5.0.0` → `headless_h5b059e6_3`, `qt6_h3f25391_603` (**no `py3XX`** ⇒ no bindings)
- `py-opencv 5.0.0` → same, `headless_hc6da5dd_3` etc., no `py3XX`

The sibling project uses `opencv = "*"`. On a fresh solve today that can pick 5.0.0 and the
environment installs cleanly but has no `cv2`. **Pin `opencv = ">=4.13,<5"`.** Revisit only
after conda-forge publishes `py312`-tagged OpenCV 5 builds, and then only with a full pass
over Method 1/2 code — OpenCV 5 is a breaking major release.

### 3. `matplotlib` and `opencv` (qt6 variant) drag a GUI stack into a headless service

Not a hard failure, a size/robustness one: `matplotlib` (metapackage) pulls PyQt/Tk;
`opencv`'s `qt6_*` variant pulls all of Qt6. Both add hundreds of MB, slow CI cache
restore, and give you `cv2.imshow`/interactive backends that will hang on a CI runner if
anything ever calls them. Use `matplotlib-base` and `build = "headless*"`.

### 4. `hydra-core` version skew between conda-forge and PyPI, plus the antlr pin

- conda-forge `hydra-core` latest **1.3.2** (plus a `1.4.0.dev1`); PyPI has **1.3.3** and
  **1.3.4**. 1.3.4 is a security release. Take it from PyPI.
- `hydra-core` requires `antlr4-python3-runtime==4.9.*` (PyPI name) while conda-forge calls
  the package `antlr-python-runtime` and has `4.9.2`/`4.9.3` plus much newer 4.10–4.13.2.
  Mixing hydra-from-conda-forge with anything that wants modern antlr is a known
  conflict class. Taking hydra from PyPI keeps that resolution entirely inside uv.
- Do **not** jump to `1.4.0.dev1`. Pin `>=1.3.4,<1.4`.

### 5. mypy 2.x changed defaults — treat as a migration even on a greenfield repo

`--local-partial-types` and `--strict-bytes` are now on by default, and
`--allow-redefinition` silently changed meaning (old behaviour is now
`--allow-redefinition-old`). Consequences to write code against from day one: `bytearray`
and `memoryview` are **not** assignable to `bytes` (relevant to image-bytes handling in the
upload endpoint), and cross-scope partial-type inference is stricter. Also: mypy 2.x
introduces a compiled `librt` runtime dependency — on conda-forge that is `python-librt`
(present); on PyPI it is `librt` (**not on conda-forge under that name**), which is another
reason to take mypy from conda-forge rather than PyPI.

### 6. ruff 0.16.0's default-rules jump (59 → 413)

Released 2026-07-23. If you configure lint via `extend-select`, upgrading will surface
hundreds of new diagnostics at once. With an explicit `select = [...]` list you are
immune. 0.16.0 also formats Python code blocks inside Markdown by default — relevant
because this project commits a lot of Markdown with fenced Python (`docs/methods/*.md`).
Either accept it or add `exclude`/`*.md` handling deliberately.

### 7. pytest 9's two config tables are mutually exclusive

`[tool.pytest]` (native TOML) and `[tool.pytest.ini_options]` (INI-compat) cannot both be
present. The sibling project uses `ini_options`; if you copy from it, convert fully rather
than partially.

### 8. Python 3.12 is now the *lower* edge, not the safe middle

conda-forge's latest Python is 3.14.6, and several pinned packages have already raised
their floor to 3.12 (`numpy 2.5.1`, `scipy 1.18.0` both require `>=3.12`). onnxruntime
1.24+ requires `>=3.11`. So 3.12 is fine today, but the margin is shrinking from below,
not from above — expect to plan a 3.13 bump within a few releases. Nothing in the current
pin set forces it.

---

## CI Notes

**Action versions verified 2026-07-24:** `prefix-dev/setup-pixi` latest **v0.10.0**
(2026-06-29); pixi latest **v0.73.0** (2026-07-15).

```yaml
name: ci
on:
  push: { branches: [main] }
  pull_request:

concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true

jobs:
  check:
    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-latest, macos-14]   # linux-64 + osx-arm64, matching workspace.platforms
    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v5

      - uses: prefix-dev/setup-pixi@v0.10.0
        with:
          pixi-version: v0.73.0     # pin explicitly; never float in CI
          environments: default
          locked: true              # fail if pixi.lock is stale vs pixi.toml
          cache: true               # keyed on pixi.lock hash
          cache-write: ${{ github.ref == 'refs/heads/main' }}
          activate-environment: false

      - run: pixi run lint
      - run: pixi run format-check
      - run: pixi run typecheck
      - run: pixi run test
```

**`--locked` vs `--frozen` — the rule**

- `locked: true` ⇒ `pixi install --locked`: verifies `pixi.lock` is **consistent with
  `pixi.toml`** and fails loudly if it is not. **This is what CI should use.** It turns
  "someone edited pixi.toml and forgot to commit the lock" into a red build instead of a
  silent re-solve that makes CI test a different environment than the developer did.
- `frozen: true` ⇒ `pixi install --frozen`: installs the lock **without** checking it
  against the manifest. Faster, and useful for a deploy/build job that must be perfectly
  reproducible, but it will happily install a stale lock. **Do not use it for the gate
  job.**
- `setup-pixi`'s default is already `locked: true` when a `pixi.lock` exists. Set it
  explicitly anyway so the intent survives a future default change.
- Never both. Setting `frozen` and `locked` together is contradictory.

**Other CI gotchas**

1. **Pin `pixi-version`.** The default floats to latest pixi; a pixi release can change the
   lockfile format or solver behaviour and turn a green PR red with no code change.
   Bump it in a dedicated PR.
2. **Cache keying.** Caching is on by default and keyed on the `pixi.lock` hash. It is
   therefore only correct if `pixi.lock` is committed — commit it. Restrict `cache-write`
   to `main` so PR branches read the cache but do not each write a new multi-hundred-MB
   entry.
3. **`post-cleanup` defaults to `false`.** That is right for GitHub-hosted runners. If you
   ever move to self-hosted, set it `true` to avoid leaking the env (and credentials)
   between jobs.
4. **`macos-14` not `macos-latest`** for the arm64 job — it is the oldest arm64 image, so
   it is what actually exercises the `__osx` floor. If you take fallback (B) from
   *Known Incompatibilities #1*, `macos-13` would break, which is the signal you want.
5. **Don't install the `export` environment in the gate job.** It pulls torch +
   transformers (multi-GB). Give model export its own manually-triggered or scheduled
   workflow with `environments: export`.
6. **Tests must not require ONNX weights.** Weights are gitignored (INFRA-11), so the CI
   gate can only run tests marked `not slow` unless you add a weight-download step. Two
   jobs: a fast gate on every PR, and a nightly/`workflow_dispatch` job that runs
   `pixi run fetch-models && pixi run test` in full. Keep the ≥80% coverage gate on the
   fast job by unit-testing inferencers against synthetic ONNX graphs rather than real
   weights.
7. **`samples-check` in CI** (see skeleton) enforces DOC-02 — but only in the weights-
   enabled job, since rendering samples needs models.
8. **Branch protection (INFRA-07)** should require the `check` status on *both* matrix legs,
   not just `check`. Otherwise a macOS-only solver failure can merge.

---

## Open Questions / Low Confidence Items

| # | Item | Confidence | What to do |
|---|---|---|---|
| 1 | Does `build = "headless*"` glob-match correctly in pixi's matchspec parser? Verified that `headless_py312*` builds **exist** for `opencv` 4.13.0 on both target subdirs; did **not** verify pixi/rattler glob handling by running a solve. | MEDIUM | Run `pixi install` in Phase 1 and check `pixi list opencv`. If the glob is rejected, fall back to `opencv = ">=4.13,<5"` and accept qt6, or depend on `libopencv = { version = ">=4.13,<5", build = "headless*" }` and let `opencv` follow. |
| 2 | Brief says "pixi 0.62.x"; latest is **0.73.0**. The skeleton uses the *current* form (inline-table platforms, `[workspace]`). | HIGH on 0.73, N/A for 0.62 | Install 0.73.0. If 0.62.x is a hard requirement, `[system-requirements]` is the form to use there and the deprecation note does not apply — but there is no reason to stay on 0.62. |
| 3 | Is `[project]` still accepted as an alias for `[workspace]`? Current docs mention only `[workspace]` and give no deprecation note for `[project]`. | LOW | Irrelevant in practice — use `[workspace]`. Do not write `[project]` in a new manifest. |
| 4 | Exact upgrade date for `ruff 0.16.0` on conda-forge (PyPI 2026-07-23, cf at 0.15.22 as of 2026-07-24). | MEDIUM | Check `pixi search ruff` at Phase 1 start; if 0.16 is there, use `>=0.16,<0.17` and bump the pre-commit rev to `v0.16.0` in the same commit. |
| 5 | Whether the `export`-feature tool versions (torch 2.13, transformers **5.x**, optimum 2.2, ultralytics 8.4) actually produce working ONNX for DINOv2 / SuperPoint / FastSAM. transformers 5.x is a major break from the 4.x APIs most export guides assume. | LOW | Do not pin these hard now. Resolve during Phase 6/7 under the `library-review` skill, as IDEA §10 already requires. Record the working versions in `models/MANIFEST.json` alongside each model hash (feeds EVAL-09 provenance). |
| 6 | `onnxruntime 1.27.x` on conda-forge (upstream v1.27.0 2026-06-19, v1.27.1 2026-07-11; cf at 1.26.0). | MEDIUM | Not needed. Stay on 1.26 from conda-forge; revisit only if a 1.27 feature is required. |
| 7 | CoreML EP *performance* for DINOv2 / FastSAM on M-series. Verified only that the provider is **compiled in and importable** (feedstock asserts `'CoreMLExecutionProvider' in ort.get_available_providers()`); no benchmark run. | LOW | Make the provider list a config field on `ONNXInferencer` and measure CPU vs CoreML in the Phase 8 latency breakdown (EVAL-11). Treat CoreML as an experiment, not an assumption — it is known to fall back to CPU for unsupported ops. |
| 8 | Whether `pytest-asyncio` is needed. Depends on whether API tests go through `httpx.ASGITransport` (no) or call `async def` handlers directly (yes). | MEDIUM | Decide in Phase 3; `httpx` is in the dev feature either way. |
| 9 | numpy 2.5's own removals/deprecations vs the project's array code. Verified ABI/metadata compatibility across the stack; did **not** read numpy 2.4/2.5 release notes for API removals. | MEDIUM | Cheap to settle: `pixi run lint` with ruff's `NPY` rules enabled (in the `select` list above) flags most legacy numpy API use automatically. |
| 10 | conda-forge `onnxruntime` builds carry build-string suffix `_cpu`; a hypothetical future need for a GPU EP on linux-64 CI would need `onnxruntime = { version = "...", build = "*cuda*" }` or similar. | LOW | Out of scope — project is local-first CPU/CoreML. Note only. |

---

## Sources

Fetched live on 2026-07-24.

**PyPI JSON API** (per-package and per-version metadata: latest version, `requires_python`, `requires_dist`, and the full wheel file list used for the macOS tag table)
- `https://pypi.org/pypi/onnxruntime/json` and `/{1.23.2,1.24.4,1.25.1,1.26.0,1.27.0}/json`
- `https://pypi.org/pypi/numpy/json`
- `https://pypi.org/pypi/scipy/json`
- `https://pypi.org/pypi/scikit-learn/json`
- `https://pypi.org/pypi/fastapi/json`
- `https://pypi.org/pypi/uvicorn/json`
- `https://pypi.org/pypi/pydantic/json`
- `https://pypi.org/pypi/loguru/json`
- `https://pypi.org/pypi/ruff/json`
- `https://pypi.org/pypi/mypy/json`
- `https://pypi.org/pypi/pytest/json`
- `https://pypi.org/pypi/pytest-cov/json`
- `https://pypi.org/pypi/pytest-asyncio/json`
- `https://pypi.org/pypi/hydra-core/json` and `/1.3.4/json`
- `https://pypi.org/pypi/matplotlib/json`
- `https://pypi.org/pypi/pillow/json`
- `https://pypi.org/pypi/opencv-python/json`
- `https://pypi.org/pypi/httpx/json`
- `https://pypi.org/pypi/python-multipart/json`
- `https://pypi.org/pypi/onnx/json`
- `https://pypi.org/pypi/onnxslim/json`
- `https://pypi.org/pypi/huggingface-hub/json`
- `https://pypi.org/pypi/optimum/json`
- `https://pypi.org/pypi/torch/json`
- `https://pypi.org/pypi/transformers/json`
- `https://pypi.org/pypi/ultralytics/json`

**anaconda.org conda-forge API** (channel availability, version lists, and per-file `subdir` / `build` / `depends` metadata — this is where the py312 build enumeration, the opencv-5 no-bindings finding, and the `__osx >=11.0` / `numpy >=1.23,<3` constraints come from)
- `https://api.anaconda.org/package/conda-forge/onnxruntime` and `.../onnxruntime/files`
- `https://api.anaconda.org/package/conda-forge/opencv` and `.../opencv/files`
- `https://api.anaconda.org/package/conda-forge/py-opencv` and `.../py-opencv/files`
- `https://api.anaconda.org/package/conda-forge/libopencv` and `.../libopencv/files`
- `https://api.anaconda.org/package/conda-forge/numpy` and `.../numpy/files`
- `https://api.anaconda.org/package/conda-forge/mypy` and `.../mypy/files`
- `https://api.anaconda.org/package/conda-forge/hydra-core` and `.../hydra-core/files`
- `https://api.anaconda.org/package/conda-forge/antlr4-python3-runtime/files`
- `https://api.anaconda.org/package/conda-forge/{scipy,scikit-learn,matplotlib-base,pillow,fastapi,uvicorn,uvicorn-standard,pydantic,loguru,ruff,pytest,pytest-cov,pytest-asyncio,python,starlette,httpx,coverage,omegaconf,pre-commit,onnx,onnxslim,huggingface_hub,optimum,ast-serialize,librt,mypy_extensions,pathspec,python-multipart,pixi}`

**conda-forge feedstocks**
- `https://raw.githubusercontent.com/conda-forge/opencv-feedstock/main/recipe/build.sh` — confirms `-DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib/modules` (contrib included)
- `https://raw.githubusercontent.com/conda-forge/onnxruntime-feedstock/main/recipe/build.sh` — confirms `--use_coreml` on Apple Silicon
- `https://raw.githubusercontent.com/conda-forge/onnxruntime-feedstock/main/recipe/meta.yaml` — confirms the `CoreMLExecutionProvider` import test and licence composition

**pixi**
- `https://pixi.prefix.dev/latest/reference/pixi_manifest/`
- `https://pixi.prefix.dev/latest/workspace/system_requirements/` — default `__osx = "13.0"` for osx-arm64 and osx-64
- `https://raw.githubusercontent.com/prefix-dev/pixi/main/docs/reference/pixi_manifest.md` — inline-table `platforms`, `[system-requirements]` deprecation, `[tasks]` field list, `[environments]` fields
- `https://api.github.com/repos/prefix-dev/pixi/releases` — v0.73.0, 2026-07-15
- `https://github.com/prefix-dev/setup-pixi` and `https://raw.githubusercontent.com/prefix-dev/setup-pixi/main/action.yml` — v0.10.0, input list and defaults

**GitHub Releases / Tags API**
- `https://api.github.com/repos/astral-sh/ruff/releases` and `.../releases/tags/0.16.0`
- `https://api.github.com/repos/pytest-dev/pytest/releases` and `.../releases/tags/9.0.0`
- `https://api.github.com/repos/facebookresearch/hydra/releases` — 1.3.3 (2026-06-11), 1.3.4 (2026-07-04) release bodies
- `https://api.github.com/repos/microsoft/onnxruntime/releases`
- `https://api.github.com/repos/opencv/opencv/releases` — 5.0.0 (2026-06-06), 4.14.0 (2026-07-19)
- `https://api.github.com/repos/pre-commit/pre-commit/releases`
- `https://api.github.com/repos/astral-sh/ruff-pre-commit/tags`
- `https://api.github.com/repos/pre-commit/pre-commit-hooks/tags`
- `https://api.github.com/repos/pre-commit/mirrors-mypy/tags`

**mypy 2.x**
- `https://mypy-lang.blogspot.com/` (2.0 released 2026-05-06, 2.1, 2.2, 2.3 notes)
- `https://mypy-lang.blogspot.com/2026/05/mypy-20-relased.html`
- `https://mypy.readthedocs.io/en/stable/changelog.html`

**Local files read for baseline/context**
- `/Users/ortizeg/1Projects/⛹️‍♂️ Next Play/code/object-search-exploration/.planning/IDEA.md`
- `/Users/ortizeg/1Projects/⛹️‍♂️ Next Play/code/basketball-2d-to-3d/pixi.toml`
- `/Users/ortizeg/1Projects/⛹️‍♂️ Next Play/code/basketball-2d-to-3d/pyproject.toml`
