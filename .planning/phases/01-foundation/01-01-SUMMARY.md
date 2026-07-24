---
phase: 01-foundation
plan: 01
subsystem: infrastructure
tags: [pixi, ruff, mypy, pytest, pre-commit, github-actions, scaffold]
requires: []
provides:
  - pixi environment with the proven version baseline (osx-arm64 + linux-64)
  - pyproject.toml as single source of truth for ruff/mypy/pytest/coverage
  - src-layout object_search package with py.typed
  - Loguru sink configuration (setup_logging)
  - three proven-real quality gates (print ban, logging ban, 80% coverage floor)
  - green CI workflow (job name `quality`)
  - environment guard tests that fail if cv2 bindings disappear
affects:
  - every subsequent phase inherits these gates
tech-stack:
  added:
    python: 3.12.13
    numpy: 2.5.1
    opencv: 4.13.0 (conda-forge, headless)
    onnxruntime: 1.23.2 (PyPI)
    scipy: 1.18.0
    scikit-learn: 1.9.0
    pillow: 12.3.0
    matplotlib-base: 3.11.1
    pydantic: 2.13.4
    fastapi: 0.139.2
    loguru: 0.7.3
    hydra-core: 1.3.4
    ruff: 0.16.0
    mypy: 2.3.0
    pytest: 9.1.1
    pytest-cov: 7.1.0
  patterns:
    - pixi-only environment management; every command is a pixi task
    - conda-forge owns the whole native graph for one coherent numpy 2 ABI
    - explicit ruff select list (immune to ruff default-rule-set changes)
    - mypy strict with per-module escape hatches, never a global ignore
key-files:
  created:
    - pixi.toml
    - pixi.lock
    - pyproject.toml
    - .pre-commit-config.yaml
    - .gitignore
    - README.md
    - src/object_search/__init__.py
    - src/object_search/py.typed
    - src/object_search/log.py
    - tests/__init__.py
    - tests/conftest.py
    - tests/test_log.py
    - tests/test_environment.py
    - .github/workflows/ci.yml
    - .github/pull_request_template.md
    - .vscode/settings.json
  modified: []
decisions:
  - opencv pinned >=4.13,<5 with build headless* -- the <5 bound prevents a green install with no cv2
  - onnxruntime kept at ==1.23.2 from PyPI; conda-forge >=1.26 alternative recorded as a TODO
  - hydra-core >=1.3.4 from PyPI for the instantiate() security patch
  - CI is a single ubuntu-latest job so the required check name stays exactly `quality`
  - environment guard tests added so a dependency bump cannot silently remove cv2
metrics:
  duration: 16m
  completed: 2026-07-24
  tasks: 6
  tasks_complete: 5
  files: 19
  commits: 8
status: complete-with-blocker
---

# Phase 1 Plan 01: Pixi Scaffold, Quality Gates, CI, and Branch Protection Summary

Repository scaffold with three **demonstrably non-advisory** quality gates — Ruff
(line-length 100, `print()` and stdlib `logging` both banned), MyPy strict, and pytest with
an 80% coverage floor — enforced locally by pre-commit and remotely by a green CI job.
Branch protection is the one item that could not be applied: it is a paid GitHub feature for
private repositories.

## What was built

**Environment (INFRA-01).** `pixi.toml` on the modern `[workspace]` table, `python = "3.12.*"`,
platforms `["osx-arm64", "linux-64"]`, 11 tasks. conda-forge owns the entire native graph
(numpy, opencv, scipy, scikit-learn, pillow, matplotlib-base) so the numpy 2 ABI is
consistent across `cv2`, `scipy` and `sklearn`. `pixi.lock` is committed — this is an
application, so CI installs `--locked` rather than re-solving.

**Tooling as one source of truth (INFRA-02/03/05/06).** `pyproject.toml` configures ruff,
mypy, pytest and coverage. There is no second place where a gate can be quietly relaxed. The
ruff `select` list is explicit rather than `extend-select`, which makes it immune to ruff
0.16.0's default-rule-set jump from 59 to 413 rules. MyPy is `strict = true` with
`ignore_missing_imports = false` globally and the escape hatch scoped to exactly
`cv2/onnxruntime/sklearn/scipy`. `disallow_any_explicit` was deliberately left unset, per
the phase context: `Diagnostics` payloads legitimately need object-typed mappings, and the
flag would force a pile of ignores that make strict mode dishonest.

**Package + logging (INFRA-02/05).** src-layout `object_search` with `py.typed` (PEP 561).
`log.py` is the only place Loguru sinks are configured; `setup_logging` opens with
`logger.remove()` so it is idempotent, and its docstring states the rule that matters — call
it once at an entry point, never from a library module, because `logger.add()` in an imported
module appends a handler per importer and the duplication stays invisible until output is
unreadable.

**Hooks, editor, README (INFRA-04).** pre-commit with the standard hygiene hooks plus
ruff/ruff-format, and mypy as a *local* hook through `pixi run typecheck`. The mirror hook
was rejected on purpose: `mirrors-mypy` builds an isolated venv that cannot see onnxruntime,
pydantic, fastapi or the conda-forge numpy, so strict mode there reports a different and
largely bogus error set than the pixi env. The config carries the version-coupling rule as a
comment — the ruff hook `rev` must equal the ruff pinned in `pixi.toml` (both 0.16.0).

**CI (INFRA-07, first half).** Job `quality` on `ubuntu-latest` runs the four gates as four
separate steps, so a red build names the gate that failed. `pixi-version` is pinned to
`v0.62.2` (the version that produced the lock) because setup-pixi otherwise floats to latest
and a pixi release can change lockfile format or solver behaviour and turn a green PR red
with no code change.

## Resolved versions

| Package | Resolved | Source |
| --- | --- | --- |
| python | 3.12.13 | conda-forge |
| numpy | 2.5.1 | conda-forge |
| opencv / libopencv / py-opencv | 4.13.0 `headless_*_14` | conda-forge |
| onnxruntime | 1.23.2 (`macosx_13_0_arm64` wheel) | PyPI |
| scipy | 1.18.0 | conda-forge |
| scikit-learn | 1.9.0 | conda-forge |
| pillow | 12.3.0 | conda-forge |
| matplotlib-base | 3.11.1 | conda-forge |
| pydantic | 2.13.4 | PyPI |
| fastapi | 0.139.2 | PyPI |
| loguru | 0.7.3 | PyPI |
| hydra-core | 1.3.4 | PyPI |
| ruff / mypy | 0.16.0 / 2.3.0 | PyPI |
| pytest / pytest-cov | 9.1.1 / 7.1.0 | PyPI |

The lock also covers linux-64 with
`onnxruntime-1.23.2-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl` and
`py-opencv-4.13.0-headless_h75b7ce4_14.conda`.

## Gate proofs — the point of this plan

Each gate was **deliberately violated** and confirmed to fail. A gate never observed failing
is an advisory gate.

| Gate | Violation | Result | Exit |
| --- | --- | --- | --- |
| `print()` banned | added `print("x")` to `src/object_search/__init__.py` | `T201 \`print\` found` | 1 |
| stdlib `logging` banned | added a **used** `import logging` (`_LOG = logging.getLogger(__name__)`) | `TID251 \`logging\` is banned: Use loguru...` — the only error reported | 1 |
| coverage floor | added a module with uncovered branches | `FAIL Required test coverage of 80% not reached. Total coverage: 41.67%` | 1 |
| baseline (all reverted) | — | `All checks passed!` / `Total coverage: 100.00%` | 0 |

The `logging` proof used a *used* import specifically so the failure could not be attributed
to `F401` unused-import. `TID251` was the sole error, which is what proves the banned-api
rule — not dead-code detection — is doing the work.

Local gate output, all exit 0: `All checks passed!` / `6 files already formatted` /
`Success: no issues found in 2 source files` / `10 passed`, `Total coverage: 100.00%`.

## CI

- First run, green on the first attempt:
  <https://github.com/ortizeg/object-search-exploration/actions/runs/30116711957>
- Final run, green:
  <https://github.com/ortizeg/object-search-exploration/actions/runs/30116790235>

On linux-64 CI: `platform linux -- Python 3.12.13`, `10 passed`,
`Required test coverage of 80% reached. Total coverage: 100.00%`. The environment guard tests
passing on linux confirms the lock is genuinely cross-platform, not just locally green.

**PR:** <https://github.com/ortizeg/object-search-exploration/pull/1> — left open for review,
not merged.

## Deviations from Plan

**1. [Rule 2 - Correctness] opencv pinned `>=4.13,<5` with `build = "headless*"`**
- **Found during:** Task 1, corroborated by a coordinator correction mid-task
- **Issue:** The plan's `opencv = ">=4.11"` is unbounded. conda-forge publishes `opencv 5.0.0`
  with no `py3XX` build tag on any subdir — no Python bindings. An unbounded pin can resolve
  to a green `pixi install` followed by `ModuleNotFoundError: cv2`.
- **Fix:** `<5` upper bound (mandatory) plus the `headless` variant to keep Qt6 out of a
  service that never calls `cv2.imshow`. The `headless*` glob solved correctly; no fallback
  needed.
- **Worth recording:** the "no `py3XX` ⇒ no bindings" heuristic is **not reliable at 4.13.0**.
  The resolved `opencv 4.13.0 headless_h5b059e6_14` also lacks `py312`, yet bindings are
  present because a separate `py-opencv` package supplies them. The empirical `import cv2`
  check, not the build string, is the real test — which is why `tests/test_environment.py`
  now exists.
- **Commit:** 0a627a7

**2. [Rule 2 - Security] hydra-core `>=1.3.4,<1.4` instead of `>=1.3`**
- 1.3.4 is a security release (blocklist for security-sensitive `_target_` in
  `instantiate()`); conda-forge is stuck at 1.3.2, so PyPI is correct. **Commit:** 0a627a7

**3. [Rule 3 - Blocking] `prefix-dev/setup-pixi@v0.10.0` instead of the plan's `v0.8.x`**
- v0.8.x is two minor lines stale. Version existence was verified against the GitHub API
  rather than assumed. `pixi-version` pinned to `v0.62.2` to match the lock. **Commit:** 705c236

**4. [Rule 1 - Correctness] `ruff-check` hook id, not `ruff`**
- The bare `ruff` id is deprecated as of ruff 0.16. **Commit:** 86bb718

**5. [Rule 2] Added `tests/test_environment.py` (not in the plan)**
- A successful solve is not evidence the bindings work, and the opencv-5 trap makes that gap
  concrete. Asserts `import cv2` on the 4.x line, the presence of `matchTemplate`/`SIFT_create`/
  `watershed` (contrib included), the onnxruntime pin, numpy 2.x, and a real `cv2.cvtColor`
  call to catch numpy-2 ABI breakage. **Commit:** 3a18f54

**6. [Rule 1] `actions/upload-artifact` v4 → v7**
- The first green run flagged `Node.js 20 is deprecated`. Fixed while green. **Commit:** 921ca39

**7. [Minor] `numpy = ">=2.0,<3"`**
- The `<3` guard matches what conda-forge `libopencv` and `onnxruntime` declare themselves
  (`numpy >=1.23,<3`). **Commit:** 0a627a7

**8. [Process] CI is a single `ubuntu-latest` job, not an OS matrix**
- The research suggested a macOS leg, but the required status check must be named exactly
  `quality` and a matrix produces per-leg names like `quality (ubuntu-latest)`. **Recorded
  gap: a macOS-only solver failure would not be caught by CI.**

**9. [Process] PR opened before CI was confirmed green**
- Unavoidable ordering, not a shortcut: the `push` trigger is scoped to `main`, so a branch
  push fires nothing and no `pull_request` event can exist until the PR does. `gh run list`
  was empty until the PR was created.

**10. [Ordering] Task 1's `pixi install` needed Task 2 and 3 files to exist**
- `object-search = { path = ".", editable = true }` cannot build without `pyproject.toml` and
  the package skeleton. Files were created in dependency order and then committed in plan-task
  order, so the intermediate commits 0a627a7 and 20fe8c0 are not independently installable.

## Known Blocker: branch protection not applied (Task 6)

**Task 6 is the one incomplete task.** Both APIs refuse on plan grounds:

```text
$ gh api -X PUT repos/ortizeg/object-search-exploration/branches/main/protection --input protection.json
gh: Upgrade to GitHub Pro or make this repository public to enable this feature. (HTTP 403)

$ gh api -X POST repos/ortizeg/object-search-exploration/rulesets --input ruleset.json
gh: Upgrade to GitHub Pro or make this repository public to enable this feature. (HTTP 403)

$ gh api repos/ortizeg/object-search-exploration/branches/main --jq '{protected}'
{"protected":false}
```

Branch protection and rulesets are paid features for **private** repositories. The repo was
**not** made public as a workaround: repository visibility is the user's decision, not a
mechanical implementation detail (deviation Rule 4).

The negative test (`git push origin main` must be rejected) was **deliberately not run**: on
an unprotected `main` it would have succeeded and pushed real commits to `main`. That test is
only safe once protection exists.

**Resolution options for the user:** upgrade to GitHub Pro, make the repo public, or accept
convention-only enforcement for now (weakest — gates would be green but not *enforced*,
exactly the advisory-gate failure this phase exists to prevent).

**Ready-to-apply config once unblocked:**

```json
{
  "required_status_checks": { "strict": true, "contexts": ["quality"] },
  "enforce_admins": false,
  "required_pull_request_reviews": {
    "dismiss_stale_reviews": true,
    "require_code_owner_reviews": false,
    "required_approving_review_count": 0
  },
  "restrictions": null,
  "required_conversation_resolution": true,
  "allow_force_pushes": false,
  "allow_deletions": false
}
```

Then verify: read the protection back and confirm `contexts` contains `quality`, and confirm
a direct `git push origin main` is rejected.

## Requirements status

| ID | Status | Note |
| --- | --- | --- |
| INFRA-01 | Complete | pixi env, python 3.12.13, all commands are pixi tasks |
| INFRA-02 | Complete | src-layout + `py.typed`, pyproject single source of truth |
| INFRA-03 | Complete | ruff (100) and mypy strict both clean |
| INFRA-04 | Complete | hooks installed, `pre-commit run --all-files` clean |
| INFRA-05 | Complete | `print()` → T201 and `logging` → TID251, both proven to fail |
| INFRA-06 | Complete | 80% floor proven to fail at 41.67% |
| INFRA-07 | **Partial** | CI green; **branch protection blocked by GitHub plan** |

## Self-Check: PASSED

All 17 claimed artifacts verified present on disk, all 8 claimed commit hashes verified in
`git log`, and all 11 machine-checkable acceptance criteria verified by grep (`[workspace]`
present and no `[project]`; `onnxruntime = "==1.23.2"`; no `opencv-python-headless` anywhere;
`line-length = 100`; `T20` selected; `--cov-fail-under=80`; `pass_filenames: false`;
`pixi.lock` **not** gitignored; `models/` and `*.onnx` gitignored; README placeholder gone;
`.git/hooks/pre-commit` installed).

One acceptance criterion is **not** met and is not claimed to be: "a direct
`git push origin main` is rejected" — see the branch-protection blocker above.

## Notes for the next plan

- `main` is **not** protected. Do not assume merges are gated until the user resolves the
  plan question.
- Two follow-ups are recorded as `TODO(phase-1-followup)` in `pixi.toml`: evaluating
  conda-forge `onnxruntime >=1.26,<1.27` (sidesteps macOS wheel tags entirely and enables the
  CoreML EP; note 1.23.x was never on conda-forge, so it is a version bump, not a channel
  swap). CoreML EP is available in the current PyPI build already
  (`['CoreMLExecutionProvider', 'AzureExecutionProvider', 'CPUExecutionProvider']`).
- If a macOS CI leg is added later, `required_status_checks.contexts` must be updated to
  match both matrix leg names, or a macOS-only failure can merge.
