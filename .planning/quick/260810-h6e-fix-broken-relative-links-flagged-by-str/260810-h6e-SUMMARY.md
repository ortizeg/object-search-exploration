---
phase: quick-260810-h6e
plan: 01
subsystem: docs
tags: [docs, mkdocs, links, build-hygiene]
status: complete
requires: []
provides:
  - "Clean `mkdocs build --strict` output with zero broken-link / bad-anchor lines"
affects:
  - docs/MILESTONE-2.md
  - docs/eval/research-datasets.md
  - docs/explorations/marker-conditioned.md
  - docs/methods/dino-dense.md
  - docs/methods/mosse.md
  - docs/methods/ncc.md
  - docs/methods/propose-retrieve.md
tech-stack:
  added: []
  patterns:
    - "Directory links in mkdocs must target the built `index.md` page, not the directory."
    - "mkdocs-material slugifies `## Scope — exclusions` to `scope-exclusions` — the em-dash collapses to a SINGLE hyphen, not two."
key-files:
  created: []
  modified:
    - docs/MILESTONE-2.md
    - docs/eval/research-datasets.md
    - docs/explorations/marker-conditioned.md
    - docs/methods/dino-dense.md
    - docs/methods/mosse.md
    - docs/methods/ncc.md
    - docs/methods/propose-retrieve.md
decisions:
  - "Left the two orphan-page INFO lines (`docs/reports/*-floorplans-improvement.md`) alone — pre-existing, tracked, out of scope, and explicitly not to be silenced via mkdocs.yml."
metrics:
  duration: ~6 min
  completed: 2026-08-10
  tasks: 3
  files_changed: 7
  commits: 1
---

# Quick Task 260810-h6e: Fix Broken Relative Links Flagged by Strict MkDocs Build

Repointed six `docs/samples/<name>/` directory links at the `index.md` page mkdocs actually
builds, and corrected one in-page anchor from `#scope--exclusions` to `#scope-exclusions`,
clearing all 7 broken-link INFO lines from `mkdocs build --strict`.

## What Was Built

Seven single-link edits across seven docs pages — one changed line per file, no prose,
link text, image link, or adjacent-correct-link touched.

### Task 1 — Six samples-directory links repointed at `index.md`

mkdocs does not build a bare directory as a page, so `](../samples/ncc/)` was an
unrecognized relative link that would 404 in the published site. Each target gained an
`index.md` suffix; all six `docs/samples/<name>/index.md` targets were confirmed present on
disk before editing.

| File | Link target before | after |
|---|---|---|
| `docs/MILESTONE-2.md` | `samples/marker-conditioned/` | `samples/marker-conditioned/index.md` |
| `docs/explorations/marker-conditioned.md` | `../samples/marker-conditioned/` | `../samples/marker-conditioned/index.md` |
| `docs/methods/dino-dense.md` | `../samples/dino-dense/` | `../samples/dino-dense/index.md` |
| `docs/methods/mosse.md` | `../samples/mosse/` | `../samples/mosse/index.md` |
| `docs/methods/ncc.md` | `../samples/ncc/` | `../samples/ncc/index.md` |
| `docs/methods/propose-retrieve.md` | `../samples/propose-retrieve/` | `../samples/propose-retrieve/index.md` |

`docs/MILESTONE-2.md` sits at the docs root, so its path correctly carries no `../` prefix.

### Task 2 — `research-datasets` Scope anchor

`docs/eval/research-datasets.md:15` linked to `#scope--exclusions` (double hyphen). The
heading at line 237 is `## Scope — exclusions`, and mkdocs-material's slugifier collapses the
em-dash to a **single** hyphen, producing `id="scope-exclusions"`. Fixed the fragment only;
the visible `Scope` link text and surrounding sentence are unchanged.

### Task 3 — Strict build proof and commit

`pixi run docs-build` (`mkdocs build --strict`) exits **0**, and the output contains **zero**
`unrecognized relative link` and zero `no such anchor` lines — a stronger result than the
required "none naming the 7 fixed targets".

## Key Decisions

**Orphan-page INFO lines left untouched.** The build still emits one INFO listing two pages
absent from `nav`: `docs/reports/dino-dense-floorplans-improvement.md` and
`docs/reports/owlv2-floorplans-improvement.md`. Both are tracked, committed, pre-existing
files, and neither appears in `mkdocs.yml`. Link-target edits cannot create orphan-page
lines, so this is unrelated pre-existing state — left alone per the plan's out-of-scope
instruction, and `mkdocs.yml` was not edited to silence it.

## Deviations from Plan

### Observation (no change made) — predicted orphan page differs from actual

- **Found during:** Task 3
- **Expected:** plan anticipated a single remaining orphan INFO naming
  `docs/benchmark/floorplans-results.md`, described as a gitignored local artifact.
- **Actual:** that file does not exist in this worktree; the build instead named two tracked
  `docs/reports/*-floorplans-improvement.md` pages.
- **Assessment:** same category (page not in `nav`), same disposition (out of scope). The
  discrepancy is an environment difference between the planning checkout and this worktree,
  not a regression — verified via `git ls-files` (both tracked) and `grep mkdocs.yml`
  (neither in nav). **No NEW warning was introduced by this change.**
- **Action taken:** none. No files modified.

No Rule 1-3 auto-fixes were required. The plan executed as written.

## Verification

| Check | Result |
|---|---|
| `mkdocs build --strict` exit code | **0** |
| `unrecognized relative link` lines naming the 6 samples dirs | **0** (none in output at all) |
| `no such anchor` lines naming `scope--exclusions` | **0** (none in output at all) |
| Bare `](samples/<name>/)` links remaining in `docs/` | **0** (grep exit 1) |
| `git diff --numstat` per touched file | **1 added / 1 removed** for all 7 |
| Files in HEAD commit | **7**, all under `docs/` |
| Deletions in commit | **0** |
| `mkdocs.yml` / `docs/benchmark/` modified | **no** (empty `git diff --stat`) |
| Pre-commit hooks | all Passed / Skipped, no file rewrites |

Worktree HEAD was asserted on the per-agent branch `worktree-agent-a135e2c56420412bf`
(not detached, not a protected ref) before staging, and the worktree's
`git rev-parse --git-common-dir` resolves into `object-search-exploration` — not the stray
parent repo.

## Commits

| Hash | Message |
|---|---|
| `f2b2b03` | `docs: fix 7 broken internal links flagged by strict mkdocs build` |

## Known Stubs

None.

## Self-Check: PASSED

- All 7 modified files present on disk and staged into the commit.
- Commit `f2b2b03` confirmed present via `git log --oneline --all`.
- Working tree clean apart from the untracked `.planning/quick/260810-h6e-*/` directory
  (planning artifacts, intentionally not committed here).
