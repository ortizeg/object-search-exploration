---
phase: quick-260810-gx5
plan: 01
subsystem: docs
tags: [mkdocs, navigation, documentation]
status: complete
requires: []
provides:
  - "mkdocs nav entries for the two floor-plan improvement reports"
affects:
  - mkdocs.yml
tech-stack:
  added: []
  patterns:
    - "mkdocs nav values are docs_dir-relative; strict mode fails hard on a nav path that does not resolve"
key-files:
  created: []
  modified:
    - mkdocs.yml
decisions:
  - "Wired only the two orphan pages that actually exist; did not add a nav entry for the plan's third target (benchmark/floorplans-results.md) because that file is not in the repo and a nav entry pointing at a missing file fails the strict build."
  - "Placed 'DINO dense (floor-plans)' directly after 'DINO dense' and 'OWLv2 (floor-plans)' between 'OWLv2' and 'OWLv2 floor-plan fine-tune', matching the section's existing parenthetical-qualifier label convention."
metrics:
  duration: "~8 min"
  completed: 2026-08-10
  tasks: 2
  files: 1
---

# Quick 260810-gx5: Wire orphan mkdocs pages into nav — Summary

Two orphaned floor-plan improvement reports are now reachable from the site menu, and the
strict mkdocs build's orphan-page listing is empty; the plan's third target page does not
exist in the repo and was correctly skipped rather than wired to a dead path.

## What Was Done

### Task 1 (tracer): Add orphan pages to the mkdocs nav

Edited the `Improvement reports:` section of `mkdocs.yml` — two insertions, no deletions,
no reordering of pre-existing entries:

| Label | Target |
|-------|--------|
| `"DINO dense (floor-plans)"` | `reports/dino-dense-floorplans-improvement.md` |
| `"OWLv2 (floor-plans)"` | `reports/owlv2-floorplans-improvement.md` |

Both labels use the parenthetical-qualifier form already established in that section by
`OWLv2 (real-objects)`, and both are quoted because they contain parentheses. Values are
`docs_dir`-relative (`reports/...`, not `docs/reports/...`).

Ordering: the DINO entry sits immediately after its non-floor-plan sibling; the OWLv2 entry
sits between `OWLv2` and `OWLv2 floor-plan fine-tune`, keeping the OWLv2 floor-plan-adjacent
entries contiguous and in chronological order (improvement log before fine-tune experiment).

Verification (adapted to the two extant pages): parsed the YAML and asserted both paths appear
in the `nav` structure — passed.

### Task 2: Strict build proof

`pixi run docs-build` (invoked as `$HOME/.pixi/bin/pixi run docs-build`, since `pixi` is a shell
function not on `PATH` for non-interactive shells) exits **0** with `strict: true` unchanged.

The `The following pages exist in the docs directory, but are not included in the "nav"
configuration:` block is now **entirely absent** from the build output — before the change it
listed exactly two pages, both now wired.

Out-of-scope INFO diagnostics are byte-for-byte unchanged (7 lines, same as the pre-change
baseline): six `unrecognized relative link` notices pointing at `samples/*/` directories, and
one missing `#scope--exclusions` anchor in `eval/research-datasets.md`. These are owned by a
separate follow-up task and were deliberately left alone.

## Deviations from Plan

### 1. [Rule 3 - Blocking issue] The plan's third target page does not exist

- **Found during:** Task 1, precondition/reality check before editing
- **Issue:** The plan is titled "fix 3 orphan mkdocs pages" and instructs adding a third nav
  entry under `Evaluation:` pointing at `benchmark/floorplans-results.md`, labelled
  `"Floor-plan results"`. That file does not exist — not in the working tree, not at the plan's
  own base commit `e7600d7`, and `grep -rn floorplans-results docs mkdocs.yml` returns nothing.
  `docs/benchmark/` contains only `results.md` plus four PNGs.
- **Confirmed independently:** the pre-change `docs-build` orphan listing named exactly **two**
  pages, not three.
- **Why it blocks:** mkdocs `strict: true` fails hard on a nav entry whose path does not
  resolve. Following the plan literally would have broken the build — the very thing Task 2
  exists to prevent. Inventing the page was also excluded: the plan states "this task creates
  no Markdown."
- **Fix:** Wired the two real orphans; omitted the third entry. Documented in the commit body.
- **Files modified:** `mkdocs.yml`
- **Commit:** `0863bab`

## Known Gaps

The plan's must-have truth *"A reader browsing the rendered docs site can reach the floor-plan
benchmark results page from the Evaluation nav section"* is **not satisfied and cannot be** —
no such page exists. `docs/eval/floorplans-findings.md` (already in the `Evaluation:` nav
section) is the closest existing page and covers floor-plan findings, but it is a findings
narrative, not a results table. If a dedicated floor-plan benchmark results page is wanted,
it needs to be authored first as its own task; wiring nav is then a one-line follow-up.

All other must-have truths are satisfied.

## Verification Results

| Criterion | Result |
|-----------|--------|
| `docs-build` exits 0 under unchanged `strict: true` | PASS |
| Orphan-page listing names none of the wired pages | PASS (listing absent entirely) |
| Diff is `mkdocs.yml` only, additions only, no reordering | PASS (`1 file changed, 2 insertions(+)`) |
| Labels match section's parenthetical-qualifier convention | PASS |
| Out-of-scope INFO diagnostics untouched | PASS (identical 7 lines) |
| One commit with required `Co-Authored-By:` trailer | PASS |

## Commits

| Commit | Message |
|--------|---------|
| `0863bab` | `docs(quick-gx5): wire orphan floor-plan improvement reports into mkdocs nav` |

## Self-Check: PASSED

- `mkdocs.yml` exists and contains both new nav entries — verified by YAML parse.
- Commit `0863bab` exists in `git log`; `git diff --name-only HEAD~1 HEAD` returns `mkdocs.yml`
  alone; `git diff --diff-filter=D HEAD~1 HEAD` returns empty (no deletions).
