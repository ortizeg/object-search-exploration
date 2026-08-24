---
quick_id: 260812-mm3
status: complete
---

# Summary — floor-plans voting-mode-confound follow-up

## Verdict

**PARTIALLY — and the picture is split by class, which is itself the finding.**
`docs/reports/sparse-geo-improvement.md` found `backend="superpoint"` DISPROVEN in 4/4 floor-plans
cells, but every delta there compared `sift/single-4dof` against
`superpoint/{translation-2dof,pairwise-4dof}` — two variables changed at once, since
`SparseGeoConfig` refuses `single-4dof` for a frameless backend. This follow-up added the missing
same-voting-mode SIFT controls, at both grids the published table used (committed grid + the
SuperPoint-matched grid), on floor-plans-500 itself.

- **Door — effectively CONFIRMED.** Voting mode alone explains 75% of the published
  `translation-2dof` loss and, at `pairwise-4dof`, voting mode alone would have been a *gain* for
  SIFT (+0.022 F1) — so the true backend-specific residual (−0.004 to −0.045 F1 once isolated) is
  small or the confound framing is simply the right one. SuperPoint is actually **ahead** on AP50 at
  matched voting mode in both modes (+0.017, +0.026), echoing the real-objects spike's AP finding.
- **Window — REFUTED as the primary explanation.** Voting mode explains the `translation-2dof` gap
  almost entirely (103% — residual ≈0) but only ~52% at `pairwise-4dof` (residual −0.039 F1). AP50
  stays clearly worse for SuperPoint at matched voting mode in **both** modes (−0.08 to −0.13),
  independent of F1, and the published coverage collapse (28/28 → 26/28, ONNX/CoreML crash on
  near-zero keypoints) has no voting-mode analogue at all — that failure mode is real and
  backend-specific.

## Where it ran

At the user's explicit direction ("run on vast.ai and go for it"), the 12-run tuning sweep ran on a
rented vast.ai CPU box (offer `45169473`, machine `52447`, $0.048/hr, 40 vCPUs — classical SIFT/
RANSAC/Hough voting, no GPU benefit). Code shipped via `git archive HEAD | gzip`; the manually-sourced
floor-plans-500 dataset (symlinked into this worktree from the primary checkout, no auto-fetch
exists) shipped as a separately dereferenced tarball (`tar -czh`) since `git archive` only carries
git-tracked files. Wall-clock: 6850s (~1h54m) — `pairwise-4dof`'s O(n²) cost dominates, as expected.
Total cost ≈$0.09. Instance destroyed immediately after results were pulled back; confirmed via
`vastai show instances --raw` before finishing.

**Two real bugs found and fixed along the way**, both outside the numbers this report is built on:
1. `.gitignore`'s `/datasets/` pattern (trailing slash) does not match a symlink, only a real
   directory (documented git behavior) — added `/datasets` alongside it (commit `1f76ff3`), or the
   symlinked dataset would have shown as untracked and risked being accidentally committed.
2. The driver's per-cell `elapsed_s` was evaluated as a call argument before the timed experiment
   actually ran, so every cell recorded ~0. Fixed in commit `f76b148`; real per-cell timings for the
   report were sourced from `sweep.log`'s own log lines rather than re-running the (already
   expensive) sweep to regenerate a cosmetic field.
3. A dereferenced (`tar -czh`) macOS tarball drags along AppleDouble `._*` resource-fork sidecar
   files, which were cleaned up on the remote box before the pixi env or the sweep ever ran (they
   would have polluted any glob-based dataset file listing had they survived).

## What shipped

- `scripts/sparse_geo_floorplan_voting_mode_experiment.py` (commits `39271eb`, `f76b148`) — the
  driver: two literal grid definitions (`committed`, `sp-matched`), three conditions, a `smoke` and
  a `sweep` subcommand, all through `object_search.eval.tuning.run_domain_tuning`'s unmodified
  `grids=` seam.
- `docs/reports/sparse-geo-floorplans-voting-mode-confound.md` (commit `aeb96d6`) — the engineering
  log: reconciliation, per-class comparison tables (both grids), the isolated voting-mode-cost table,
  the split verdict, cost, regression guard, deferred work.
- `mkdocs.yml` — one nav entry; cross-links added to both prior sparse-geo reports (the real-objects
  spike's deferred item marked resolved).
- `.gitignore` — the symlink-matching fix (commit `1f76ff3`), a genuine correctness fix unrelated to
  the spike's own scope but required to keep the symlinked dataset out of git.
- **No source or config change.** `git diff --stat -- src/ conf/` is empty; `backend` stays `sift`,
  `voting_mode` stays `single-4dof`.
- Raw run artifacts (gitignored, not committed): 12 per-cell reports + consolidated `summary.json` +
  full `sweep.log`, under this quick task's own `runs/` directory.

## Verification

`pixi run quality`: Ruff clean, Ruff-format clean, MyPy strict clean (81 source files), **932
passed / 20 skipped, coverage 92.92%** (floor 80%). `pixi run docs-build`
(`mkdocs build --strict`) green with the report in nav. `git status --porcelain src/ conf/` and
`git diff --stat -- src/ conf/` both empty. Baseline reconciliation: the two headline metrics both
prior reports cite (committed-grid `single-4dof` test F1) reproduce essentially exactly; the three
flagged val-F1/allow_mirror-control disagreements are all 0.0004–0.0040, an order of magnitude below
any effect this report measures.

## For the PR (if opened)

**Suggested title:** `docs(sparse-geo): floor-plans voting-mode-confound follow-up — PARTIALLY,
split by class`

**Requirement IDs:** METHOD-04, METHOD-04a, EVAL-24, DOC-04.

**Suggested body points:**

- No source change. Closes an open hypothesis stated in two already-committed reports by adding the
  same-voting-mode SIFT controls the original floor-plans investigation never ran.
- Headline: the published "SuperPoint is worse" verdict is largely a voting-mode artifact on doors
  (SuperPoint even wins AP50 at matched voting mode) but holds up on windows, where a real
  backend-specific F1/AP50 gap and the original coverage-collapse crash both survive the correction.
- Reviewer note: the most reusable artifact here is the isolated per-class voting-mode-cost table —
  `pairwise-4dof` costs AP50 on both classes regardless of backend, and on doors it's actually a net
  F1 *gain* for SIFT over the shipped `single-4dof` default, which is a genuinely new (and separately
  scoped, not acted on here) tuning question.
- Two infra fixes rode along: a `.gitignore` gap where a trailing-slash pattern doesn't match a
  symlinked directory, and a timing bug in the new driver (argument evaluated before the timed call
  ran) — both fixed and covered by this same diff, unrelated to the confound numbers themselves.
