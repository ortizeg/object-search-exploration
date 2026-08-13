---
quick_id: 260811-p0l
status: complete
---

# Summary — SuperPoint backend spike on `sparse-geo` real-objects

## Verdict

**PARTIALLY DIVERGES from the floor-plans-500 verdict.** `docs/reports/sparse-geo-improvement.md`
found `backend="superpoint"` DISPROVEN in 4/4 cells on floor-plans-500 (F1 down 0.024–0.096, AP50
down in 3/4 cells, window coverage 28/28 → 26/28 on an ONNX/CoreML crash, 5.3–6.9× latency). This
spike ran the same backend swap over the 30-image `real-objects` set (real photographic texture)
under five conditions (SIFT baseline, two SIFT voting-mode controls, SuperPoint × two
frameless-compatible voting modes), through the project's own `benchmark._run_one` scoring path.

- **Against the shipped `sift/single-4dof` baseline**, SuperPoint still loses F1 in both voting
  modes (−0.066, −0.069) — same direction as floor-plans, smaller magnitude, and **without** an AP
  or coverage collapse (coverage stays 30/30 in every condition; AP down only 0.009–0.055 vs. down
  in 3/4 floor-plans cells).
- **Against the fairer same-voting-mode SIFT controls** (which the floor-plans report did not
  run), SuperPoint is roughly at parity on F1 (+0.013, −0.014, both within noise) and **ahead on
  AP in both modes** (+0.041, +0.090).
- **Zero errors/abstentions** across all 5 conditions × 30 images. The zero-keypoint crash's
  precondition (SuperPoint returning near-zero keypoints) never occurred — minimum observed crop
  keypoints was 17 (SIFT) / 24 (SuperPoint), vs. 0–33 on the floor-plans probe.

Causal hypothesis: the floor-plans result likely conflated two effects — a real
detector-quality gap specific to sparse line-art (which vanishes on real photo texture: crop
keypoint medians land within one keypoint of each other here, 144 vs 145) and an unmeasured
voting-mode confound (this spike's SIFT controls show switching off `single-4dof` alone costs
0.055–0.079 F1, independent of backend — floor-plans never isolated that). Stated as hypothesis,
not re-measured on floor-plans itself; carried into deferred work.

## Where it ran

At the user's request, the compute-heavy steps (fetch SuperPoint weight, smoke test, full sweep)
ran on a rented vast.ai CPU box rather than locally, to keep the spike off the user's machine.
Code was shipped via `git archive HEAD | gzip` over `scp` (no `git push` — this worktree's branch
isn't pushed to `origin`). The box installed the default pixi env, ran
`pixi run fetch-models --only superpoint` (sha256-gated), ran all 5 conditions under the default
CPU execution provider, and results were pulled back over `scp`. The instance was destroyed
immediately after.

**Process note (cost honesty):** a first attempt hung on a client-side polling bug (`vastai show
instance <id> --raw` singular didn't parse as expected) and left two instances running unattended
for ~20 minutes (~$0.03 combined) before the orchestrator found and destroyed them. The successful
run used a fresh instance and was destroyed within minutes of the sweep finishing. Full detail in
the report's "A process note on this run" section.

## What shipped

- `scripts/sparse_geo_real_objects_experiment.py` (commit `7fc33ae`) — the driver: builds a
  `MethodSpec` variant via `dataclasses.replace` + `functools.partial(SparseGeoConfig, ...)`,
  monkeypatches `benchmark.get_method` around an unmodified `_run_one` call, plus `smoke` and
  `sweep` subcommands.
- `docs/reports/sparse-geo-real-objects-superpoint-spike.md` (commit `141101f`) — the engineering
  log: question, conditions table, baseline reconciliation, pooled + per-regime results, keypoint
  counts vs. floor-plans, the crash question answered, verdict, cost, deferred work.
- `mkdocs.yml` — one nav entry under "Improvement reports", after the existing Sparse-geo line.
- **No source or config change.** `git diff --stat -- src/ conf/` is empty; `sparse-geo`'s shipped
  `backend` default stays `sift`.
- Raw run artifacts (gitignored, not committed): `runs/summary.json`, `runs/sweep.log`, and 5
  per-condition warning logs (all empty — zero warnings fired), under this quick task's own
  `runs/` directory.

## Verification

`pixi run quality`: Ruff clean, Ruff-format clean, MyPy strict clean (81 source files), **932
passed / 20 skipped, coverage 92.92%** (floor 80%). `pixi run docs-build`
(`mkdocs build --strict`) green with the report in nav. `git status --porcelain src/ conf/` and
`git diff --stat -- src/ conf/` both empty.

## For the PR (if opened)

**Suggested title:** `docs(sparse-geo): real-objects SuperPoint spike — partially diverges from
the floor-plans-500 verdict`

**Requirement IDs:** METHOD-04, METHOD-04a, EVAL-24, DOC-04.

**Suggested body points:**

- No source change. This ships an engineering-log spike testing whether the floor-plans-500
  SuperPoint verdict generalizes to a second domain (real photographic texture) via the project's
  own scoring path, with same-voting-mode SIFT controls the original report didn't run.
- Headline: SuperPoint is roughly at parity with SIFT (and ahead on AP) once the voting-mode
  confound is controlled for; still behind the shipped `single-4dof` baseline because
  `single-4dof` itself is the strongest voting mode, independent of backend.
- Zero errors/abstentions across 150 (5 conditions × 30 images) scored runs — the floor-plans
  zero-keypoint crash's precondition doesn't arise on real photos.
- `backend` stays `sift`; SuperPoint remains non-default (also independently blocked by its
  MagicLeap non-commercial licence, unrelated to how it scores).
- Reviewer note: the SIFT voting-mode controls are the most reusable finding — they suggest part
  of the floor-plans-500 investigation's SuperPoint loss may be attributable to the voting-mode
  switch it's forced into rather than to SuperPoint's keypoints, a hypothesis this spike states
  explicitly as unconfirmed on floor-plans itself and carries into deferred work.
