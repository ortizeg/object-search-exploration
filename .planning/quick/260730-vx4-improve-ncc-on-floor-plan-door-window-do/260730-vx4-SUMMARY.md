---
quick_id: 260730-vx4
status: complete
---

# Summary — improve `ncc` on the floor-plan door/window domain

## Verdict

**Root cause confirmed for doors, mixed for windows.** `NCCConfig.angles_deg`'s shipped ±35° bank
misses instances on perpendicular walls; a cardinal-only bank (0/90/180/270°) — not a wider
continuous sweep — is the fix, because floor-plan walls are discretely orthogonal. A separate,
default-off `mirror` field (horizontally-flipped template siblings) was a statistical tie for
doors and net-negative for windows.

- **floorplans-door**: test F1 0.164 (default) → 0.248 (pre-existing tuned) → **0.358** (cardinal +
  mirror=True, retain=0.65). Reproduced consistently across three independent measurements.
- **floorplans-window**: test F1 0.222 (default) → 0.401 (pre-existing tuned) → 0.350 (cardinal +
  mirror=False). A genuine, honestly-disclosed val/test generalization gap versus the pre-existing
  grid entry — not reverted, since the tuning protocol forbids conditioning grid choices on test
  outcomes.
- Two further recall levers (lower `retain_frac`, wider scale pyramid) were tested directly and
  both net F1-negative: true-positive and false-positive scores overlap in the same 0.5-0.65 band
  on this domain (unlike synthetic data, where genuine instances cluster near the 1.0 self-match),
  so no threshold or search-bank change can separate them. This is `ncc`'s real ceiling here.

## What shipped

- `src/object_search/eval/tuning.py`: additive `grids=` override on `run_domain_tuning` (backward-
  compatible, defaults to `_TUNING_GRIDS`); the previously-aliased `ncc`/`mosse` grid objects split
  into independent tuples; `_TUNING_GRIDS["ncc"]` extended with an additive cardinal-bank × mirror
  block (grid-only change — no `NCCConfig` default touched, so the synthetic regime cannot regress
  by construction).
- `src/object_search/search/ncc.py`: new default-off `mirror: bool` field on `NCCConfig`;
  `_rotated_bank` extended to yield a `cv2.flip`-mirrored template/mask sibling per angle when set.
- `scripts/ncc_floorplan_experiment.py`, `scripts/ncc_debug_visualize.py`: research/debug tooling
  (lab-bench experiment runner + a flag-driven per-image debug visualizer showing matches vs.
  ground truth, sub-threshold candidates, and the similarity heatmap).
- `docs/methods/ncc.md`, `docs/reports/ncc-improvement.md`: docs updated; a new dated "Floor-plan
  domain follow-up (2026-07-30)" section carries the full investigation, numbers, and dead ends.
- `.planning/quick/260730-vx4-.../EXPERIMENTS.md`: the append-only lab notebook backing every
  number in the report.
- `.gitignore`: quick-task `runs/`, `logs/`, and `debug/` output directories ignored (environment-
  dependent numbers and per-image renders of the licensed floor-plan images; the committed
  deliverable is the aggregate report + EXPERIMENTS.md).

## Verification

`$HOME/.pixi/bin/pixi run lint` / `typecheck` clean. Full suite: 705 passed / 20 skipped, 92.34%
coverage (floor held). Synthetic regression guard: `bench-ci` unchanged (`ncc` F1 1.000); a full
`pixi run bench "methods=[ncc]"` re-run reproduced the committed EASY/TEXTURED/VARIED/CLUTTERED
numbers exactly (overall F1 0.7878, fixed 0.9503, varied 0.5063) — zero regression.

## Not done here

- The floorplans-window generalization gap is disclosed, not resolved; a future pass could split
  the tuning grid per-dataset if it matters enough to chase.
- A more discriminative scoring mechanism (to close the true/false score overlap found in the
  recall investigation) is out of scope for `ncc` — see the parallel `mosse` investigation, whose
  whitened correlation filter is built to suppress background rather than just correlate raw
  pixels.
- No `gh pr create` — the orchestrator opens the PR after this and the sibling `mosse` task land.
