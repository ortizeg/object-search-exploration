---
quick_id: 260730-w9s
status: complete
---

# Summary — improve `mosse` on the floor-plan door/window domain

## Verdict

**Root cause confirmed for both classes — a stronger, cleaner result than the sibling `ncc`
investigation.** `MOSSEConfig.train_angles_deg`'s shipped ±35° bank (folded into `n_angle_groups`
sub-filters) misses instances on perpendicular walls; a cardinal-only bank (0/90/180/270°) with
`n_angle_groups` scaled to match (4 groups, one sharp sub-filter per cardinal) wins outright for
BOTH doors and windows — unlike `ncc`, where windows showed a mixed/regressed result. A separate,
default-off `mirror` field (verify-side-only horizontally-flipped re-score) is a clear, decisive
win for doors and mildly net-negative for windows, the same direction `ncc` found but much
stronger on doors.

- **floorplans-door**: test F1 0.201 (default) → 0.213 (pre-existing tuned) → **0.408** (full
  extended grid, argmax on val: cardinal + mirror=True + retain=0.65). `mosse`'s tuned door F1
  beats `ncc`'s (0.358) outright.
- **floorplans-window**: test F1 0.077 (default) → 0.148 (pre-existing tuned) → **0.155** (full
  extended grid: cardinal + mirror=False + retain=0.65). A real, disclosed win with NO
  generalization gap (unlike `ncc`'s window regression against its own pre-existing grid) — though
  `mosse` remains well behind `ncc`'s tuned window number in absolute terms.
- **A disclosed generalization-gap nuance for doors**: a narrower manual sweep found an even
  better door result (test F1 0.509 at `retain_frac=0.55`), but the FULL grid's honest argmax-on-val
  selects `retain_frac=0.65` instead (a legitimately higher val F1, but a more conservative frozen
  config that generalizes to a lower test F1 of 0.408). Per the tuning protocol's discipline, the
  full-grid result is what ships — not the better-looking narrower-sweep number. Both are recorded
  in EXPERIMENTS.md.

## What shipped

- `src/object_search/eval/tuning.py`: additive `grids=` override on `run_domain_tuning`
  (independently re-implemented on this branch; backward-compatible, defaults to `_TUNING_GRIDS`);
  the previously-aliased `ncc`/`mosse` grid objects split into independent tuples;
  `_TUNING_GRIDS["mosse"]` extended with an additive cardinal-bank (matched `n_angle_groups`) ×
  mirror block (grid-only change — no `MOSSEConfig` default touched, so the synthetic regime cannot
  regress by construction).
- `src/object_search/search/mosse.py`: new default-off `mirror: bool` field on `MOSSEConfig`;
  `_rotated_template_bank` (used by the coarse-to-fine `_verify_score`) extended to yield a
  `cv2.flip`-mirrored template/mask sibling per angle when set. The filter-training side
  (`_build_filter_bank`) was NOT touched — the verify-side-only flip already fully recovered the
  expected gain.
- `scripts/mosse_floorplan_experiment.py`: research/lab-bench experiment runner, mirroring the
  sibling `ncc` investigation's tool.
- `docs/methods/mosse.md`, `docs/reports/mosse-improvement.md`: docs updated; a new dated
  "Floor-plan domain follow-up (2026-07-30)" section carries the full investigation, numbers,
  the angles-per-group trap demonstration, and the doors generalization-gap nuance.
- `.planning/quick/260730-w9s-.../EXPERIMENTS.md`: the append-only lab notebook backing every
  number in the report (E0 baselines, E1 orientation sweep incl. the trap control, E2 mirror
  sweep, E3 final full-grid result).
- `.gitignore`: quick-task `runs/`, `logs/`, `debug/` output directories ignored (same pattern as
  the sibling `ncc` PR, re-added independently since this branch forked before that commit).

## Verification

`$HOME/.pixi/bin/pixi run lint` / `typecheck` clean. Full suite: 707 passed / 20 skipped, 92.34%
coverage (floor held). Synthetic regression guard: `models/` is empty in this worktree, so `mosse`
(needing no ONNX weights) was checked via a full `pixi run bench "methods=[mosse]"` re-run rather
than `bench-ci` (which excludes `mosse`) — reproduced the pre-change numbers exactly byte-for-byte
(overall F1 0.7989, fixed 0.9283, varied 0.5963), including the VARIED/CLUTTERED win over `ncc`
this report's v1 established. Zero regression.

## Not done here

- The filter-training-side mirror extension (`_build_filter_bank`) was not needed and not
  attempted — the verify-side-only flip already recovered the full measured gain.
- `mosse`'s window result, while genuinely improved, remains behind `ncc`'s tuned window number —
  `ncc` stays the stronger window method on this domain regardless of this change.
- No `gh pr create` — the orchestrator opens the PR after this and the sibling `ncc` task both land.
