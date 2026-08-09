---
quick_id: 260730-vx3
status: complete
---

# Summary — improve `sparse-geo` on the floor-plan door/window domain

## Verdict

**Both hypotheses disproven. Nothing shipped to `src/`. The recommended config is byte-for-byte the
one that was already shipped.** This is a fully negative result, measured rather than assumed, and
the engineering log is the deliverable.

The symptom under investigation: `sparse-geo` doors test F1 **0.219** with recall by symbol size
**flat at 0.14 / 0.13 / 0.14**, while windows on the same dataset climb 0.13 → 0.40 with size.
Flatness across size is the signature of a *structural* failure, not "too few pixels".

- **Hypothesis 1 — mirror acceptance: DISPROVEN at both voting modes, fully reverted**
  (commit `8ab99a2`). At the shipped `single-4dof`: door F1 −0.004, window −0.020, precision down
  on both. At `pairwise-4dof` measured against **its own no-mirror control**: door −0.008 / window
  −0.032 on test, −0.008 / −0.009 on val, with precision and AP50 down in every cell and ~2×
  latency. `allow_mirror` improved floor-plan test F1 in **zero of four** class × voting-mode
  cells. Per the plan and CLAUDE.md, a disproven idea is not parked behind an off-by-default flag,
  so the entire surface came out.
- **Hypothesis 2 — SuperPoint backend: DISPROVEN in 4/4 cells, never committed.** Both
  frameless-compatible voting modes swept. Best case door −0.024 / window −0.048 test F1; val F1
  down 0.051–0.087 in all four; AP50 down in 3/4; window coverage **28/28 → 26/28** on a hard
  ONNX/CoreML crash when SuperPoint detects zero keypoints; **5.3–6.9× latency**.
- **Combine (Step C): not applicable** — neither hypothesis survived, so there was nothing to
  combine.
- **The symptom remains OPEN.** Flat door recall-by-size is explained by neither mirror rejection
  nor keypoint-detector choice. What *is* established: the funnel collapses between correspondence
  and peak — **55 Hough peaks hypothesized for 157 ground-truth doors, from 2 664 correspondences**.
  No diagnosis is claimed that was not measured.

## Key diagnostic findings (the reusable part)

1. **The mirror gate was never where mirrored doors die.** Instrumenting 16 door val plans: the
   `det < 0` branch fires **2 times in 55 peaks**. Relaxing it alone could recover at most ~2
   detections — very nearly an inert control. The loss is at the **voting** stage: single-4dof
   computes only the orientation-preserving branch, so a mirrored instance's correspondences
   predict a wrong centre, scatter, and never form a peak at all.
2. **SIFT orientations are not mirror-consistent.** On a controlled scene, correspondences landing
   in a mirrored copy are geometrically correct only 17% of the time (vs 35% for an identical
   copy), and the reflected pose angle scatters ~80–280° instead of clustering at 180°. So
   `single-4dof` *structurally cannot* cluster a mirrored instance; `pairwise-4dof` can (IoU 0.98
   on the controlled scene) — which is why hypothesis 1 had to be tested there too, with a control.
3. **SuperPoint's premise is half-right and it does not matter.** It fires marginally on barren
   crops where SIFT finds nothing (0 → 1, 0 → 3) but finds **less than half** as much on the one
   texture-rich door crop (33 → 12); 4/5 plans abstain under *both* backends. It yields ~2.8× the
   *scene* keypoints, which is where the O(n²) latency comes from. Cost scales, benefit does not.
4. **The `min_inliers` grid-floor caveat is bounded, not waved away.** The SuperPoint argmax pinned
   to the grid floor (5) in all four conditions while the SIFT baseline picked 3. By the baseline's
   own measured gradient, extending to `min_inliers=3` is worth **+0.007–0.019 val F1** — an order
   of magnitude short of the **0.051–0.087** gap it would need to close. (Explicit extrapolation,
   labelled as such.)

## What shipped

Documentation only. **No source file changed** — `git diff --stat src/` is empty.

- `docs/reports/sparse-geo-improvement.md` (**new**): the engineering log — symptom → root-cause
  investigation → hypothesis 1 with its control → hypothesis 2 with its probe and sweep →
  "combine: not applicable" → final/recommended config (unchanged) → what was reverted with
  numbers → fairness note → cost → regression guard → a structural contrast with the `ncc`/`mosse`
  reports → deferred work and the open question.
- `docs/eval/floorplans-findings.md`: a caveat blockquote under the result tables, matching the
  existing `owlv2` block's style — states the measured outcome (both hypotheses tested, both
  negative, rows correctly unchanged, symptom open) rather than claiming an improvement. The
  six-method tables are untouched.
- `docs/ROBUSTNESS-BACKLOG.md`: an annotation under the `sparse-geo` section recording both
  investigations and their reverts, plus the context this work adds to two existing backlog items
  (DISK/ALIKED is now also a detector-quality-on-line-art question; post-hoc orientation assignment
  is worth less than it looked, since SIFT's own orientations are not mirror-consistent).
- `mkdocs.yml`: `- Sparse-geo: reports/sparse-geo-improvement.md` under `Improvement reports:`,
  after MOSSE and before DINO dense.
- `.gitignore`: `.planning/quick/*/measurements/` ignored, matching the existing `runs/` / `logs/` /
  `debug/` entries — those logs are multi-MB and carry per-image IDs from the licensed floor-plan
  dataset (T-vx3-02); the committed deliverables are the aggregate report and `RESULTS.md`.
- `.planning/quick/260730-vx3-.../RESULTS.md`: the measurement log backing every number.

**`docs/methods/sparse-geo.md` was deliberately NOT touched** — `SparseGeoConfig` gained no
surviving field and no shipped behaviour changed, so editing it would have documented a change that
did not happen.

## Regression guard

Two independent checks, both after all measurement was complete:

1. **Byte-identical source.** `git diff df64af1 -- src/object_search/search/sparse_geo.py` is empty
   and the SHA-256 matches on both sides:
   `244a2dcd64eefaf292253c44bfeb637f5b361e06f0de91df692edac57bd87883`.
2. **Exact-match synthetic regimes.** `regime_harness.py` re-run at the shipped defaults, diffed
   field by field against the pre-change capture — every precision / recall / F1 / AP / tp / fp / fn
   value matches on all four regimes (EASY 0.4545, TEXTURED 1.0000, VARIED 0.7597, CLUTTERED
   0.8634; tp/fp/fn identical). Only `p50_latency_ms` differs, which is laptop thermal/load state
   after a multi-hour ONNX sweep, not a code change.

A full `pixi run bench` was deliberately **not** run: with the method file byte-identical it would
spend hours reproducing numbers that cannot have moved, and the exact-match regime diff is the
stronger claim (the plan's Task 4 step 7 explicitly allows this).

## Verification

- `pixi run quality` green — Ruff and Ruff-format clean, MyPy strict clean, **784 passed / 16
  skipped, coverage 92.78%** against the 80% floor.
- `pixi run docs-build` green (`mkdocs build --strict`) with the new report in nav.
- Baseline reproduced to three decimals against the published findings table before any delta was
  measured (doors 0.2194 / 0.4416 / 0.1459; windows 0.3092 / 0.6275 / 0.2051).
- All four SuperPoint conditions and all four hypothesis-1 conditions tuned on `val` only, scored
  once on `test`.

## Fairness

Tuning read `val` **only**; `test` was scored exactly once per frozen config. All selections are
argmax-F1-on-val — never a test peek, and no grid decision was conditioned on a test outcome. Split
membership came from the committed manifests and was not regenerated. Acceptance rules are keyed to
fit geometry and vote-distribution shape, never to ground-truth boxes. The `pairwise-4dof`
**no-mirror control** exists specifically so a voting-mode change could not be silently credited to
mirror handling.

## Branch and suggested PR framing

**Branch:** `worktree/copper-spring`. Opening the PR is the user's follow-up step — it was not
opened here.

**Suggested title:** `docs(sparse-geo): floor-plan investigation — mirror + SuperPoint both
disproven, nothing shipped`

**Requirement IDs:** METHOD-04, METHOD-04a, DOC-04.

**Suggested body points:**

- This PR ships **no source change**. It ships the engineering log for a two-hypothesis
  investigation into `sparse-geo`'s flat floor-plan door recall, in which **both hypotheses were
  measured and both were disproven**. The prior commit on this branch (`8ab99a2`) already reverted
  hypothesis 1's implementation in full.
- **METHOD-04 / METHOD-04a** — how the criteria were verified: each hypothesis was measured in
  isolation against a baseline reproduced to three decimals before any source change; the revert
  filter (improve floor-plan test F1 without regressing synthetic regimes / AP50 / 28-28 coverage)
  was applied to each and each failed it; every non-surviving change is fully out of the diff with
  its measured reason recorded, not parked behind an off-by-default flag. `backend` still defaults
  to `sift` and SuperPoint remains permanently opt-in (MagicLeap non-commercial, gitignored
  weights). Regression guard: `sparse_geo.py` byte-identical to `df64af1` (SHA-256 + empty diff)
  and the default-config synthetic-regime metrics reproduce exactly.
- **DOC-04** — `docs/reports/sparse-geo-improvement.md` added and wired into the mkdocs nav;
  `docs/eval/floorplans-findings.md` and `docs/ROBUSTNESS-BACKLOG.md` cross-reference it without
  claiming an improvement that did not happen.
- **Reviewer note:** the most useful content here is the *negative* evidence — that the mirror gate
  fires 2 times in 55 peaks, that SIFT orientations are not mirror-consistent, that SuperPoint
  finds fewer usable exemplar keypoints than SIFT on line-art and hard-crashes at zero keypoints.
  Those rule out two plausible-sounding explanations that would otherwise stay folklore. The flat
  door recall-by-size symptom is explicitly left **open**, with candidate directions labelled as
  speculation for future work rather than as a conclusion.
