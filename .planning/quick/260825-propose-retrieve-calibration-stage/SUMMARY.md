---
phase: quick-260825-calibration
plan: 01
subsystem: search-methods
tags: [propose-retrieve, floorplans, eval, calibration, iterate-measure-revert, negative-result]
status: complete
requires: []
provides:
  - "Per-GT-box diagnostic trace of propose-retrieve's retrieval/calibration stage on floorplans-door, decomposing the crowded-bucket recall gap into matched/below_threshold/nms_suppressed/no_proposal"
  - "similarity_floor swept below its shipped 0.70 default at proposal_conf=0.10 (0.55/0.60/0.65) — closes the prior report's explicitly stated 'not measured' gap"
  - "Evidence that the gmm calibrator's adaptive component is nearly inert on this domain (the fixed similarity_floor decides almost every case) and that DINOv2 embedding discriminability, not calibration logic, is the crowded-bucket ceiling"
  - "docs/reports/propose-retrieve-floorplans-improvement.md — new 2026-08-25 follow-on section recording the negative result"
affects:
  - docs/reports/propose-retrieve-floorplans-improvement.md
  - docs/methods/propose-retrieve.md
  - docs/ROBUSTNESS-BACKLOG.md
  - src/object_search/search/propose_retrieve.py
tech-stack:
  added: []
  patterns:
    - "Per-GT-box pipeline tracing (propose -> embed -> calibrate -> NMS, classifying each box's fate) isolates a stage's OWN loss rate from an upstream stage's — cleaner than inferring it from the gap between two aggregate recall numbers."
    - "Re-reading a threshold's own code path (here: max(gmm_cut, floor), floor used directly on a degenerate fit) can show an 'adaptive' component is actually inert in practice — worth checking before assuming the adaptive logic is where the noise lives."
key-files:
  created:
    - scripts/propose_retrieve_calibration_experiment.py
    - .planning/quick/260825-propose-retrieve-calibration-stage/EXPERIMENTS.md
    - .planning/quick/260825-propose-retrieve-calibration-stage/SUMMARY.md
  modified:
    - docs/reports/propose-retrieve-floorplans-improvement.md (new dated follow-on section)
    - docs/methods/propose-retrieve.md (ROBUSTNESS BACKLOG bullet)
    - docs/ROBUSTNESS-BACKLOG.md (ROBUSTNESS BACKLOG bullet)
    - src/object_search/search/propose_retrieve.py (ROBUSTNESS BACKLOG docstring bullet, mirrored)
decisions:
  - "No code or config change ships. similarity_floor was swept down to 0.55 at proposal_conf=0.10 and pooled val F1 fell monotonically at every step (0.542 -> 0.480 -> 0.393 -> 0.322) — the shipped 0.70 default is confirmed as the argmax across the full plausible range, not just the previously-swept upper half."
  - "A crowding-conditional similarity_floor was considered and explicitly NOT pursued, even though floor=0.65 alone wins the crowded bucket (+0.028 F1) — it would need to dispatch on ground truth the method cannot observe at inference time, and would introduce config-driven dispatch inside a method, which .claude/CLAUDE.md's method-module conventions rule out."
  - "Hypothesis 3 from the task brief (pre-embedding objectness/top-K filtering) was argued from existing T2 evidence rather than given a fresh ~4h trial — it reduces to re-raising proposal_conf, which T2 already measured to hurt recall monotonically."
  - "Test was NOT read — per the tune-on-val/read-test-once discipline, test is read only for a finalist that beats val baseline, and none did."
metrics:
  duration: "~7 hours (remote box provisioning/sync + one 56-plan diagnostic trace + three parallel 56-plan val trials, each ~4h wall clock 3-way contended on an 11.2-effective-core box)"
  completed: 2026-08-25
  tasks: 7
  files_changed: 5
  commits: 1
---

# Quick Task 260825: propose-retrieve retrieval/calibration-stage investigation (floorplans-door)

Investigated the crowded-bucket retrieval/calibration gap the prior floor-plan pass
(`260812-m8m`) flagged as its next lead (proposal-stage recall 0.639 vs end-to-end recall 0.262 in
the 11+-door bucket) and closed it as a **well-evidenced negative result**: no threshold lever
recovers the gap without a net pooled-F1 cost, and the evidence points to DINOv2 embedding
discriminability — not a miscalibrated gmm cut — as the actual ceiling.

## What Was Built

### Diagnostic — a per-GT-box calibration trace

`scripts/propose_retrieve_calibration_experiment.py` traces every ground-truth box through the
actual pipeline (`propose` -> `embed_regions` -> `calibration.calibrate` -> threshold ->
`nms.nms`) and classifies its fate as `matched` / `below_threshold` / `nms_suppressed` /
`no_proposal`. Run over all 56 val plans (527 GT boxes) at the shipped finalist config, it found
the retrieval-stage loss rate (isolated from proposal supply) rises from 0.095 (sparse) to 0.322
(crowded), that the gmm's adaptive cut is nearly inert (the fixed `similarity_floor` decides
almost every case, by construction of the method's own `max(gmm_cut, floor)` / degenerate-fallback
logic), and that true/background cosine-score separation compresses with crowding (0.373 ->
0.287) — the signature of an embedding-discriminability ceiling, not a calibration-logic defect.

### Candidate lever — `similarity_floor` swept below 0.70

The one lever the diagnostic motivated (a lower floor might rescue the crowded bucket, since its
`below_threshold` GT boxes score 0.64-0.66, comfortably below the fixed 0.70 floor) was measured
directly: three val trials at `proposal_conf=0.10`, `similarity_floor` in {0.55, 0.60, 0.65}.
Pooled F1 fell monotonically at every step (0.542 -> 0.480 -> 0.393 -> 0.322); the shipped 0.70
default is confirmed as the val argmax across the entire now-measured {0.55-0.85} range. The
crowded bucket alone does improve at floor=0.65 (F1 0.282 -> 0.310), but the same move costs the
sparse and medium buckets far more (-0.137 and -0.072 F1, on 14 and 36 plans vs 6), so the pooled
metric the repo's tuning methodology optimises correctly rejects it.

### Report

`docs/reports/propose-retrieve-floorplans-improvement.md` gained a new dated follow-on section
recording the full negative result, and the ROBUSTNESS BACKLOG (in `propose_retrieve.py`,
`docs/methods/propose-retrieve.md`, and `docs/ROBUSTNESS-BACKLOG.md`, kept mirrored per
convention) gained a matching bullet.

## Key Decisions

See frontmatter `decisions`. The central one: a crowding-conditional floor was explicitly
considered and rejected on architectural grounds (config-driven dispatch inside a method), not
just measured-and-lost — even though it would have been the more literally "successful" outcome
for the crowded bucket in isolation.

## Verification

| Check | Result |
|---|---|
| `pixi run lint` | Passed |
| `pixi run typecheck` (mypy strict) | Passed, unchanged file count (no `src/` production code changed) |
| `pixi run test` | Unchanged — no `src/` production code touched |
| `pixi run docs-build` (`mkdocs build --strict`) | Passed |
| floorplans-window / chipset/textured/synthetic/real-objects guardrails | Not re-run — nothing shipped that could affect them (no config default, no grid entry, no code path change) |
| Git guard (`--git-common-dir`) | Verified before commit — resolves under `object-search-exploration` |

## Known Stubs

None.

## Self-Check: PASSED

- `docs/reports/propose-retrieve-floorplans-improvement.md`'s new section reads coherently and its
  anchor (`#follow-on-260825-calibration`, via `attr_list` rather than an auto-slug guess) resolves
  under `mkdocs build --strict`.
- ROBUSTNESS BACKLOG mirrored in all three locations (module docstring, method doc, top-level
  backlog doc) with consistent numbers.
- `runs/` JSON artifacts intentionally left uncommitted per `.gitignore`'s
  `.planning/quick/*/runs/` rule; every cited number traces to an artifact on the vast.ai instance
  (`48124756`) that produced it.
- No PR opened by this step — a separate, explicit step per the task's own instructions.
