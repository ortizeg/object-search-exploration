---
phase: 08-evaluation-and-docs
plan: 02
subsystem: docs
tags: [charts, matplotlib, determinism, readme, method-docs, robustness-backlog, milestone-2, limitations]
status: complete

# Dependency graph
requires:
  - phase: 08-evaluation-and-docs
    provides: benchmark runner + results.json, metrics, Wilson interval, the four registered methods
  - phase: 02-07
    provides: docs/samples renderer (registry-iterating), the four method docs, propose()/embed_regions() seams
provides:
  - deterministic committed benchmark charts (Agg, suppressed PNG Software chunk) + results.md tables
  - README rewritten to the shipped Milestone 1 state with a four-method sample grid + real findings
  - config-schema drift guard tying every method doc to its config_model JSON Schema
  - aggregated robustness backlog (four methods + lattice verification + deferred Methods 4/6)
  - docs/MILESTONE-2.md pointing at the real shipped seams
  - docs/LIMITATIONS.md stating INFRA-07, MobileSAM, AGPL, non-commercial weights, real underperformance
affects: []

# Tech tracking
tech-stack:
  added: [matplotlib-base charts entrypoint]
  patterns:
    - "Deterministic PNG render: Agg backend + metadata={'Software': None} so committed charts regenerate byte-identically"
    - "results.md prints n/a for abstentions (None), never 0 — the abstention convention carried into the tables"
    - "Docs guarded by a test: every config_model JSON Schema field must appear in the method doc"

key-files:
  created:
    - src/object_search/eval/charts.py
    - tests/test_charts.py
    - tests/test_method_docs.py
    - docs/benchmark/results.md
    - docs/benchmark/metrics_by_method.png
    - docs/benchmark/crossover_by_scale.png
    - docs/benchmark/latency_by_canvas.png
    - docs/benchmark/thumbs_wilson.png
    - docs/MILESTONE-2.md
    - docs/LIMITATIONS.md
  modified:
    - README.md
    - docs/ROBUSTNESS-BACKLOG.md
    - pixi.toml
    - .planning/REQUIREMENTS.md
    - .planning/ROADMAP.md

decisions:
  - "Charts render from a full-sweep results.json (all four methods, 12 images); results.json stays gitignored, the PNGs + results.md are the committed deliverables"
  - "Thumbs/Wilson chart renders an honest n=0 empty-state panel rather than a fabricated bar when no human ratings exist"
  - "Method docs were already complete; the plan's config-reference regeneration is delivered as a drift-guarding test instead of a generator script"

metrics:
  completed: 2026-07-25
  tasks: 5
  files: 15

status: complete
---

# Phase 8 Plan 2: Committed charts, README, method docs, robustness backlog, Milestone 2, limitations Summary

Rendered the committed, byte-deterministic benchmark charts from real full-sweep data, rewrote the README to the shipped Milestone 1 state with a four-method sample grid and the honest findings, guarded the four method docs against config-schema drift, aggregated the robustness backlog, specified Milestone 2 against the real shipped seams, and stated the project's limitations without softening. This is the final plan of the project.

## What shipped

- **`charts.py` (EVAL-06)** — four figures rendered headlessly (matplotlib **Agg**) from `results.json`: pooled precision/recall/F1/AP bars, the NCC-vs-sparse-geo scale-bucket crossover, p50 latency across the canvas ramp (the EVAL-19 scaling story), and a thumbs/Wilson chart with an honest empty-state. Determinism: the PNG `Software` metadata chunk is suppressed so re-renders are byte-identical, asserted by `tests/test_charts.py`. Also emits `docs/benchmark/results.md`. Wired `pixi run bench-charts`.
- **README (DOC-03)** — rewritten from the stale "Phase 1 in progress" to Milestone 1 complete: quickstart, the four-methods table, a side-by-side sample grid for all four methods (registry-iterating renderer), the benchmark summary with committed charts and the real findings, a link to LIMITATIONS, project layout, dev commands.
- **Method-doc drift guard (DOC-04)** — the four docs already carried algorithm, explicit pre/post-processing, schema-derived config reference, failure modes, and mirrored ROBUSTNESS BACKLOG; `tests/test_method_docs.py` now asserts every registered method's doc has the required sections and names every `config_model` JSON Schema field, so the config reference cannot drift.
- **ROBUSTNESS-BACKLOG.md (DOC-05)** — appended the cross-cutting items: lattice fitting as post-detection verification (the highest-leverage item for shelf/PCB/tile — recovers misses and kills false positives by adding arrangement information) and the deferred Methods 4 and 6 with reasoning intact.
- **MILESTONE-2.md (DOC-06)** — the marker-conditioned proposal feature, pointing at the real shipped seams: the `exploration` column (`store/schema.py`), the registry, `propose()`/`embed_regions()`, and the UI mode selector (`frontend/index.html`).
- **LIMITATIONS.md** — INFRA-07 partial (branch-protection 403 on a free private repo), MobileSAM not shipped and why, FastSAM AGPL-3.0 embedded in the `.onnx`, SuperPoint non-commercial research-only weights, the canvas-height CSS polish item, the empty human-rating scoreboard, and the real benchmark findings with numbers.

## Real benchmark findings (12-image demo set, IoU 0.5)

| method | precision | recall | F1 | mean AP | p50 |
| --- | --- | --- | --- | --- | --- |
| `ncc` | 0.913 | 0.922 | 0.918 | 0.484 | 238 ms |
| `sparse-geo` | 0.833 | 0.097 | 0.174 | 0.083 | 76 ms |
| `dino-dense` | 0.276 | 0.078 | 0.121 | 0.190 | 2259 ms |
| `propose-retrieve` | 0.748 | 0.951 | 0.838 | 0.635 | 291 ms |

- `propose-retrieve` is the strongest general retriever (best AP 0.635, best recall 0.951).
- `ncc` wins the fixed-scale regime (recall 0.989) but collapses on varied scale (0.30) and scales badly with canvas (5.7 s at 6000×4000).
- `sparse-geo` abstains on 11/12 low-texture chips (< 20 SIFT keypoints) — the **NCC-vs-sparse-geo crossover**, reported as the expected finding.
- `dino-dense` underperforms (F1 0.121) and is the slowest — stride-14 tokens too coarse for tiny instances.

## Deviations from Plan

- **[Rule 3 — Environment] `gsd-tools` shim absent.** This repo has no `gsd-core/bin/gsd-tools.cjs` and `gsd-tools` is not on PATH, so the SDK state handlers (`state.advance-plan`, `requirements.mark-complete`, `roadmap.update-plan-progress`, `state.record-metric`) were unavailable. STATE.md, ROADMAP.md, and REQUIREMENTS.md were updated by direct edit instead. No effect on deliverables.
- **[Task 1] Task-3 config-reference generator delivered as a test, not a script.** All four method docs already contained every schema field, so rather than a one-shot generator that would immediately go stale, the sync is enforced by `tests/test_method_docs.py` (a CI-run drift guard). Same guarantee, continuously checked.
- **[Task 1] `results.json` is gitignored** in this repo, so the committed deliverables are the PNGs and `results.md`, not the raw JSON. Charts regenerate from a local `pixi run bench` output.

## Verification

- All four quality gates green; **440 passed, 5 skipped, coverage 93.13%** (floor 80%).
- Charts byte-identical-on-rerender test passes (`test_charts_are_byte_identical_on_rerender`).
- README references all four `docs/samples/<method>/`; LIMITATIONS contains INFRA-07, AGPL, MobileSAM, and the real numbers; MILESTONE-2 names the real seams.
- PR #16 open into `main`, CI green (quality + model-free chipset benchmark subset). Left open, not merged, per instruction.

## Self-Check: PASSED

- Created files exist on disk (charts.py, both test files, MILESTONE-2.md, LIMITATIONS.md, four PNGs, results.md).
- Commits present: ed8ddef, b802bee, 5833fcc, 5346c13, bb90555.
