---
planStatus:
  planId: plan-real-objects-eval-report
  title: Real-Objects vs Synthetic Comparative Evaluation Report
  status: draft
  planType: research
  priority: medium
  owner: ortizeg
  stakeholders: []
  tags: [eval, benchmark, reporting]
  created: "2026-08-06"
  updated: "2026-08-06T00:00:00.000Z"
  progress: 0
---

# Real-Objects vs Synthetic Comparative Evaluation Report

## Objective

PR #48 (merged into `main` as part of this session) added `real-objects` — 30 real-photo composite
images (10 objects × 3 regimes: plain/varied/cluttered) wired into `object_search.eval.labels` via
`real_objects_image_ids()`. It rides along in the existing full sweep (`resolve_run_set()` already
unions it in), but there is no **dedicated** report for it and no **written analysis** of how the
six methods' scores on real photographic pixels compare to their scores on the fully-synthetic
sets (chipset/textured) that the existing `benchmark-report.html` covers. This plan produces both:
a standalone `real-objects-report.html` (same visual language as the existing report) and a prose
findings doc that actually explains *why* scores move the way they do between the two surfaces —
plus, conditionally, a per-method remediation plan if any method's real-objects score collapses in
a way not explained by ordinary real-photo noise.

## Non-goals

- No change to any search method (`ncc`, `mosse`, `sparse-geo`, `dino-dense`, `propose-retrieve`,
  `owlv2-oneshot`) as part of this PR. If §5 (remediation plan) is triggered, it is a **plan
  document only** — implementing the fix is explicitly out of scope here, same convention as the
  existing `docs/reports/*-improvement.md` docs record spikes that *did* change a method, vs. this
  doc which may propose one without doing it.
- No change to the existing `docs/reports/benchmark-report.html` / `scripts/build_report.py`. Its
  `REGIMES`/`regime_of()` stay chipset+textured-only; real-objects rows already ride along
  unused in its `results.json` input (confirmed: current committed report has zero `real-` matches)
  and stay that way.
- Not wired into the TINY-DENSE/mAP@[.5:.95]/Bradley-Terry spec-only items from `EVAL-DESIGN.md` —
  same practical-size-cut scope as the shipped report.

## Design

### 1. Scoping the sweep to real-objects only

`BenchmarkConfig.resolve_run_set()` (`src/object_search/eval/benchmark.py`) currently has one
branch (`ci: bool`) that switches the entire run set. I'll add a second, orthogonal boolean in the
same style:

```python
real_objects_only: bool = False
```

`resolve_run_set()` gains a branch ahead of the existing union logic:

```python
if self.real_objects_only:
    return self.methods, real_objects_image_ids()
```

— full six-method list (the config default already matches the task's requested methods), image
set = exactly `real_objects_image_ids()`, nothing unioned in. `ci` and `real_objects_only` are both
bools with independent defaults; `ci=True` still wins if both were somehow set (checked first,
unchanged).

A new `conf/benchmark-real-objects.yaml` mirrors `conf/benchmark.yaml`'s structure (same
`defaults:`/`hydra:` boilerplate to silence Hydra's own logging and stay in `.`) but sets
`real_objects_only: true` and `out: docs/benchmark/real-objects-results.json`. Hydra's
`@hydra.main` binds one `config_name` at decoration time, but it accepts `--config-name=...` as a
CLI override, so no code change is needed to select it — a new pixi task does that:

```
bench-real-objects = "python -m object_search.eval.benchmark --config-name=benchmark-real-objects"
```

**Why a config field instead of a CLI-only `image_ids` override:** hardcoding the 30 ids into the
YAML (or passing them on the CLI) would silently go stale the moment `real-objects` regenerates
with a different object added/dropped (as already happened once — 4 objects were swapped during
dataset construction). `real_objects_only` re-derives the id list from disk every run, the same
guarantee `chipset_image_ids()`/`textured_image_ids()` already give the existing paths.

### 2. The dedicated report

New `scripts/build_real_objects_report.py`, adapted from `scripts/build_report.py` (same
`STYLE`/`METHODS`/`COLOR`/`ORDER`/bootstrap-CI/`b64` approach — the repo's established pattern is
one self-contained script per report, e.g. `build_floorplans_report.py` already duplicates rather
than shares against `build_report.py`) with:

- `REGIMES` = `PLAIN` / `VARIED` / `CLUTTERED` (3, not 4 — no EASY/TEXTURED split; `real-plain-*`
  already plays both those roles per `DATASETS.md`).
- `regime_of()` matches on the `real-plain-` / `real-varied-` / `real-cluttered-` id prefixes.
- Reads `docs/benchmark/real-objects-results.json` (new, gitignored like `results.json`).
- Writes `docs/reports/real-objects-report.html` (new, committed, same self-contained inline-SVG /
  base64-JPEG-overlay style as the existing report — no network, no JS).
- Overlay gallery: one representative object per regime, including the `ping-pong-ball` stress
  object (the documented textureless/rotationally-symmetric case) for at least one regime, since
  it is the single most diagnostic image in the set.
- New pixi task: `report-real-objects = "python scripts/build_real_objects_report.py"`.

### 3. Running both sweeps

- `pixi run bench` (existing default) — unchanged invocation, already unions
  chipset+textured+real-objects+scatter-scaled+cluttered-distractors per the current
  `resolve_run_set()`. This is the "full synthetic sweep" data source for the comparison (I'll
  filter its `per_image` rows by id prefix to isolate chipset/textured from real-objects rather
  than re-deriving a synthetic-only sweep, since one sweep already contains both — see note below).
- `pixi run bench-real-objects` (new) — the dedicated, real-objects-only run, feeding
  `real-objects-report.html`.
- `pixi run report` / `pixi run report-real-objects` — regenerate both committed HTML reports.
- `pixi run bench-charts` — unchanged, still chipset/textured/scatter/cluttered charts only.

Both sweeps score the *same* (method, image, config) cells when they overlap (real-objects rows),
and reproducibility is a hard invariant here, so the numbers are expected to agree byte-for-byte;
running both is for two independent artifacts (the general report's `results.json` vs. the
dedicated report's own file), not because the numbers could differ.

**Environment note:** this worktree has no `pixi` env solved yet and no `models/` (gitignored,
~8 GB of ONNX weights normally produced by `pixi run fetch-models`). The primary checkout at
`.../object-search-exploration/models/` already has `dinov2_small.onnx`, `fastsam_s.onnx`, and
`owlv2_base_patch16.onnx` — exactly the three weight files the full sweep needs (`sparse-geo`
defaults to the classical SIFT backend, no weights). I'll symlink that directory into this worktree
rather than re-downloading, then `pixi install`. This only reads from the main checkout; nothing
there is modified.

### 4. Findings doc — the actual analysis

`docs/reports/real-objects-findings.md`, styled like `docs/eval/floorplans-findings.md`: how it was
produced, a dataset-statistics section, per-method comparison tables (synthetic pooled vs.
real-objects pooled, and per-regime PLAIN/VARIED/CLUTTERED vs. their EASY/TEXTURED/VARIED/CLUTTERED
synthetic analogues), then **prose** answering the brief's actual questions per method:

- Is the method's rank preserved between synthetic and real-objects?
- How much does its absolute P/R/F1/AP drop (or not), regime by regime?
- A causal hypothesis for the delta, grounded in something inspectable — not generic hand-waving:
  candidate log stats (e.g. mosse's proposal-recall-vs-match-recall diagnostic used in its own
  improvement log), the ground-truth-quality difference (FastSAM soft-mask AABB vs. exact rendered
  AABB — the ping-pong-ball's documented shadow-wedge artifact is a concrete, named instance of
  this), real JPEG/lighting noise NCC/MOSSE's flat-window degenerate case is sensitive to, texture
  differences a keypoint method depends on, etc. I will actually pull the worst-N real-objects
  per-image rows per method from `real-objects-results.json` and look at them before writing any
  causal claim.

### 5. Conditional remediation plan(s)

After the findings doc's per-method deltas are computed: for any method whose real-objects score
drops far more than the other five *and* the findings analysis cannot pin the drop on ordinary
photographic-noise/GT-imperfection causes above, I inspect that method's overlays on its worst
real-objects images (via the new report's overlay gallery, extended with extra ad hoc overlays
through `search/common/viz.py` if the report's fixed gallery doesn't cover the worst case), form a
concrete hypothesis, and write `docs/reports/<method>-real-objects-improvement.md` (plan only, no
code change), following the existing `docs/reports/*-improvement.md` measured-iteration-log
convention (symptom → measurement setup → hypothesis → proposed levers → expected effect). This is
data-dependent and cannot be scoped further until the sweep has run.

## Testing plan

- `tests/test_eval_benchmark.py`: two new tests mirroring the existing `ci=True` pair —
  `test_real_objects_only_subset_is_real_objects_images_all_methods()` (asserts `resolve_run_set()`
  returns the full method tuple and exactly `real_objects_image_ids()`, no chipset/textured/
  scatter-scaled ids leak in) and a `run_benchmark` round-trip test with `real_objects_only=True`
  over a couple of real-objects images asserting `coverage`/`per_image` shape, following
  `test_ci_benchmark_writes_results_with_per_slice_breakdowns`'s pattern. These run with
  `ncc`/`sparse-geo`-only `methods=(...)` overrides so they stay in the model-free CI-runnable set
  (no ONNX weights needed), consistent with `[[memory: CI coverage runs without weights]]`.
- `scripts/build_real_objects_report.py` gets no dedicated pytest (matches
  `build_report.py`/`build_floorplans_report.py`: `pyproject.toml`'s coverage `source` is
  `src/object_search` only, `scripts/` is not gated).
- `pixi run quality` (Ruff + MyPy strict + pytest w/ coverage floor) must stay green.

## Deliverables checklist

1. `src/object_search/eval/benchmark.py` — `real_objects_only` field + `resolve_run_set()` branch.
2. `conf/benchmark-real-objects.yaml` — new Hydra config.
3. `pixi.toml` — `bench-real-objects` + `report-real-objects` tasks.
4. `scripts/build_real_objects_report.py` — new report builder.
5. `docs/reports/real-objects-report.html` — committed output of #4.
6. `docs/reports/real-objects-findings.md` — the prose comparison + analysis (§4).
7. `docs/reports/<method>-real-objects-improvement.md` — 0+ conditional remediation plan(s) (§5).
8. New/updated tests in `tests/test_eval_benchmark.py`.
9. `docs/DATASETS.md` — one-line update pointing at the new report once it exists (mirrors how it
   already links `benchmark-report.html`).

## Open questions for the user

See the accompanying `AskUserQuestion` prompt.
