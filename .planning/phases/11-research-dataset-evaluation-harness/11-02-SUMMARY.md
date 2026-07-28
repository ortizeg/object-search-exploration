---
phase: 11-research-dataset-evaluation-harness
plan: 02
subsystem: eval
tags: [research-datasets, converters, fscd147, fscd-lvis, rpine, dedup, seeded-carve, split-manifests]
status: complete
requirements: [EVAL-21, EVAL-22]
provides:
  - object_search.eval.converters.convert_fscd147
  - object_search.eval.converters.dedup_fscd147
  - object_search.eval.converters.convert_fscd_lvis
  - object_search.eval.converters.convert_rpine
  - object_search.eval.splits.carve_val
  - object_search.eval.splits.NativeSplits
  - object_search.eval.splits.build_manifest
  - object_search.eval.splits.build_all_manifests
  - object_search.eval.splits.write_split_manifest
  - DATASET_REGISTRY entries fscd147 / fscd_lvis / rpine / pucpr_plus
  - committed manifests conf/datasets/{fscd147,fscd_lvis,rpine,pucpr_plus}.split.json
requires:
  - object_search.eval.converters.carpk (the 11-01 converter contract mirrored)
  - object_search.eval.labels._parse_sidecar (the single GT reader, D-10)
  - object_search.eval.splits.ResearchSplitManifest (11-01 frozen schema)
  - object_search.provenance.file_sha256 (pixel-identical duplicate detection)
affects: [conf/datasets/, tests/fixtures/research/]
key-files:
  created:
    - src/object_search/eval/converters/fscd147.py
    - src/object_search/eval/converters/fscd_lvis.py
    - src/object_search/eval/converters/rpine.py
    - conf/datasets/fscd147.split.json
    - conf/datasets/fscd_lvis.split.json
    - conf/datasets/rpine.split.json
    - conf/datasets/pucpr_plus.split.json
    - tests/fixtures/research/fscd147/ (10 images + annotations.json + split.json + README)
    - tests/fixtures/research/fscd_lvis/ (8 images + annotations.json + split.json + README)
    - tests/fixtures/research/rpine/ (8 images + 8 annotation txts + split.json + README)
    - tests/test_research_dedup.py
    - tests/test_research_converters.py
    - tests/test_research_splits.py
  modified:
    - src/object_search/eval/converters/__init__.py
    - src/object_search/eval/datasets.py
    - src/object_search/eval/splits.py
decisions:
  - "Train<->test leaks are caught STRUCTURALLY (an id in >1 split) rather than by a transcribed 11-filename list; the real FSC-147 leaks are, by definition, exactly the ids that appear in both train and test, so the structural check removes all 11 without a list that could drift. arXiv:2409.15953 cited; _DOCUMENTED_TRAIN_TEST_LEAK_IDS left empty for any leak documented outside the split files."
  - "One shared carve_val helper in splits.py (Rule of Three: RPINE + FSCD-LVIS-unseen are the two concrete uses); np.random.default_rng only, sorted-canonical input so order and process cannot affect the draw."
  - "PUCPR+ reuses the CARPK converter (same author release / native format) and gets a test-only manifest with empty test ids until its licence-gated archive is fetched."
  - "build_all_manifests takes NativeSplits (pure); FSCD-147 dedup runs in the caller BEFORE the manifest is built, so leaked/duplicate ids never enter a split."
metrics:
  completed: 2026-07-26
  tasks: 3
  files: 20
status_detail: complete
---

# Phase 11 Plan 02: Three research converters + dedup + seeded split manifests Summary

The proven 11-01 seam is expanded from one dataset to four. FSCD-147, FSCD-LVIS (unseen) and RPINE
each gain a converter that emits the **existing** `*.gt.json` sidecar schema (no second GT reader,
D-10); FSC-147's documented contamination is de-duplicated on load; one shared seeded `carve_val`
helper produces deterministic val slices for the two datasets with no official val; and all
four/five datasets have committed, schema-valid, seed-reproducible split manifests. Everything runs
offline on tiny committed native-format fixtures — the licence-gated real archives are a pending
user_setup step and no real-data results are claimed.

## Task-by-task

- **Task 1 — FSCD-147 converter + de-duplication (auto, tdd): DONE.**
  `converters/fscd147.py`: `convert_fscd147` parses FSC-147's COCO-style `annotations.json` (per-object
  boxes for val/test only — train pseudo-boxes skipped, D-06), converts each `[x1,y1,x2,y2]` to the
  half-open `BBox` at the boundary, and maps the 3 native exemplar polygons to `exemplar_indices`
  (first == `exemplar_index`). `dedup_fscd147` drops (a) every id in more than one split — the
  structural signature of the 11 documented train↔test leaks — and (b) pixel-identical duplicates by
  `file_sha256`, keeping the lexicographically-first copy, **before** any manifest is built. Committed
  `conf/datasets/fscd147.split.json` is `val_strategy="native"`, de-duplicated, over the fixture ids.
  Fixture plants a byte-identical `dup-a`/`dup-b` pair and a `leak` id in both train and test.

- **Task 2 — FSCD-LVIS + RPINE converters and the shared carve_val (auto, tdd): DONE.**
  `converters/fscd_lvis.py` (unseen protocol) emits **only** the exemplar-category boxes as GT — the
  other-category boxes are the distractors and are intentionally excluded, so returning one scores as
  a false positive (the distractor-rejection signal). `converters/rpine.py` emits all-repeats boxes and
  samples up to three `exemplar_indices` from the GT, seeded with `np.random.default_rng` (per-image
  FNV-1a offset so each image is independent yet byte-stable). `splits.carve_val(train_ids, *, seed,
  val_fraction)` is the single shared val-carver: sorts input to a canonical order, draws a
  `default_rng(seed)` permutation, returns `(train_remainder, val)` — the test list is never passed in
  or touched. Registry entries added for `fscd_lvis`, `rpine`, `pucpr_plus`.

- **Task 3 — Generate + commit all manifests; prove seed-stability (auto, tdd): DONE.**
  `splits.build_manifest` / `build_all_manifests` apply the per-dataset `_VAL_STRATEGY`
  (native / seeded-carve / test-only) and `write_split_manifest` serializes with sorted keys +
  trailing newline (config-hash-stable, D-11). Committed `fscd_lvis.split.json`, `rpine.split.json`
  (seeded-carve) and `pucpr_plus.split.json` (test-only). `test_research_splits.py` proves carved val
  is byte-identical across two same-seed builds AND the test split is identical across two DIFFERENT
  seeds (seed reaches only the train↔val partition — EVAL-22 success criterion 3), that carpk/pucpr_plus
  are test-only, that fscd147 is native, and that every committed manifest round-trips through the frozen
  schema with sorted-key JSON equality.

## Load-bearing correctness (as required)

- **FSCD-147 dedup (D-07):** `test_research_dedup.py` asserts the planted duplicate copy and the
  planted leaked id are absent after dedup, the leaked id is gone from the **test** split, no id remains
  in more than one split, and `removed_count == leaks + (dup_copies − canonical) == 2`.
- **Seeded val carving (D-03):** `test_research_converters.py` asserts `carve_val(seed=7) ==
  carve_val(seed=7)` byte-for-byte, `carve_val(seed=7)` val ≠ `carve_val(seed=8)` val (50-id list, so
  robust), the train↔val partition is exact and disjoint, carving is order-independent, and the test id
  list is never seen by the carver. `test_research_splits.py` proves the same at the manifest level and
  that FSCD-147 uses its native triple (no carving).
- **Test-only (D-04):** carpk/pucpr_plus manifests have empty `train` and `val`.
- **One GT reader (D-10):** every converter only writes JSON; all three fixtures are asserted by
  loading back through `load_research_ground_truth` → the single `_parse_sidecar`.

## Deviations from Plan

- **[Rule 3 — blocking] Added `conf/datasets/pucpr_plus.split.json`.** Task 3's acceptance test asserts
  a committed pucpr_plus manifest is test-only, but the plan's `files_modified` listed only fscd_lvis
  and rpine manifests. Added the test-only pucpr_plus manifest (empty test until its archive is
  fetched — real ids land on fetch, mirroring the carpk fixture-id pattern from 11-01). Not a stub: it
  declares the correct protocol; the empty test list is expected for a dataset with no committed
  fixture this wave.
- **Structural leak detection instead of a hard-coded 11-filename list.** The plan text suggested a
  "hard-coded documented 11-id leak list". The real 11 FSC-147 filenames are distributed *with* the
  licence-gated data, not with our code, and a transcribed list can silently drift. Since the 11 leaks
  are by definition ids that appear in both train and test, `dedup_fscd147` removes all of them
  structurally (an id in >1 split), which is stronger and needs no maintenance. arXiv:2409.15953 and the
  count `11` are cited in-source (`DOCUMENTED_FSC147_TRAIN_TEST_LEAK_COUNT`), and an explicit
  `documented_leak_ids` parameter remains for any leak ever published outside the split files.

No architectural (Rule 4) changes; no auth gates.

## Pending USER action — real licence-gated data (user_setup)

The real FSCD-147, FSCD-LVIS and RPINE archives are licence-gated (VinAI / Counting-DETR and TMR
research terms) and were **not** fetched; all tests use the committed synthetic fixtures. Two follow-ups
are needed when a human drops a real archive (both local-only, never CI):

1. **Accept each licence and drop the archive** at the path printed by `pixi run fetch-datasets --list`
   (e.g. `datasets/_incoming/fscd147/`).
2. **Real-fetch raw-root resolution is not yet wired for the FSC-style layouts.** The registry entries
   and the dataset→converter dispatch (`_CONVERTERS`) are in place, but `datasets._resolve_raw_root`
   still recognises only CARPK's `Annotations/` layout, so `fetch()` degrades gracefully to
   "no data found" for the three new datasets. Wiring the FSC-style raw-root detection (and building the
   real manifests via `build_all_manifests` over the fetched id lists) is the natural companion to the
   licence drop and should be done then, against the real tree. The converters themselves are proven on
   fixtures and are reachable directly and via `build_all_manifests`.

There is **no `build-manifests` pixi task** yet: `object_search.eval.splits.build_all_manifests` is the
public entry point, and the committed fixture manifests were generated by calling it at
`RESEARCH_VAL_SEED`. A pixi task folding it into `fetch-datasets` as a post-fetch step is the right home
once real-fetch resolution lands (documented above).

## Known Stubs

None. `pucpr_plus.split.json` carries an empty test list by design (test-only, no committed fixture this
wave; real ids on fetch) — the same ratified fixture-vs-real pattern as 11-01's carpk manifest, not a
stub that blocks the plan goal.

## Threat surface

No new trust boundaries beyond the plan's register. T-11-06 (box coords) handled by half-open `BBox`
conversion + `BBox`'s `>=1` w/h validation; T-11-07 (train↔test leakage) by `dedup_fscd147` proven in
`test_research_dedup.py`; T-11-08 (non-reproducible carve) by the `carve_val` seed-stability tests;
T-11-09 (fabricated boxes from dots) by skipping dot-only / distractor annotations rather than
synthesizing boxes (D-06), asserted by the FSCD-LVIS target-only box count.

## Verification output (pasted real results)

- `pixi run lint` → `All checks passed!`
- `pixi run format-check` → `134 files already formatted`
- `pixi run typecheck` → `Success: no issues found in 72 source files` (mypy strict, no new ignores)
- `pixi run test` → `531 passed, 19 skipped`; `Required test coverage of 80% reached. Total coverage:
  88.62%`
- `pixi run quality` (umbrella: lint + format-check + typecheck + test) → green.
- New-module coverage: `fscd147.py` 90%, `fscd_lvis.py` 84%, `rpine.py` 82%, `splits.py` 88%,
  `datasets.py` 81% — all above the 80% floor.
- Task greps: `arXiv:2409.15953` + the count `11` live in `fscd147.py`; `dedup_fscd147` documents and
  runs "before any split manifest is built"; `carve_val` is the single val-carve implementation;
  `grep -c default_rng src/object_search/eval/splits.py` → `3`; no `cv2.setRNGSeed` call anywhere
  (only referenced in docstrings as the thing NOT used).
- `pixi run fetch-datasets --list` → `5 registered dataset(s)`: carpk, fscd147, fscd_lvis, pucpr_plus,
  rpine — each with licence + `datasets/_incoming/<key>` drop path.
- `git ls-files datasets/` → empty (no raw dataset bytes tracked); fixtures under
  `tests/fixtures/research/` and manifests under `conf/datasets/` are tracked and not gitignored.

## Self-Check: PASSED

- Created files exist on disk: three converter modules, four new `conf/datasets/*.split.json`, three
  fixture trees, three test modules (verified via `git status`).
- New symbols import and run: the full suite (531 passed) exercises `convert_fscd147`, `dedup_fscd147`,
  `convert_fscd_lvis`, `convert_rpine`, `carve_val`, `build_all_manifests`, `write_split_manifest` and
  the five registry entries.
- Commits: not created by this executor (orchestrator commits atomically; changes left in the working
  tree as instructed).
