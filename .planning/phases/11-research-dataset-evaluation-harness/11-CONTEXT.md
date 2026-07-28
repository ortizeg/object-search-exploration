# Phase 11 Context — Research-dataset evaluation harness

**Source:** This session's literature survey and design discussion (2026-07-26), captured in
`11-RESEARCH.md` (dataset survey, links, strengths/weaknesses, splits, metric definitions).
Seeded in place of a `gsd-phase-researcher` run because the domain research was done live with the
user. Builds directly on Phase 8's eval layer (`src/object_search/eval/`).

## Domain

The benchmark today scores four methods on **self-generated** synthetic sets (chipset EVAL-19,
textured EVAL-20) where ground truth is exact by construction. That proves the *crossover* story
but says nothing about how the methods do on **real, published** data, and produces no number
comparable to the few-shot counting/detection literature. Phase 11 adds four **external research
datasets** as a held-out evaluation surface, keeps the synthetic sets as the tuning surface, and
reports the **literature's own metrics** so our numbers sit next to published work.

The project's task is unusual: **one hand-drawn exemplar box → every other instance of that same
object in the same image.** That is single-image exemplar search, not dataset-level counting. So
dataset fit is judged by whether the set has **per-instance boxes** (our harness scores boxes with
IoU) and **exemplar-style queries** — not by whether it is a famous counting benchmark.

## Locked Decisions

1. **Four datasets, each doing a distinct job** (do not add redundant sets):
   - **RPINE** — the closest match to our task: all repetitions in a single image are box-annotated,
     with box exemplars. Core "does it find *all* repeats in one image" set.
   - **FSCD-147** — box-annotated FSC-147 (val/test). Category diversity + leaderboard comparability.
   - **FSCD-LVIS (unseen split)** — multi-class crowded scenes; the only **distractor-rejection**
     stress (several repeated things share an image).
   - **CARPK (+ PUCPR+)** — dense near-identical cars, drone view; **cross-domain generalization**.

2. **Protocol: tune on val, freeze test, report on test.** This is the field-standard protocol and
   gives leaderboard-comparable numbers. "Tuning" here = config/threshold sweeps — our four methods
   are **training-free / config-tuned**, not gradient-trained — so this is cheap and reproducible.

3. **Where no official val split exists, carve one from train deterministically** (seeded from
   config), and **never touch test**. Applies to RPINE and FSCD-LVIS-unseen. FSCD-147 ships a native
   train/val/test triple — use it as-is.

4. **CARPK + PUCPR+ are TEST-ONLY.** No tuning on them at all. In the class-agnostic-counting
   literature they exist to measure cross-domain generalization from whatever was tuned elsewhere —
   which is exactly the "tune on synthetic/FSC-val, evaluate on a different domain" experiment the
   user wanted, done cleanly.

5. **Every method runs at BOTH 1 exemplar and 3 exemplars.**
   - **1 exemplar** = the product's real operating point (the UI draws one box). Our headline UX number.
   - **3 exemplars** = the published-benchmark convention. Comparable to leaderboards.
   Report both; they are different questions, not redundancy.

6. **Box-only scoring; no fabricated boxes.** Datasets whose native labels are **dots** (original
   FSC-147, crowd-counting sets, TRANCOS, cell sets) are **excluded** from box-IoU P/R rather than
   given synthesized boxes. That is why we take **FSCD-147/FSCD-LVIS** (which add real boxes), not
   raw FSC-147 dots. Count-only sets are out of scope for this phase (see Scope Fence).

7. **FSC-147/FSCD-147 must be de-duplicated on load.** Documented contamination: pixel-identical
   duplicates (159 images appearing 334×) and **11 images that leak across train↔test**. Scrub both
   before scoring or precision/recall/counts are inflated. A test asserts the leaked ids are gone.

8. **Datasets are large and gitignored, exactly like `models/`.** Add `datasets/` to `.gitignore`.
   Acquire via a new `pixi run fetch-datasets` task that records **SHA-256 + source URL + licence**
   in a provenance manifest (mirror the `fetch-models` provenance discipline). No raw dataset file is
   ever tracked in git. Respect each dataset's licence — record it, do not re-host.

9. **Literature-standard metrics, added alongside the existing ones** (EVAL-24):
   - **Detection/localization** (task-native, extend `eval/metrics.py`): Precision, Recall, F1 (exist)
     plus **COCO-style AP@[.5:.95:.05], AP50, AP75** (today `average_precision` is single-IoU 0.5 —
     generalize it to an IoU sweep; keep AP50 == the current number).
   - **Counting** (comparability): **MAE, RMSE, NAE** over predicted-vs-true instance counts (new).
   Reported per **method × dataset × {1,3 exemplars} × {val,test}**.

10. **Reuse the one-loader philosophy, don't fork it.** `eval/labels.py` reads a single `*.gt.json`
    sidecar format. Prefer **converters** that translate each research dataset's native annotations
    (COCO-style JSON for FSCD-*, CSV/`.mat` for CARPK, RPINE's boxes) into that same sidecar format
    plus a **split manifest**, so the benchmark gains research sets without a second GT reader. If a
    dataset genuinely cannot fit the sidecar, add a narrow adapter — but justify it, per Rule of Three.

11. **Reproducibility, per repo rules.** Same image + box + method + config ⇒ identical results.
    All exemplar sampling and val-carving seed from config via `np.random.default_rng(seed)`
    (NOT `cv2.setRNGSeed`, which does not affect anything here). Split membership and exemplar choice
    must be byte-stable across runs.

## Canonical References

**Code this phase extends (read before planning tasks against them):**
- `src/object_search/eval/metrics.py` — `match_predictions`, `precision_recall_f1`,
  `average_precision` (all-point AP @ IoU 0.5). EVAL-24 adds the IoU sweep + MAE/RMSE/NAE here.
- `src/object_search/eval/labels.py` — the single `*.gt.json` loader and `_GT_ROOTS`. Research
  converters emit this format; add roots/manifest discovery here.
- `src/object_search/eval/benchmark.py` — `BenchmarkConfig`, `resolve_run_set`,
  `chipset_image_ids()` / `textured_image_ids()`, `_run_one`, `_aggregate`. Add research image-id
  sources + the val/test + exemplar-count sweep dimensions.
- `conf/benchmark.yaml` — Hydra config; add research-dataset sweep entries (kept OUT of the
  model-free CI subset, which stays chipset-only).
- `src/object_search/provenance.py` + the `fetch-models` task — the SHA-256/provenance pattern to
  mirror for `fetch-datasets`.
- `.gitignore` — add `datasets/`.

**Research + external sources:** `11-RESEARCH.md` (this phase) has the full survey with links and
per-dataset splits. Also `.planning/research/PITFALLS.md` (seed/reproducibility, AP convention,
NULL-vs-zero) and `08-CONTEXT.md` (the AP-convention and abstention-is-not-zero decisions this phase
must stay consistent with).

## Specifics — dataset splits (drive the loaders/manifests)

| Dataset | Train | Val | Test | Handling |
|---|---|---|---|---|
| FSCD-147 | 3,659 | 1,286 | 1,190 | Native triple; val/test have human boxes (train boxes are pseudo). Dedup first. |
| FSCD-LVIS (unseen) | 3,959 | — | 2,242 | Carve seeded val from train; report on test. |
| RPINE | 3,925 | — | 435 | Carve seeded val from train; report on test. |
| CARPK | 989 | — | 459 | Test-only (no tuning). |
| PUCPR+ | 100 | (25) | — | Test-only cross-domain probe. |

Exemplar convention: sample exemplar box(es) from each image's GT (seeded). FSCD-* ship 3 exemplar
boxes/image — honor them for the 3-exemplar run; derive the 1-exemplar run by taking the first.

## Scope Fence

**In:** `pixi run fetch-datasets` + provenance/SHA-256 + `.gitignore`; per-dataset converters to the
`*.gt.json` sidecar format + committed split manifests; seeded val-carving; seeded exemplar sampler
(1 & 3); FSC-147 dedup; COCO AP sweep (AP/AP50/AP75) + MAE/RMSE/NAE in `eval/metrics.py`; benchmark
wiring for method × dataset × {1,3} × {val,test}; report table; `docs/eval/research-datasets.md`
(DOC-07, seed from `11-RESEARCH.md`).

**Out:** dot-only / count-only datasets (crowd counting, TRANCOS, cell sets) — no fabricated boxes
this phase. Any new search method. Training/fine-tuning any model. Re-hosting dataset images.
Changing the human-rating store or the synthetic generators. A public leaderboard submission.

## Risk Summary

- **Licences differ per dataset and some restrict redistribution.** Record licence + source in
  provenance; fetch from the official source; never commit or re-host raw images. Flag any set whose
  licence blocks even local research use and drop it rather than guess.
- **Native annotation formats vary** (COCO JSON, `.mat`, CSV). The converter is the load-bearing,
  test-worthy part: assert converted box counts and a few known images against the source.
- **Research sets cannot run in CI** (large, may need weights). Keep the model-free CI subset
  chipset-only; gate research sweeps behind `fetch-datasets`, like the full benchmark gates on
  `fetch-models`.
- **1-exemplar is genuinely harder than 3** — expect lower numbers at 1. That is a real finding to
  report honestly, not a bug to tune away.
- **AP convention drift.** Keep all-point interpolation (COCO-style, per `08-CONTEXT.md` decision 8);
  AP50 must equal the existing single-IoU-0.5 number so old and new reports reconcile.
