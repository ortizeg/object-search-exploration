# Research datasets (DOC-07)

The committed benchmark sets are **synthetic with exact ground truth by construction** (see
[`DATASETS.md`](../DATASETS.md)): they prove the NCC-vs-learned *crossover*, but they say nothing
about how the four methods do on **real, published** data, and they produce no number comparable to
the few-shot counting / detection literature. Phase 11 adds four **external research datasets** as a
held-out evaluation surface, keeps the synthetic sets as the tuning surface, and reports the
**literature's own metrics** so our numbers sit next to published work.

The project's task is unusual: **one hand-drawn exemplar box → every other instance of that same
object in the same image.** That is single-image exemplar search, not dataset-level counting. So a
dataset earns its place here only if it has **per-instance bounding boxes** (our harness scores boxes
with IoU) and **exemplar-style queries** — not by being a famous counting benchmark. Dot-only sets
are excluded rather than given fabricated boxes (see [Metrics](#metrics) and
[Scope](#scope--exclusions)).

The rendered numbers live in the [research report](../reports/research-report.html)
(`pixi run report-research`); the raw `docs/benchmark/research-results.json` it reads is gitignored
and regenerable, because the numbers depend on the licence-gated archives and, for the learned
methods, on fetched ONNX weights.

> **Status — offline fixture smoke-run.** The real dataset images are **licence-gated and not
> fetched in this repo.** Every acceptance check runs on the committed offline fixtures under
> `tests/fixtures/research/`, so the harness, the metrics, and the report table are all exercised
> end-to-end without any download. The real report regenerates via `pixi run` once a human accepts
> each licence and drops the archives (`pixi run fetch-datasets --list` prints where). **No
> real-dataset numbers are claimed or committed.**

---

## The four datasets

Each does a **distinct** job; none is redundant (D-01).

### RPINE — "Repeated Patterns IN Everywhere"

- **Purpose:** the closest match to our task — **all** repetitions in a single image are
  box-annotated, with box exemplars. The core "does it find *all* repeats in one image" set.
- **Source:** paper <https://arxiv.org/html/2508.17636> · project
  <https://chipmunk-g4.github.io/TMR/>
- **Annotation type:** per-instance bounding boxes; multiple annotators/image; box exemplars.
  Includes non-object and "nameless part" repeats, not just clean object categories.
- **Splits:** 3,925 train / — / 435 test (no official val → carve a **seeded** val from train).
- **Strengths:** exact-task fit; diverse domains (nature, textiles, architecture); multiple
  annotations per image.
- **Weaknesses / biases:** annotation subjectivity (what counts as "a repeat"); hard on tiny
  exemplars and highly-textured patterns — which is exactly the difficulty we want to measure.

### FSCD-147 — box-annotated FSC-147

- **Purpose:** category diversity + comparability to published few-shot counting/detection numbers.
- **Source:** VinAI <https://research.vinai.io/few-shot-object-counting-and-detection/> ·
  Counting-DETR <https://github.com/VinAIResearch/Counting-DETR> · paper
  <https://arxiv.org/pdf/2207.10988>
- **Annotation type:** FSC-147 extended with **bounding boxes for every object** in val/test; 3
  exemplar boxes/image. Train boxes are pseudo-labels; **val/test boxes are human** (D-06: only the
  human box splits are scored).
- **Splits:** 3,659 train / 1,286 val / 1,190 test (89/29/29 disjoint classes, open-set). Native
  triple — used as-is.
- **Strengths:** 147 classes, extreme count range (7–3,731 objects/image), open-set test classes,
  leaderboard comparability.
- **Weaknesses / biases:** one labelled class/image → never tests distractor rejection; documented
  contamination — **159 images appear as 334 pixel-identical duplicates** and **11 images leak
  train↔test** (<https://arxiv.org/pdf/2409.15953>). Both are **de-duplicated on load** before any
  scoring (D-07), or precision/recall/counts inflate.

### FSCD-LVIS (unseen split) — multi-class crowded scenes

- **Purpose:** the **distractor-rejection** stress — several repeated classes share an image, so it
  measures whether a method finds the *right* object, the gap FSC-147 leaves.
- **Source:** Counting-DETR <https://github.com/VinAIResearch/Counting-DETR> · split figures
  <https://arxiv.org/html/2511.08048>
- **Annotation type:** box exemplars + per-instance boxes, 377 LVIS classes. Only the
  exemplar-category boxes are scored as GT; other-category boxes are the **distractors** and are
  intentionally excluded, so returning one scores as a false positive.
- **Splits:** 3,959 train / — / 2,242 test (unseen protocol, no val → carve a **seeded** val from
  train). The unseen split is the standard Counting-DETR generalization eval.
- **Strengths:** real multi-class clutter.
- **Weaknesses / biases:** noisier labels; lower ceiling numbers.

### CARPK (+ PUCPR+) — dense cars, cross-domain probe

- **Purpose:** **test-only** cross-domain generalization — dense, near-identical instances at scale
  in a domain nothing was tuned on (D-04).
- **Source:** ICCV'17
  <https://openaccess.thecvf.com/content_ICCV_2017/papers/Hsieh_Drone-Based_Object_Counting_ICCV_2017_paper.pdf>
  · project (LPN) <https://lafi.github.io/LPN/>
- **Annotation type:** per-car bounding boxes (~90k cars). CARPK is drone @ 40 m, 720×1280, 4
  parking lots; PUCPR+ is a fixed slanted building-camera view.
- **Splits:** CARPK 989 train / — / 459 test; PUCPR+ ~100 / (25) / — . **Both test-only** — no
  tuning on them at all.
- **Strengths:** boxes for every instance; the standard cross-domain probe in the class-agnostic
  counting literature.
- **Weaknesses / biases:** one appearance (cars), one viewpoint regime — narrow; a probe, not a
  primary diversity signal.

| Dataset | Train | Val | Test | Val strategy | Role |
|---|---|---|---|---|---|
| FSCD-147 | 3,659 | 1,286 | 1,190 | native (dedup first) | diversity + comparability |
| FSCD-LVIS (unseen) | 3,959 | — | 2,242 | seeded carve from train | distractor rejection |
| RPINE | 3,925 | — | 435 | seeded carve from train | single-image all-repeats (task fit) |
| CARPK | 989 | — | 459 | test-only | cross-domain generalization |
| PUCPR+ | ~100 | (25) | — | test-only | cross-domain generalization |

---

## Metrics

Every cell reports the literature's own columns, added **alongside** the harness's existing
precision/recall/F1 (EVAL-24, D-09). Rates render as percentages; abstentions render as **n/a**,
never `0` (a method that honestly returns nothing has *undefined* precision — scoring it `0` would
punish an abstention as a wrong answer).

**Detection / localization** (FSCD-147 / Counting-DETR standard):

- **Precision / Recall / F1** — greedy IoU matching, each GT matched at most once (the EVAL-16
  duplicate rule: two boxes on one instance are 1 TP + 1 FP, never 2 TP).
- **AP** — COCO-style average precision over the IoU sweep **[0.50, 0.55, …, 0.95]** (ten
  thresholds), all-point interpolation.
- **AP50** — AP at IoU 0.5. **AP75** — AP at IoU 0.75.
- **Reconciling note:** `AP50` is computed by evaluating AP at the single IoU 0.5, so it is *exactly*
  the project's pre-existing single-IoU-0.5 AP number — old and new reports reconcile with no drift.

**Counting** (FSC-147 / class-agnostic-counting standard), over predicted-vs-true instance counts:

- **MAE** = mean(|pred − true|) — the primary count error.
- **RMSE** = sqrt(mean((pred − true)²)) — penalises large misses more.
- **NAE** = mean(|pred − true| / true) — normalised; computed **only over images with `true > 0`**
  (a zero true count has no denominator and is skipped explicitly, never fabricated).

Reported per **method × dataset × {1, 3 exemplars} × {val, test}**.

---

## Protocol

**Tune on val → freeze test → report on test.** This is the field-standard protocol and yields
leaderboard-comparable numbers. "Tuning" here is config/threshold sweeps — the four methods are
**training-free / config-tuned**, not gradient-trained — so it is cheap and reproducible (D-02).

- **Seeded val carve.** Where no official val split exists (**RPINE** and **FSCD-LVIS unseen**), a
  val slice is carved from train **deterministically**, seeded from config via
  `np.random.default_rng(seed)` (never `cv2.setRNGSeed`, which controls nothing here — D-11); test is
  never touched. FSCD-147 ships a native train/val/test triple and is used as-is.
- **CARPK / PUCPR+ are test-only.** No tuning on them at all (D-04). In the class-agnostic-counting
  literature they exist to measure **cross-domain generalization** from whatever was tuned elsewhere
  — exactly the "tune elsewhere, evaluate on a different domain" experiment. The sweep therefore
  emits **zero val cells** for them.

### 1 vs 3 exemplars, and how a method is run at 3 (k-shot late fusion)

Every method is scored at **both** operating points, and both are reported — they are different
questions, not redundancy (D-05):

- **1 exemplar** — the product's real operating point (the UI draws one box). The headline UX number.
- **3 exemplars** — the published-benchmark convention. Comparable to the leaderboards.

The methods each take a **single** exemplar box; none knows how to consume three. The 3-exemplar
number is produced by **k-shot late fusion in the eval layer**: run the single-exemplar method
**once per exemplar**, **union** the resulting matches *and* sub-threshold candidates across the runs,
then **NMS-dedupe** overlapping detections (deterministic tie-break `(-score, y, x)`; each GT still
matched at most once). The 1-exemplar run is the k=1 special case of this same runner, so 1 and 3
exemplars share one code path. Crucially, **`SearchFn` and all four method files are unchanged** —
`@register_method` remains the only indirection.

The sampled exemplar boxes **remain in the ground truth and are scored like any other instance** (a
correct method re-detects its own exemplar; the FSC-147/FSCD count convention counts exemplars). So
the recall denominator (`len(gt.boxes)`) is **identical** between the 1- and 3-exemplar runs, which is
what makes the 1-vs-3 numbers directly comparable — the exact comparison D-05 exists to make. Native
exemplar boxes are honoured where a dataset ships them (FSCD-* provide three); otherwise the exemplars
are a **seeded** draw from the GT, and the 1-exemplar set is always the first of the 3-exemplar set.

> **1-exemplar is genuinely harder than 3** — expect lower numbers at 1. That is a real finding to
> report honestly, not a bug to tune away.

## Scope — exclusions

**Box-only scoring; no fabricated boxes** (D-06). Datasets whose native labels are **dots** — original
FSC-147, crowd-counting sets (ShanghaiTech, UCF-QNRF, NWPU-Crowd, JHU-CROWD++), TRANCOS, cell sets —
are **excluded** from box-IoU precision/recall rather than given synthesized boxes. That is why we take
**FSCD-147 / FSCD-LVIS** (which add real human boxes), not raw FSC-147 dots. Count-only sets can be
revisited later as a count-MAE-only stress; they are out of scope this phase.

Raw dataset images are large and licence-restricted: they are **gitignored** (`/datasets/`), fetched
via `pixi run fetch-datasets`, and their SHA-256 + source URL + licence are recorded in a provenance
manifest (D-08). No raw dataset file is ever tracked in git, and no dataset is re-hosted.
