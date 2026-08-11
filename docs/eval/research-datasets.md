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
[Scope](#scope-exclusions)).

The rendered numbers live in the [research report](../reports/research-report.html)
(`pixi run report-research`); the raw `docs/benchmark/research-results.json` it reads is gitignored
and regenerable, because the numbers depend on the licence-gated archives and, for the learned
methods, on fetched ONNX weights.

> **Status.** The real dataset **images and raw per-image results stay gitignored** (manual /
> licence-gated, or an export drop for floor plans); every acceptance check runs on the committed
> offline fixtures under `tests/fixtures/research/`, so the harness, metrics, tuning pass, and report
> table are exercised end-to-end without any download. Numbers regenerate via `pixi run` once a human
> supplies each dataset (`pixi run fetch-datasets --list` prints where each drop goes). **Measured
> findings from a real floor-plan run** (which method wins, per-symbol-size recall, tuning gains) are
> recorded in prose in [floorplans-findings.md](floorplans-findings.md) — metrics only, no images or
> raw data.

---

## The research datasets

Each does a **distinct** job; none is redundant (D-01). The first four are Phase 11's held-out
surface; floor plans (below) was added later as the **target-domain** set.

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

### FSCD-LVIS (unseen split) — LVIS-category counting, cross-domain generalization

- **Purpose (intended):** the **distractor-rejection** stress — several classes share an image, so a
  method must find the *right* one. **Verified caveat:** in the mirrored **unseen** split every image
  carries exactly **one** category (2,242/2,242 test images single-class), i.e. **no distractors as
  delivered** — so the unseen release exercises open-category *generalization*, not distractor
  rejection. Multi-class clutter lives in the SEEN split (`instances_*.json`) — a future variant.
- **Source:** Counting-DETR <https://github.com/VinAIResearch/Counting-DETR> · split figures
  <https://arxiv.org/html/2511.08048> · HF `ChipmunkG4/FSCD-147_FSCD-LVIS_temp`
- **Annotation type:** COCO `unseen_instances_{train,test}.json` (xywh boxes, **single** target
  category per image); no explicit exemplar boxes, so the harness samples exemplars from GT
  (RPINE-style, via `convert_rpine`).
- **Splits:** 3,959 train / — / 2,242 test (unseen protocol, no val → carve a **seeded** val from
  train). The unseen split is the standard Counting-DETR generalization eval.
- **Strengths:** open-category generalization; LVIS visual diversity.
- **Weaknesses / biases:** single-class as delivered (no distractor test); no native exemplars.

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

### Floor plans (Roboflow floor-plans-500) — the target domain

- **Purpose:** the **target domain**. Architectural floor plans where the product's framing is
  literal — draw one `door` (or `window`), find every other instance in the *same* plan. This is the
  one research set whose images look like the intended application, so it is where "which method
  should we ship" is actually decided.
- **Source:** Roboflow Universe <https://universe.roboflow.com/university-y9nbi/floor-plans-500>,
  exported in COCO format.
- **Licence:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) — confirmed 2026-08-10
  directly against the creator's (`university-y9nbi`) own Roboflow listing (a third-party Kaggle
  mirror's "MIT" tag was checked and discounted — a re-uploader cannot unilaterally relicense
  someone else's data). Permits redistributing derivatives with attribution; this is why the
  TP/FP/FN overlay gallery in [floorplans-findings.md](floorplans-findings.md) is committed rather
  than gitignored. Raw dataset images themselves stay gitignored regardless (`/datasets/`,
  `requires_manual` — see below), not because of licence doubt but because they are large and this
  repo never re-hosts a full third-party dataset, only attributed derivative excerpts.
- **Annotation type:** per-symbol bounding boxes. The export carries `bathroom` / `door` /
  `perimeter` / `stairs` / `window`; we use **`door`** and **`window`** — dense repeated stamped
  symbols present in every plan (door ~9/plan, window ~7–8/plan). `perimeter` / `bathroom` /
  `stairs` are region-ish or one-per-room and are **not** exemplar-search targets, so they are
  excluded.
- **Per-class single-class datasets.** The harness ground truth is single-class, so the multi-class
  export is converted **once per class** into two datasets — `floorplans-door` and
  `floorplans-window` — over the same plans. An exemplar door is then scored against exactly the
  doors: recall's denominator is the door count, never doors + windows.
- **Splits:** native `train 197 / valid 56 / test 28`. **Train is intentionally not converted** —
  the exemplar-search methods do no training — so the manifest's train is empty; **val** (native)
  tunes and **test** is the frozen surface.
- **Strengths:** real target-domain imagery; dense repeated instances; native val, so no carve.
- **Weaknesses / biases:** small dataset (28 test plans → watch sample noise); symbols are small
  relative to the full plan, which stresses the min/max box-area gates of some methods.

| Dataset | Train | Val | Test | Val strategy | Role |
|---|---|---|---|---|---|
| FSCD-147 | 3,659 | 1,286 | 1,190 | native (dedup first) | diversity + comparability |
| FSCD-LVIS (unseen) | 3,959 | — | 2,242 | seeded carve from train | distractor rejection |
| RPINE | 3,925 | — | 435 | seeded carve from train | single-image all-repeats (task fit) |
| CARPK | 989 | — | 459 | test-only | cross-domain generalization |
| PUCPR+ | ~100 | (25) | — | test-only | cross-domain generalization |
| floorplans-door | — | 56 | 28 | native | **target domain** (doors) |
| floorplans-window | — | 56 | 28 | native | **target domain** (windows) |

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
- **Floor plans ship a native val**, so tuning happens on it directly (no carve, no contamination),
  and the domain-tuning pass below uses it.

### Domain threshold tuning (`pixi run tune-floorplans`)

The default sweep scores every method at its **default** config — "out of the box on floor plans".
The tuning pass answers the shipping question: **how good is each method once its acceptance
threshold is adapted to this domain, and how much adaptation did it need?**

1. **Tune on val.** For each method, sweep its single acceptance knob over a small explicit grid on
   `floorplans-{door,window}` **val** and pick the config maximizing **F1 @ IoU 0.5** — the
   operating-point metric the product cares about (find all the doors without junk). Each method
   gates acceptance differently, so the knob is method-specific and hand-listed in
   `src/object_search/eval/tuning.py`:

   | Method | Tuned knob |
   |---|---|
   | `ncc`, `mosse`, `dino-dense`, `owlv2-oneshot` | `retain_frac` (calibrated score floor) |
   | `sparse-geo` | `min_inliers` (geometric acceptance) |
   | `propose-retrieve` | `similarity_floor` (retrieval cosine floor) |

2. **Freeze → report tuned-vs-default on test.** The frozen config and the default config are each
   scored once on **test**. Reporting both side by side shows which method wins on floor plans *and*
   how much domain tuning each needed (a method that barely moves is robust; one that jumps was
   mis-calibrated for this domain). Tuning **never reads test**, and the tuned config is always an
   instance of the method's own frozen `config_model`, so no method file is touched — the tuned
   config feeds through the additive `config` param on `run_research_benchmark`.

Output: `docs/benchmark/floorplans-{door,window}-tuning-results.json` (gitignored, regenerable).

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

## Obtaining the datasets

`fetch-datasets` has two source kinds (see `object_search.eval.datasets`).

### HuggingFace (automatic) — RPINE, FSCD-147, FSCD-LVIS

These live on ungated HF dataset repos, so `pixi run fetch-datasets` downloads them anonymously (via
`huggingface_hub`, with the Xet backend disabled) and **normalizes-in-fetch**: it reshapes the real HF
layout into the tree each existing converter already expects and runs the converter unchanged. Verified
against the real data (counts match the canonical datasets):

| Dataset | HF repo (dataset) | Real layout | Verified counts |
|---|---|---|---|
| RPINE | `ChipmunkG4/RPINE` (the TMR authors) | `<split>/images/*.jpg` + `<split>/labels/*.txt` (`x1 y1 x2 y2` px GT) + `<split>/exemplars.json` (queries); splits train/val | 3925 train imgs/labels 1:1; converter emits one sidecar/image |
| FSCD-147 | `ChipmunkG4/FSCD-147_FSCD-LVIS_temp` → `FSCD_147.zip` | self-contained: `FSC147/images_384_VarV2/*.jpg` + COCO `instances_{val,test}.json` (xywh boxes) + `annotation_FSC147_384.json` (exemplars) | **1286 val + 1190 test** sidecars — exact canonical match |
| FSCD-LVIS (unseen) | same repo → `FSCD_LVIS.zip` (6.3 GB) | `FSCD_LVIS/images/*` + COCO `unseen_instances_{train,test}.json` (single-class xywh); normalized to `convert_rpine` | **2242 test** sidecars — canonical match |

Notes:
- **FSC-147 needs no separate entry** — its images are bundled inside `FSCD_147.zip`.
- **RPINE ships real query exemplars** (`exemplars.json`); the converter currently samples exemplars
  from GT — wiring the real exemplars through is a documented follow-up.
- **Rate-limiting / token.** Anonymous HF downloads are IP-rate-limited (429). For the large pulls
  (FSCD-LVIS, all of RPINE) set an `HF_TOKEN` in the environment (`huggingface-cli login`, or export
  `HF_TOKEN`) before `pixi run fetch-datasets`; `_hf_download` honours it and degrades gracefully
  (logs + returns `None`, never crashes the sweep) when a download fails.

### Manual (licence-gated drop) — CARPK, PUCPR+

CARPK/PUCPR+ have **no clean HuggingFace source** (the one CARPK repo is a third-party private Kaggle
re-mirror in COCO format whose own notice asks it not be reused). They stay `requires_manual`: accept
the licence, obtain the archive yourself, and drop it (or an extracted `Images/` + `Annotations/` tree)
at `datasets/_incoming/<carpk|pucpr_plus>/`, then re-run `pixi run fetch-datasets`. Routes:
- **Kaggle** — `kambojharyana/carpk-coco` (or the original CARPK upload), with your own Kaggle account
  and after accepting the dataset's terms. Note it is COCO-reformatted, not the native `Annotations/`
  + `Images/` layout the converter reads.
- **Official** — the NTU CARPK/PUCPR+ terms-of-use page linked from https://lafi.github.io/LPN/ (request
  form). This is the native format the converter expects.

### Manual (COCO export drop) — floor plans

`floorplans-door` / `floorplans-window` also stay `requires_manual`, but for a different reason: the
data is exported from [Roboflow Universe](https://universe.roboflow.com/university-y9nbi/floor-plans-500),
not licence-gated behind a request form. Export the dataset in **COCO** format, then drop the extracted
`train/valid/test` tree at `datasets/_incoming/floorplans/` and re-run `pixi run fetch-datasets`. Both
class keys read the **one** dropped tree (converted once per class). On a GPU box, scp the export up
first — `scripts/gpu_bench.sh` documents the exact command and then includes floor plans in the sweep
and the `tune-floorplans` pass automatically.

## Improving these results

The first real-dataset run (T4, N=20/dataset/split — see `docs/reports/research-report.html`) is weak
across the board (F1 0.04–0.35). The per-method precision/recall shape says *why*, and points at the
fixes. These are ordered by expected payoff.

### 1. Actually run the tune-on-val protocol (the biggest lever)

The run used each method's **default config** — thresholds and calibration picked on the *synthetic*
chip/textured sets — and reported straight on test. The whole `val → test` protocol this harness was
built for (sweep each method's threshold/config on `val`, freeze, report on `test`) was **never
exercised**. The P/R shape shows how much that costs, because almost every method is mis-thresholded
for real images:

| Method | P | R | Read |
|---|---|---|---|
| propose-retrieve | 0.71 | 0.25 | too **conservative** — accurate when it fires, misses most. **Lower** the retrieval threshold. |
| sparse-geo | 0.80 | 0.04 | extremely conservative — **abstains on ~8 of 20 images** (the ≥20-keypoint gate) and verifies too strictly. |
| ncc | 0.37 | 0.24 | mis-scaled threshold; best-behaved but under-recalling. |
| owlv2-oneshot | 0.11 | 0.41 | the opposite — **over-detects** (threshold too low). **Raise** the score threshold. |
| mosse | 0.16 | 0.14 | low both; correlation-peak threshold untuned for real texture. |
| dino-dense | 0.08 | 0.04 | **both** low → the dense-similarity threshold/decoding is genuinely miscalibrated for real scenes; not just a threshold nudge (see §3). |

A per-dataset `val` sweep of the single threshold each method already exposes (`config_model`) —
lower it for the conservative methods, raise it for owlv2 — is the highest-payoff change and is exactly
what the harness supports; it just needs a driver that grid-searches `val` and writes the winning config
per (method, dataset).

### 2. Kill the sample noise — larger N (or the full sweep)

N=20 images against datasets with hundreds of instances per image is directional at best; a single hard
image swings a cell. The classical methods are CPU-bound over the large real images (why the full sweep
is ~2 days on one T4), so scaling means either a bigger/multi-GPU box, capping the classical methods to
a subset while the learned methods run full, or simply leaving `scripts/gpu_bench.sh` to run overnight at
a larger N.

### 3. Fix `dino-dense` specifically (P=0.08, R=0.04)

Both metrics near zero means the boxes are *wrong*, not merely missing — a decoding/calibration failure
on real scenes, not a threshold nudge. Likely culprits: the cosine-similarity threshold and the
connected-components → box step tuned for clean synthetic lattices, and the mean-pooled crop prototype
being unrepresentative on cluttered real crops. Inspect the similarity-map diagnostics on a few real
images; recalibrate the threshold on `val`; consider multi-scale scene inference.

### 4. Use the datasets' native exemplars

RPINE ships real **query exemplars** (`exemplars.json`) and FSCD-* ship 3 exemplar boxes/image; the run
**sampled exemplars from GT** instead. Real exemplars are the intended query and should improve every
retrieval/template method — wiring them through (the `convert_rpine` TODO) is a clean, isolated win.

### 5. Give `owlv2` a fair GPU run

`owlv2-base` at 960px OOMs the T4's 16 GB, so it ran on CPU (18 s/img). Re-export at a smaller `imgsz`
(fits 16 GB), or run on a ≥24 GB GPU, to get a real GPU latency and let its recall-heavy behaviour be
threshold-tuned into usable precision.

### 6. Report per-slice, not just pooled

The pooled F1 hides where each method wins. Slicing by instance count, object scale, and clutter (as the
synthetic benchmark already does) would turn "everything is bad" into "method X holds up on sparse
scenes, collapses on dense" — the actionable signal.
