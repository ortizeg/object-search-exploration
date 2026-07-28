# Phase 11 Research — Counting/detection datasets & metrics for exemplar search

**Method:** live web survey (2026-07-26) of the class-agnostic counting / few-shot detection
literature, filtered for our task: *one exemplar box → every instance of that object in the same
image*, scored with **per-instance boxes**. Not a `gsd-phase-researcher` run — seeded from the
session's survey with sources inline.

## Task framing (why "counting datasets" ≠ what we need)

Most "counting" datasets provide **dot annotations + a global count** and are scored with count
error (MAE/RMSE). Our harness scores **boxes** with IoU (precision/recall/F1/AP). So the datasets
that matter are the ones with **per-instance bounding boxes** and **exemplar-style queries**. That
selects the four below and excludes the dot-only sets (kept as out-of-scope; see §Excluded).

## Anchor survey

- **A Survey on Class-Agnostic Counting (2025)** — taxonomy of ~30 methods; standardizes on
  **FSC-147 + CARPK** as the gold evaluation pair. https://arxiv.org/abs/2501.19184

## Selected datasets (Tier 1 — per-instance boxes + exemplars)

### RPINE — "Repeated Patterns IN Everywhere"
- **What:** 4,362 images (**3,925 train / 435 test**; no official val). Box exemplars, and **all
  repetitions in each image are box-annotated** (3 annotators/image), including non-object and
  "nameless part" repeats — not just clean object categories.
- **For us:** the single closest match — the task literally is "detect all instances of a given
  pattern from an input image." Multi-pattern per image, low semantic bias.
- **Strengths:** exact-task fit; diverse domains (nature, textiles, architecture); multiple
  annotations per image.
- **Weaknesses:** annotation subjectivity (what counts as "a repeat"); hard on tiny exemplars and
  highly-textured patterns — which is the difficulty we want.
- **Links:** paper/HTML https://arxiv.org/html/2508.17636 · project https://chipmunk-g4.github.io/TMR/

### FSCD-147 — box-annotated FSC-147
- **What:** FSC-147 (**3,659 train / 1,286 val / 1,190 test**, 89/29/29 disjoint classes, open-set)
  extended with **bounding boxes for every object** in val/test; 3 exemplar boxes/image; 7–3,731
  objects/image. Train boxes are pseudo-labels; **val/test boxes are human**.
- **For us:** category diversity + comparability to published few-shot counting/detection numbers.
- **Strengths:** 147 classes, extreme count range, open-set test classes, leaderboard comparability.
- **Weaknesses (must handle):** one labelled class/image → never tests distractor rejection;
  **159 images appear as 334 pixel-identical duplicates**, and **11 images leak train↔test** →
  dedup before use. (Duplication/leakage documented in "Mind the Prompt", https://arxiv.org/pdf/2409.15953)
- **Links:** VinAI https://research.vinai.io/few-shot-object-counting-and-detection/ ·
  Counting-DETR (ECCV'22) https://github.com/VinAIResearch/Counting-DETR ·
  paper https://arxiv.org/pdf/2207.10988

### FSCD-LVIS — multi-class crowded scenes
- **What:** ~6,196 images, 377 classes from LVIS, box exemplars + boxes. Two protocols:
  **seen split** 4,000 / 1,181 / 1,014 (has val); **unseen split** 3,959 train / 2,242 test
  (no val — the standard Counting-DETR generalization eval). Use **unseen** for the headline number;
  carve a seeded val from train.
- **For us:** the **distractor-rejection** test — multiple repeated classes share an image, so it
  measures whether a method finds the *right* object, the gap FSC-147 leaves.
- **Strengths:** real multi-class clutter.
- **Weaknesses:** noisier labels; lower ceiling numbers.
- **Links:** Counting-DETR https://github.com/VinAIResearch/Counting-DETR ·
  split figures https://arxiv.org/html/2511.08048

### CARPK (+ PUCPR+) — dense cars, cross-domain probe
- **What:** CARPK **989 train / 459 test**, ~90k cars, per-car boxes, 4 parking lots, drone @ 40 m,
  720×1280. PUCPR+ ~16k cars, fixed slanted building-camera view (~100 train / 25 val in some
  reports). Single-class, dense, near-identical instances.
- **For us:** **test-only** cross-domain generalization; dense small repeats at scale.
- **Strengths:** boxes for every instance; the standard cross-domain probe in the CAC literature.
- **Weaknesses:** one appearance (cars), one viewpoint regime — narrow; use as a probe, not a
  primary diversity signal.
- **Links:** ICCV'17 https://openaccess.thecvf.com/content_ICCV_2017/papers/Hsieh_Drone-Based_Object_Counting_ICCV_2017_paper.pdf ·
  project (LPN) https://lafi.github.io/LPN/

## Metrics used in the literature (drives EVAL-24)

**Counting (FSC-147 / class-agnostic-counting standard):**
- **MAE** = mean(|pred_count − true_count|) — primary.
- **RMSE** = sqrt(mean((pred_count − true_count)²)) — primary.
- **NAE** = mean(|pred − true| / true) — normalized, secondary. (SRE sometimes reported.)
- Sources: https://arxiv.org/html/2403.01418v2 · https://arxiv.org/pdf/2502.10677

**Detection (FSCD-147 / Counting-DETR standard):**
- **AP** = COCO-style average precision over **IoU 0.5:0.95 step 0.05**, all-point interpolation.
- **AP50** = AP at IoU 0.5 · **AP75** = AP at IoU 0.75.
- Source: https://arxiv.org/pdf/2207.10988 (FSCD / Counting-DETR reports MAE, RMSE for counting and
  AP, AP50, AP75 for detection).

**Our harness today** (`eval/metrics.py`) computes P/R/F1 and all-point AP at a **single** IoU 0.5.
EVAL-24 = generalize AP to the IoU sweep (AP/AP50/AP75, keep AP50 == today's number) and add
MAE/RMSE/NAE. Keep the abstention-is-None and EVAL-16 duplicate rules intact.

## Excluded this phase (dot-only / count-only — no boxes to IoU-score)

Kept out to avoid fabricating boxes from dots; can revisit as count-MAE-only stress later.
- Crowd counting: ShanghaiTech, UCF-QNRF, NWPU-Crowd, JHU-CROWD++ (dots; heads overlap).
- TRANCOS (traffic, dots + ROI); RSOC (remote sensing; mixed dots/boxes); VGG Cells / MBM (dots).
- Original **FSC-147** (dots only) — we take **FSCD-147** for the boxes instead.

## Split/handling summary (for loaders + manifests)

| Dataset | Train | Val | Test | Val strategy | Role |
|---|---|---|---|---|---|
| FSCD-147 | 3,659 | 1,286 | 1,190 | native | diversity + comparability (dedup first) |
| FSCD-LVIS (unseen) | 3,959 | — | 2,242 | seeded carve from train | distractor rejection |
| RPINE | 3,925 | — | 435 | seeded carve from train | single-image all-repeats (task fit) |
| CARPK | 989 | — | 459 | test-only | cross-domain generalization |
| PUCPR+ | 100 | (25) | — | test-only | cross-domain generalization |

## RESEARCH COMPLETE
