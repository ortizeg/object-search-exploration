# Limitations

Stated plainly, without softening. The whole value of this project is saying which method
actually works and where the harness falls short; a flattering account would defeat it. Read
this before drawing conclusions from the benchmark or before sharing the repo.

## Governance

### INFRA-07 — branch protection is only partially satisfied

Branch protection on `main` is **not enforced**, and cannot be on this repo as it stands.
Both the branch-protection API and the rulesets API return:

```
403 Upgrade to GitHub Pro or make this repository public
```

GitHub Free does not offer protected branches on **private** repositories, and `main` reports
`"protected": false`. CI (`lint`, `format-check`, `typecheck`, `test`) **does** run on every pull
request, and every phase has in fact gone through a PR — but the *server-side enforcement* that a
human could not bypass does not exist. Since the single user is also the only person with push
access, the practical gap is small, but it is a real gap and INFRA-07 is **not** ticked as fully
done. Closing it requires an owner decision — upgrade to GitHub Pro, or make the repo public
(which has the AGPL dimension below, so it must not be done implicitly). The ready-to-apply
protection JSON is preserved in `.planning/phases/01-foundation/01-01-SUMMARY.md`.

## Models that did not ship as planned

### MobileSAM was not built as a second proposal backend

Method 5 ships with **FastSAM as its only working proposal backend**. MobileSAM was intended as a
second backend but did not ship, for a concrete reason: the ONNX SAM decoder accepts **one prompt
per call**, so "everything mode" means roughly **1024 sequential decoder calls** plus a
hand-ported `SamAutomaticMaskGenerator` (grid-prompt sampling, mask NMS, stability scoring). That
is a phase of work, not a config-switchable backend. The proposal stage was therefore built behind
a `ProposalBackend` protocol (`src/object_search/search/proposals.py`) so MobileSAM can be added
later without restructuring — but as of Milestone 1 the protocol has a single implementation.
This is a deviation from the brief, recorded here and in the robustness backlog.

Relatedly, `awarebayes/MobileSamONNX` (referenced in the source research) is a **HOLD**: 0 stars,
a 4-day project untouched for 14 months, single author, when first-party export scripts and a
better-provenanced MIT artifact (`Acly/MobileSAM`) exist.

## Licensing constraints on the weights

The project **source is MIT**, but two model weights are not, and both are gitignored. These
constrain how the repo may be shared:

### FastSAM is AGPL-3.0, and the exported `.onnx` embeds that licence

The FastSAM export used by Method 5 carries an **AGPL-3.0** licence string embedded in the `.onnx`
file itself. "Export-time-only dependency" protects the runtime dependency graph but **not the
weights**. Private, local use triggers nothing. But **publishing this repo publicly, or
network-exposing the FastAPI app, would fire AGPL §13** (the network-use / "affero" clause),
requiring the complete corresponding source of the whole service under AGPL. The repo is private
and the app is local-only, so Milestone 1 is fine — but this is the reason "make the repo public"
is not a free action.

### SuperPoint weights are non-commercial, research-only

The SuperPoint weights for `sparse-geo`'s learned backend are **MagicLeap non-commercial
research-only**, and the DERIVATIVES clause covers the exported ONNX file. Acceptable as scoped
because the weights are gitignored and never redistributed (INFRA-11), but they **must never be
redistributed** and cannot be used commercially. DISK / ALIKED are the permissive-licence swaps if
that ever needs to change.

## Known UI polish item (does not affect correctness)

From `.planning/POLISH-BACKLOG.md`:

- **The canvas `#stage` overflows the viewport height.** Verified in a real browser: the canvas
  renders taller than the viewport (e.g. 759×1529 CSS px for an 800×600 image) because the stage
  element stretches to a flex column taller than the viewport, so the fit-scaled image is
  letterboxed with large empty bands and extends below the fold. **The coordinate transform is
  unaffected** — round-trip draws map to correct image pixels to 5.68e-14 px with separate
  scaleX/scaleY — so this is layout polish only, not a data-correctness bug. Fix: constrain the
  stage to the available viewport height. A dedicated `design-review` / `design-deslop` pass would
  catch it.
- **The `/stats` latency panel shows total percentiles, not the preprocess/inference/postprocess
  split.** The three-way breakdown *is* stored, so the data exists; only the dashboard aggregation
  is pooled to a total. UI-06 ("latency percentiles") is satisfied; the split is an enhancement.

## What the benchmark actually found

Real numbers over the committed 12-image demo set (chipset repeats + scale/clutter synthetics,
IoU 0.5). Full tables and charts: [`benchmark/results.md`](benchmark/results.md).

| method | precision | recall | F1 | mean AP | p50 latency |
| --- | --- | --- | --- | --- | --- |
| `ncc` | 0.913 | 0.922 | 0.918 | 0.484 | 238 ms |
| `sparse-geo` | 0.833 | 0.097 | 0.174 | 0.083 | 76 ms |
| `dino-dense` | 0.276 | 0.078 | 0.121 | 0.190 | 2259 ms |
| `propose-retrieve` | 0.748 | 0.951 | 0.838 | 0.635 | 291 ms |

- **`dino-dense` underperforms on this set** — F1 **0.121**, the weakest of the four, and the
  slowest (p50 **2.26 s**, up to ~5 s on the largest canvases). The stride-14 DINOv2 token grid is
  too coarse to localise the small chips: the similarity map cannot separate adjacent tiny
  instances, so recall craters (0.078). It is a poor fit for the small-repeated-instance regime the
  chipset represents, and the benchmark says so. The robustness backlog records the fixes
  (sliding-window inference, FeatUp upsampling, SAM box refinement).

- **`sparse-geo` abstains on 11 of 12 images** — pooled recall **0.097**. The chips are
  near-identical and **low-texture**, so the exemplar crop yields fewer than the 20 SIFT keypoints
  the method requires, and it **correctly declines rather than guess** (its precision when it does
  fire is 0.833). This is the **NCC-vs-sparse-geo crossover** the literature predicts and CONTEXT
  calls out as an expected finding: keypoint matching plus Hough voting has insufficient
  discriminative power precisely when instances are small and near-identical — exactly where NCC is
  strongest. It is made visible in `crossover_by_scale.png`, not averaged away.

- **`ncc` wins the fixed-scale regime decisively but does not generalise.** Fixed-scale recall
  **0.989**; varied-scale recall **0.30**. It has no scale/rotation invariance beyond its
  brute-force pyramid + rotation bank, and its cost grows steeply with canvas size (**5.7 s** at
  6000×4000) because the correlation is over the full scene at every pyramid level.

- **`propose-retrieve` is the strongest general retriever** — best AP (**0.635**) and best recall
  (**0.951**) — at a roughly canvas-independent latency (proposals dominate, not correlation).
  Its precision (0.748) is the cost: FastSAM over-proposes, so some retrieved regions are
  false positives.

None of these is presented as a solved problem. The harness exists so an ML practitioner can pick
the right method per image, change a config, re-run the chipset, and watch the number move.

## Human-rating scoreboard is empty

The subjective layer — thumbs ratings, per-match verdicts, paired comparisons, the Wilson
intervals and the Bradley-Terry ranking — is wired end to end and tested on synthetic win/loss
records, but **starts with `n = 0` real ratings** because rating is a manual activity. The thumbs
chart renders an honest empty-state panel until runs are rated in the UI. Any ranking shown before
then would be built on no data, which is the exact false-certainty the Wilson lower-bound and the
Bradley-Terry regularisation exist to prevent.
