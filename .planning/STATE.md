---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
current_phase: 8
current_phase_name: Evaluation & docs
status: executing
stopped_at: Completed 09-02-PLAN.md (Milestone 2 complete)
last_updated: "2026-07-25T12:35:00.000Z"
last_activity: 2026-07-25
last_activity_desc: "Executed 09-02: schema-driven second UI exploration mode, presence-driven marker overlay, marker demo assets + byte-identical sample gallery, docs flipped to built. Milestone 2 complete (M2-05). Four gates green, 93% coverage. PR #18 open. Human-verify (browser marker flow) pending an orchestrator run; FastSAM CoreML-EP failure on this host deferred (CPU provider works)."
progress:
  total_phases: 9
  completed_phases: 8
  total_plans: 18
  completed_plans: 18
deviation_branch_protection: INFRA-07 partial — branch protection unavailable on a free private repo (403 'Upgrade to GitHub Pro or make this repository public'). Skipped by decision; CI on PRs works, server-side enforcement does not. Not a blocker for any other work.
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-07-24)

**Core value:** Given one hand-drawn exemplar box, return all matching instances in the
image — through any of four interchangeable methods — and accumulate enough evidence to say
which method actually works, on which kind of image, and at what latency.
**Current focus:** Phase 1 — Foundation

## Current Position

Phase: 1 of 8 (Foundation)
Plan: 2 of 2 in current phase
Status: Both Phase 1 plans executed; INFRA-07 (branch protection) deferred — private-repo 403
Last activity: 2026-08-09 — Quick task 260808-w8c: two further, independently-motivated levers on
260808-dla's contrastive-crop recipe, sequenced so the cheap one was measured before any GPU spend.
Lever A (crop context-margin padding) was tested inference-only against the already-trained checkpoint
(zero retraining, moved to a vast.ai GPU after a local CPU sweep proved too slow): helps door at
margin=0.15 (+21%) but hurts window at every margin tried, so no margin beats 0.0 on both classes and
none was carried into training. Lever B (rotation/mirror-augmented SupCon crop positives) then trained
as contrastive-crop-v2: door F1 improves to 0.253 tuned / 0.433 default (best door numbers of any
fine-tuned arm, vs 0.229/0.391 for contrastive-crop), window dips slightly to 0.204 (vs 0.216). The
crop/scene self-score mechanism holds and slightly improves (independent single-exemplar diagnostic:
+0.896 vs +0.859, the highest of five checkpoints). Neither arm reaches propose-retrieve (0.459 door
F1) / ncc (0.403 window F1), so the floor-plan recommendation is unchanged. A local fetch-datasets
sweep stalled on an unrelated HF Hub timeout (FSCD-LVIS); worked around via --only scoping to just the
two floor-plans datasets needed. Full suite green (859 passed, 93.89% coverage). contrastive-crop-v2
remains opt-in, not the shipped default; the choice between contrastive-crop and contrastive-crop-v2
is itself class-dependent (v2 better for door, v1 marginally better for window).

Prior activity: 2026-08-07 — Quick tasks 260730-vx4 (ncc) + 260730-w9s (mosse): floor-plan
orientation/mirror follow-up, merged together into one PR against a rebased main. Both confirmed
the rotation-bank-too-narrow hypothesis via a cardinal 0/90/180/270 bank (+ mirror), additive
_TUNING_GRIDS entries, no shipped defaults touched. ncc doors F1 0.164->0.358 (windows: a disclosed
val/test generalization gap, 0.401->0.350, not reverted); ncc also tested two further recall levers
(lower retain_frac, wider scale pyramid) and found both net-negative -- its real ceiling on this
domain (true/false-positive scores overlap, unlike synthetic's clean separation). mosse confirmed
the hypothesis more cleanly for BOTH classes (doors F1 0.201->0.408, beating ncc; windows
0.077->0.155, no generalization gap but still behind ncc's tuned window number), via a cardinal
bank with n_angle_groups scaled to match (the angles-per-group invariant) plus a verify-side-only
mirror knob; mosse also has a disclosed doors val/test nuance (narrower sweep found F1 0.509,
honest full-grid argmax ships 0.408 instead). Zero synthetic-regime regression on either method.
New scripts/ncc_debug_visualize.py debug tool. Full suite green.

Prior activity: 2026-08-08 — Quick task 260730-vx3 (sparse-geo): the same floor-plan flat-recall symptom attacked on
Method 2, and unlike ncc/mosse it did NOT pay off — a fully negative, fully measured result. Two
hypotheses, both disproven, both out of the diff: (1) mirror acceptance (allow_mirror + reflected
pose votes) lost F1 in 4/4 class x voting-mode cells — at single-4dof (door -0.004, window -0.020)
and at pairwise-4dof against its own no-mirror control (door -0.008, window -0.032) — with precision
and AP50 down everywhere and ~2x latency; reverted in full, commit 8ab99a2. (2) SuperPoint backend
lost F1 in 4/4 cells (best case door -0.024, window -0.048), AP50 down in 3/4, window coverage
28/28 -> 26/28 on a hard ONNX/CoreML crash at zero detected keypoints, 5.3-6.9x latency; never
committed, so its revert was a no-op in src/. sparse_geo.py verified byte-identical to pre-task
df64af1 (SHA-256 + empty diff) and the default-config synthetic regimes reproduce EXACTLY
(tp/fp/fn identical on all four). Key diagnostics worth keeping: the det<0 mirror gate fires only
2 times in 55 peaks (it was never where mirrored doors die — the loss is at the voting stage), and
SIFT orientations are NOT mirror-consistent, so single-4dof structurally cannot cluster a mirrored
instance. The flat door recall-by-size symptom remains OPEN — the funnel collapses between
correspondence and peak (55 peaks for 157 GT doors from 2664 correspondences). Docs-only change:
new docs/reports/sparse-geo-improvement.md + nav, findings-page and ROBUSTNESS-BACKLOG
cross-references. Structural contrast recorded: ncc/mosse are template-correlation methods with no
built-in rotation invariance, so a cardinal rotation bank was their winning lever; sparse-geo's SIFT
keypoints are rotation-invariant by construction, so this investigation correctly scoped to
reflection instead — reflection just turned out not to be the answer either.

Progress: [█████████░] 88%

### Quick Tasks Completed

| # | Description | Date | Commit | Directory |
|---|-------------|------|--------|-----------|
| 260726-lct | MkDocs Material docs site + GitHub Pages + UI walkthrough | 2026-07-26 | (pending) | [260726-lct](./quick/260726-lct-set-up-mkdocs-material-docs-site-github-/) |
| 260727-fpe | Floor-plan (Roboflow) target-domain eval + per-method threshold tuning | 2026-07-27 | 5a3a477 | [260727-fpe](./quick/260727-fpe-floorplans-domain-eval/) |
| 260729-dh6 | Floor-plan eval enrichment: per-slice analysis + aggressive tuning + dino-dense OOM fix | 2026-07-29 | 8f8192c | [260729-dh6](./quick/260729-dh6-floor-plan-eval-per-slice-analysis-recal/) |
| 260730-vx4 | ncc floor-plan orientation/mirror follow-up: cardinal bank + mirror, doors F1 0.164->0.358 | 2026-08-01 | 357abe3 | [260730-vx4](./quick/260730-vx4-improve-ncc-on-floor-plan-door-window-do/) |
| 260801-8zy | Fine-tune OWLv2 on floor-plans train data — measured negative result, regresses doors | 2026-08-04 | (pending) | [260801-8zy](./quick/260801-8zy-fine-tune-owlv2-on-the-floor-plans-train/) |
| 260730-w9s | mosse floor-plan orientation/mirror follow-up: cardinal bank + verify-mirror, doors F1 0.201->0.408, windows 0.077->0.155 | 2026-08-07 | c84428b | [260730-w9s](./quick/260730-w9s-improve-mosse-on-floor-plan-door-window-/) |
| 260730-vx3 | sparse-geo floor-plan investigation: mirror acceptance AND SuperPoint backend BOTH disproven and reverted; no source change, docs-only report | 2026-08-08 | (pending) | [260730-vx3](./quick/260730-vx3-improve-sparse-geo-src-object-search-sea/) |
| 260805-hg1 | SupCon contrastive loss for OWLv2 floor-plans fine-tune — sharper negative result, diagnosed crop/scene calibration break | 2026-08-08 | (pending) | [260805-hg1](./quick/260805-hg1-add-a-supervised-contrastive-loss-varian/) |
| 260808-dla | Crop-context SupCon fix — closes the calibration break, best fine-tuned OWLv2 arm measured (door F1 0.229, window F1 0.216) | 2026-08-08 | (pending) | [260808-dla](./quick/260808-dla-add-crop-context-supervision-to-the-owlv/) |
| 260808-w8c | Crop-margin sweep (split result, not adopted) + rotation-augment fix v2 — best door F1 0.253/0.433, window dips slightly to 0.204 | 2026-08-09 | (pending) | [260808-w8c](./quick/260808-w8c-crop-context-margin-padding-rotation-mir/) |

## Performance Metrics

**Velocity:**

- Total plans completed: 0
- Average duration: —
- Total execution time: —

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| - | - | - | - |

**Recent Trend:**

- Last 5 plans: —
- Trend: —

*Updated after each plan completion*
**Per-Plan Metrics:**

| Plan | Duration | Tasks | Files |
|------|----------|-------|-------|
| Phase 1 P1 | 16m | 6 tasks | 19 files |
| Phase 07 P01 | 40m | 3 tasks | 8 files |

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table. Decisions carried in from the
project brief that most affect near-term work:

- [Init]: Requirement IDs are inherited verbatim from `.planning/IDEA.md` §7, never
  renumbered — docs, commits, PR bodies, and code comments all reference the same IDs.

- [Init]: `ONNXInferencer` **design** is ported from the sibling `basketball-2d-to-3d`
  project; a dependency on that project is not added.

- [Init]: The API and UI land between method 1 and method 2, so every later method is
  human-testable the day it lands.

- [Init]: Phases ship as 2 PRs each against a protected `main`.

### Pending Todos

None yet.

### Assumption: PRs are merged as they go green, not left stacked

The brief says the user reviews the PRs when everything is done. Two readings were possible:
leave all ~16 PRs open, or merge each once CI is green.

**Chosen: merge each phase PR after CI passes, then branch the next phase from `main`.**
Rationale — with 16 PRs across 8 sequentially-dependent phases, leaving them open forces each
branch to target the previous branch, producing a 16-deep stack in which every PR's diff is
polluted by its ancestors and none can be read independently. Merging as we go keeps each PR a
self-contained, readable unit in `main`'s history, which is exactly what makes them reviewable
after the fact. Branch protection is configured with **0 required approvals** (single-user repo),
so the PR flow and the CI gate are still enforced — the merge is not bypassing a review that was
ever going to happen.

Consequence for the reviewer: use `gh pr list --state merged` (or the PR list in the web UI) to
review; the PRs are numbered in phase order.

### Blockers/Concerns

- **GSD agent coverage is partial.** `gsd-project-researcher`, `gsd-research-synthesizer`,
  and `gsd-verifier` are not installed in `~/.claude/agents`. `gsd-planner`,
  `gsd-phase-researcher`, `gsd-plan-checker`, `gsd-executor`, `gsd-roadmapper`,
  `gsd-code-reviewer`, and `gsd-code-fixer` **are** available. Project-level research was
  run with general-purpose agents instead; per-phase research and plan-checking work
  normally. Phase verification is performed against the ROADMAP success criteria directly
  rather than by `gsd-verifier`.

- **BLOCKED — branch protection on `main` is not available on this repo, so INFRA-07 is only
  partially satisfied.** Both the branch-protection API and the rulesets API return
  `403 Upgrade to GitHub Pro or make this repository public`. GitHub Free does not offer
  protected branches on **private** repositories. `main` reports `"protected": false`.

  Three ways to close it, all requiring a decision that is the repo owner's to make:

  1. **Upgrade to GitHub Pro** — costs money.
  2. **Make the repository public** — an irreversible outward-facing action, and one with a
     licence dimension (see the FastSAM AGPL note below), so it must not be done implicitly.

  3. **Accept convention-only enforcement** — what is in effect now.

  **Chosen for now: option 3.** CI still runs `lint`, `format-check`, `typecheck`, and `test` on
  every pull request, and every phase has in fact gone through a PR. What is missing is only the
  server-side *enforcement* that a human could not bypass. Since the single user is also the
  only person with push access, the practical gap is small — but it is a real gap and INFRA-07
  should not be ticked as fully done until one of options 1 or 2 is taken.

  The ready-to-apply protection JSON is preserved in
  `.planning/phases/01-foundation/01-01-SUMMARY.md`, so closing this later is a one-command
  change rather than a re-derivation.

  Note: the "direct push to `main` is rejected" verification was deliberately **not** run, because
  on an unprotected `main` it would have succeeded and pushed. Reporting it as passing would have
  been false; reporting it as failing would have been misleading.

- **FastSAM is AGPL-3.0 and the exported `.onnx` embeds that licence string.** Verified during
  project research. "Export-time-only dependency" protects the runtime dependency graph but
  **not the weights**. Private local use triggers nothing; **publishing this repo publicly or
  network-exposing the FastAPI app would fire AGPL §13.** The repo is private and the app is
  local-only, so Milestone 1 proceeds with FastSAM as planned — but this constrains how the
  repo may later be shared, and the constraint must be restated in `LICENSES.md` and
  `docs/methods/propose-retrieve.md`.

- **MobileSAM will NOT be built as a working second backend in Milestone 1.** Research found
  that the ONNX SAM decoder accepts one prompt per call, so "everything mode" means ~1024
  sequential decoder calls plus a hand-ported `SamAutomaticMaskGenerator` — a phase of work,
  not a config-switchable backend. METHOD-06 is therefore satisfied with FastSAM as the
  implemented backend and the proposal stage written behind a `ProposalBackend` seam so
  MobileSAM can be added without restructuring. This is a **deviation from the brief** and is
  recorded in the Phase 7 docs and the robustness backlog.

- **`awarebayes/MobileSamONNX` is a HOLD** — 0 stars, a 4-day project untouched for 14 months,
  single author, while first-party export scripts and a better-provenanced MIT artifact
  (`Acly/MobileSAM`) exist. The IDEA.md §14 reference to it should be dropped.

- **SuperPoint weights are MagicLeap non-commercial research-only**, and the DERIVATIVES clause
  covers the ONNX file. Acceptable as scoped because weights are gitignored (INFRA-11), but
  they must never be redistributed. DISK/ALIKED is the permissive swap if that ever changes.

- Branch protection on main NOT applied (INFRA-07 partial): private repo on free GitHub plan returns HTTP 403 for both the branch-protection and rulesets APIs. Needs user decision: upgrade to GitHub Pro, make repo public, or accept convention-only enforcement. Ready-to-apply JSON is in 01-01-SUMMARY.md.

### Corrections to IDEA.md found during research

The brief is authoritative on intent, but three of its technical references are stale. These
corrections are load-bearing:

1. **§5/§14 SuperPoint source is stale.** `fabio-sim/LightGlue-ONNX` `main` no longer exports
   SuperPoint standalone — its CLI now emits only the fused extractor+matcher pipeline. The
   correct source is the frozen **v1.0.0 release asset `superpoint.onnx`** (~5.03 MiB), which
   is genuinely standalone and has a **variable** keypoint count. Variable is better here: the
   METHOD-04c low-keypoint guard reads the count directly, so a fixed top-K export would
   defeat it.

2. **§5 Method 3's "stride-14 tokens" needs an explicit resize policy.** DINOv2 silently drops
   trailing pixels when a side is not a multiple of 14 (a 225 px side yields 224 px of
   content), producing a systematic spatial offset rather than an error. Input sides must be
   snapped to a multiple of 14 and the scale factors returned to the caller.

3. **§14's `sefaburak*` DINOv2 references are superseded** by `onnx-community/dinov2-small-ONNX`
   (Apache-2.0, fully dynamic input, `last_hidden_state [B, floor(H/14)*floor(W/14)+1, 384]`),
   which was runtime-verified at six resolutions.

4. **§8's reproducibility constraint cannot be honoured as written for RANSAC, and pretending
   otherwise would be the exact kind of lie the constraint exists to prevent.** The brief says
   "any stochastic step (RANSAC, SAM prompt sampling) takes an explicit seed from config."
   Research established, by reading the OpenCV source, that `cv2.setRNGSeed` has **no effect** on
   RANSAC: `modules/calib3d/src/ptsetreg.cpp` hardcodes `RNG rng((uint64)-1)` (lines 171 and 284),
   and a maintainer has confirmed this is deliberate. `theRNG()` is additionally thread-local.

   Consequence: OpenCV's RANSAC **is** reproducible — same input gives same output — but the seed
   is **not user-controllable**. Adding a `ransac_seed` config field that silently does nothing
   would be worse than having none, because it would advertise a control that does not exist.

   **Resolution for Phase 5:** implement the per-peak similarity/affine RANSAC **in NumPy inside
   `sparse_geo.py`** with an explicit `np.random.default_rng(config.seed)`. For a 4-DoF similarity
   fitted from 2-point samples this is on the order of 30 readable lines, it satisfies the
   constraint honestly, it makes the sampling visible in the one file the user will read — which
   is the whole point of the self-contained-module rule — and it removes the dependency on an
   OpenCV behaviour we cannot control. Where `cv2.estimateAffinePartial2D` is still used (e.g. as
   a cross-check), document that its seed is fixed internally and not configurable.

5. **§8's reproducibility constraint is also mis-aimed at thread counts.** Research measured ONNX
   Runtime thread count, OpenCV thread count, BLAS thread count, and argmax tie order as all
   producing **bit-identical** results, and found `use_deterministic_compute` is a **no-op on the
   CPU execution provider**. Meanwhile `cv2.setNumThreads(1)` is *silently ignored* on macOS GCD
   (only `0` does anything). So thread pinning must not be described as a determinism measure.
   What actually threatens reproducibility, and therefore what gets pinned: **set/dict iteration
   order, NMS tie-breaking, config-hash key ordering, and library-version drift.**
   Additional real threat found: `cv2.matchTemplate` results depend on the **search extent** —
   cropping the search region changes 73% of the returned floats — so Method 1 must always
   correlate over the full scene.

## Deferred Items

- Milestone 2 (marker-conditioned region proposal) — specified in `docs/MILESTONE-2.md`
  during Phase 8, built later.

- Method 4 (exemplar-conditioned detectors/counters) and Method 6 (one-shot personalized
  segmentation) from the source research — see REQUIREMENTS.md § v2.

- Lattice fitting as post-detection verification — documented in the robustness backlog.

## Session

**Last session:** 2026-07-25T11:20:00.000Z
**Stopped at:** Completed 08-02-PLAN.md — Phase 8 complete (EVAL-06, DOC-03/04/05/06); PR #16 open, CI green. Final plan of the project.
**Resume file:** None
