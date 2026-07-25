---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
current_phase: 8
current_phase_name: Evaluation & docs
status: executing
stopped_at: Completed 08-02-PLAN.md
last_updated: "2026-07-25T11:20:00.000Z"
last_activity: 2026-07-25
last_activity_desc: "Executed 08-02: committed charts, README, method-doc drift guard, robustness backlog, Milestone 2 spec, LIMITATIONS. Phase 8 complete — final plan of the project. PR #16 open, CI green."
progress:
  total_phases: 8
  completed_phases: 8
  total_plans: 16
  completed_plans: 16
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
Last activity: 2026-07-25 — Executed 01-02: schemas, registry, ONNXInferencer, synthetic +
chip benchmark, fetch-models, demo assets. All gates green (90.34% coverage). PR #2 open, CI
green. INFRA-07 branch protection unavailable on a private repo.

Progress: [█████████░] 88%

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
