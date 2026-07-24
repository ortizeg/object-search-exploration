---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
current_phase: 1
current_phase_name: Foundation
status: executing
stopped_at: "Completed 01-01-PLAN.md (Task 6 blocked: branch protection needs GitHub Pro or a public repo)"
last_updated: "2026-07-24T18:28:36.420Z"
last_activity: 2026-07-24
last_activity_desc: Project initialized; PROJECT.md, REQUIREMENTS.md, ROADMAP.md
progress:
  total_phases: 4
  completed_phases: 0
  total_plans: 2
  completed_plans: 1
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
Plan: 1 of 2 in current phase
Status: Ready to execute
Last activity: 2026-07-24 — Project initialized; PROJECT.md, REQUIREMENTS.md, ROADMAP.md
written and committed; private GitHub repo created and `main` pushed.

Progress: [█████░░░░░] 50%

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

### Blockers/Concerns

- **GSD agent coverage is partial.** `gsd-project-researcher`, `gsd-research-synthesizer`,
  and `gsd-verifier` are not installed in `~/.claude/agents`. `gsd-planner`,
  `gsd-phase-researcher`, `gsd-plan-checker`, `gsd-executor`, `gsd-roadmapper`,
  `gsd-code-reviewer`, and `gsd-code-fixer` **are** available. Project-level research was
  run with general-purpose agents instead; per-phase research and plan-checking work
  normally. Phase verification is performed against the ROADMAP success criteria directly
  rather than by `gsd-verifier`.

- **Branch protection on `main` is deferred to the end of Phase 1 plan 01-01**, because
  required status checks cannot be configured until the CI workflow has run at least once
  and registered its check names.

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

## Deferred Items

- Milestone 2 (marker-conditioned region proposal) — specified in `docs/MILESTONE-2.md`
  during Phase 8, built later.

- Method 4 (exemplar-conditioned detectors/counters) and Method 6 (one-shot personalized
  segmentation) from the source research — see REQUIREMENTS.md § v2.

- Lattice fitting as post-detection verification — documented in the robustness backlog.

## Session

**Last session:** 2026-07-24T18:28:36.415Z
**Stopped at:** Completed 01-01-PLAN.md (Task 6 blocked: branch protection needs GitHub Pro or a public repo)
**Resume file:** None
