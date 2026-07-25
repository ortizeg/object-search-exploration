# Phase 8 Context — Evaluation & Docs

**Source:** `.planning/IDEA.md` §7 (EVAL-*, DOC-*), §7a (evaluation design), §11 (Milestone 2 spec),
§12 (deferred work), plus `.planning/research/PITFALLS.md` (Wilson / Bradley-Terry edge cases).

## Domain

Turn "which method is better" into a number, with the crossover the research predicts **visible
rather than hidden** — and leave the repo readable to someone opening it cold.

## Locked Decisions

1. **Hydra drives the benchmark CLI only.** This is the one place Hydra is used, because sweeping
   method × config × image is exactly what it is for. The API path keeps plain frozen Pydantic
   configs. `hydra-core` is pinned `>=1.3.4` from PyPI (the 1.3.4 release is a security patch adding
   a `_target_` blocklist to `instantiate()`; conda-forge is stuck at 1.3.2).
2. **The benchmark must demonstrate the NCC-vs-sparse-geo crossover, not hide it.** The literature
   is explicit that when instances are small and nearly identical, almost all of Method 2's
   tentative matches are wrong and Hough's discriminative power is insufficient — precisely the
   regime where Method 1 is strongest. That is an **expected finding** and a large part of why four
   methods exist. Include synthetic slices on both sides of the crossover and report per-slice, not
   just pooled.
3. **Per-slice reporting is the point** (EVAL-10). The deliverable is *"Method 3 wins once instance
   scale varies more than 1.5×, Method 1 wins below that"*, not *"Method 3 scored 0.71."* Slice by
   true instance count, scale range, rotation range, clutter level, and exemplar keypoint count.
4. **Derived metrics still come from the query layer** (EVAL-07). Phase 8 reads the Phase 3 views;
   it does not add stored metric columns. Remember the SQLite traps: `CAST(... AS REAL)` before any
   division, `SUM` not `TOTAL` so NULL propagates.
5. **Wilson score interval** for every rate, with `n` always beside it (EVAL-14). Handle the closed
   forms at `n = 0`, `p = 0`, `p = 1` without dividing by zero or emitting bounds outside `[0, 1]`.
6. **Bradley-Terry over paired comparisons** (EVAL-15), not a comparison of four independent means —
   it is far more statistically efficient and needs far fewer ratings to separate the methods.
   **Complete separation is the failure mode**: if one method never loses, the MLE diverges to
   infinity. Regularize — add a small symmetric prior (equivalent to a fraction of a win and a loss
   between every pair) and say in the docs that this is what keeps an undefeated method from
   scoring `+inf`. Report the ranking with uncertainty, not as a bare ordering.
7. **Paired comparison mode runs one box through all four methods in a single request** (EVAL-05),
   so ratings are directly comparable rather than confounded by different boxes. It writes to the
   `paired_comparisons` table Phase 3 already created.
8. **AP computation must state its convention.** All-point interpolation vs 11-point changes the
   number materially. Pick all-point (the modern COCO-style convention), state it, and note that
   AP here is computed from the **sub-threshold candidate log** (EVAL-08), which is what makes a
   real PR curve possible from a single operating point's worth of rating effort.
9. **Charts are committed** (EVAL-06), generated headlessly by matplotlib with a deterministic
   render (no embedded timestamps), regenerable by one command.
10. **Ground-truth labels** (EVAL-02): exact and free for the synthetic set (the generator emits
    them). For the basketball frames and any photo, hand-label a small set in the same sidecar JSON
    format the generator uses, so the eval harness has one loader. If a category has no labels,
    report coverage honestly rather than silently excluding it.

11. **The chip-insertion set (EVAL-19) is the primary objective signal.** Ten generated images,
    each a white canvas with one *distinct* randomly-generated chip pasted `N ∈ {5, 10, 15}` times
    at random **strictly non-overlapping** positions, across canvas sizes ramping 320×240 →
    6000×4000. Built by the Phase 1 generator; consumed here.

    Why it carries so much weight in this phase:
    - **Ground truth is exact by construction** — we pasted every instance, so there is nothing to
      label and nothing to dispute. Precision, recall, and AP are computable at zero human cost.
    - **Non-overlap removes the duplicate/fragment judgement call entirely.** The EVAL-16
      convention (two boxes on one instance = 1 TP + 1 FP) still governs human rating, but on this
      set the matching is unambiguous, so a disagreement between methods is a real difference in
      method quality rather than an artefact of how overlap was scored.
    - **It is re-runnable after every change**, which makes it the parameter-tuning harness. This
      is the intended workflow: change a config, re-run the chipset benchmark, see the number move.
    - **It is the natural model-free CI benchmark subset.** The images regenerate deterministically
      from seeds and need no ONNX weights, so the full chipset benchmark can run in CI for `ncc`
      and for `sparse-geo`'s classical backend — satisfying the "benchmark subset that runs in CI"
      requirement without gating on `fetch-models`.
    - **The canvas-size ramp is a latency and scaling story on its own.** Reporting per-canvas-size
      latency across 320×240 → 6000×4000 shows where each method's cost actually goes, which a
      single pooled latency number would hide.

    Two honesty constraints carried from the generator: the sidecar records the **achieved**
    instance count, not the requested `N` (if rejection sampling could not place all of them,
    fewer were placed and the ground truth says so) — the benchmark must read that field, not
    assume `N`. And every method is queried with the **same** designated exemplar index per image,
    so the comparison is not confounded by different query boxes.

    Expect this set to show the NCC-vs-sparse-geo crossover clearly: the chips are near-identical
    repeated instances at a fixed scale, which is exactly the regime where the literature says
    Hough's discriminative power is insufficient and NCC is strongest. Report that as the expected
    finding it is.

## Canonical References

- `.planning/IDEA.md` §7a — the derived-metric formulas and the rationale for each logged field
- `.planning/IDEA.md` §11 — the Milestone 2 pipeline and exactly which Milestone 1 components it
  reuses; `docs/MILESTONE-2.md` (DOC-06) is essentially §11 expanded with references to the real
  code that landed
- `.planning/IDEA.md` §12 — deferred work, whose reasoning must survive into
  `docs/ROBUSTNESS-BACKLOG.md`
- `.planning/STATE.md` — the accumulated deviation list, which the final docs must reflect honestly

## Specifics — documentation deliverables

- **DOC-03** — README showing sample runs for all four methods **side by side**. Uses the Phase 2
  sample renderer, which iterates the registry, so all four appear with no per-method code.
- **DOC-04** — per-method pages for `ncc`, `sparse-geo`, `dino-dense`, `propose-retrieve`, each with
  algorithm, **explicit pre/post-processing**, config reference generated from the JSON Schema,
  known failure modes, and the robustness backlog mirrored from the module docstring.
- **DOC-05** — `docs/ROBUSTNESS-BACKLOG.md` aggregating every method's backlog, plus the
  cross-cutting items: **lattice verification** (documented, not built — likely the single
  highest-leverage robustness item for the shelf/PCB/tile images, since fitting the lattice
  post-detection recovers misses and kills false positives more effectively than tuning the
  detector), and the deferred Methods 4 and 6 with their reasoning intact.
- **DOC-06** — `docs/MILESTONE-2.md`: the marker-conditioned region proposal feature. Find every
  instance of a marker (arrow, dot, caret, highlighter blob) by reusing any Milestone 1 method,
  estimate each marker's reference point and orientation (arrow tip and direction, or centroid with
  no direction for a symmetric marker — from PCA on the mask, or recovered from the
  similarity/affine transform Method 2 already fits per instance), reuse Method 5's proposal stage
  directly, then score proposals by distance from the reference point, alignment with the marker's
  direction, objectness, and a size prior. State what it reuses vs what is new, and point at the
  actual seams that shipped: the `exploration` column, the registry, `propose()`/`embed_regions()`,
  and the UI mode selector.

## Honest reporting requirements

The final docs and PR must state, without softening:

- **INFRA-07 is partially satisfied** — branch protection is unavailable on a free private repo
  (`403 Upgrade to GitHub Pro or make this repository public`). CI runs on every PR; server-side
  enforcement does not exist.
- **MobileSAM did not ship** as a working backend, and why.
- **FastSAM's AGPL-3.0** constrains publishing the repo or exposing the API.
- **SuperPoint weights are non-commercial research-only.**
- Any method that underperforms should be reported as underperforming. The project's whole value is
  saying which method actually works; a flattering benchmark would defeat it.

## Scope Fence

**In:** ground-truth labels, `eval/` (metrics, benchmark runner, paired comparison, Bradley-Terry),
committed charts and tables, README, four method docs, aggregated robustness backlog,
`docs/MILESTONE-2.md`.

**Out:** Milestone 2 implementation. Any new search method. Corpus search. FAISS.

## Risk Summary

- **Bradley-Terry with sparse real ratings.** There may be very few human ratings by Phase 8, since
  rating is a manual activity. The benchmark's objective metrics do not need ratings; the paired
  Bradley-Terry does. Make the code correct and tested on synthetic win/loss records, and report
  honestly that the human-rating `n` is small rather than presenting a confident ranking built on
  four comparisons.
- **The benchmark needs all four methods' models present**, so it cannot run in CI. Provide a
  synthetic-only benchmark subset that runs model-free in CI, and gate the full run behind
  `fetch-models`.
- **Coverage.** `eval/` is arithmetic-heavy and therefore easy to test well — take the opportunity
  to test the metric edge cases (R=0 abstention, all-NULL aggregates, single-method Bradley-Terry)
  rather than only the happy path.
