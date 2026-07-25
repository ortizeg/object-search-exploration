# Milestone 2 — Marker-Conditioned Region Proposal (DOC-06)

**Status: specified, not built.** This is the next big feature. It is documented now, and
Milestone 1 was built with the seams it needs, so Milestone 2 **adds an exploration rather than
forking the app**. Every seam this spec relies on has already shipped — the references below
point at the real code, not a plan.

## The feature

Given a crop of a **marker** — an arrow, a dot, a caret, a highlighter blob — find every instance
of that marker in the image, then for each one find the **best object region proposal near it**.
The marker points at things; the system resolves what it is pointing at. Where Milestone 1 asks
"find every other instance of *this object*", Milestone 2 asks "find every object *this marker
gesture* is indicating".

## Pipeline

1. **Find markers.** Reuse any Milestone 1 method wholesale, unchanged: `sparse-geo` or `ncc` for
   rigid synthetic markers, `dino-dense` for hand-drawn ones. The marker crop is just an exemplar;
   the existing `search()` returns every marker instance as a set of boxes.
2. **Estimate each marker's reference point and orientation.** For an arrow, the tip and its
   direction; for a symmetric marker, the centroid and no direction. Orientation comes from **PCA
   on the marker's mask**, or is **recovered directly from the similarity/affine transform that
   `sparse-geo` already fits per instance** (each accepted instance carries its transform in the
   diagnostics payload, so the rotation is already computed).
3. **Propose objects nearby.** **Reuse Method 5's proposal stage directly** — this is the whole
   reason it was built as an independently callable unit.
4. **Score and pick.** Rank proposals per marker by (a) distance from the reference point,
   (b) alignment with the marker's direction, (c) objectness, and (d) a size prior; return the
   best-scoring proposal for each marker.

## What it reuses vs what is new

### Reuses — the shipped seams

| Seam | Where it lives today | How Milestone 2 uses it |
| --- | --- | --- |
| The **`exploration` column** | `src/object_search/store/schema.py` — `runs.exploration TEXT NOT NULL DEFAULT 'same-image-search'`, threaded through the metrics views; `runs.py` deliberately omits it on insert so the DEFAULT applies | A second exploration writes rows with `exploration = 'marker-conditioned'`; **no schema migration** — the scoreboard and stats views already group by it |
| The **method registry** | `src/object_search/search/registry.py` — `@register_method`, `list_methods()` | Step 1 calls any registered method to find markers; nothing new registers to find them |
| **`propose()`** | `src/object_search/search/proposals.py` — the module-level `propose()` and the `ProposalBackend` protocol (FastSAM today, MobileSAM seam already cut) | Step 3 calls it directly to get class-agnostic region proposals near each marker |
| **`embed_regions()`** | `src/object_search/search/propose_retrieve.py` — the standalone region-embedding stage | Available if a proposal needs to be matched by appearance as well as position |
| **The UI mode selector** | `frontend/index.html` — the `<select id="exploration">` "Exploration" control that already sits **above** the method selector | Gains a second option; the method selector below it keeps working unchanged |
| `ONNXInferencer`, all frozen schemas, the run/rating store, the stats + Wilson + Bradley-Terry layer | across `inference/`, `schemas/`, `store/` | Consumed as-is |

### New — the only Milestone 2 code

- **Marker orientation estimation** (PCA on the mask, or reading `sparse-geo`'s per-instance
  transform) — the one genuinely new piece of computer vision.
- **The proposal scoring function** — distance + direction-alignment + objectness + size-prior.
- **A second UI mode** wired into the existing exploration selector, and a thin exploration entry
  that composes the four steps above.

Everything else is a call into code that already exists and is already tested. That is the payoff
of the Milestone 1 seams: the marker-conditioned feature is a small, readable addition, not a
rewrite.

## References

- `.planning/IDEA.md` §11 — the source specification this expands.
- `docs/methods/propose-retrieve.md` — the `propose()` / `embed_regions()` stages this reuses.
- `docs/methods/sparse-geo.md` — the per-instance similarity/affine transform reused for
  orientation.
