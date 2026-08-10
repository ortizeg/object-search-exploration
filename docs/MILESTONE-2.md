# Milestone 2 — Marker-Conditioned Region Proposal (DOC-06)

**Status: built.** Milestone 2 shipped as a second *exploration*, not a fork of the app — exactly
what the Milestone 1 seams were cut for. The exploration registry, the marker-conditioned pipeline,
the orientation estimator, `GET /explorations` + `/search` routing, the schema-driven second UI
mode, the marker→proposal overlay, the synthetic marker demo set, and the committed sample runs are
all implemented and tested. The references below point at the real, shipped code.

- **Exploration:** [`docs/explorations/marker-conditioned.md`](explorations/marker-conditioned.md)
  — the full pipeline, the two orientation paths, the scoring formula, and the config reference.
- **Sample runs:** [`docs/samples/marker-conditioned/`](samples/marker-conditioned/index.md) — committed,
  byte-identical on re-render (`pixi run samples`).

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
| The **`exploration` column** | `src/object_search/store/schema.py` — `runs.exploration TEXT NOT NULL DEFAULT 'same-image-search'`, threaded through the metrics views; `runs.py` deliberately omits it on insert so the DEFAULT applies | `routes_search.py` **persists marker runs with `exploration = 'marker-conditioned'`** — no schema migration; the scoreboard and stats views group by it as shipped |
| The **method registry** | `src/object_search/search/registry.py` — `@register_method`, `list_methods()` | `explorations/marker_conditioned.py` step 1 **calls `get_method(config.marker_method).search(...)`** to find markers; nothing new registers to find them |
| **`propose()`** | `src/object_search/search/proposals.py` — the module-level `propose()` and the `ProposalBackend` protocol (FastSAM today, MobileSAM seam already cut) | `marker_conditioned.py` step 3 **calls `propose(image, config.proposal)` once per scene** for class-agnostic region proposals |
| **`embed_regions()`** | `src/object_search/search/propose_retrieve.py` — the standalone region-embedding stage | Not used in the shipped scorer (position + direction + objectness + size); recorded in the exploration's ROBUSTNESS BACKLOG as the per-proposal appearance-matching upgrade |
| **The UI mode selector** | `frontend/index.html` — the `<select id="exploration">` "Exploration" control that already sits **above** the method selector | **Populated from `GET /explorations`** by `frontend/js/main.js` + `explorations.js`; choosing the marker mode rebuilds the config form from that exploration's own JSON schema. The method selector stays for the same-image (method-wrapper) mode |
| `ONNXInferencer`, all frozen schemas, the run/rating store, the stats + Wilson + Bradley-Terry layer | across `inference/`, `schemas/`, `store/` | Consumed as-is |

### New — the only Milestone 2 code (all shipped)

- **The exploration registry** — `explorations/registry.py`, `@register_exploration` /
  `list_explorations` / `exploration_schemas`, the deliberate mirror of the method registry, plus
  the `same-image-search` adapter (`explorations/same_image_search.py`) that re-labels the
  Milestone 1 flow as one exploration among others.
- **Marker orientation estimation** — `explorations/markers.py`: PCA-on-the-mask with an
  arrowhead-mass tip heuristic, or reading `sparse-geo`'s per-instance affine (`atan2(c, a)` with the
  exemplar-tip 180° tiebreak). The one genuinely new piece of computer vision.
- **The proposal scoring function** — `explorations/marker_conditioned.py`: distance +
  direction-alignment + objectness + size-prior, config-exposed weights.
- **`GET /explorations` + `/search` routing** — `api/routes_explorations.py`, `api/routes_search.py`.
- **The schema-driven second UI mode + marker→proposal overlay** — `frontend/js/explorations.js`,
  `main.js`, `overlay.js`.
- **Synthetic arrow/dot markers + committed sample runs** — `synthetic/generator.py` (marker mode),
  `samples.py` (the exploration-registry-iterating renderer).

Everything else is a call into code that already existed and was already tested. That is the payoff
of the Milestone 1 seams: the marker-conditioned feature is a small, readable addition, not a rewrite.

### What actually shipped vs the spec

- **Marker finder for the demo.** The default `marker_method` is `sparse-geo` (rotation-invariant),
  but the committed sample gallery uses `ncc` because classical `sparse-geo` abstains on the
  low-texture synthetic arrows (< 20 keypoints). `ncc` is model-free but not rotation-invariant, so
  the randomly-rotated `arrows` demo resolves only the exemplar-orientation instance — an honest
  data/method-fit limitation recorded in the exploration doc and the robustness backlog, not hidden.
- **`embed_regions()`** is a documented backlog upgrade, not part of the shipped scorer.
- **MobileSAM** remains the documented clean-licence alternative to FastSAM; still deferred.

## References

- `.planning/IDEA.md` §11 — the source specification this expands.
- `docs/methods/propose-retrieve.md` — the `propose()` / `embed_regions()` stages this reuses.
- `docs/methods/sparse-geo.md` — the per-instance similarity/affine transform reused for
  orientation.
