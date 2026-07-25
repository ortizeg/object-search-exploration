# Phase 9 Context — Milestone 2: Marker-Conditioned Region Proposal

**Source:** `docs/MILESTONE-2.md` (the spec that shipped in Phase 8), `.planning/IDEA.md` §11, and
the real shipped seams inspected on `main`.

## Domain

The next big feature, and the payoff of every seam Milestone 1 deliberately left. Given a crop of
a **marker** (arrow, dot, caret, highlighter blob), find every instance of that marker in the
image, then for each one return the **best object region proposal near it**. The marker points at
things; the system resolves what it is pointing at.

Milestone 1 asks "find every other instance of *this object*." Milestone 2 asks "find every object
*this marker gesture* is indicating." It **adds an exploration, it does not fork the app** — every
seam it needs already shipped and is tested.

## The shipped seams this builds on (verified on main)

| Seam | Location | Use |
|------|----------|-----|
| `exploration` column | `store/schema.py:39` — `runs.exploration TEXT NOT NULL DEFAULT 'same-image-search'`, threaded through the metric views; `runs.py` omits it on insert so the DEFAULT applies | Marker runs persist with `exploration='marker-conditioned'` — **no migration**; the scoreboard already groups by it |
| method registry | `search/registry.py` — `register_method`, `get_method`, `list_methods`, `method_schemas` | Step 1 finds markers by calling any registered method unchanged |
| `propose()` | `search/proposals.py:80` — `propose(image, config, *, backend=None) -> list[Proposal]`, standalone | Step 3 gets class-agnostic proposals near each marker |
| `embed_regions()` | `search/propose_retrieve.py:301` — standalone | Available if a proposal must also match by appearance |
| `Match.transform` | `schemas/search.py:59` — flattened 2×3 affine (6 floats, row-major `[a,b,tx,c,d,ty]`) | sparse-geo fills this per accepted instance; **rotation = atan2(c, a)** gives marker orientation for free |
| UI exploration `<select>` | `frontend/index.html:24` — `#exploration`, already ABOVE the method selector | Gains a second option; the method selector below keeps working |

## Locked Decisions

1. **Explorations become a registry-level concept, mirroring methods.** Add
   `explorations/registry.py` with `@register_exploration`, `get_exploration`, `list_explorations`,
   `exploration_schemas()` — the exact shape of the method registry, so the API and UI stay
   schema-driven. `same-image-search` is registered as the default exploration wrapping the
   existing method flow (a thin adapter); `marker-conditioned` is the new one. Adding an
   exploration = one new file + one import, same rule as methods (INFRA-10 generalized).
2. **The marker-conditioned exploration returns a `SearchResult`**, so it persists through the
   existing `RunRecord`/store/stats layer unchanged. Its `matches` are the **best proposal per
   marker**; its `diagnostics` carry the marker boxes, per-marker reference point + orientation,
   and the full proposal set (rendered by the existing overlay by field presence).
3. **Orientation — the one genuinely new CV piece — has two paths, preferring the free one:**
   - If the marker-finding method supplied a `transform` (sparse-geo does), recover rotation
     directly as `atan2(c, a)` from the 2×3 affine, and resolve the 180° ambiguity using the
     exemplar marker's own known tip mapped through the transform.
   - Otherwise (ncc, dino-dense — no transform), estimate the axis by **PCA on the marker
     region's foreground mask**, and resolve the tip via an arrowhead-mass heuristic (the end
     whose perpendicular width profile narrows is the tip). For a symmetric marker, return the
     centroid and **no direction** (spec-required).
4. **Scoring is a documented weighted sum**, config-exposed: `distance` from the reference point
   (nearer better), `direction` alignment with the marker's pointing vector (only when a direction
   exists), `objectness` (from the proposal), and a `size_prior`. Return the best-scoring proposal
   per marker. Weights are `Field`-described so they drive the UI form.
5. **`propose()` is called ONCE per image**, not once per marker — the proposal set is shared and
   each marker scores against it. The proposal stage dominates latency (EVAL-11 finding), so
   calling it per-marker would be wasteful and is explicitly avoided.
6. **Reproducibility** holds: the pipeline is deterministic given the marker-finding method's
   determinism plus a config seed for any sampled step. No unseeded randomness.
7. **Self-contained, readable module** for the exploration, numbered step comments matching
   `docs/explorations/marker-conditioned.md`, a ROBUSTNESS BACKLOG section, explicit
   pre/post-processing — the same conventions as a method module.

## Specifics

**`MarkerConditionedConfig`** (frozen, Field-described): `marker_method: str = "sparse-geo"`,
`marker_config: dict` (the chosen method's config, validated against its `config_model`),
`proposal: FastSAMConfig`, `w_distance/w_direction/w_objectness/w_size: float`,
`size_prior_frac: float` (expected object size relative to marker), `max_markers: int`, `seed: int`.

**Pipeline (`run(image, marker_exemplar, config) -> SearchResult`):**
1. `markers = get_method(config.marker_method).search(image, marker_exemplar, marker_config).matches`
   (reuse a M1 method wholesale; the exemplar box is the marker crop).
2. Per marker: reference point + orientation (per decision 3).
3. `proposals = propose(image, config.proposal)` — once.
4. Per marker: score every proposal, pick the best; assemble it as a `Match`. Diagnostics carry
   markers, reference points, orientations (as arrows), and the full proposal set.
   Outcome `empty` (with a note) if no markers were found — never a silent empty.

**API:** add `GET /explorations` (mirrors `/methods`: name, description, config schema). Extend
`POST /search` with an optional `exploration: str = "same-image-search"`; when
`marker-conditioned`, route to the exploration and persist the run with that exploration tag. Keep
`/methods` and the existing method path untouched. No method name hardcoded in the api package.

**Synthetic markers (for exact-GT tests):** extend the generator with an arrow-marker mode — an
arrow of known tip, direction, and position, optionally with a target object a known distance away
in the pointing direction. Same-seed byte-identical; the tip/direction are exact test oracles for
orientation estimation.

## Scope Fence

**In:** the exploration registry + `same-image-search` adapter, `marker_conditioned` exploration
(orientation, scoring, pipeline), synthetic arrow markers, `GET /explorations` + `/search`
routing + persistence, the second UI mode + schema-driven exploration form + marker→proposal
overlay, committed marker demo assets + sample runs, docs (update `MILESTONE-2.md` status, add
`docs/explorations/marker-conditioned.md`, README mention).

**Out:** MobileSAM (still the documented deviation). Any change to the four Milestone 1 methods.
Corpus/cross-image marker search. A learned marker detector (markers are found by reusing M1).

## Risk Summary

- **Tip/direction from PCA is 180°-ambiguous.** The arrowhead-mass heuristic resolves it for
  arrows; test it against synthetic arrows with exact GT, and fall back to "no direction"
  (centroid only) when the heuristic is low-confidence, recording that in diagnostics rather than
  guessing a wrong direction.
- **The scoring weights are a real design surface.** Ship sane defaults, expose them in the form,
  and document what each does — do not bury magic constants.
- **Model dependence:** the proposal stage needs FastSAM. Keep the orientation + scoring logic
  model-free and unit-tested; make the end-to-end marker→proposal test skip-when-absent, like the
  other learned-method tests.
- **Latency:** find-markers (a full method run) + one propose() call. Attribute them separately in
  the latency breakdown so the cost is visible.
