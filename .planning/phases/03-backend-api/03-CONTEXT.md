# Phase 3 Context — Backend API + SQLite Run/Rating Store

**Source:** `.planning/IDEA.md` §6 (architecture), §7 (API-*, EVAL-*), **§7a (the evaluation
design — read it in full; it is the rationale for almost every decision in this phase)**,
plus `.planning/research/PITFALLS.md` (SQLite NULL semantics, Wilson interval).

## Domain

The persistence and HTTP layer. This phase decides what evidence the project can ever produce,
because a rating not captured today cannot be reconstructed later. Two things make it more
than CRUD: the **null discipline** on human counts, and **logging sub-threshold candidates** so
one rating session yields a whole PR curve instead of one operating point.

## Locked Decisions

### The two rules that must not be regressed

1. **All human count fields are nullable and stored EMPTY.** `wrong_count` and `missed_count`
   are `INTEGER NULL` with **no `DEFAULT 0`** in the DDL, `int | None = None` in the Pydantic
   model, and no coercion in the route handler. `null` = "not assessed"; `0` = "assessed, none".
   Collapsing them defaults every unreviewed run to **perfect precision and recall** — the exact
   way a scoreboard lies, and it gets worse the faster ratings are entered.
   A test must submit a bare thumbs-up and assert the aggregates ignore that run entirely.
2. **Derived metrics are computed in queries/views, never stored as columns.** No `precision`,
   `recall`, `f1`, `tp`, `fp`, or `expected` column exists anywhere. They come from SQL views
   over `retrieved`, per-match verdicts, and `missed_count`, with **NULL propagating**.

### Everything else

3. **SQLite, one file** (`runs.db`, gitignored), accessed through `sqlite3` from the stdlib. No
   ORM, no Alembic. Schema version lives in `PRAGMA user_version`; migrations are an ordered
   list of DDL steps applied idempotently at startup. `PRAGMA foreign_keys = ON` must be set on
   every connection — SQLite defaults it OFF and silently ignores FK constraints otherwise.
4. **`WAL` journal mode** so a read (the stats dashboard) never blocks a write (a search).
5. **Sub-threshold candidates (EVAL-08)** are stored in their own `candidates` table, one row
   per candidate, with `run_id`, `rank`, `score`, and the box. The run row stores
   `threshold_applied`. Top ~50 per run.
6. **Provenance on every run (EVAL-09):** `git_sha`, `method_version`, `config_hash`,
   `model_hashes` (JSON). Ratings from before and after a change are never pooled — the stats
   view groups by `method` **and** exposes provenance so a caller can slice by it.
7. **Latency is three columns** (`preprocess_ms`, `inference_ms`, `postprocess_ms`), never one
   total (EVAL-11). Total is derived.
8. **Outcome is an explicit column** with values `ok` / `empty` / `error` (EVAL-12). Precision is
   **undefined**, not zero, when `retrieved == 0`. Views must exclude abstentions from precision
   aggregates rather than averaging a zero into them.
9. **`rating_completeness` and `fp_source` are DERIVED in the view** (EVAL-13), not stored — they
   are functions of which fields are non-null. Storing them would let them drift out of sync
   with the fields they describe, which is the same class of bug as rule 2.
10. **Wilson score interval** for the thumbs-up rate (EVAL-14), computed in Python (SQLite lacks
    the needed math), with `n` always returned alongside. Handle `n = 0`, `p = 0`, and `p = 1`
    without dividing by zero or returning bounds outside `[0, 1]`.
11. **No method names anywhere in the API layer** (API-01). `GET /methods` is a loop over
    `registry.method_schemas()`. A test greps the `api/` package for each known method name and
    asserts zero hits.
12. **ONNX sessions load once in the FastAPI `lifespan`** and are reused (API-07). In Phase 3
    there are no ONNX-backed methods yet, so the lifespan builds an empty, typed session
    registry with the wiring in place — not a `TODO`.
13. **`exploration` is a first-class column** on `runs` from day one, defaulting to
    `"same-image-search"`. This is the Milestone 2 seam: a second exploration adds rows, not a
    schema migration.
14. **Structured errors (API-08).** A method that raises is caught at the route boundary and
    returned as a typed error body with the run still persisted (`outcome='error'`), because an
    error is evidence too. Never a bare 500 with a stack trace.

## Canonical References

- `.planning/IDEA.md` §7a — the full derivation of what gets logged and why, including the
  derived-metric formulas
- `.planning/research/PITFALLS.md` — SQLite NULL semantics in aggregates (`AVG` ignores NULL,
  `SUM` of all-NULL is NULL, `COUNT(col)` vs `COUNT(*)`), and the exact Wilson formula
- `src/object_search/schemas/records.py` — `Rating`, `RunRecord`, `Provenance`, `SliceMetadata`
  from Phase 1, including the already-implemented derived properties

## Specifics

### Derived metrics — the exact semantics the views must implement

```
R         = retrieved                                  # stored on the run
FP        = count(verdict = incorrect)                 # if per-match verdicts CONFIRMED
          | wrong_count                                # elif bare count entered
          | NULL                                       # else -> precision unavailable
TP        = R - FP                                     # NULL-propagating
FN        = missed_count                               # NULL -> recall unavailable
expected  = TP + FN                                    # inferred, NEVER entered
precision = TP / R                                     # UNDEFINED when R = 0 (abstention)
recall    = TP / (TP + FN)
F1        = 2PR / (P + R)
```

Null propagates rather than defaulting. A run missing `wrong_count` contributes to recall
aggregates **and to nothing else**. Every aggregate reports the `n` it was actually computed
over — and the threshold-sweep sample size is reported **separately** from the precision sample
size, because a run rated with a bare `wrong_count` has no score attribution and cannot feed
the sweep.

### Four SQLite behaviours that will silently break these views

All four were measured during research. Each one produces a plausible-looking wrong number
rather than an error, so each needs an explicit test.

1. **Integer division truncates.** `TP / R` with both columns `INTEGER` returns `0` for every
   case where `TP < R` — a precision of 0.87 becomes `0`. **Every derived ratio must cast to
   REAL**: `CAST(tp AS REAL) / r`. This is the most dangerous item in the phase because it makes
   precision look catastrophically bad rather than absent.
2. **`TOTAL()` returns `0.0` where `SUM()` returns `NULL`.** Use `SUM()` throughout. `TOTAL()`
   would convert "no assessed runs" into "zero false positives", which is precisely the
   null-becomes-zero failure the whole design exists to prevent — reintroduced via a SQL
   function name.
3. **`CHECK` constraints pass on NULL.** `CHECK (wrong_count >= 0)` does **not** reject NULL, and
   that is exactly what we want here — but it also means a CHECK can never enforce that a value
   was supplied. Presence rules live in the Pydantic layer and the views, not in CHECK.
4. **Pure-DDL migrations run in Python are not automatically transactional.** Wrap the migration
   sequence in an explicit `BEGIN`/`COMMIT` so a failure halfway cannot leave a half-migrated
   schema with `user_version` already bumped.

Per-match verdicts win over a conflicting bare count, and the discrepancy is **flagged**, not
silently reconciled (EVAL-18). Validation: `0 <= wrong_count <= R`, `missed_count >= 0`.

### Endpoints

- `GET /methods` → name, description, version, config JSON Schema per registered method
- `GET /images` → demo image list with dimensions and whether ground truth exists;
  `POST /images` → ad-hoc upload
- `POST /search` → `{image_id, exemplar, method, config, exploration?}` → `SearchResult`,
  persisting a `RunRecord` with provenance, latency breakdown, slice metadata, and candidates
- `POST /ratings` → records a rating against a run; validates against that run's `retrieved`
- `GET /stats` → per-method scoreboard: thumbs-up rate with Wilson CI and `n`, precision/recall
  with their own separate `n`s, latency percentiles (p50/p90/p99) from the breakdown,
  abstention and error counts, and the threshold-sweep-eligible count
- `GET /runs/{id}` → full run for the UI to re-render

### Tables (shape, not final DDL)

`runs` — id, exploration, image_id, method, method_version, exemplar box, config_json,
config_hash, outcome, error_kind, error_message, retrieved, threshold_applied,
preprocess_ms, inference_ms, postprocess_ms, git_sha, model_hashes_json,
slice_* columns (all NULL-able), diagnostics_json, created_at.

`matches` — run_id, idx, box, score, is_exemplar.
`candidates` — run_id, rank, box, score.
`ratings` — run_id, thumbs_up, wrong_count (NULL, no default), missed_count (NULL, no default),
verdicts_confirmed, unratable, note, created_at.
`match_verdicts` — rating_id, match_index, correct.
`paired_comparisons` — Phase 8 writes these; create the table now so Phase 8 is additive.

Store boxes as four integer columns, not JSON — they are queried and filtered.
Store `diagnostics_json` as TEXT; note in the docstring that heatmap PNGs are the bulky part
and that a size cap (drop the heatmap above N KB) keeps the DB usable.

## Deferred

- Alembic-style migration tooling — overkill for a single-user local DB; the ordered-DDL
  approach is explicitly chosen and its limits documented.
- Bradley-Terry ranking and the benchmark runner — Phase 8, but `paired_comparisons` exists now.
- Threshold-sweep computation itself — Phase 8 reads `candidates`; Phase 3 only guarantees the
  data is there.

## Scope Fence

**In:** the store (schema, migrations, queries, views, Wilson), the FastAPI app and all six
endpoint groups, provenance/latency/slice capture, typed errors, lifespan session registry.

**Out:** the frontend (Phase 4 — Phase 3 ships no HTML), any new search method, the benchmark.

## Risk Summary

- **The null discipline is easy to break at three layers**, and breaking any one silently
  produces a lying scoreboard. Defend it with a test at each layer: DDL has no `DEFAULT 0`
  (assert by reading `PRAGMA table_info`), the Pydantic model defaults to `None`, and a
  round-trip through the real HTTP endpoint preserves `None`.
- **SQLite `AVG` silently ignores NULL**, which is *usually* what we want but makes it easy to
  report an average over an unstated subset. Every view must emit its own `n` next to every
  rate — not one shared `n`.
- **`PRAGMA foreign_keys` defaults OFF.** Set it per connection or the FK constraints are
  decorative.
- **Coverage.** SQL-heavy code needs real round-trip tests against a temp DB, not mocks.
  Mocked DB tests here would pass while the actual DDL carried a `DEFAULT 0`.
