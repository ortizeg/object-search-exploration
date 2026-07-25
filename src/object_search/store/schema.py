"""The DDL for the run/rating store, as versioned string constants.

Two rules are enforced *here*, at the layer that is hardest to accidentally undo:

1. **The human-count columns carry no ``DEFAULT``** (EVAL-17). ``wrong_count`` and
   ``missed_count`` are plain ``INTEGER`` -- nullable, with no default -- so a row
   inserted without them stores ``NULL`` = "not assessed", never ``0`` = "assessed,
   none found". A ``DEFAULT 0`` here would make every unreviewed run claim perfect
   precision and recall. A test reads ``PRAGMA table_info(ratings)`` and asserts the
   ``dflt_value`` of both columns is ``None``.
2. **No derived metric is a stored column** (EVAL-07). There is no ``precision``,
   ``recall``, ``f1``, ``tp``, ``fp`` or ``expected`` column on any *table*. Those live
   only in the VIEWs below, computed from the raw columns with ``NULL`` propagating.

The ``CHECK`` constraints on the count columns are written with an explicit
``IS NULL OR ...`` branch. A bare ``CHECK (wrong_count >= 0)`` already passes on ``NULL``
(SQLite treats a ``NULL`` check result as satisfied, PITFALLS §7.1) -- spelling out the
``NULL`` branch documents that this is *intended*: presence is enforced in Pydantic and
the API, never in a CHECK.

Wide-but-flat is a deliberate choice here over the split-payload table in PITFALLS §7.8:
the plan lists these columns on ``runs`` directly, and the diagnostics blob -- the only
genuinely fat field -- is size-capped in :mod:`object_search.store.runs` before it is
stored.
"""

from __future__ import annotations

SCHEMA_VERSION = 1
"""Target ``PRAGMA user_version``. Bump when adding a migration step below."""

DEFAULT_EXPLORATION = "same-image-search"
"""The Milestone 2 seam: a second exploration adds rows, not a schema migration."""


_RUNS = """
CREATE TABLE runs (
    id                            INTEGER PRIMARY KEY,
    exploration                   TEXT    NOT NULL DEFAULT 'same-image-search',
    image_id                      TEXT    NOT NULL,
    method                        TEXT    NOT NULL,
    method_version                TEXT    NOT NULL,
    exemplar_x                    INTEGER NOT NULL,
    exemplar_y                    INTEGER NOT NULL,
    exemplar_w                    INTEGER NOT NULL,
    exemplar_h                    INTEGER NOT NULL,
    exemplar_label                TEXT,
    config_json                   TEXT    NOT NULL,
    config_hash                   TEXT    NOT NULL,
    outcome                       TEXT    NOT NULL CHECK (outcome IN ('ok', 'empty', 'error')),
    error_kind                    TEXT,
    error_message                 TEXT,
    retrieved                     INTEGER NOT NULL,
    threshold_applied             REAL,
    preprocess_ms                 REAL    NOT NULL,
    inference_ms                  REAL    NOT NULL,
    postprocess_ms                REAL    NOT NULL,
    git_sha                       TEXT    NOT NULL,
    python_version                TEXT    NOT NULL,
    numpy_version                 TEXT    NOT NULL,
    cv2_version                   TEXT    NOT NULL,
    onnxruntime_version           TEXT    NOT NULL,
    ort_providers                 TEXT    NOT NULL,
    model_hashes_json             TEXT    NOT NULL,
    pixi_lock_sha256              TEXT    NOT NULL,
    provenance_created_at         TEXT    NOT NULL,
    slice_true_instance_count     INTEGER,
    slice_scale_min               REAL,
    slice_scale_max               REAL,
    slice_rotation_min            REAL,
    slice_rotation_max            REAL,
    slice_clutter                 REAL,
    slice_exemplar_keypoint_count INTEGER,
    diagnostics_json              TEXT,
    created_at                    TEXT    NOT NULL
)
"""

_MATCHES = """
CREATE TABLE matches (
    id             INTEGER PRIMARY KEY,
    run_id         INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    idx            INTEGER NOT NULL,
    x              INTEGER NOT NULL,
    y              INTEGER NOT NULL,
    w              INTEGER NOT NULL,
    h              INTEGER NOT NULL,
    score          REAL    NOT NULL,
    is_exemplar    INTEGER NOT NULL DEFAULT 0,
    transform_json TEXT
)
"""

_CANDIDATES = """
CREATE TABLE candidates (
    id     INTEGER PRIMARY KEY,
    run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    rank   INTEGER NOT NULL,
    x      INTEGER NOT NULL,
    y      INTEGER NOT NULL,
    w      INTEGER NOT NULL,
    h      INTEGER NOT NULL,
    score  REAL    NOT NULL
)
"""

# wrong_count / missed_count: INTEGER, nullable, NO DEFAULT. This is the EVAL-17 line.
# The CHECKs spell out the NULL branch so the "CHECK passes on NULL" behaviour is on
# purpose, not an accident (PITFALLS §7.1).
_RATINGS = """
CREATE TABLE ratings (
    id                 INTEGER PRIMARY KEY,
    run_id             INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    thumbs_up          INTEGER NOT NULL,
    wrong_count        INTEGER,
    missed_count       INTEGER,
    verdicts_confirmed INTEGER NOT NULL DEFAULT 0,
    unratable          INTEGER NOT NULL DEFAULT 0,
    note               TEXT,
    created_at         TEXT    NOT NULL,
    CHECK (wrong_count  IS NULL OR wrong_count  >= 0),
    CHECK (missed_count IS NULL OR missed_count >= 0)
)
"""

_MATCH_VERDICTS = """
CREATE TABLE match_verdicts (
    id          INTEGER PRIMARY KEY,
    rating_id   INTEGER NOT NULL REFERENCES ratings(id) ON DELETE CASCADE,
    match_index INTEGER NOT NULL,
    correct     INTEGER NOT NULL
)
"""

# Created now so Phase 8 (paired comparison / Bradley-Terry) is purely additive.
_PAIRED_COMPARISONS = """
CREATE TABLE paired_comparisons (
    id         INTEGER PRIMARY KEY,
    exploration TEXT   NOT NULL DEFAULT 'same-image-search',
    image_id   TEXT    NOT NULL,
    exemplar_x INTEGER NOT NULL,
    exemplar_y INTEGER NOT NULL,
    exemplar_w INTEGER NOT NULL,
    exemplar_h INTEGER NOT NULL,
    method_a   TEXT    NOT NULL,
    method_b   TEXT    NOT NULL,
    winner     TEXT    NOT NULL CHECK (winner IN ('a', 'b', 'tie')),
    created_at TEXT    NOT NULL
)
"""

MIGRATIONS: list[tuple[int, str]] = [
    (
        1,
        ";\n".join(
            [
                _RUNS,
                _MATCHES,
                _CANDIDATES,
                _RATINGS,
                _MATCH_VERDICTS,
                _PAIRED_COMPARISONS,
            ]
        ),
    ),
]
"""Ordered ``(version, ddl)`` steps. Each is applied inside its own explicit
``BEGIN``/``COMMIT`` by :func:`object_search.store.migrations.migrate`, because a
pure-DDL sequence run through Python's ``sqlite3`` is **not** auto-transactional
(PITFALLS §7.9)."""


# --------------------------------------------------------------------------- views
#
# Derived metrics live ONLY here (EVAL-07). Every ratio CASTs its numerator to REAL
# first -- integer division would truncate a precision of 0.87 to 0 (PITFALLS §7.4) --
# and every aggregate uses SUM, never TOTAL (TOTAL turns "nothing assessed" into 0.0,
# reintroducing the null-becomes-zero lie via a function name, PITFALLS §7.3).

# Per-run false-positive resolution. Per-match verdicts win over a bare count (EVAL-18);
# fp is NULL when neither is available, which is what makes precision "unavailable"
# rather than zero downstream.
_VIEW_RUN_FP = """
CREATE VIEW run_fp AS
SELECT
    r.id          AS run_id,
    r.method      AS method,
    r.exploration AS exploration,
    r.outcome     AS outcome,
    r.retrieved   AS retrieved,
    r.preprocess_ms + r.inference_ms + r.postprocess_ms AS total_ms,
    ra.id                 AS rating_id,
    ra.thumbs_up          AS thumbs_up,
    ra.verdicts_confirmed AS verdicts_confirmed,
    ra.wrong_count        AS wrong_count,
    ra.missed_count       AS missed_count,
    CASE
        WHEN ra.verdicts_confirmed = 1
             AND (SELECT COUNT(*) FROM match_verdicts mv WHERE mv.rating_id = ra.id) > 0
            THEN 'per_match'
        WHEN ra.wrong_count IS NOT NULL
            THEN 'count'
        ELSE NULL
    END AS fp_source,
    CASE
        WHEN ra.verdicts_confirmed = 1
             AND (SELECT COUNT(*) FROM match_verdicts mv WHERE mv.rating_id = ra.id) > 0
            THEN (SELECT SUM(CASE WHEN mv.correct = 0 THEN 1 ELSE 0 END)
                  FROM match_verdicts mv WHERE mv.rating_id = ra.id)
        WHEN ra.wrong_count IS NOT NULL
            THEN ra.wrong_count
        ELSE NULL
    END AS fp
FROM runs r
LEFT JOIN ratings ra ON ra.run_id = r.id
"""

# tp/fn/expected/precision/recall, all NULL-propagating. Built on run_fp because SQLite
# cannot reference a sibling column alias (fp) within the same SELECT list.
_VIEW_RUN_METRICS = """
CREATE VIEW run_metrics AS
SELECT
    f.run_id,
    f.method,
    f.exploration,
    f.outcome,
    f.retrieved,
    f.total_ms,
    f.thumbs_up,
    f.fp_source,
    f.fp,
    (f.retrieved - f.fp)                       AS tp,
    f.missed_count                             AS fn,
    ((f.retrieved - f.fp) + f.missed_count)    AS expected,
    CASE
        WHEN f.fp IS NULL      THEN 'not_assessed'
        WHEN f.retrieved = 0   THEN 'undefined_abstention'
        ELSE 'ok'
    END AS precision_status,
    CAST(f.retrieved - f.fp AS REAL) / NULLIF(f.retrieved, 0)                       AS precision,
    CAST(f.retrieved - f.fp AS REAL) / NULLIF((f.retrieved - f.fp) + f.missed_count, 0) AS recall
FROM run_fp f
"""

# Thumbs aggregate: n_rated = COUNT(thumbs_up), NOT COUNT(*) -- COUNT(*) would count
# unrated runs as thumbs-down (PITFALLS §8.2 note 2). The Wilson interval is computed in
# Python from n_up / n_rated (SQLite lacks the inverse-normal it needs).
_VIEW_METHOD_THUMBS = """
CREATE VIEW method_thumbs AS
SELECT
    r.method                                        AS method,
    SUM(CASE WHEN ra.thumbs_up = 1 THEN 1 ELSE 0 END) AS n_up,
    COUNT(ra.thumbs_up)                             AS n_rated,
    COUNT(*)                                        AS n_runs
FROM runs r
LEFT JOIN ratings ra ON ra.run_id = r.id
GROUP BY r.method
"""

VIEWS: list[tuple[str, str]] = [
    ("run_fp", _VIEW_RUN_FP),
    ("run_metrics", _VIEW_RUN_METRICS),
    ("method_thumbs", _VIEW_METHOD_THUMBS),
]
"""``(name, create_ddl)`` for every derived-metric view. Dropped and recreated
idempotently after each migration, because rebuilding a base table would otherwise leave
a view silently referencing a dropped column (PITFALLS §7.9)."""
