"""Persist a :class:`Rating`, keeping the null-vs-zero distinction intact end to end.

This is the store-layer half of EVAL-17. The rule that governs every line here:
``wrong_count`` and ``missed_count`` are written **exactly as the model carries them** --
``None`` stays ``None`` (SQL ``NULL`` = "not assessed"), ``0`` stays ``0`` ("assessed,
none found"). There is no ``or 0``, no ``COALESCE``, no ``.get(..., 0)`` anywhere in this
module, because any one of them silently turns every unreviewed run into a perfect score.

The presence/bounds rule (``0 <= wrong_count <= retrieved``) cannot live on the column --
a ``CHECK`` against the run's ``retrieved`` count would pass on ``NULL`` (PITFALLS §7.1)
and cannot reach another table anyway -- so it lives in
:func:`validate_rating_against_run`, which the route layer (plan 03-02) calls after
loading the run. Per-match verdicts win over a conflicting bare count, and the
disagreement is *flagged*, never silently reconciled (EVAL-18).
"""

from __future__ import annotations

import sqlite3
from datetime import datetime

from loguru import logger

from object_search.schemas.records import MatchVerdict, Rating


def insert_rating(conn: sqlite3.Connection, rating: Rating) -> int:
    """Write a rating and its per-match verdicts in one transaction; return the new id.

    ``wrong_count`` and ``missed_count`` are passed straight through: a ``None`` becomes
    a SQL ``NULL``. Per-match verdicts are written only when the rating carries them.
    """
    with conn:
        cursor = conn.execute(
            """
            INSERT INTO ratings (
                run_id, thumbs_up, wrong_count, missed_count,
                verdicts_confirmed, unratable, note, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                rating.run_id,
                int(rating.thumbs_up),
                # No coercion: None -> NULL (EVAL-17). Do not "fix" this to `or 0`.
                rating.wrong_count,
                rating.missed_count,
                int(rating.verdicts_confirmed),
                int(rating.unratable),
                rating.note,
                rating.created_at.isoformat(),
            ),
        )
        rating_id = int(cursor.lastrowid or 0)

        if rating.per_match_verdicts is not None:
            conn.executemany(
                """
                INSERT INTO match_verdicts (rating_id, match_index, correct)
                VALUES (?, ?, ?)
                """,
                [
                    (rating_id, verdict.match_index, int(verdict.correct))
                    for verdict in rating.per_match_verdicts
                ],
            )

    logger.debug("stored rating {} for run {}", rating_id, rating.run_id)
    return rating_id


def get_rating(conn: sqlite3.Connection, rating_id: int) -> Rating:
    """Reconstruct a :class:`Rating`, per-match verdicts included.

    ``NULL`` count columns come back as ``None``, preserving the not-assessed distinction.

    Raises:
        KeyError: If no rating with ``rating_id`` exists.
    """
    row = conn.execute("SELECT * FROM ratings WHERE id = ?", (rating_id,)).fetchone()
    if row is None:
        raise KeyError(f"no rating with id {rating_id}")

    verdict_rows = conn.execute(
        "SELECT match_index, correct FROM match_verdicts WHERE rating_id = ? ORDER BY match_index",
        (rating_id,),
    ).fetchall()
    verdicts = (
        tuple(
            MatchVerdict(match_index=vr["match_index"], correct=bool(vr["correct"]))
            for vr in verdict_rows
        )
        if verdict_rows
        else None
    )

    return Rating(
        run_id=row["run_id"],
        thumbs_up=bool(row["thumbs_up"]),
        # None stays None; a stored 0 stays 0. The two are distinct facts (EVAL-17).
        wrong_count=row["wrong_count"],
        missed_count=row["missed_count"],
        per_match_verdicts=verdicts,
        verdicts_confirmed=bool(row["verdicts_confirmed"]),
        unratable=bool(row["unratable"]),
        note=row["note"],
        created_at=datetime.fromisoformat(row["created_at"]),
    )


def validate_rating_against_run(rating: Rating, retrieved: int) -> tuple[bool, str | None]:
    """Check a rating against the run's returned-box count (EVAL-18).

    Enforces ``0 <= wrong_count <= retrieved`` (and the per-match index bounds) via the
    model's own :meth:`Rating.validate_against_retrieved`, then *flags* -- does not
    reconcile -- a disagreement between confirmed per-match verdicts and a bare
    ``wrong_count``.

    Args:
        rating: The submitted rating.
        retrieved: ``len(run.result.matches)`` for the run being rated.

    Returns:
        ``(True, None)`` when consistent, otherwise ``(False, reason)`` with a message
        fit to show a user. The bounds violation is reported before the discrepancy.
    """
    ok, reason = rating.validate_against_retrieved(retrieved)
    if not ok:
        return (False, reason)
    if rating.has_fp_discrepancy:
        return (
            False,
            f"wrong_count={rating.wrong_count} disagrees with {rating.false_positives} "
            f"false positive(s) from confirmed per-match verdicts; flagged, not "
            f"reconciled (EVAL-18)",
        )
    return (True, None)
