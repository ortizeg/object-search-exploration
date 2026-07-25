"""Rating persistence tests -- the EVAL-17 heart of the phase, at the store layer.

Every assertion here defends the null-vs-zero distinction through a real database round
trip: NULL means "not assessed", 0 means "assessed, none found", and the two must stay
distinguishable in the raw row, not just in the Pydantic model.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from object_search.schemas.records import MatchVerdict, Rating
from object_search.store.db import open_store
from object_search.store.ratings import (
    get_rating,
    insert_rating,
    validate_rating_against_run,
)
from object_search.store.runs import insert_run
from tests.store_helpers import make_run_record


@pytest.fixture
def db(tmp_path: Path) -> sqlite3.Connection:
    return open_store(tmp_path / "runs.db")


def _run(db: sqlite3.Connection, n_matches: int = 3) -> int:
    return insert_run(db, make_run_record(n_matches=n_matches))


def test_bare_thumbs_up_stores_null_counts(db: sqlite3.Connection) -> None:
    """A one-click thumbs-up must leave both count columns NULL in the raw row."""
    run_id = _run(db)
    rating_id = insert_rating(db, Rating(run_id=run_id, thumbs_up=True))

    row = db.execute(
        "SELECT wrong_count, missed_count FROM ratings WHERE id = ?", (rating_id,)
    ).fetchone()
    assert row["wrong_count"] is None
    assert row["missed_count"] is None
    # And the model round-trips as None, not 0.
    loaded = get_rating(db, rating_id)
    assert loaded.wrong_count is None
    assert loaded.missed_count is None


def test_explicit_zero_is_distinct_from_null(db: sqlite3.Connection) -> None:
    """An explicit 'all correct' (0) must read back as 0, never collapsed into NULL."""
    run_id = _run(db)
    rating_id = insert_rating(db, Rating(run_id=run_id, thumbs_up=True, wrong_count=0))

    row = db.execute(
        "SELECT wrong_count, missed_count FROM ratings WHERE id = ?", (rating_id,)
    ).fetchone()
    assert row["wrong_count"] == 0  # assessed, none found
    assert row["missed_count"] is None  # not assessed
    loaded = get_rating(db, rating_id)
    assert loaded.wrong_count == 0
    assert loaded.wrong_count is not None


def test_full_rating_round_trips(db: sqlite3.Connection) -> None:
    run_id = _run(db)
    rating = Rating(
        run_id=run_id,
        thumbs_up=False,
        wrong_count=1,
        missed_count=2,
        note="one false positive, two missed",
    )
    rating_id = insert_rating(db, rating)
    loaded = get_rating(db, rating_id)
    assert loaded == rating


def test_per_match_verdicts_round_trip(db: sqlite3.Connection) -> None:
    run_id = _run(db)
    rating = Rating(
        run_id=run_id,
        thumbs_up=True,
        verdicts_confirmed=True,
        per_match_verdicts=(
            MatchVerdict(match_index=0, correct=True),
            MatchVerdict(match_index=1, correct=False),
            MatchVerdict(match_index=2, correct=True),
        ),
    )
    rating_id = insert_rating(db, rating)
    loaded = get_rating(db, rating_id)
    assert loaded.per_match_verdicts == rating.per_match_verdicts
    assert loaded.verdicts_confirmed is True


def test_negative_count_is_rejected_by_check_constraint(db: sqlite3.Connection) -> None:
    """CHECK passes on NULL but must still reject a real negative (PITFALLS §7.1)."""
    run_id = _run(db)
    with pytest.raises(sqlite3.IntegrityError), db:
        db.execute(
            "INSERT INTO ratings (run_id, thumbs_up, wrong_count, created_at) "
            "VALUES (?, 1, -1, '2026-01-01T00:00:00+00:00')",
            (run_id,),
        )


def test_store_layer_contains_no_count_coercion() -> None:
    """grep guard: no `or 0`, `= 0` default, or COALESCE on the count columns."""
    store_dir = Path("src/object_search/store")
    for path in store_dir.glob("*.py"):
        text = path.read_text().lower()
        assert "coalesce(ra.wrong_count" not in text
        assert "coalesce(ra.missed_count" not in text
        assert "wrong_count or 0" not in text
        assert "missed_count or 0" not in text


def test_validate_flags_over_retrieved_wrong_count(db: sqlite3.Connection) -> None:
    rating = Rating(run_id=1, thumbs_up=False, wrong_count=5)
    ok, reason = validate_rating_against_run(rating, retrieved=3)
    assert ok is False
    assert reason is not None
    assert "exceeds" in reason


def test_validate_flags_verdict_count_discrepancy(db: sqlite3.Connection) -> None:
    """Confirmed per-match verdicts and a bare wrong_count that disagree are flagged,
    and both are kept in the store (EVAL-18)."""
    run_id = _run(db)
    rating = Rating(
        run_id=run_id,
        thumbs_up=False,
        wrong_count=2,  # human typed 2 wrong...
        verdicts_confirmed=True,
        per_match_verdicts=(
            MatchVerdict(match_index=0, correct=False),  # ...but only 1 verdict is wrong
            MatchVerdict(match_index=1, correct=True),
            MatchVerdict(match_index=2, correct=True),
        ),
    )
    rating_id = insert_rating(db, rating)

    ok, reason = validate_rating_against_run(rating, retrieved=3)
    assert ok is False
    assert reason is not None and "disagrees" in reason

    # Both signals are preserved -- the store did not reconcile them away.
    loaded = get_rating(db, rating_id)
    assert loaded.wrong_count == 2
    assert loaded.per_match_verdicts is not None
    assert sum(1 for v in loaded.per_match_verdicts if not v.correct) == 1


def test_validate_accepts_a_consistent_rating(db: sqlite3.Connection) -> None:
    rating = Rating(run_id=1, thumbs_up=True, wrong_count=1, missed_count=0)
    ok, reason = validate_rating_against_run(rating, retrieved=3)
    assert ok is True
    assert reason is None
