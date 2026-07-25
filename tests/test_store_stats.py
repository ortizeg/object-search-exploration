"""Derived-metric view and scoreboard tests, against a real temp-file database.

The four SQLite traps are asserted structurally (CAST before every division, SUM not
TOTAL) and behaviourally (a real 3/4 precision comes back as 0.75, not 0; a bare-thumbs
run contributes to no precision/recall sample; NULL propagates instead of defaulting).
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import pytest

from object_search.schemas.records import MatchVerdict, Rating
from object_search.schemas.search import MethodError, SearchOutcome
from object_search.store import schema
from object_search.store.db import open_store
from object_search.store.ratings import insert_rating
from object_search.store.runs import insert_run
from object_search.store.stats import scoreboard
from tests.store_helpers import make_run_record


@pytest.fixture
def db(tmp_path: Path) -> sqlite3.Connection:
    return open_store(tmp_path / "runs.db")


def _run(
    db: sqlite3.Connection,
    *,
    method: str = "ncc",
    outcome: SearchOutcome = SearchOutcome.OK,
    n_matches: int = 3,
) -> int:
    error = MethodError(kind="boom", message="crashed") if outcome is SearchOutcome.ERROR else None
    return insert_run(
        db,
        make_run_record(method=method, outcome=outcome, n_matches=n_matches, error=error),
    )


def _by_method(board: list) -> dict:
    return {stats.method: stats for stats in board}


# ------------------------------------------------------------------ structural (grep)


def test_views_cast_before_every_metric_division() -> None:
    metrics_ddl = dict(schema.VIEWS)["run_metrics"]
    # Every '/' that is a metric division is immediately preceded by a CAST(...) numerator.
    for match in re.finditer(r"/\s*NULLIF", metrics_ddl):
        prefix = metrics_ddl[: match.start()]
        assert "CAST(" in prefix, "a metric division lacks a CAST(... AS REAL) numerator"
    assert "CAST(" in metrics_ddl


def test_views_use_sum_not_total() -> None:
    all_view_ddl = "\n".join(ddl for _, ddl in schema.VIEWS).lower()
    assert "sum(" in all_view_ddl
    assert "total(" not in all_view_ddl  # TOTAL turns "not assessed" into 0.0 (§7.3)


def test_method_thumbs_counts_thumbs_up_not_star() -> None:
    thumbs_ddl = dict(schema.VIEWS)["method_thumbs"]
    assert "COUNT(ra.thumbs_up)" in thumbs_ddl


# ------------------------------------------------------------------ behavioural


def test_precision_is_a_real_fraction_not_zero(db: sqlite3.Connection) -> None:
    """A 3/4 precision must come back as 0.75, proving the CAST to REAL works (§7.4)."""
    run_id = _run(db, n_matches=4)
    insert_rating(db, Rating(run_id=run_id, thumbs_up=True, wrong_count=1))

    row = db.execute("SELECT precision FROM run_metrics WHERE run_id = ?", (run_id,)).fetchone()
    assert row["precision"] == 0.75
    assert isinstance(row["precision"], float)


def test_scoreboard_excludes_bare_thumbs_from_precision_and_recall(
    db: sqlite3.Connection,
) -> None:
    # A: fully assessed (precision + recall available).
    a = _run(db, n_matches=4)
    insert_rating(db, Rating(run_id=a, thumbs_up=True, wrong_count=1, missed_count=2))
    # B: bare thumbs-up -- feeds thumbs, but NOT precision or recall.
    b = _run(db, n_matches=3)
    insert_rating(db, Rating(run_id=b, thumbs_up=True))
    # C: abstention (retrieved=0) -- excluded from precision, not averaged as 0.
    c = _run(db, outcome=SearchOutcome.EMPTY, n_matches=0)
    insert_rating(db, Rating(run_id=c, thumbs_up=False))
    # D: error, unrated -- not in the thumbs sample at all.
    _run(db, outcome=SearchOutcome.ERROR, n_matches=0)

    stats = _by_method(scoreboard(db))["ncc"]

    # precision/recall each computed over exactly one run (A).
    assert stats.precision_n == 1
    assert stats.precision_mean == 0.75
    assert stats.recall_n == 1
    assert stats.recall_mean == pytest.approx(0.6)

    # thumbs sample excludes the unrated error run D (3 rated: A, B, C).
    assert stats.thumbs_n == 3
    assert stats.thumbs_n_up == 2  # A and B up, C down

    assert stats.abstention_count == 1
    assert stats.error_count == 1
    assert stats.threshold_sweep_eligible_count == 0


def test_all_null_counts_yield_no_precision(db: sqlite3.Connection) -> None:
    """A method with only bare-thumbs ratings has precision_n=0 and a None mean, not 0."""
    for _ in range(3):
        run_id = _run(db, method="blank")
        insert_rating(db, Rating(run_id=run_id, thumbs_up=True))

    stats = _by_method(scoreboard(db))["blank"]
    assert stats.precision_n == 0
    assert stats.precision_mean is None
    assert stats.recall_n == 0
    assert stats.recall_mean is None


def test_confirmed_verdicts_count_toward_threshold_sweep(db: sqlite3.Connection) -> None:
    run_id = _run(db, n_matches=3)
    insert_rating(
        db,
        Rating(
            run_id=run_id,
            thumbs_up=True,
            verdicts_confirmed=True,
            per_match_verdicts=(
                MatchVerdict(match_index=0, correct=True),
                MatchVerdict(match_index=1, correct=False),
                MatchVerdict(match_index=2, correct=True),
            ),
        ),
    )
    stats = _by_method(scoreboard(db))["ncc"]
    assert stats.threshold_sweep_eligible_count == 1
    # FP resolved from verdicts (1 wrong of 3) -> precision 2/3.
    assert stats.precision_n == 1
    assert stats.precision_mean == pytest.approx(2 / 3)


def test_scoreboard_ranks_by_wilson_lower_bound(db: sqlite3.Connection) -> None:
    """1/1 must rank BELOW 50/100 -- ranking by lower bound, not raw rate (EVAL-14)."""
    # method "lucky": one run, thumbs up (raw rate 1.0, but tiny n).
    lucky = _run(db, method="lucky")
    insert_rating(db, Rating(run_id=lucky, thumbs_up=True))
    # method "solid": 100 runs, 50 up (raw rate 0.5, but large n).
    for i in range(100):
        run_id = _run(db, method="solid")
        insert_rating(db, Rating(run_id=run_id, thumbs_up=(i < 50)))

    board = scoreboard(db)
    order = [stats.method for stats in board]
    assert order.index("solid") < order.index("lucky")


def test_unrated_method_sorts_last_with_none_ci(db: sqlite3.Connection) -> None:
    rated = _run(db, method="rated")
    insert_rating(db, Rating(run_id=rated, thumbs_up=True))
    _run(db, method="unrated")  # no rating at all

    board = _by_method(scoreboard(db))
    assert board["unrated"].thumbs_n == 0
    assert board["unrated"].thumbs_rate is None
    assert board["unrated"].thumbs_ci_lower is None
    # unrated ranks after rated.
    order = [s.method for s in scoreboard(db)]
    assert order.index("rated") < order.index("unrated")
