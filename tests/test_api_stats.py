"""Task 4: ``GET /stats`` over HTTP carries an honest ``n`` and a Wilson interval per rate.

The Phase 3 scoreboard criteria, exercised through the real ``TestClient``:

* a per-match + ``missed_count`` rating yields the correct precision, recall and expected count
  from the query layer (EVAL-07/EVAL-13) -- the numbers come out of the SQL views, and no
  derived metric was stored as a column;
* every rate on ``/stats`` ships beside the ``n`` it was computed over and a Wilson interval,
  and a method with ``n = 0`` thumbs renders that interval as ``null`` -- absent, not ``[0, 1]``
  (EVAL-14).

Runs are seeded through the store into the app's own database so the match counts are exact;
the rating and the scoreboard read both travel over HTTP.
"""

from __future__ import annotations

import pytest
from starlette.testclient import TestClient

from object_search.schemas.search import SearchOutcome
from object_search.store.db import connect
from object_search.store.runs import insert_run
from tests.store_helpers import make_run_record


def _seed_run(
    client: TestClient,
    *,
    n_matches: int,
    method: str = "ncc",
    outcome: SearchOutcome = SearchOutcome.OK,
) -> int:
    conn = connect(client.app.state.db_path)  # type: ignore[attr-defined]
    try:
        return insert_run(
            conn,
            make_run_record(method=method, n_matches=n_matches, outcome=outcome),
        )
    finally:
        conn.close()


def test_per_match_and_missed_yields_correct_precision_recall_expected(
    api_client: TestClient,
) -> None:
    """Confirmed verdicts (1 wrong of 4) + missed_count=2 -> P=0.75, R=0.6, expected=5."""
    run_id = _seed_run(api_client, n_matches=4)

    rated = api_client.post(
        "/ratings",
        json={
            "run_id": run_id,
            "thumbs_up": True,
            "missed_count": 2,
            "verdicts_confirmed": True,
            "per_match_verdicts": [
                {"match_index": 0, "correct": True},
                {"match_index": 1, "correct": False},
                {"match_index": 2, "correct": True},
                {"match_index": 3, "correct": True},
            ],
        },
    )
    assert rated.status_code == 200, rated.text

    # Precision and recall come off /stats, computed in the view -- TP=3, R=4, FN=2.
    ncc = {row["method"]: row for row in api_client.get("/stats").json()}["ncc"]
    assert ncc["precision_n"] == 1
    assert ncc["precision_mean"] == pytest.approx(0.75)
    assert ncc["recall_n"] == 1
    assert ncc["recall_mean"] == pytest.approx(0.6)
    # Per-match verdicts make the run threshold-sweep eligible; a bare count would not.
    assert ncc["threshold_sweep_eligible_count"] == 1

    # expected-count (TP + FN = 5) is derived in the run_metrics view, never a stored column.
    conn = connect(api_client.app.state.db_path)  # type: ignore[attr-defined]
    try:
        expected = conn.execute(
            "SELECT expected FROM run_metrics WHERE run_id = ?", (run_id,)
        ).fetchone()["expected"]
        run_columns = {row["name"] for row in conn.execute("PRAGMA table_info(runs)")}
        rating_columns = {row["name"] for row in conn.execute("PRAGMA table_info(ratings)")}
    finally:
        conn.close()
    assert expected == 5

    # No derived metric was stored: precision/recall/f1/expected exist only in views.
    forbidden = {"precision", "recall", "f1", "expected", "tp", "fp"}
    assert forbidden.isdisjoint(run_columns)
    assert forbidden.isdisjoint(rating_columns)


def test_every_rate_carries_n_and_a_wilson_interval(api_client: TestClient) -> None:
    """A rated method exposes a thumbs n and finite Wilson bounds inside [0, 1]."""
    run_id = _seed_run(api_client, n_matches=3)
    rated = api_client.post("/ratings", json={"run_id": run_id, "thumbs_up": True})
    assert rated.status_code == 200, rated.text

    ncc = {row["method"]: row for row in api_client.get("/stats").json()}["ncc"]
    assert ncc["thumbs_n"] == 1
    assert ncc["thumbs_rate"] == 1.0
    lower = ncc["thumbs_ci_lower"]
    upper = ncc["thumbs_ci_upper"]
    assert lower is not None and upper is not None
    assert 0.0 <= lower <= upper <= 1.0
    # A 1/1 does not fabricate certainty: the lower bound is well below 1.
    assert lower < 1.0


def test_method_with_zero_thumbs_renders_null_interval_not_zero_one(
    api_client: TestClient,
) -> None:
    """An unrated method has thumbs_n=0 and a null interval -- absent, not [0, 1]."""
    _seed_run(api_client, method="blank", n_matches=3)  # a run, but no rating

    blank = {row["method"]: row for row in api_client.get("/stats").json()}["blank"]
    assert blank["thumbs_n"] == 0
    assert blank["thumbs_rate"] is None
    assert blank["thumbs_ci_lower"] is None
    assert blank["thumbs_ci_upper"] is None


def test_empty_stats_is_an_empty_list(api_client: TestClient) -> None:
    """No runs at all -> an empty scoreboard, not an error."""
    response = api_client.get("/stats")
    assert response.status_code == 200
    assert response.json() == []
