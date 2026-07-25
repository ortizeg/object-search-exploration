"""Task 4: ``POST /ratings`` over HTTP keeps the null discipline and the bounds flag intact.

These are the Phase 3 success criteria exercised end to end through the real ``TestClient``:
a bare thumbs-up submitted over HTTP is stored with **NULL** counts (not ``0``) and contributes
to no precision/recall aggregate, and a ``wrong_count`` that exceeds what the run returned is a
``422`` with the flag rather than a silent clamp (EVAL-17, EVAL-18).

Runs are seeded through the store directly into the same database the app serves
(``client.app.state.db_path``) so each test controls the exact ``retrieved`` count; the rating
itself always travels over HTTP, which is the path the invariant must survive.
"""

from __future__ import annotations

from starlette.testclient import TestClient

from object_search.store.db import connect
from object_search.store.runs import insert_run
from tests.store_helpers import make_run_record

# A concrete exemplar taken from chipset-01's ground truth (box 0, 24x24 near the top edge).
_CHIPSET_IMAGE = "chipset/chipset-01.png"
_CHIPSET_EXEMPLAR = {"box": {"x": 293, "y": 12, "w": 24, "h": 24}, "label": "chip"}


def _seed_run(client: TestClient, *, n_matches: int, method: str = "ncc") -> int:
    """Insert a run with a known match count straight into the app's store; return its id."""
    conn = connect(client.app.state.db_path)  # type: ignore[attr-defined]
    try:
        return insert_run(conn, make_run_record(method=method, n_matches=n_matches))
    finally:
        conn.close()


def test_bare_thumbs_up_stores_null_counts_and_is_ignored(api_client: TestClient) -> None:
    """A thumbs-up with no counts POSTed over HTTP stays NULL and feeds no precision/recall n."""
    # A run created over the real HTTP search path -- the whole loop is HTTP end to end.
    search = api_client.post(
        "/search",
        json={
            "image_id": _CHIPSET_IMAGE,
            "exemplar": _CHIPSET_EXEMPLAR,
            "method": "ncc",
            "config": {},
        },
    )
    assert search.status_code == 200, search.text
    run_id = search.json()["run_id"]

    # Bare thumbs-up: no wrong_count, no missed_count in the body at all.
    rated = api_client.post("/ratings", json={"run_id": run_id, "thumbs_up": True})
    assert rated.status_code == 200, rated.text
    assert rated.json()["rating_id"] >= 1

    # The stored row carries SQL NULL, not 0 -- the not-assessed distinction survived HTTP.
    conn = connect(api_client.app.state.db_path)  # type: ignore[attr-defined]
    try:
        row = conn.execute(
            "SELECT wrong_count, missed_count FROM ratings WHERE run_id = ?", (run_id,)
        ).fetchone()
    finally:
        conn.close()
    assert row["wrong_count"] is None
    assert row["missed_count"] is None

    # And the scoreboard ignores it: this run feeds neither the precision nor the recall n.
    board = {row["method"]: row for row in api_client.get("/stats").json()}
    ncc = board["ncc"]
    assert ncc["thumbs_n"] == 1
    assert ncc["thumbs_n_up"] == 1
    assert ncc["precision_n"] == 0
    assert ncc["precision_mean"] is None
    assert ncc["recall_n"] == 0
    assert ncc["recall_mean"] is None


def test_wrong_count_greater_than_retrieved_is_422_not_clamped(api_client: TestClient) -> None:
    """``wrong_count > retrieved`` is rejected with the flag, never silently reduced to R."""
    run_id = _seed_run(api_client, n_matches=3)

    response = api_client.post(
        "/ratings",
        json={"run_id": run_id, "thumbs_up": False, "wrong_count": 4},
    )
    assert response.status_code == 422, response.text
    error = response.json()["error"]
    assert error["kind"] == "rating_rejected"
    # The message names the offending count and the run's box count -- the flag, not a clamp.
    assert "4" in error["message"]
    assert "3" in error["message"]

    # Nothing was stored: a rejected rating leaves no half-written row.
    conn = connect(api_client.app.state.db_path)  # type: ignore[attr-defined]
    try:
        count = conn.execute(
            "SELECT COUNT(*) AS n FROM ratings WHERE run_id = ?", (run_id,)
        ).fetchone()["n"]
    finally:
        conn.close()
    assert count == 0


def test_per_match_verdict_conflicting_with_wrong_count_is_flagged(api_client: TestClient) -> None:
    """Confirmed verdicts disagreeing with a bare wrong_count are flagged, not reconciled."""
    run_id = _seed_run(api_client, n_matches=3)

    # Verdicts say 1 wrong; the bare count says 2 -- a contradiction the API must flag (EVAL-18).
    response = api_client.post(
        "/ratings",
        json={
            "run_id": run_id,
            "thumbs_up": True,
            "wrong_count": 2,
            "verdicts_confirmed": True,
            "per_match_verdicts": [
                {"match_index": 0, "correct": True},
                {"match_index": 1, "correct": False},
                {"match_index": 2, "correct": True},
            ],
        },
    )
    assert response.status_code == 422, response.text
    error = response.json()["error"]
    assert error["kind"] == "rating_rejected"
    assert "disagree" in error["message"]


def test_rating_an_unknown_run_is_404(api_client: TestClient) -> None:
    response = api_client.post("/ratings", json={"run_id": 999999, "thumbs_up": True})
    assert response.status_code == 404
    assert response.json()["error"]["kind"] == "run_not_found"


def test_assessed_zero_is_distinct_from_unassessed_null(api_client: TestClient) -> None:
    """An explicit ``wrong_count=0`` is stored as 0, not folded into NULL -- the mirror rule."""
    run_id = _seed_run(api_client, n_matches=3)

    rated = api_client.post(
        "/ratings",
        json={"run_id": run_id, "thumbs_up": True, "wrong_count": 0, "missed_count": 0},
    )
    assert rated.status_code == 200, rated.text

    # 0 means "assessed, none found" and must round-trip as 0, feeding a perfect precision.
    board = {row["method"]: row for row in api_client.get("/stats").json()}
    ncc = board["ncc"]
    assert ncc["precision_n"] == 1
    assert ncc["precision_mean"] == 1.0
    assert ncc["recall_n"] == 1
    assert ncc["recall_mean"] == 1.0

    # And the stored columns are a real 0, not NULL -- the mirror image of the bare-thumbs rule.
    conn = connect(api_client.app.state.db_path)  # type: ignore[attr-defined]
    try:
        row = conn.execute(
            "SELECT wrong_count, missed_count FROM ratings WHERE run_id = ?", (run_id,)
        ).fetchone()
    finally:
        conn.close()
    assert row["wrong_count"] == 0
    assert row["missed_count"] == 0


def test_wrong_count_equal_to_retrieved_is_accepted(api_client: TestClient) -> None:
    """The bound is inclusive: ``wrong_count == retrieved`` (every box wrong) is valid."""
    run_id = _seed_run(api_client, n_matches=2)
    rated = api_client.post(
        "/ratings",
        json={"run_id": run_id, "thumbs_up": False, "wrong_count": 2},
    )
    assert rated.status_code == 200, rated.text
