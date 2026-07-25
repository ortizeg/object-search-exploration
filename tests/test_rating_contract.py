"""Plan 04-02: the rating widget's payload contract, proven from the Python side.

The frontend has no JS test runner by design (adding one would mean npm, which the project
constraints forbid). The load-bearing logic in ``frontend/js/payload.js`` --
``buildRatingPayload`` -- is therefore kept a *pure* function, and this suite closes the loop
the JS cannot: it POSTs the **exact** bodies that function produces, for three representative
widget states, to the real ``/ratings`` endpoint, and asserts the stored row and the
``/stats`` aggregate match. Each body below is annotated with the widget state that yields it.

It also carries the phase's single highest-risk guard as a hard test: ``payload.js`` must
contain no ``|| 0`` and no ``?? 0`` (UI-08 / EVAL-17). That one character collapses "not
assessed" (``null``) into "assessed, none" (``0``) and makes every unreviewed run claim
perfect precision and recall -- the exact failure the whole store layer exists to prevent.
"""

from __future__ import annotations

import pytest
from starlette.testclient import TestClient

from object_search.api.static import frontend_dir
from object_search.store.db import connect
from object_search.store.runs import insert_run
from tests.store_helpers import make_run_record


def _seed_run(client: TestClient, *, n_matches: int, method: str = "ncc") -> int:
    """Insert a run with a known ``retrieved`` count straight into the app's store."""
    conn = connect(client.app.state.db_path)  # type: ignore[attr-defined]
    try:
        return insert_run(conn, make_run_record(method=method, n_matches=n_matches))
    finally:
        conn.close()


def _stats_for(client: TestClient, method: str) -> dict:
    board = {row["method"]: row for row in client.get("/stats").json()}
    return board[method]


def test_payload_js_has_no_zero_coercion() -> None:
    """UI-08's highest-risk line: no ``|| 0`` / ``?? 0`` may appear in the submit builder."""
    source = (frontend_dir() / "js" / "payload.js").read_text(encoding="utf-8")
    assert "|| 0" not in source, "`|| 0` in payload.js collapses null (not assessed) into 0"
    assert "?? 0" not in source, "`?? 0` in payload.js collapses null (not assessed) into 0"


def test_bare_thumbs_up_body_stores_null_and_feeds_no_precision(api_client: TestClient) -> None:
    """Widget state: thumbs up, no mode chosen, both counts untouched.

    ``buildRatingPayload`` omits both counts entirely, so the body is just the run and the
    thumb. The counts must land as SQL NULL and feed neither precision nor recall.
    """
    run_id = _seed_run(api_client, n_matches=3)

    # Exactly what buildRatingPayload({thumbsUp:true, wrongMode:null, wrongCount:"",
    # missedCount:""}) produces -- no wrong_count, no missed_count keys at all.
    body = {"run_id": run_id, "thumbs_up": True}
    assert "wrong_count" not in body and "missed_count" not in body

    rated = api_client.post("/ratings", json=body)
    assert rated.status_code == 200, rated.text

    conn = connect(api_client.app.state.db_path)  # type: ignore[attr-defined]
    try:
        row = conn.execute(
            "SELECT wrong_count, missed_count FROM ratings WHERE run_id = ?", (run_id,)
        ).fetchone()
    finally:
        conn.close()
    assert row["wrong_count"] is None
    assert row["missed_count"] is None

    ncc = _stats_for(api_client, "ncc")
    assert ncc["thumbs_n"] == 1
    assert ncc["thumbs_n_up"] == 1
    assert ncc["precision_n"] == 0
    assert ncc["precision_mean"] is None
    assert ncc["recall_n"] == 0
    assert ncc["recall_mean"] is None


def test_confirmed_verdicts_body_feeds_per_match_precision(api_client: TestClient) -> None:
    """Widget state: per-match mode, box #1 clicked wrong, "Confirm verdicts" pressed.

    ``buildRatingPayload`` emits ``verdicts_confirmed: true`` and a full per-match array
    (one entry per match, ``correct`` false only for the clicked index). With one wrong of
    three retrieved, precision is (3-1)/3 and recall stays unavailable (missed untouched).
    """
    run_id = _seed_run(api_client, n_matches=3)

    body = {
        "run_id": run_id,
        "thumbs_up": True,
        "verdicts_confirmed": True,
        "per_match_verdicts": [
            {"match_index": 0, "correct": True},
            {"match_index": 1, "correct": False},
            {"match_index": 2, "correct": True},
        ],
    }
    rated = api_client.post("/ratings", json=body)
    assert rated.status_code == 200, rated.text

    conn = connect(api_client.app.state.db_path)  # type: ignore[attr-defined]
    try:
        rating = conn.execute(
            "SELECT id, wrong_count, verdicts_confirmed FROM ratings WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        verdicts = conn.execute(
            "SELECT match_index, correct FROM match_verdicts WHERE rating_id = ? "
            "ORDER BY match_index",
            (rating["id"],),
        ).fetchall()
    finally:
        conn.close()
    # A per-match rating never carries a bare wrong_count -- the modes are exclusive.
    assert rating["wrong_count"] is None
    assert rating["verdicts_confirmed"] == 1
    assert [(v["match_index"], v["correct"]) for v in verdicts] == [(0, 1), (1, 0), (2, 1)]

    ncc = _stats_for(api_client, "ncc")
    assert ncc["precision_n"] == 1
    assert ncc["precision_mean"] == pytest.approx(2.0 / 3.0)
    # Recall was never assessed (missed_count untouched) -- it must stay unavailable, not 0.
    assert ncc["recall_n"] == 0
    assert ncc["recall_mean"] is None
    # Per-match evidence is the sweep-eligible kind (EVAL-18).
    assert ncc["threshold_sweep_eligible_count"] == 1


def test_all_correct_body_sends_explicit_zero(api_client: TestClient) -> None:
    """Widget state: count mode, "All correct (0)" and "None missed (0)" pressed.

    Those buttons write an explicit "0" into the inputs, so ``buildRatingPayload`` sends
    ``wrong_count: 0`` and ``missed_count: 0`` -- assessed-none, which round-trips as a real
    0 (not NULL) and yields perfect precision and recall. This is the mirror of the bare
    thumbs-up case: 0 is evidence, null is not.
    """
    run_id = _seed_run(api_client, n_matches=3)

    body = {"run_id": run_id, "thumbs_up": True, "wrong_count": 0, "missed_count": 0}
    rated = api_client.post("/ratings", json=body)
    assert rated.status_code == 200, rated.text

    conn = connect(api_client.app.state.db_path)  # type: ignore[attr-defined]
    try:
        row = conn.execute(
            "SELECT wrong_count, missed_count FROM ratings WHERE run_id = ?", (run_id,)
        ).fetchone()
    finally:
        conn.close()
    assert row["wrong_count"] == 0
    assert row["missed_count"] == 0

    ncc = _stats_for(api_client, "ncc")
    assert ncc["precision_n"] == 1
    assert ncc["precision_mean"] == pytest.approx(1.0)
    assert ncc["recall_n"] == 1
    assert ncc["recall_mean"] == pytest.approx(1.0)
