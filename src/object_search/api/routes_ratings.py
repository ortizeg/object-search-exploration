"""``POST /ratings`` -- record a human's assessment of a run, null discipline intact.

This route is the HTTP end of the project's most-guarded invariant (EVAL-17). The request
body is the domain :class:`~object_search.schemas.records.Rating` verbatim, precisely so the
HTTP layer invents no second contract: its ``wrong_count`` and ``missed_count`` default to
``None``, a submitted body that omits them stays ``None`` (SQL ``NULL`` = "not assessed"),
and **this handler does not coerce** -- no ``or 0``, no default, no ``COALESCE``. A single
``= 0`` anywhere on this path would make every unreviewed run claim perfect precision and
recall, which is the exact failure the whole store layer exists to prevent.

The one thing the route adds on top of the store is the cross-row bounds check
(``0 <= wrong_count <= retrieved``), because ``retrieved`` is a property of the *run*, not of
the rating, and so cannot live as a field constraint on ``Rating``. The run is loaded, its
returned-box count read, and :func:`validate_rating_against_run` applied (EVAL-18): a count
that exceeds what the run returned is a ``422`` with the flag -- **never a silent clamp** --
and a confirmed per-match verdict set that disagrees with a bare ``wrong_count`` is likewise
flagged rather than reconciled, because the two answers contradicting each other is itself
information.
"""

from __future__ import annotations

from fastapi import APIRouter, Request

from object_search.api.errors import APIError
from object_search.api.schemas import RatingResponse
from object_search.schemas.records import Rating
from object_search.store.db import connect
from object_search.store.ratings import insert_rating, validate_rating_against_run
from object_search.store.runs import get_run

router = APIRouter(tags=["ratings"])


@router.post("/ratings", response_model=RatingResponse)
def post_rating(request: Request, rating: Rating) -> RatingResponse:
    """Validate a rating against its run and persist it, counts stored exactly as submitted.

    Raises:
        APIError: 404 ``run_not_found`` if the rated run does not exist; 422
            ``rating_rejected`` if the counts violate ``0 <= wrong_count <= retrieved`` or a
            confirmed per-match verdict set disagrees with a bare ``wrong_count`` (EVAL-18) --
            the flag is surfaced, the value is never clamped.
    """
    conn = connect(request.app.state.db_path)
    try:
        try:
            run = get_run(conn, rating.run_id)
        except KeyError as exc:
            raise APIError(404, "run_not_found", str(exc)) from exc

        # retrieved is a property of the run, not the rating -- hence the check lives here
        # rather than as a field constraint on Rating (records.py::validate_against_retrieved).
        retrieved = len(run.result.matches)
        ok, reason = validate_rating_against_run(rating, retrieved)
        if not ok:
            raise APIError(422, "rating_rejected", reason or "rating failed validation")

        # No coercion: a None count is written as SQL NULL by the store (EVAL-17).
        rating_id = insert_rating(conn, rating)
    finally:
        conn.close()

    return RatingResponse(rating_id=rating_id)
