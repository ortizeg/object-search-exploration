"""``GET /stats`` -- the per-method scoreboard, every rate beside the ``n`` it was computed over.

This route is a thin pass-through to :func:`object_search.store.stats.scoreboard`: all of the
care lives in the store's derived-metric views, and the endpoint only opens a connection and
serialises the result. That is deliberate -- the scoreboard's honesty (a separate ``n`` for the
thumbs, precision and recall samples; a Wilson interval that is ``None`` rather than ``[0, 1]``
when nobody has rated a method; ``NULL`` propagating so an unassessed run counts toward nothing)
is a property of the SQL and the aggregation, not of the HTTP layer, and duplicating any of it
here would be a second place for it to drift.

Each :class:`~object_search.store.stats.MethodStats` row carries its own ``thumbs_n`` /
``precision_n`` / ``recall_n`` because they are genuinely different subsets: a run rated with a
bare thumbs-up feeds the thumbs sample but neither precision nor recall (EVAL-13/EVAL-14). The
UI renders a ``None`` rate as an em dash rather than a fabricated score.
"""

from __future__ import annotations

from fastapi import APIRouter, Request

from object_search.store.db import connect
from object_search.store.stats import MethodStats, scoreboard

router = APIRouter(tags=["stats"])


@router.get("/stats", response_model=list[MethodStats])
def get_stats(request: Request) -> list[MethodStats]:
    """Return the per-method scoreboard, ranked best-first by the thumbs Wilson lower bound."""
    conn = connect(request.app.state.db_path)
    try:
        return scoreboard(conn)
    finally:
        conn.close()
