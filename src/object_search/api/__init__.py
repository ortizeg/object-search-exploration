"""The FastAPI HTTP layer: ``/methods`` ``/images`` ``/search`` ``/ratings`` ``/stats``.

Every search becomes a persisted, provenance-stamped HTTP call here, so the UI (Phase 4)
and the benchmark (Phase 8) share one contract. Two invariants run through the package:

* **No method name is written in this package** (API-01). ``/methods`` is a loop over
  :func:`object_search.search.registry.method_schemas`; ``/search`` resolves a method by the
  key the caller sends. A test greps ``api/`` for every registered name and asserts none.
* **The null discipline survives the wire** (EVAL-17). ``POST /ratings`` accepts the frozen
  :class:`~object_search.schemas.records.Rating` directly, whose count fields default to
  ``None``, and the route does not coerce -- a bare thumbs-up stores ``NULL``.
"""

from object_search.api.app import app, create_app

__all__ = ["app", "create_app"]
