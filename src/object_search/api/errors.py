"""Typed, structured HTTP errors -- never a bare 500 with a stack trace (API-08).

Two shapes of failure travel through this module, and they are deliberately different:

* **Request-level errors** -- an unknown method, a config that fails validation, an image
  that does not exist, an upload that is not a decodable image. These are the caller's
  problem, so they are raised as an :class:`APIError` carrying an HTTP status and a
  machine-readable ``kind``, and rendered by :func:`install_error_handlers` as
  ``{"error": {kind, message, detail}}``. A 4xx, always typed.
* **Method-level errors** -- a registered search method that *raises* mid-run. Those are
  evidence (EVAL-12), not 4xx client mistakes: the search route catches them, persists the
  run with ``outcome='error'`` and a :class:`~object_search.schemas.search.MethodError`,
  and returns the run like any other. That path lives in ``routes_search``; this module is
  only the request-level half.

The one rule both halves share: a failure is a structured object with a stable ``kind`` a
client can branch on, never an HTML traceback and never a naked ``500``.
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic import BaseModel, ConfigDict


class APIErrorBody(BaseModel):
    """The JSON body of a request-level error, under the top-level ``error`` key.

    Attributes:
        kind: Short machine-readable tag (``"unknown_method"``, ``"invalid_config"``,
            ``"image_not_found"``, ``"invalid_image"``, ``"rating_rejected"``,
            ``"run_not_found"``). Grouping by kind is what lets a client branch without
            string-matching a human message.
        message: Human-readable detail, safe to show a user.
        detail: Optional structured extras -- e.g. the per-field Pydantic errors behind an
            ``invalid_config`` 422, so the UI form can highlight the offending fields.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: str
    message: str
    detail: list[dict[str, object]] | None = None


class APIError(Exception):
    """A request-level failure that renders as a typed 4xx body.

    Raise this instead of :class:`fastapi.HTTPException` so every error in the API shares
    one envelope and one ``kind`` vocabulary. The registered handler turns it into a
    :class:`JSONResponse`; nothing else needs to know the wire shape.
    """

    def __init__(
        self,
        status_code: int,
        kind: str,
        message: str,
        detail: list[dict[str, object]] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.body = APIErrorBody(kind=kind, message=message, detail=detail)


async def _handle_api_error(request: Request, exc: Exception) -> JSONResponse:
    """Render an :class:`APIError` as ``{"error": {...}}`` at its declared status.

    Typed as ``exc: Exception`` to match Starlette's handler signature; it is only ever
    registered for :class:`APIError`, and a non-``APIError` (which cannot occur) is
    re-raised rather than silently mishandled.
    """
    if not isinstance(exc, APIError):  # pragma: no cover -- registered only for APIError
        raise exc
    logger.info("api error {} ({}): {}", exc.status_code, exc.body.kind, exc.body.message)
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.body.model_dump(mode="json")},
    )


def install_error_handlers(app: FastAPI) -> None:
    """Register the :class:`APIError` handler on ``app``.

    Called once from :func:`object_search.api.app.create_app`, so every route gets the
    typed-error envelope for free.
    """
    app.add_exception_handler(APIError, _handle_api_error)
