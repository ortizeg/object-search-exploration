"""Static serving of the frontend, plus the raw scene bytes the canvas draws.

The route modules (``routes_*``) speak JSON; this module is the one place that serves files
from disk. It does three things and wires them into ``create_app``:

1. Mounts the vanilla-ES-module frontend (``frontend/``) at ``/app`` with ``html=True``, so
   ``/app/`` serves ``index.html`` and ``/app/dev/selfcheck.html`` serves the transform proof.
   No build step, no bundler -- the modules are served exactly as written (Phase 4 decision 1).
2. Redirects ``/`` to ``/app/`` so the bare origin lands on the UI.
3. Serves the decoded scene bytes at ``GET /image?image_id=...`` so the canvas can draw the
   exact pixels a search runs against. ``GET /images`` returns only *metadata*; the canvas
   needs the image itself, and rendering it into the canvas (rather than an ``<img>`` with
   ``object-fit``) is what lets the viewport own the letterbox transform (PITFALLS §9.7).
   Path resolution reuses :func:`resolve_image_path`, so the same containment check that
   protects ``/search`` protects this route -- a crafted ``image_id`` cannot escape its base.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from object_search.api.errors import APIError
from object_search.api.images import ImageNotFoundError, resolve_image_path
from object_search.provenance import repo_root


def frontend_dir() -> Path:
    """The ``frontend/`` directory shipped with the repo (served as static files)."""
    return repo_root() / "frontend"


def install_static(app: FastAPI) -> None:
    """Register the ``/`` redirect and the ``/image`` route, then mount ``/app``.

    The mount is added last: it claims the whole ``/app`` subtree, so the explicit routes
    (``/`` and ``/image``) are registered first to keep them readable as top-level routes.
    """

    @app.get("/", include_in_schema=False)
    def root_redirect() -> RedirectResponse:
        """Send the bare origin to the app shell (trailing slash so module paths resolve)."""
        return RedirectResponse(url="/app/")

    @app.get("/image", include_in_schema=False)
    def raw_image(request: Request, image_id: str) -> FileResponse:
        """Serve the raw bytes of a scene image so the canvas can draw it.

        Raises:
            APIError: 404 ``image_not_found`` if the id escapes its base or names no file.
        """
        try:
            path = resolve_image_path(image_id, request.app.state.uploads_dir)
        except ImageNotFoundError as exc:
            raise APIError(404, "image_not_found", str(exc)) from exc
        return FileResponse(path)

    app.mount("/app", StaticFiles(directory=frontend_dir(), html=True), name="frontend")
