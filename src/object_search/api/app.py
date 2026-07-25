"""The FastAPI application factory and the module-scope ``app`` uvicorn serves.

``create_app`` is the seam every test uses: it takes the store path and the uploads
directory as arguments, so a test drives a fresh temp database through the real HTTP stack
with no global state. The module-scope ``app = create_app()`` at the bottom is what
``uvicorn object_search.api.app:app`` (the ``serve`` pixi task) imports.

Importing :mod:`object_search.search` for its registration side effect is deliberate and
load-bearing: it runs each method's ``@register_method`` decorator, so ``GET /methods`` and
``POST /search`` see the installed methods. The API layer names no method itself (API-01) --
it only imports the package whose ``__init__`` lists them.

CORS is fully permissive on purpose: this is a local, single-user exploration harness, the
frontend (Phase 4) is served from a file:// or a different local port, and there is no
credentialed cross-origin surface to protect.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import for the @register_method side effect (API-01: the API names no method itself).
from object_search import search as _search  # noqa: F401
from object_search.api.errors import install_error_handlers
from object_search.api.lifespan import lifespan
from object_search.api.routes_images import router as images_router
from object_search.api.routes_methods import router as methods_router
from object_search.api.routes_ratings import router as ratings_router
from object_search.api.routes_search import router as search_router
from object_search.api.routes_stats import router as stats_router
from object_search.provenance import repo_root


def create_app(
    *,
    db_path: str | Path | None = None,
    uploads_dir: str | Path | None = None,
    log_level: str = "INFO",
) -> FastAPI:
    """Build a fully wired FastAPI app.

    Args:
        db_path: SQLite run/rating store path. Defaults to ``<repo>/runs.db`` (gitignored).
            The lifespan migrates it on startup; request handlers open one connection each.
        uploads_dir: Where ``POST /images`` writes ad-hoc uploads. Defaults to
            ``<repo>/runtime/uploads``. Created on first write.
        log_level: Loguru level the lifespan configures, e.g. ``"INFO"`` or ``"WARNING"``.

    Returns:
        A FastAPI instance with the lifespan, permissive CORS, the typed-error handler, and
        every route group mounted.
    """
    app = FastAPI(
        title="Object Search Exploration API",
        summary="Exemplar-based object search: methods, search, ratings, and a scoreboard.",
        lifespan=lifespan,
    )

    app.state.db_path = str(db_path) if db_path is not None else str(repo_root() / "runs.db")
    app.state.uploads_dir = (
        Path(uploads_dir) if uploads_dir is not None else repo_root() / "runtime" / "uploads"
    )
    app.state.log_level = log_level
    # Set now so app.state.sessions is always present; the lifespan replaces it on startup.
    app.state.sessions = {}

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
        allow_credentials=False,
    )

    install_error_handlers(app)

    app.include_router(methods_router)
    app.include_router(images_router)
    app.include_router(search_router)
    app.include_router(ratings_router)
    app.include_router(stats_router)

    return app


app = create_app()
