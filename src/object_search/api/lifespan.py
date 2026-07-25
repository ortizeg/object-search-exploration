"""The FastAPI lifespan: one place where logging, the store, and ONNX sessions are set up.

Three startup facts this handler guarantees, each with a rule behind it:

* **``setup_logging`` is called exactly once** (INFRA-05). The lifespan is *an* entry point,
  and entry points are the only place that may configure Loguru sinks -- a library module
  that called it would append a duplicate handler per import.
* **The store is migrated once, up front.** :func:`object_search.store.db.open_store` applies
  the idempotent ordered-DDL migrations, so the very first request meets an up-to-date
  schema rather than racing the migration. The migrating connection is then closed: request
  handlers open their own short-lived connection each (``check_same_thread=False``, one per
  request), which is why the path -- not a shared connection -- is what lives on ``app.state``.
* **ONNX sessions load once and are reused** (API-07). A model loaded per request would pay
  the multi-hundred-millisecond session-init cost on every call. In Phase 3 no ONNX-backed
  method is registered yet, so the registry builds **empty** -- but the wiring is real, not a
  ``TODO``: it iterates :data:`~object_search.inference.models.MODEL_REGISTRY`, loads a
  session for every weight file that is present, and skips the absent ones with an INFO log.
  Phases 5/6/7 register methods that read ``app.state.sessions[key]``; adding a model then is
  a fetch, not an API change.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from loguru import logger

from object_search.inference.models import MODEL_REGISTRY, models_dir
from object_search.log import setup_logging
from object_search.store.db import open_store

# Keyed by model key (e.g. "dinov2-small") -> a loaded onnxruntime.InferenceSession. The
# value is typed Any because onnxruntime ships no usable stubs (see the mypy overrides); the
# key vocabulary is MODEL_REGISTRY's, so a method asks for a session by the same key it
# fetched the weights under.
SessionRegistry = dict[str, Any]


def build_session_registry() -> SessionRegistry:
    """Load an ONNX session for every model whose weights are present; skip the rest.

    Real wiring even though it returns empty in Phase 3: a method added in a later phase
    fetches its weights, and this loop then loads the session at startup with no code change
    here. ``onnxruntime`` is imported lazily inside the present-weights branch so a
    weightless install (the Phase 3 state) never pays its import cost.

    Returns:
        ``{model_key: InferenceSession}`` for every registered model found on disk.
    """
    sessions: SessionRegistry = {}
    directory = models_dir()
    for key, spec in MODEL_REGISTRY.items():
        weight = directory / spec.dest
        if not weight.is_file():
            logger.info(
                "model {!r}: weights absent at {}; no registered method needs it yet, skipping",
                key,
                weight,
            )
            continue
        import onnxruntime as ort  # lazy: never imported on a weightless install

        logger.info("model {!r}: loading ONNX session from {}", key, weight)
        sessions[key] = ort.InferenceSession(str(weight), providers=ort.get_available_providers())
    logger.info("session registry built with {} loaded session(s)", len(sessions))
    return sessions


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Configure logging, migrate the store, and build the session registry, once each.

    Reads ``db_path`` and ``log_level`` from ``app.state`` (set by
    :func:`object_search.api.app.create_app`) and writes ``sessions`` back onto it.
    """
    setup_logging(app.state.log_level)
    db_path = app.state.db_path
    logger.info("opening and migrating store at {}", db_path)
    conn = open_store(db_path)
    conn.close()  # migration done; request handlers open one connection per request
    app.state.sessions = build_session_registry()
    logger.info("lifespan startup complete")
    yield
    logger.info("lifespan shutdown")
