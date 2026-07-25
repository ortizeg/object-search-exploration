"""``POST /search`` (run + persist) and ``GET /runs/{id}`` (re-read a persisted run).

Every search is a persisted, provenance-stamped event -- that is the whole point of routing
it through here rather than calling a method directly. One request:

1. resolves the method from the registry (unknown -> 404 listing the known names);
2. validates the raw config against that method's ``config_model`` (invalid -> 422 with the
   field errors, never a 500);
3. loads the scene image (missing -> 404);
4. captures :meth:`Provenance.capture` and the ground-truth-seeded slice metadata;
5. runs the method **inside a try/except** so a method that raises becomes a persisted
   ``outcome='error'`` run with a typed :class:`MethodError`, not a bare stack trace (API-08);
6. persists a :class:`RunRecord` -- provenance, the three-way latency breakdown, slice
   metadata, matches and the sub-threshold candidate log (EVAL-08) -- and returns the result
   with the new run id.

A successful run's latency is the method's own three-stage breakdown (only the method can
attribute its preprocess / inference / postprocess split); the error path, where the method
never returned one, records the measured wall-clock so an ``outcome='error'`` run still has a
non-fabricated latency.
"""

from __future__ import annotations

from time import perf_counter

from fastapi import APIRouter, Request
from loguru import logger
from pydantic import ValidationError

from object_search import provenance
from object_search.api.errors import APIError
from object_search.api.images import ImageNotFoundError, load_image_bgr, slice_metadata_for
from object_search.api.schemas import SearchRequest, SearchResponse
from object_search.explorations import UnknownExplorationError, get_exploration
from object_search.schemas.records import DEFAULT_EXPLORATION, Provenance, RunRecord
from object_search.schemas.search import (
    LatencyBreakdown,
    MethodError,
    SearchOutcome,
    SearchResult,
)
from object_search.search import SearchFn, UnknownMethodError, get_method
from object_search.store.db import connect
from object_search.store.runs import get_run, insert_run

router = APIRouter(tags=["search"])


def _field_errors(exc: ValidationError) -> list[dict[str, object]]:
    """Flatten a Pydantic ``ValidationError`` into a JSON-safe per-field error list.

    Only ``loc`` / ``msg`` / ``type`` are kept; the raw ``ctx`` can hold exception objects
    that are not JSON-serializable, and those three are what a UI form needs to point at the
    offending field.
    """
    return [
        {
            "loc": [str(part) for part in error.get("loc", ())],
            "msg": str(error.get("msg", "")),
            "type": str(error.get("type", "")),
        }
        for error in exc.errors()
    ]


@router.post("/search", response_model=SearchResponse)
def post_search(request: Request, body: SearchRequest) -> SearchResponse:
    """Run a search, persist it with full provenance, and return the result plus run id.

    Dispatch is registry-driven, not name-driven: the default exploration keeps the exact
    Milestone 1 path (resolve the method, run it), and any other exploration is resolved from the
    exploration registry and run through its ``run`` callable. Either way the run is persisted with
    ``body.exploration`` as its tag, so a marker run lands under its own exploration tag with no
    schema migration while the default path is byte-for-byte unchanged.
    """
    is_default = body.exploration == DEFAULT_EXPLORATION

    # 1. Resolve the callable, its config model and its version -- from the method registry on the
    #    default path, from the exploration registry otherwise. Both registries share the exact
    #    ``(image, exemplar, config) -> SearchResult`` call shape, so the run flow below is common.
    fn: SearchFn
    if is_default:
        try:
            method_spec = get_method(body.method)
        except UnknownMethodError as exc:
            raise APIError(404, "unknown_method", str(exc)) from exc
        config_model = method_spec.config_model
        version = method_spec.version
        fn = method_spec.fn
    else:
        try:
            exploration_spec = get_exploration(body.exploration)
        except UnknownExplorationError as exc:
            raise APIError(404, "unknown_exploration", str(exc)) from exc
        config_model = exploration_spec.config_model
        version = exploration_spec.version
        fn = exploration_spec.fn

    try:
        config = config_model.model_validate(body.config)
    except ValidationError as exc:
        raise APIError(
            422,
            "invalid_config",
            f"config failed validation for exploration {body.exploration!r}",
            detail=_field_errors(exc),
        ) from exc

    try:
        image = load_image_bgr(body.image_id, request.app.state.uploads_dir)
    except ImageNotFoundError as exc:
        raise APIError(404, "image_not_found", str(exc)) from exc

    config_json = provenance.canonical_config_json(config)
    config_hash = provenance.config_hash(config)
    captured = Provenance.capture(method_version=version, config_hash=config_hash)
    slice_metadata = slice_metadata_for(body.image_id)

    started = perf_counter()
    try:
        result = fn(image, body.exemplar, config)
    except Exception as exc:
        elapsed_ms = (perf_counter() - started) * 1000.0
        logger.warning(
            "{} {!r} raised on run against {!r}: {}: {}",
            "method" if is_default else "exploration",
            body.method if is_default else body.exploration,
            body.image_id,
            type(exc).__name__,
            exc,
        )
        result = SearchResult(
            method=body.method,
            method_version=version,
            outcome=SearchOutcome.ERROR,
            matches=(),
            latency=LatencyBreakdown(
                preprocess_ms=0.0, inference_ms=elapsed_ms, postprocess_ms=0.0
            ),
            threshold_applied=None,
            error=MethodError(
                kind=type(exc).__name__,
                message=str(exc) or type(exc).__name__,
            ),
        )

    run = RunRecord(
        image_id=body.image_id,
        exemplar=body.exemplar,
        method=body.method,
        exploration=body.exploration,
        config_json=config_json,
        config_hash=config_hash,
        result=result,
        slice_metadata=slice_metadata,
        provenance=captured,
    )

    conn = connect(request.app.state.db_path)
    try:
        run_id = insert_run(conn, run)
    finally:
        conn.close()

    return SearchResponse(run_id=run_id, result=result)


@router.get("/runs/{run_id}", response_model=RunRecord)
def get_run_route(request: Request, run_id: int) -> RunRecord:
    """Re-read a persisted run in full -- provenance, latency, matches, candidates.

    Raises:
        APIError: 404 ``run_not_found`` if no run has that id.
    """
    conn = connect(request.app.state.db_path)
    try:
        try:
            return get_run(conn, run_id)
        except KeyError as exc:
            raise APIError(404, "run_not_found", str(exc)) from exc
    finally:
        conn.close()
