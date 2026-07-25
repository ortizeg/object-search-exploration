"""``GET /methods`` -- the registered methods and their config schemas (API-01).

The single most important property of this file is what it does **not** contain: any method
name. The route is a loop over :func:`object_search.search.registry.method_schemas`, which
renders each registered method's ``config_model`` as JSON Schema. A method is added by
registering it in the ``search`` package; this route, and the UI form it feeds (UI-07),
pick it up with no edit here. A test greps this whole package for every registered name and
asserts zero hits.
"""

from __future__ import annotations

from fastapi import APIRouter

from object_search.search import MethodInfo, method_schemas

router = APIRouter(tags=["methods"])


@router.get("/methods", response_model=list[MethodInfo])
def get_methods() -> list[MethodInfo]:
    """List every registered method: name, description, version, and config JSON Schema."""
    return list(method_schemas())
