"""``GET /explorations`` -- the registered explorations and their config schemas.

The single most important property of this file is what it does **not** contain: any exploration
name. The route is a loop over :func:`object_search.explorations.exploration_schemas`, which renders
each registered exploration's ``config_model`` as JSON Schema. An exploration is added by
registering it in the ``explorations`` package; this route, and the UI form it feeds, pick it up
with no edit here. A test greps this whole package for every registered name and asserts zero hits.
"""

from __future__ import annotations

from fastapi import APIRouter

from object_search.explorations import ExplorationInfo, exploration_schemas

router = APIRouter(tags=["explorations"])


@router.get("/explorations", response_model=list[ExplorationInfo])
def get_explorations() -> list[ExplorationInfo]:
    """List every registered exploration: name, description, version, and config JSON Schema."""
    return list(exploration_schemas())
