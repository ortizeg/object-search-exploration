"""Request and response models for the API layer that are not already a domain schema.

Most of the wire contract is reused verbatim from :mod:`object_search.schemas` -- a search
returns a :class:`~object_search.schemas.search.SearchResult`, a rating is posted as a
:class:`~object_search.schemas.records.Rating` -- precisely so the HTTP layer invents no
second source of truth. The models here are only the few shapes that exist *at* the HTTP
boundary and nowhere else: the image catalogue entry, the search request envelope, and the
search response that pairs a result with the run id it was persisted under.

Reusing :class:`Rating` directly as the ratings request body is deliberate (EVAL-17): its
``wrong_count`` / ``missed_count`` default to ``None``, so a body that omits them stores
``NULL``. A bespoke request model would risk a well-meaning ``= 0`` sneaking the null-becomes-
zero bug back in at the one layer the whole design is trying to protect.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from object_search.schemas.geometry import ExemplarBox
from object_search.schemas.search import SearchResult


class ImageInfo(BaseModel):
    """One entry in the ``GET /images`` catalogue.

    Attributes:
        id: Stable image identifier -- the path relative to ``assets/demo`` for a demo image
            (``"chipset/chipset-01.png"``) or ``"uploads/<name>"`` for an ad-hoc upload. This
            is exactly what ``POST /search`` takes as ``image_id``.
        width: Image width in pixels.
        height: Image height in pixels.
        has_ground_truth: Whether a ground-truth sidecar exists for this image. True for the
            synthetic and chipset sets (they ship ``.gt.json`` files), false for basketball
            frames and uploads. The eval layer and the UI use this to tell which images can
            be scored objectively rather than only by human rating.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    width: int = Field(ge=1)
    height: int = Field(ge=1)
    has_ground_truth: bool


class SearchRequest(BaseModel):
    """The body of ``POST /search``.

    ``config`` is an untyped JSON object here and validated against the resolved method's
    ``config_model`` inside the route -- the API cannot name a method's config type without
    naming the method (API-01), so the per-method validation is deferred to request time.

    Attributes:
        image_id: Which image to search, as returned by ``GET /images``.
        exemplar: The box the user drew (plus optional label).
        method: Registry key of the method to run, as returned by ``GET /methods``.
        config: Raw config object; validated against the method's ``config_model``.
        exploration: The exploration this run belongs to; defaults to the same-image search
            on the store side (the Milestone 2 seam), so callers omit it in Phase 3.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    image_id: str = Field(min_length=1)
    exemplar: ExemplarBox
    method: str = Field(min_length=1)
    config: dict[str, object] = Field(default_factory=dict)
    exploration: str | None = None


class SearchResponse(BaseModel):
    """The body of a successful (or persisted-error) ``POST /search``.

    Attributes:
        run_id: The row id the run was persisted under. ``GET`` it back, or rate it via
            ``POST /ratings``.
        result: What the method returned -- matches, latency breakdown, candidates,
            diagnostics, and (on a method failure) the typed ``error`` with
            ``outcome='error'``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: int
    result: SearchResult
