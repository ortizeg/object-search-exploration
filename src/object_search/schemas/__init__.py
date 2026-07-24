"""Frozen Pydantic contracts shared by every layer.

Everything that crosses a layer boundary -- method to API, API to UI, API to store, store to
statistics -- is one of these models. They are all frozen, so a value that has been handed
across a boundary cannot be mutated behind the sender's back, and ``extra="forbid"`` so a
typo'd field is a load-time error rather than a silently ignored key.

Import from this package, not from the submodules::

    from object_search.schemas import BBox, ExemplarBox, SearchResult
"""

from object_search.schemas.geometry import BBox, ExemplarBox, Point
from object_search.schemas.search import (
    Candidate,
    Correspondence,
    Diagnostics,
    HeatmapPayload,
    HoughPeak,
    LatencyBreakdown,
    Match,
    MethodError,
    SearchOutcome,
    SearchResult,
)

__all__ = [
    "BBox",
    "Candidate",
    "Correspondence",
    "Diagnostics",
    "ExemplarBox",
    "HeatmapPayload",
    "HoughPeak",
    "LatencyBreakdown",
    "Match",
    "MethodError",
    "Point",
    "SearchOutcome",
    "SearchResult",
]
