"""The explorations package: marker geometry plus (Task 3+) the exploration registry.

An "exploration" is a registry-level concept mirroring a search method: one whole way of
turning an exemplar box into a persistable :class:`SearchResult`. The same-image search of
Milestone 1 is one exploration; the marker-conditioned search of Milestone 2 is another.

Task 2 lands the marker geometry estimator; Task 3 adds the registry and the one-import-per-
exploration wiring below it.
"""

from object_search.explorations.markers import (
    MarkerGeometry,
    estimate_geometry,
    foreground_mask,
    theta_from_transform,
)

__all__ = [
    "MarkerGeometry",
    "estimate_geometry",
    "foreground_mask",
    "theta_from_transform",
]
