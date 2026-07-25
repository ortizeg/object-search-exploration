"""The explorations package: the exploration registry plus one-import-per-exploration wiring.

An *exploration* is a registry-level concept mirroring a search method: one whole way of turning
an exemplar box into a persistable :class:`SearchResult`. The same-image search of Milestone 1 is
one exploration; the marker-conditioned search of Milestone 2 is another. This is INFRA-10
generalised -- adding an exploration is exactly one new file plus one import line below.
"""

# -- Exploration registrations (INFRA-10 generalised) ---------------------------------
# One import per exploration, each purely for its @register_exploration side effect. The import
# list below *is* the set of installed explorations -- no plugin scan, no auto-discovery. Each
# import is "unused" by name (hence noqa: F401) but load-bearing: importing the module runs its
# @register_exploration decorator, which -- plus the new file -- is the whole cost of adding one.
from object_search.explorations import (
    marker_conditioned,  # noqa: F401  (registers "marker-conditioned")
    same_image_search,  # noqa: F401  (registers "same-image-search", the default adapter)
)
from object_search.explorations.markers import (
    MarkerGeometry,
    estimate_geometry,
    foreground_mask,
    theta_from_transform,
)
from object_search.explorations.registry import (
    DuplicateExplorationError,
    ExplorationFn,
    ExplorationInfo,
    ExplorationSpec,
    UnknownExplorationError,
    exploration_schemas,
    get_exploration,
    has_exploration,
    list_explorations,
    register_exploration,
    unregister,
)

__all__ = [
    "DuplicateExplorationError",
    "ExplorationFn",
    "ExplorationInfo",
    "ExplorationSpec",
    "MarkerGeometry",
    "UnknownExplorationError",
    "estimate_geometry",
    "exploration_schemas",
    "foreground_mask",
    "get_exploration",
    "has_exploration",
    "list_explorations",
    "register_exploration",
    "theta_from_transform",
    "unregister",
]
