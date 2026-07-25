"""The search package: the method registry, plus the one-import-per-method wiring.

Half of INFRA-10 lives here. A method becomes discoverable by adding exactly one import
line below, which loads the module for its ``@register_method`` side effect. There is no
plugin scan and no auto-discovery -- the import list *is* the set of installed methods, and
you can read it top to bottom to know what the app can do.

There are no methods yet: Phase 2 adds Method 1 and, with it, the first import here, e.g.::

    from object_search.search import ncc  # noqa: F401  (registers "ncc")

Until then the block is intentionally empty except for this note.
"""

from object_search.search.registry import (
    DuplicateMethodError,
    MethodInfo,
    MethodSpec,
    SearchFn,
    UnknownMethodError,
    get_method,
    has_method,
    list_methods,
    method_schemas,
    register_method,
)

# -- Method registrations (INFRA-10) --------------------------------------------------
# One import per method, each purely for its @register_method side effect. Phase 2 adds a
# line like `from object_search.search import ncc` with a trailing noqa-F401 (the import is
# "unused" by name but load-bearing for its registration side effect).

__all__ = [
    "DuplicateMethodError",
    "MethodInfo",
    "MethodSpec",
    "SearchFn",
    "UnknownMethodError",
    "get_method",
    "has_method",
    "list_methods",
    "method_schemas",
    "register_method",
]
