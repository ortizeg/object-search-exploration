"""The search package: the method registry, plus the one-import-per-method wiring.

Half of INFRA-10 lives here. A method becomes discoverable by adding exactly one import
line below, which loads the module for its ``@register_method`` side effect. There is no
plugin scan and no auto-discovery -- the import list *is* the set of installed methods, and
you can read it top to bottom to know what the app can do.

Phase 2 adds Method 1, and with it the first registration import::

    from object_search.search import ncc  # noqa: F401  (registers "ncc")

Every later method appends exactly one more such line -- that plus the new file is the whole
cost of adding a method (INFRA-10).
"""

# -- Method registrations (INFRA-10) --------------------------------------------------
# One import per method, each purely for its @register_method side effect. The import list
# below *is* the set of installed methods -- no plugin scan, no auto-discovery. Each import
# is "unused" by name (hence noqa: F401) but load-bearing: importing the module runs its
# @register_method decorator, which is the entire cost of adding a method.
from object_search.search import (
    ncc,  # noqa: F401  (registers "ncc")
    sparse_geo,  # noqa: F401  (registers "sparse-geo")
)
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
    unregister,
)

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
    "unregister",
]
