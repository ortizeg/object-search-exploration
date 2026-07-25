"""Synthetic scene generation with exact ground truth (EVAL-03).

Import from here::

    from object_search.synthetic import synthesize, SyntheticSpec, DEMO_SPECS
"""

from object_search.synthetic.generator import (
    DEMO_SPECS,
    SyntheticImage,
    SyntheticSpec,
    save,
    synthesize,
)

__all__ = [
    "DEMO_SPECS",
    "SyntheticImage",
    "SyntheticSpec",
    "save",
    "synthesize",
]
