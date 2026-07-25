"""Synthetic scene generation with exact ground truth (EVAL-03).

Import from here::

    from object_search.synthetic import synthesize, SyntheticSpec, DEMO_SPECS
"""

from object_search.synthetic.generator import (
    DEMO_SPECS,
    MARKER_DEMO_SPECS,
    MarkerGT,
    MarkerImage,
    MarkerSpec,
    SyntheticImage,
    SyntheticSpec,
    save,
    save_marker_image,
    synthesize,
    synthesize_markers,
)

__all__ = [
    "DEMO_SPECS",
    "MARKER_DEMO_SPECS",
    "MarkerGT",
    "MarkerImage",
    "MarkerSpec",
    "SyntheticImage",
    "SyntheticSpec",
    "save",
    "save_marker_image",
    "synthesize",
    "synthesize_markers",
]
