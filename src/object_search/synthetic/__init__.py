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
from object_search.synthetic.real_insertion import (
    REAL_BACKGROUND_MANIFEST,
    REAL_BUSY_BACKGROUND_MANIFEST,
    REAL_INSERTION_SPECS,
    REAL_OBJECT_MANIFEST,
    Cutout,
    PhotoProvenance,
    RealInsertionImageSpec,
    extract_cutout,
    fetch_real_photos,
    generate_real_insertion_image,
    write_real_insertion,
)

__all__ = [
    "DEMO_SPECS",
    "MARKER_DEMO_SPECS",
    "REAL_BACKGROUND_MANIFEST",
    "REAL_BUSY_BACKGROUND_MANIFEST",
    "REAL_INSERTION_SPECS",
    "REAL_OBJECT_MANIFEST",
    "Cutout",
    "MarkerGT",
    "MarkerImage",
    "MarkerSpec",
    "PhotoProvenance",
    "RealInsertionImageSpec",
    "SyntheticImage",
    "SyntheticSpec",
    "extract_cutout",
    "fetch_real_photos",
    "generate_real_insertion_image",
    "save",
    "save_marker_image",
    "synthesize",
    "synthesize_markers",
    "write_real_insertion",
]
