"""Native-annotation converters: each research dataset -> the one ``*.gt.json`` sidecar (D-10).

A research dataset joins the benchmark by **translating** its native annotation format into the
single ``*.gt.json`` schema :mod:`object_search.eval.labels` already reads, plus a committed split
manifest. That is the whole reason the benchmark gains a research set without a second ground-truth
reader: the converter is the load-bearing, test-worthy seam, and the loader is untouched.

Each converter is a plain function ``convert_<dataset>(raw_root, out_root) -> list[Path]`` that
reads the gitignored raw tree and writes co-located sidecars under ``out_root``. There is still no
base class and no registry decorator here on purpose (Rule of Three): the four converters share a
*shape* (read native annotations -> emit the one ``*.gt.json`` schema), but each native format is
different enough -- plain-text boxes (CARPK), COCO-style JSON with exemplar polygons (FSCD-147,
FSCD-LVIS), all-repeats boxes (RPINE) -- that a shared abstraction would hide more than it saves.
The dataset->converter dispatch that *does* exist lives once, in :mod:`object_search.eval.datasets`.
"""

from __future__ import annotations

from object_search.eval.converters.carpk import convert_carpk
from object_search.eval.converters.fscd147 import (
    Fscd147DedupResult,
    Fscd147Splits,
    convert_fscd147,
    dedup_fscd147,
)
from object_search.eval.converters.fscd_lvis import convert_fscd_lvis
from object_search.eval.converters.rpine import convert_rpine

__all__ = [
    "Fscd147DedupResult",
    "Fscd147Splits",
    "convert_carpk",
    "convert_fscd147",
    "convert_fscd_lvis",
    "convert_rpine",
    "dedup_fscd147",
]
