"""Native-annotation converters: each research dataset -> the one ``*.gt.json`` sidecar (D-10).

A research dataset joins the benchmark by **translating** its native annotation format into the
single ``*.gt.json`` schema :mod:`object_search.eval.labels` already reads, plus a committed split
manifest. That is the whole reason the benchmark gains a research set without a second ground-truth
reader: the converter is the load-bearing, test-worthy seam, and the loader is untouched.

Each converter is a plain function ``convert_<dataset>(raw_root, out_root) -> list[Path]`` that
reads the gitignored raw tree and writes co-located sidecars under ``out_root``. There is no base
class and no registry here on purpose (Rule of Three): the CARPK converter is the only one this
plan adds, and a shared abstraction is not justified until a third converter demands it.
"""

from __future__ import annotations

from object_search.eval.converters.carpk import convert_carpk

__all__ = ["convert_carpk"]
