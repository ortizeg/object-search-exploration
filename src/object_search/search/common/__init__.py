"""Shared *offerings* for search methods -- never requirements.

Every module in this package is a leaf utility a method **may** import or **may** inline its
own variant of. Nothing here is mandatory: the single most important convention in the repo
is that each search method reads top-to-bottom as one self-contained file, and that rule
overrides DRY. A method that reads better with its own three-line peak picker should keep it.

Submodules are imported directly and independently::

    from object_search.search.common import nms
    from object_search.search.common.peaks import extract_peaks

This package deliberately does **not** eagerly import its submodules, so importing one
offering never drags the others (and their heavier deps -- scikit-learn, matplotlib) into a
method that doesn't want them.
"""
