"""The SQLite run/rating store: schema, migrations, persistence, views and statistics.

The layer that decides what evidence the project can ever produce. Two rules run through
every module and neither may be regressed (see :mod:`object_search.schemas.records`):

1. **Human count fields are nullable and stored empty** (EVAL-17). ``NULL`` = "not
   assessed"; ``0`` = "assessed, none". No ``DEFAULT 0`` in the DDL, no coercion in the
   store. Breaking this makes every unreviewed run claim a perfect score.
2. **Derived metrics are computed in views, never stored** (EVAL-07). Precision, recall,
   FP, TP and expected live only in the SQL views, with ``NULL`` propagating.

Open a ready-to-use store with :func:`open_store`; everything else composes from there.
"""

from object_search.store.db import connect, open_store
from object_search.store.migrations import create_views, migrate
from object_search.store.ratings import (
    get_rating,
    insert_rating,
    validate_rating_against_run,
)
from object_search.store.runs import get_run, insert_run
from object_search.store.stats import MethodStats, scoreboard
from object_search.store.wilson import wilson_interval

__all__ = [
    "MethodStats",
    "connect",
    "create_views",
    "get_rating",
    "get_run",
    "insert_rating",
    "insert_run",
    "migrate",
    "open_store",
    "scoreboard",
    "validate_rating_against_run",
    "wilson_interval",
]
