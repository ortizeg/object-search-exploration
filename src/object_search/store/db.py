"""The one connection factory, and the operational PRAGMAs every connection needs.

SQLite's defaults are wrong for this project in three ways that all fail silently
(PITFALLS §7.10), so **every** connection must be opened through :func:`connect`:

* ``PRAGMA foreign_keys`` defaults **OFF** and is connection-scoped -- an ``ON DELETE
  CASCADE`` is decorative until it is turned on, so deleting a run would orphan its
  rating rather than removing it. We assert it came back ``1`` so a build where the
  pragma is unsupported fails loudly rather than dropping constraints.
* ``PRAGMA journal_mode`` defaults to ``delete``, where a writer blocks all readers.
  ``WAL`` is the mode where the stats dashboard (a read) never blocks a search (a
  write). WAL is a persistent, file-level setting, unlike the others.
* ``check_same_thread`` defaults ``True``. FastAPI runs non-async ``def`` endpoints in
  a worker threadpool, so a connection must be usable from a thread other than the one
  that created it -- but only ever one request at a time (one connection per request).

WAL does not work on a network filesystem or inside a cloud-sync folder; keep ``runs.db``
on local disk.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from loguru import logger

_BUSY_TIMEOUT_MS = 30_000


def connect(path: str | Path) -> sqlite3.Connection:
    """Open a SQLite connection with this project's mandatory PRAGMAs applied.

    Args:
        path: Filesystem path to the database file. Use a real file (not ``:memory:``)
            for anything that must survive the process; WAL is a no-op in memory.

    Returns:
        A connection with ``row_factory = sqlite3.Row``, foreign keys enforced, WAL
        journaling, and a 30 s busy timeout.

    Raises:
        RuntimeError: If ``PRAGMA foreign_keys`` did not come back ``1`` -- rather than
            silently running with FK constraints disabled.
    """
    conn = sqlite3.connect(str(path), check_same_thread=False, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute(f"PRAGMA busy_timeout = {_BUSY_TIMEOUT_MS}")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    enabled = conn.execute("PRAGMA foreign_keys").fetchone()[0]
    if enabled != 1:
        conn.close()
        raise RuntimeError(
            "PRAGMA foreign_keys did not enable on this SQLite build; refusing to run "
            "with FK constraints silently disabled (PITFALLS §7.10)."
        )
    logger.debug(f"opened sqlite connection to {path} (WAL, FK on)")
    return conn


def open_store(path: str | Path) -> sqlite3.Connection:
    """Connect and bring the schema up to the current version in one call.

    The common entry point for callers that just want a ready-to-use store; migrations
    are idempotent, so this is safe to call on every startup.
    """
    from object_search.store.migrations import migrate

    conn = connect(path)
    migrate(conn)
    return conn
