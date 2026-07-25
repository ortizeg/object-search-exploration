"""Schema version management via ``PRAGMA user_version`` and explicit transactions.

There is no Alembic and no ORM: migrations are an ordered list of DDL steps
(:data:`object_search.store.schema.MIGRATIONS`) applied idempotently at startup, with
the schema version stored in SQLite's ``PRAGMA user_version`` header field.

The one trap this module exists to defeat (PITFALLS §7.9): **a pure-DDL migration run
through Python's ``sqlite3`` is not automatically transactional.** The legacy driver
opens an implicit ``BEGIN`` only before ``INSERT``/``UPDATE``/``DELETE``, not before
``CREATE TABLE``. So a three-statement migration that fails on statement three, followed
by ``rollback()``, would leave the first two tables in place with ``user_version``
already bumped -- a half-migrated schema that reads as complete. Every migration step
here is therefore wrapped in an **explicit** ``BEGIN``/``COMMIT``, and ``user_version``
is set inside that same transaction, so a failure rolls the whole step back atomically.

Statements are executed one at a time rather than via ``executescript``, because
``executescript`` issues its own ``COMMIT`` first and would break the explicit
transaction boundary.
"""

from __future__ import annotations

import sqlite3

from loguru import logger

from object_search.store import schema


def _target_version(migrations: list[tuple[int, str]]) -> int:
    return max((version for version, _ in migrations), default=0)


def _statements(ddl: str) -> list[str]:
    """Split a multi-statement DDL blob into individual, non-empty statements."""
    return [stmt for stmt in (s.strip() for s in ddl.split(";")) if stmt]


def create_views(
    conn: sqlite3.Connection,
    views: list[tuple[str, str]] | None = None,
) -> None:
    """Drop and recreate every derived-metric view, idempotently.

    Run after the table migrations so a rebuilt base table can never leave a view
    pointing at a dropped column. SQLite does not validate a view body at creation time,
    so the accompanying smoke test (``SELECT * FROM <view> LIMIT 1``) is what actually
    proves each view still resolves.
    """
    views = views if views is not None else schema.VIEWS
    conn.execute("BEGIN")
    try:
        for name, ddl in views:
            conn.execute(f"DROP VIEW IF EXISTS {name}")
            conn.execute(ddl)
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    logger.debug(f"recreated {len(views)} derived-metric view(s)")


def migrate(
    conn: sqlite3.Connection,
    migrations: list[tuple[int, str]] | None = None,
    views: list[tuple[str, str]] | None = None,
) -> None:
    """Bring ``conn`` up to the target schema version, atomically per step.

    Args:
        conn: An open connection (typically from :func:`object_search.store.db.connect`).
        migrations: Ordered ``(version, ddl)`` steps; defaults to
            :data:`object_search.store.schema.MIGRATIONS`. Overridable so the rollback
            behaviour can be tested with a deliberately broken step.
        views: ``(name, ddl)`` views to (re)create after migrating; defaults to
            :data:`object_search.store.schema.VIEWS`.

    Raises:
        RuntimeError: If the database's ``user_version`` is newer than this build knows
            about -- refusing to run against a schema from the future.
        sqlite3.Error: Propagated after the failing step is rolled back; ``user_version``
            is left untouched so the migration can be retried once fixed.
    """
    migrations = migrations if migrations is not None else schema.MIGRATIONS
    target = _target_version(migrations)
    current: int = conn.execute("PRAGMA user_version").fetchone()[0]

    if current > target:
        raise RuntimeError(
            f"database is at schema v{current}, newer than this build's v{target}; "
            f"refusing to run against a schema from the future."
        )

    for version, ddl in migrations:
        if version <= current:
            continue
        logger.info(f"applying migration to schema v{version}")
        conn.execute("BEGIN")
        try:
            for statement in _statements(ddl):
                conn.execute(statement)
            # user_version cannot be parameterised; version is a trusted int constant.
            conn.execute(f"PRAGMA user_version = {int(version)}")
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            logger.error(f"migration to v{version} failed and was rolled back")
            raise

    create_views(conn, views)
