"""Schema, migration and run-persistence tests, all against a real temp-file database.

Mocks are deliberately avoided: a mocked connection would pass while the actual DDL
carried a ``DEFAULT 0`` on a count column, which is precisely the bug this suite exists to
catch (PITFALLS §7, EVAL-17).
"""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

import pytest
from pydantic import BaseModel

from object_search import provenance as prov
from object_search.schemas.records import Rating, SliceMetadata
from object_search.schemas.search import (
    Diagnostics,
    HeatmapPayload,
    MethodError,
    SearchOutcome,
)
from object_search.store import migrations, schema
from object_search.store.db import connect, open_store
from object_search.store.ratings import insert_rating
from object_search.store.runs import get_run, insert_run
from tests.store_helpers import make_run_record


@pytest.fixture
def db(tmp_path: Path) -> sqlite3.Connection:
    return open_store(tmp_path / "runs.db")


# ---------------------------------------------------------------- Task 1: schema / DDL


def test_fresh_connection_has_foreign_keys_on(tmp_path: Path) -> None:
    conn = connect(tmp_path / "runs.db")
    assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1


def test_count_columns_have_no_default(db: sqlite3.Connection) -> None:
    """EVAL-17 at the DDL layer: wrong_count/missed_count must carry NO default.

    A DEFAULT 0 here would make every unreviewed run store 0 instead of NULL and claim
    perfect precision/recall. dflt_value is None precisely when there is no default.
    """
    info = {row["name"]: row for row in db.execute("PRAGMA table_info(ratings)")}
    assert info["wrong_count"]["dflt_value"] is None
    assert info["missed_count"]["dflt_value"] is None
    # And they must be nullable (notnull == 0).
    assert info["wrong_count"]["notnull"] == 0
    assert info["missed_count"]["notnull"] == 0


def test_no_derived_metric_is_a_stored_column() -> None:
    """EVAL-07: no precision/recall/f1/tp/fp/expected column on any TABLE.

    The words appear legitimately as view-output aliases, so this greps the table DDL
    (MIGRATIONS) only, not the views.
    """
    table_ddl = "\n".join(ddl for _, ddl in schema.MIGRATIONS).lower()
    forbidden = ["precision", "recall", "expected"]
    for token in forbidden:
        assert token not in table_ddl, f"forbidden derived column-word {token!r} in table DDL"
    for token in ["f1", "tp", "fp"]:
        assert not re.search(rf"\b{token}\b", table_ddl), f"forbidden column {token!r}"


def test_exploration_defaults_to_same_image_search(db: sqlite3.Connection) -> None:
    info = {row["name"]: row for row in db.execute("PRAGMA table_info(runs)")}
    assert info["exploration"]["dflt_value"] == "'same-image-search'"
    # And the default actually fires on insert (insert_run omits the column).
    run_id = insert_run(db, make_run_record())
    stored = db.execute("SELECT exploration FROM runs WHERE id = ?", (run_id,)).fetchone()
    assert stored["exploration"] == "same-image-search"


def test_migrate_is_idempotent(db: sqlite3.Connection) -> None:
    assert db.execute("PRAGMA user_version").fetchone()[0] == schema.SCHEMA_VERSION
    migrations.migrate(db)  # second run must be a no-op, not an error
    assert db.execute("PRAGMA user_version").fetchone()[0] == schema.SCHEMA_VERSION


def test_migration_source_uses_explicit_transaction() -> None:
    source = Path(migrations.__file__).read_text()
    assert 'conn.execute("BEGIN")' in source
    assert 'conn.execute("COMMIT")' in source
    assert 'conn.execute("ROLLBACK")' in source


def test_failed_migration_rolls_back_atomically(tmp_path: Path) -> None:
    """A pure-DDL migration is not auto-transactional; a mid-step failure must not leave
    a half-migrated schema with user_version already bumped (PITFALLS §7.9)."""
    conn = connect(tmp_path / "runs.db")
    broken = [
        (1, "CREATE TABLE good_one (x INTEGER)"),
        (2, "CREATE TABLE half (x INTEGER); CREATE TABLE oops (;"),
    ]
    with pytest.raises(sqlite3.Error):
        migrations.migrate(conn, migrations=broken, views=[])
    # Step 1 committed, step 2 rolled back entirely.
    assert conn.execute("PRAGMA user_version").fetchone()[0] == 1
    tables = {
        row["name"] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    }
    assert "good_one" in tables
    assert "half" not in tables  # the successful statement before the failure is gone too


def test_every_view_resolves(db: sqlite3.Connection) -> None:
    """SQLite does not validate a view body at creation; only a SELECT proves it."""
    for name, _ in schema.VIEWS:
        # View names are trusted module constants, not user input (S608 false positive).
        db.execute(f"SELECT * FROM {name} LIMIT 1").fetchall()  # noqa: S608


def test_deleting_a_run_cascades_to_its_rating(db: sqlite3.Connection) -> None:
    run_id = insert_run(db, make_run_record())
    insert_rating(db, Rating(run_id=run_id, thumbs_up=True))
    with db:
        db.execute("DELETE FROM runs WHERE id = ?", (run_id,))
    remaining = db.execute(
        "SELECT COUNT(*) AS n FROM ratings WHERE run_id = ?", (run_id,)
    ).fetchone()["n"]
    assert remaining == 0  # cascade fired -> FK enforcement is live


# ------------------------------------------------------- Task 2: run persistence (-k run)


def test_run_round_trips_losslessly(db: sqlite3.Connection) -> None:
    record = make_run_record(
        n_matches=3,
        n_candidates=50,
        slice_metadata=SliceMetadata(
            true_instance_count=5,
            instance_scale_min=0.8,
            # rotation and clutter deliberately left None -- must survive as None, not 0.
        ),
    )
    run_id = insert_run(db, record)
    loaded = get_run(db, run_id)

    assert loaded == record.model_copy(update={"id": run_id})
    # The None slice fields stayed None (not coerced to 0).
    assert loaded.slice_metadata.rotation_min_deg is None
    assert loaded.slice_metadata.clutter_level is None
    assert loaded.slice_metadata.true_instance_count == 5
    # Candidates survived with their raw scores and order.
    assert len(loaded.result.candidates) == 50
    assert loaded.result.candidates[0].score == record.result.candidates[0].score


def test_run_config_json_is_stored_with_sorted_keys(db: sqlite3.Connection) -> None:
    class _Cfg(BaseModel):
        threshold: float
        backend: str
        alpha: int

    canonical = prov.canonical_config_json(_Cfg(threshold=0.7, backend="sift", alpha=2))
    record = make_run_record().model_copy(update={"config_json": canonical})
    run_id = insert_run(db, record)

    stored = db.execute("SELECT config_json FROM runs WHERE id = ?", (run_id,)).fetchone()[
        "config_json"
    ]
    keys = list(json.loads(stored).keys())
    assert keys == sorted(keys)


def test_error_run_persists_error_and_zero_matches(db: sqlite3.Connection) -> None:
    record = make_run_record(
        outcome=SearchOutcome.ERROR,
        n_matches=0,
        error=MethodError(kind="exemplar_out_of_bounds", message="box off image"),
    )
    run_id = insert_run(db, record)
    loaded = get_run(db, run_id)

    assert loaded.result.outcome is SearchOutcome.ERROR
    assert loaded.result.error is not None
    assert loaded.result.error.kind == "exemplar_out_of_bounds"
    assert loaded.result.matches == ()
    assert (
        db.execute("SELECT retrieved FROM runs WHERE id = ?", (run_id,)).fetchone()["retrieved"]
        == 0
    )


def test_empty_run_persists_zero_matches(db: sqlite3.Connection) -> None:
    record = make_run_record(outcome=SearchOutcome.EMPTY, n_matches=0)
    run_id = insert_run(db, record)
    loaded = get_run(db, run_id)
    assert loaded.result.outcome is SearchOutcome.EMPTY
    assert loaded.result.matches == ()
    assert loaded.result.error is None


def test_oversized_diagnostics_drops_only_the_heatmap(db: sqlite3.Connection) -> None:
    big_png = "A" * (300 * 1024)  # comfortably over the 256 KB cap
    diagnostics = Diagnostics(
        notes=("kept",),
        similarity_heatmap=HeatmapPayload(
            png_b64=big_png, width=100, height=100, vmin=0.0, vmax=1.0
        ),
    )
    record = make_run_record(diagnostics=diagnostics)
    run_id = insert_run(db, record)
    loaded = get_run(db, run_id)

    assert loaded.result.diagnostics.similarity_heatmap is None  # dropped
    assert loaded.result.diagnostics.notes == ("kept",)  # rest retained


def test_get_run_raises_for_missing_id(db: sqlite3.Connection) -> None:
    with pytest.raises(KeyError):
        get_run(db, 999)
