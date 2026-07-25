"""Task 1: the app factory, the lifespan, and the session registry (API-07).

These tests drive the real ASGI lifespan through Starlette's ``TestClient`` context manager
(``with TestClient(app) as client:`` is what actually runs startup/shutdown), so they assert
the wiring the way production hits it rather than by calling the contextmanager by hand.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from starlette.testclient import TestClient

import object_search.api.lifespan as lifespan_module
from object_search.api.app import create_app
from object_search.store.db import connect


def test_lifespan_migrates_store_and_builds_empty_session_registry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = tmp_path / "runs.db"
    calls: list[str] = []
    real_setup = lifespan_module.setup_logging

    def counting_setup(level: str = "INFO", log_file: Path | None = None) -> None:
        calls.append(level)
        real_setup(level, log_file)

    monkeypatch.setattr(lifespan_module, "setup_logging", counting_setup)

    app = create_app(db_path=db, uploads_dir=tmp_path / "uploads", log_level="WARNING")

    with TestClient(app):
        # setup_logging ran exactly once, at startup.
        assert calls == ["WARNING"]
        # The store was migrated: the file exists and carries the target schema version.
        assert db.is_file()
        conn = connect(db)
        try:
            version = conn.execute("PRAGMA user_version").fetchone()[0]
        finally:
            conn.close()
        assert version == 1
        # The session registry is a real dict on app.state, empty because Phase 3 ships no
        # ONNX weights -- the wiring exists, the load list is simply empty (API-07).
        assert isinstance(app.state.sessions, dict)
        assert app.state.sessions == {}


def test_sessions_are_reused_not_rebuilt_across_requests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    build_calls: list[int] = []
    real_build = lifespan_module.build_session_registry

    def counting_build() -> dict[str, object]:
        build_calls.append(1)
        return real_build()

    monkeypatch.setattr(lifespan_module, "build_session_registry", counting_build)

    app = create_app(db_path=tmp_path / "runs.db", uploads_dir=tmp_path / "uploads")

    with TestClient(app):
        first = app.state.sessions
        second = app.state.sessions
        # The registry is built once in lifespan and the same object is served every time,
        # never rebuilt per request (API-07).
        assert first is second
    assert sum(build_calls) == 1


def test_build_session_registry_skips_absent_weights(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Point models_dir at an empty temp dir so no weights are present, and assert the loop
    # skips every registered model rather than raising.
    def empty_models_dir() -> Path:
        return tmp_path

    monkeypatch.setattr(lifespan_module, "models_dir", empty_models_dir)
    sessions = lifespan_module.build_session_registry()
    assert sessions == {}
