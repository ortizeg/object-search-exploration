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


def test_lifespan_migrates_store_and_loads_present_weights(
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
    # Point models_dir at an empty temp dir so the session registry is deterministically empty
    # regardless of which gitignored weights the developer has fetched locally (Phase 5 makes
    # superpoint.onnx genuinely loadable, so relying on the real models/ dir being empty is not
    # hermetic). The load-list-is-empty intent is still asserted below.
    empty_models = tmp_path / "models"
    empty_models.mkdir()
    monkeypatch.setattr(lifespan_module, "models_dir", lambda: empty_models)

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
        # The session registry is a real dict on app.state (API-07). Phase 6 lands the first
        # ONNX weight (dinov2_small.onnx) and Phase 5 lands superpoint.onnx, so the registry is
        # no longer inherently empty: it holds a session for exactly the registered models whose
        # weights are present, and skips the absent ones. models_dir is monkeypatched to an empty
        # temp dir above so the outcome is deterministic (present == {}) regardless of what the
        # developer has fetched; the assertion expresses the general "registry MATCHES on-disk
        # presence" invariant rather than hardcoding either state.
        sessions = app.state.sessions
        assert isinstance(sessions, dict)
        models_directory = lifespan_module.models_dir()
        present = {
            key
            for key, spec in lifespan_module.MODEL_REGISTRY.items()
            if (models_directory / spec.dest).is_file()
        }
        assert set(sessions) == present


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
