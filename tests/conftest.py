"""Shared fixtures for the object_search test suite."""

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from starlette.testclient import TestClient

from object_search.api.app import create_app
from object_search.log import setup_logging


@pytest.fixture(autouse=True, scope="session")
def _quiet_logging() -> Iterator[None]:
    """Configure Loguru once per session at WARNING so test output is not flooded.

    Autouse and session-scoped: this is the single entry-point call to ``setup_logging`` for
    the whole suite, which is exactly the contract that function documents.
    """
    setup_logging("WARNING")
    yield


@pytest.fixture
def rng() -> np.random.Generator:
    """Seeded NumPy generator, so anything stochastic in a test is reproducible.

    Reproducibility is a project constraint: same image + box + method + config must give
    identical results, so tests never use unseeded randomness.
    """
    return np.random.default_rng(0)


@pytest.fixture
def tmp_models_dir(tmp_path: Path) -> Path:
    """Empty stand-in for the gitignored ``models/`` directory, for inferencer tests."""
    models = tmp_path / "models"
    models.mkdir()
    return models


@pytest.fixture
def api_client(tmp_path: Path) -> Iterator[TestClient]:
    """A ``TestClient`` over the real app with a fresh temp store and uploads directory.

    Entering the ``TestClient`` context runs the real ASGI lifespan, so the store is migrated
    and the (empty, in Phase 3) session registry is built exactly as in production. Each test
    gets its own database, so run ids and stats never leak between tests.
    """
    app = create_app(
        db_path=tmp_path / "runs.db",
        uploads_dir=tmp_path / "uploads",
        log_level="WARNING",
    )
    with TestClient(app) as client:
        yield client
