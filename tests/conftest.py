"""Shared fixtures for the object_search test suite."""

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

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
