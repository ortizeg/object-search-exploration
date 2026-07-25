"""Tests for the SearchMethod decorator registry (INFRA-10)."""

from collections.abc import Iterator

import numpy as np
import numpy.typing as npt
import pytest
from pydantic import BaseModel, ConfigDict, Field

from object_search.schemas.geometry import BBox, ExemplarBox
from object_search.schemas.search import (
    LatencyBreakdown,
    Match,
    SearchOutcome,
    SearchResult,
)
from object_search.search import registry
from object_search.search.registry import (
    DuplicateMethodError,
    UnknownMethodError,
    get_method,
    has_method,
    list_methods,
    method_schemas,
    register_method,
)


@pytest.fixture(autouse=True)
def _isolated_registry() -> Iterator[None]:
    """Snapshot and restore the module-level registry around every test.

    The registry is process-global by design (it is the single source of installed
    methods). Tests register throwaway methods, so each test must leave it as it found it.
    """
    saved = dict(registry._REGISTRY)
    registry._REGISTRY.clear()
    try:
        yield
    finally:
        registry._REGISTRY.clear()
        registry._REGISTRY.update(saved)


class _DummyConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="Match cut-off.")


def _make_result(name: str) -> SearchResult:
    return SearchResult(
        method=name,
        method_version="1.0.0",
        outcome=SearchOutcome.OK,
        matches=(Match(box=BBox(x=0, y=0, w=4, h=4), score=1.0, is_exemplar=True),),
        latency=LatencyBreakdown(preprocess_ms=0.0, inference_ms=0.0, postprocess_ms=0.0),
        threshold_applied=0.5,
    )


def test_register_and_get_returns_spec_with_config_schema() -> None:
    @register_method(
        name="dummy",
        description="A test method.",
        version="1.0.0",
        config_model=_DummyConfig,
    )
    def search(
        image: npt.NDArray[np.uint8],
        exemplar: ExemplarBox,
        config: BaseModel,
    ) -> SearchResult:
        return _make_result("dummy")

    assert has_method("dummy")
    spec = get_method("dummy")
    assert spec.name == "dummy"
    assert spec.config_model is _DummyConfig
    assert spec.module == __name__

    # The function is returned unchanged and is callable through the spec.
    scene = np.zeros((8, 8, 3), dtype=np.uint8)
    result = spec.fn(scene, ExemplarBox(box=BBox(x=0, y=0, w=4, h=4)), _DummyConfig())
    assert result.outcome is SearchOutcome.OK

    schemas = method_schemas()
    assert len(schemas) == 1
    info = schemas[0]
    assert info.name == "dummy"
    assert info.config_schema  # non-empty JSON Schema
    assert "threshold" in info.config_schema["properties"]  # type: ignore[index]


def test_duplicate_name_raises() -> None:
    def _register(desc: str) -> None:
        @register_method(
            name="clash",
            description=desc,
            version="1.0.0",
            config_model=_DummyConfig,
        )
        def search(
            image: npt.NDArray[np.uint8],
            exemplar: ExemplarBox,
            config: BaseModel,
        ) -> SearchResult:
            return _make_result("clash")

    _register("first")
    with pytest.raises(DuplicateMethodError, match="already registered"):
        _register("second")


def test_get_unknown_raises_listing_known_names() -> None:
    @register_method(
        name="alpha",
        description="a",
        version="1.0.0",
        config_model=_DummyConfig,
    )
    def search(
        image: npt.NDArray[np.uint8],
        exemplar: ExemplarBox,
        config: BaseModel,
    ) -> SearchResult:
        return _make_result("alpha")

    with pytest.raises(UnknownMethodError, match="alpha"):
        get_method("nope")


def _register_dummy(name: str) -> None:
    @register_method(
        name=name,
        description=name,
        version="1.0.0",
        config_model=_DummyConfig,
    )
    def search(
        image: npt.NDArray[np.uint8],
        exemplar: ExemplarBox,
        config: BaseModel,
    ) -> SearchResult:
        return _make_result(name)


def test_list_methods_is_sorted_by_name() -> None:
    for name in ("zulu", "alpha", "mike"):
        _register_dummy(name)

    names = [spec.name for spec in list_methods()]
    assert names == ["alpha", "mike", "zulu"]
