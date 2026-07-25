"""The exploration registry mirrors the method registry (INFRA-10 generalised)."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import numpy.typing as npt
import pytest
from pydantic import BaseModel, ConfigDict, Field

from object_search.explorations import registry
from object_search.explorations.registry import (
    DuplicateExplorationError,
    UnknownExplorationError,
    exploration_schemas,
    get_exploration,
    has_exploration,
    list_explorations,
    register_exploration,
)
from object_search.schemas.geometry import BBox, ExemplarBox
from object_search.schemas.search import (
    LatencyBreakdown,
    Match,
    SearchOutcome,
    SearchResult,
)


@pytest.fixture
def _isolated_registry() -> Iterator[None]:
    """Snapshot and restore the module-level registry around a test that registers dummies.

    Opt-in (not autouse) on purpose: the two "real installed set" tests below must see the
    genuine registrations, which a blanket clear would hide.
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

    radius: int = Field(default=3, ge=1, description="A knob.")


def _make_result(name: str) -> SearchResult:
    return SearchResult(
        method=name,
        method_version="1.0.0",
        outcome=SearchOutcome.OK,
        matches=(Match(box=BBox(x=0, y=0, w=4, h=4), score=1.0),),
        latency=LatencyBreakdown(preprocess_ms=0.0, inference_ms=0.0, postprocess_ms=0.0),
        threshold_applied=None,
    )


def test_register_and_get_returns_spec_with_config_schema(_isolated_registry: None) -> None:
    @register_exploration(
        name="dummy-exploration",
        description="A test exploration.",
        version="1.0.0",
        config_model=_DummyConfig,
    )
    def run(
        image: npt.NDArray[np.uint8],
        exemplar: ExemplarBox,
        config: BaseModel,
    ) -> SearchResult:
        return _make_result("dummy-exploration")

    assert has_exploration("dummy-exploration")
    spec = get_exploration("dummy-exploration")
    assert spec.name == "dummy-exploration"
    assert spec.config_model is _DummyConfig

    schemas = exploration_schemas()
    assert len(schemas) == 1
    assert schemas[0].config_schema  # non-empty JSON Schema
    assert "radius" in schemas[0].config_schema["properties"]  # type: ignore[index]


def test_duplicate_name_raises(_isolated_registry: None) -> None:
    def _register() -> None:
        @register_exploration(
            name="clash",
            description="d",
            version="1.0.0",
            config_model=_DummyConfig,
        )
        def run(
            image: npt.NDArray[np.uint8],
            exemplar: ExemplarBox,
            config: BaseModel,
        ) -> SearchResult:
            return _make_result("clash")

    _register()
    with pytest.raises(DuplicateExplorationError, match="already registered"):
        _register()


def test_unknown_lists_known_names(_isolated_registry: None) -> None:
    @register_exploration(
        name="alpha",
        description="a",
        version="1.0.0",
        config_model=_DummyConfig,
    )
    def run(
        image: npt.NDArray[np.uint8],
        exemplar: ExemplarBox,
        config: BaseModel,
    ) -> SearchResult:
        return _make_result("alpha")

    with pytest.raises(UnknownExplorationError, match="alpha"):
        get_exploration("nope")


def test_list_is_sorted_by_name(_isolated_registry: None) -> None:
    for name in ("zulu", "alpha", "mike"):

        @register_exploration(
            name=name,
            description=name,
            version="1.0.0",
            config_model=_DummyConfig,
        )
        def run(
            image: npt.NDArray[np.uint8],
            exemplar: ExemplarBox,
            config: BaseModel,
            _name: str = name,
        ) -> SearchResult:
            return _make_result(_name)

    assert [spec.name for spec in list_explorations()] == ["alpha", "mike", "zulu"]


def test_same_image_search_is_registered_by_default() -> None:
    # Importing the package runs the registration side effects (no isolation here on purpose:
    # we want the real installed set).
    import object_search.explorations as explorations_pkg

    names = {spec.name for spec in explorations_pkg.list_explorations()}
    assert "same-image-search" in names


def test_same_image_search_adapter_delegates_to_a_method() -> None:
    """The default adapter runs a registered method end to end (uses ncc, no model)."""
    import object_search.explorations as explorations_pkg
    import object_search.search as _search  # noqa: F401  (installs methods)
    from object_search.explorations.same_image_search import SameImageSearchConfig

    spec = explorations_pkg.get_exploration("same-image-search")
    scene = np.zeros((48, 48, 3), dtype=np.uint8)
    scene[10:20, 10:20] = 255
    exemplar = ExemplarBox(box=BBox(x=10, y=10, w=10, h=10))
    result = spec.fn(scene, exemplar, SameImageSearchConfig(method="ncc", config={}))
    assert isinstance(result, SearchResult)
    assert result.method == "ncc"
