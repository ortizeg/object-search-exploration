"""The default exploration: a thin adapter over the Milestone 1 method flow.

Registering the existing "find every instance of *this object*" search as an exploration is what
makes the four Milestone 1 methods *one exploration among others* rather than a special case the
API and UI must know about by name. The adapter carries a ``method`` key plus that method's own
config and simply delegates to :func:`object_search.search.get_method`, so the same-image search
and the marker-conditioned search are dispatched through one uniform registry.

This module is deliberately tiny -- the whole point is that it adds *nothing* to the method flow;
it only re-labels it as an exploration.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field

from object_search.explorations.registry import register_exploration
from object_search.schemas.geometry import ExemplarBox
from object_search.schemas.search import SearchResult
from object_search.search import get_method

_EXPLORATION_VERSION = "1.0.0"


class SameImageSearchConfig(BaseModel):
    """Config for the default exploration: which method to run, and that method's own config.

    Attributes:
        method: Registry key of the Milestone 1 method to run (e.g. ``"ncc"``). Validated against
            the method registry at run time, exactly as ``POST /search`` validates it today.
        config: The chosen method's raw config, validated against its ``config_model`` at run
            time. Untyped here for the same reason the API request is: the exploration cannot name
            a method's config type without naming the method.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    method: str = Field(default="ncc", description="Registry key of the method to run.")
    config: Mapping[str, object] = Field(
        default_factory=dict, description="The method's own config; validated against its schema."
    )


@register_exploration(
    name="same-image-search",
    description="Find every other instance of the drawn object in the same image (Milestone 1).",
    version=_EXPLORATION_VERSION,
    config_model=SameImageSearchConfig,
)
def run(
    image: npt.NDArray[np.uint8],
    exemplar: ExemplarBox,
    config: BaseModel,
) -> SearchResult:
    """Delegate to the configured Milestone 1 method -- the adapter adds nothing else."""
    # The registry types config as BaseModel; the registered config_model guarantees the concrete
    # type. Narrow it once here and fail loudly if the contract is ever violated (mirrors methods).
    if not isinstance(config, SameImageSearchConfig):
        raise TypeError(
            f"same-image-search requires a SameImageSearchConfig, got {type(config).__name__}"
        )
    spec = get_method(config.method)
    method_config = spec.config_model.model_validate(config.config)
    return spec.fn(image, exemplar, method_config)
