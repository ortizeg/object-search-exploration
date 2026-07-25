"""The exploration registry -- INFRA-10 generalised from methods to whole explorations.

An *exploration* is one complete way of turning an exemplar box into a persistable
:class:`SearchResult`: the same-image search of Milestone 1 is one, the marker-conditioned
search of Milestone 2 is another. This registry is the deliberate mirror of
``search/registry.py`` -- same surface, same rules -- so the API and UI stay schema-driven and
adding an exploration costs exactly one new file plus one import in ``explorations/__init__.py``.

Keep this file small and obvious, exactly like the method registry it mirrors. The
``@register_exploration`` decorator is the only indirection; there is no plugin scan and no
config-driven dispatch.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field

from object_search.schemas.geometry import ExemplarBox
from object_search.schemas.search import SearchResult


@runtime_checkable
class ExplorationFn(Protocol):
    """The shape every exploration's ``run`` function has -- the whole shared contract.

    Identical in spirit to ``SearchFn``: an exploration takes the scene, the drawn exemplar and
    its own config, and returns a :class:`SearchResult` that persists through the unchanged store.
    """

    def __call__(
        self,
        image: npt.NDArray[np.uint8],
        exemplar: ExemplarBox,
        config: BaseModel,
    ) -> SearchResult:
        """Run the exploration over ``image`` seeded by ``exemplar`` under ``config``."""
        ...


@dataclass(frozen=True)
class ExplorationSpec:
    """A registered exploration: its identity, its config model, and the callable that runs it."""

    name: str
    description: str
    version: str
    config_model: type[BaseModel]
    fn: ExplorationFn
    module: str


class ExplorationInfo(BaseModel):
    """The API-facing description of an exploration -- name, docs, and its config JSON Schema.

    ``config_schema`` is the exploration's ``config_model`` rendered as JSON Schema, exactly what
    the UI turns into a config form. Because it is derived here, the API layer never hardcodes an
    exploration name or a config field.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    description: str
    version: str
    config_schema: Mapping[str, object] = Field(default_factory=dict)


class DuplicateExplorationError(ValueError):
    """Raised when two explorations register under the same name -- never a silent overwrite."""


class UnknownExplorationError(KeyError):
    """Raised when a name is looked up that no exploration registered under."""


_REGISTRY: dict[str, ExplorationSpec] = {}


def register_exploration(
    *,
    name: str,
    description: str,
    version: str,
    config_model: type[BaseModel],
) -> Callable[[ExplorationFn], ExplorationFn]:
    """Register the decorated ``run`` function under ``name``.

    Raises:
        DuplicateExplorationError: If ``name`` is already registered. A silent overwrite would
            let two explorations share one scoreboard grouping, quietly corrupting the very
            comparisons this project exists to make.
    """

    def decorator(fn: ExplorationFn) -> ExplorationFn:
        if name in _REGISTRY:
            raise DuplicateExplorationError(
                f"exploration {name!r} is already registered by "
                f"{_REGISTRY[name].module!r}; names must be unique"
            )
        _REGISTRY[name] = ExplorationSpec(
            name=name,
            description=description,
            version=version,
            config_model=config_model,
            fn=fn,
            module=getattr(fn, "__module__", "?"),
        )
        return fn

    return decorator


def unregister(name: str) -> None:
    """Remove the exploration registered under ``name`` -- the partner of registration.

    Registration is normally a permanent import-time side effect, so production code never calls
    this; it exists for **test isolation** so a throwaway exploration registered for one test does
    not leak into the global registry a later test enumerates.

    Raises:
        UnknownExplorationError: If no exploration registered under ``name``.
    """
    try:
        del _REGISTRY[name]
    except KeyError:
        raise UnknownExplorationError(f"cannot unregister unknown exploration {name!r}") from None


def has_exploration(name: str) -> bool:
    """True if an exploration registered under ``name``."""
    return name in _REGISTRY


def get_exploration(name: str) -> ExplorationSpec:
    """Return the :class:`ExplorationSpec` for ``name``.

    Raises:
        UnknownExplorationError: If no exploration registered under ``name``. The message lists
            the known names so a typo is diagnosable rather than mysterious.
    """
    try:
        return _REGISTRY[name]
    except KeyError:
        known = ", ".join(sorted(_REGISTRY)) or "(none registered)"
        raise UnknownExplorationError(
            f"unknown exploration {name!r}; known explorations: {known}"
        ) from None


def list_explorations() -> tuple[ExplorationSpec, ...]:
    """Every registered exploration, sorted by name so API output is deterministic."""
    return tuple(_REGISTRY[name] for name in sorted(_REGISTRY))


def exploration_schemas() -> tuple[ExplorationInfo, ...]:
    """Every exploration as an :class:`ExplorationInfo`, sorted by name."""
    return tuple(
        ExplorationInfo(
            name=spec.name,
            description=spec.description,
            version=spec.version,
            config_schema=spec.config_model.model_json_schema(),
        )
        for spec in list_explorations()
    )
