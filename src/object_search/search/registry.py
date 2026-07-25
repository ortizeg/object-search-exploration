"""The one abstraction that matters: the ``SearchMethod`` registry (INFRA-10).

This is the *only* indirection in the codebase. Adding a search method is one new file in
``search/`` that ends in a ``@register_method(...)`` decorator, plus one import line in
``search/__init__.py`` so the module is loaded for its registration side effect. Nothing
else is shared between methods -- no base class, no plugin scan, no config-driven dispatch.

Keep this file small and obvious on purpose. If it stops being readable top-to-bottom, the
"no hidden control flow" convention has been violated.
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
class SearchFn(Protocol):
    """The shape every method's ``search`` function has -- and the *whole* shared contract."""

    def __call__(
        self,
        image: npt.NDArray[np.uint8],
        exemplar: ExemplarBox,
        config: BaseModel,
    ) -> SearchResult:
        """Find every instance of ``exemplar`` in ``image`` under ``config``."""
        ...


@dataclass(frozen=True)
class MethodSpec:
    """A registered method: its identity, its config model, and the callable that runs it."""

    name: str
    description: str
    version: str
    config_model: type[BaseModel]
    fn: SearchFn
    module: str


class MethodInfo(BaseModel):
    """The API-facing description of a method (API-01) -- name, docs, and its config schema.

    ``config_schema`` is the method's ``config_model`` rendered as JSON Schema, which is
    exactly what UI-07 turns into a config form. Because it is derived here, the API layer
    never hardcodes a method name or a config field.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    description: str
    version: str
    config_schema: Mapping[str, object] = Field(default_factory=dict)


class DuplicateMethodError(ValueError):
    """Raised when two methods register under the same name -- never a silent overwrite."""


class UnknownMethodError(KeyError):
    """Raised when a name is looked up that no method registered under."""


_REGISTRY: dict[str, MethodSpec] = {}


def register_method(
    *,
    name: str,
    description: str,
    version: str,
    config_model: type[BaseModel],
) -> Callable[[SearchFn], SearchFn]:
    """Register the decorated ``search`` function under ``name``.

    Raises:
        DuplicateMethodError: If ``name`` is already registered. A silent overwrite would
            let two methods share one scoreboard row, which would quietly corrupt every
            comparison this project exists to make.
    """

    def decorator(fn: SearchFn) -> SearchFn:
        if name in _REGISTRY:
            raise DuplicateMethodError(
                f"method {name!r} is already registered by "
                f"{_REGISTRY[name].module!r}; names must be unique"
            )
        _REGISTRY[name] = MethodSpec(
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
    """Remove the method registered under ``name`` -- the symmetric partner of registration.

    Registration is normally a permanent, import-time side effect, so production code never
    calls this. It exists for **test isolation**: a throwaway method registered for one test
    (an always-raising stub exercising the error path, say) must not leak into the global
    registry that a later test enumerates, or that later test sees a phantom method it cannot
    run. A registration with no clean-up is the same shared-mutable-state trap the registry's
    duplicate check guards against, one level up.

    Raises:
        UnknownMethodError: If no method registered under ``name`` -- removing a name that was
            never registered is a bug worth surfacing, not a silent no-op.
    """
    try:
        del _REGISTRY[name]
    except KeyError:
        raise UnknownMethodError(f"cannot unregister unknown method {name!r}") from None


def has_method(name: str) -> bool:
    """True if a method registered under ``name``."""
    return name in _REGISTRY


def get_method(name: str) -> MethodSpec:
    """Return the :class:`MethodSpec` for ``name``.

    Raises:
        UnknownMethodError: If no method registered under ``name``. The message lists the
            known names so a typo is diagnosable rather than mysterious.
    """
    try:
        return _REGISTRY[name]
    except KeyError:
        known = ", ".join(sorted(_REGISTRY)) or "(none registered)"
        raise UnknownMethodError(f"unknown method {name!r}; known methods: {known}") from None


def list_methods() -> tuple[MethodSpec, ...]:
    """Every registered method, sorted by name so API output is deterministic."""
    return tuple(_REGISTRY[name] for name in sorted(_REGISTRY))


def method_schemas() -> tuple[MethodInfo, ...]:
    """Every method as a :class:`MethodInfo` (API-01), sorted by name."""
    return tuple(
        MethodInfo(
            name=spec.name,
            description=spec.description,
            version=spec.version,
            config_schema=spec.config_model.model_json_schema(),
        )
        for spec in list_methods()
    )
