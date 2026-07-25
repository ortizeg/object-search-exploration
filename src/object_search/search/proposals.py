"""The class-agnostic proposal stage -- an independently callable unit (Method 5, Phase 7).

This is one half of the Milestone 2 seam. :func:`propose` takes a raw scene and returns a list of
class-agnostic region :class:`~object_search.inference.Proposal` objects. It **knows nothing about
exemplars or retrieval** -- the exemplar-search method (``propose_retrieve.py``, plan 07-02)
composes this with the embedding stage and does nothing these two units cannot do alone. Phase 7's
defining success criterion is that this unit is callable *directly*, not only through ``search()``,
and a test in ``tests/test_proposals.py`` exercises exactly that.

Why a ``ProposalBackend`` protocol with a single implementation
---------------------------------------------------------------
FastSAM is the only proposal backend built in Milestone 1. The :class:`ProposalBackend` protocol
exists anyway so a second backend (MobileSAM) can be slotted in later **without restructuring** --
which is the deferred deviation recorded in ``docs/library-reviews/fastsam.md`` and the phase
CONTEXT: MobileSAM's ONNX decoder takes one prompt per call, so "everything mode" is ~1024
sequential calls plus a ported automatic-mask generator (a phase of work, not a backend swap). The
protocol is the seam that keeps that future cheap; the ``config: BaseModel`` signature is the same
backend-agnostic contract the ``SearchFn`` registry protocol uses.

The abstraction is deliberately thin: :func:`propose` is a five-line delegation, not a framework.
Per the Rule of Three it is a protocol-plus-one-impl only because the second implementation is a
*known, named, deferred* backend -- not speculative generality.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel

from object_search.inference import FastSAMConfig, FastSAMInferencer, Proposal, models


@runtime_checkable
class ProposalBackend(Protocol):
    """The shape every proposal backend has -- and the whole shared contract.

    A backend turns one BGR scene into class-agnostic region proposals under its own config.
    ``config`` is typed as :class:`~pydantic.BaseModel` (backend-agnostic); each backend narrows it
    to its own config type, exactly as the ``SearchFn`` registry protocol does. Runtime-checkable
    so a test can assert a concrete backend structurally satisfies it.
    """

    def propose(
        self,
        image: npt.NDArray[np.uint8],
        config: BaseModel,
    ) -> list[Proposal]:
        """Return class-agnostic region proposals for ``image`` under ``config``."""
        ...


def default_backend(
    model_path: Path | str | None = None,
    providers: list[str] | None = None,
) -> FastSAMInferencer:
    """Construct the Milestone 1 proposal backend: a :class:`FastSAMInferencer`.

    Args:
        model_path: Path to ``fastsam_s.onnx``. ``None`` uses the gitignored registry location
            (``models/`` + the ``fastsam-s`` spec's ``dest``), which must have been produced by
            ``pixi run -e export export-fastsam``.
        providers: ONNX Runtime execution providers. ``None`` pins ``CPUExecutionProvider``
            -- NOT the runtime default. On macOS the runtime default puts CoreML first, whose
            kernels are non-deterministic and, empirically, fail to build an execution plan for
            some input shapes ("Error in building plan"). Reproducibility is a hard project
            constraint (same input => identical results), so the proposal backend pins CPU
            exactly as ``dino_dense`` does. Pass an explicit list to override.

    Raises:
        FileNotFoundError: If the weight is absent -- surfaced here rather than swallowed, so the
            "export the AGPL weight first" step is loud.
    """
    path = (
        Path(model_path)
        if model_path is not None
        else (models.models_dir() / models.MODEL_REGISTRY["fastsam-s"].dest)
    )
    resolved_providers = providers if providers is not None else ["CPUExecutionProvider"]
    return FastSAMInferencer(path, providers=resolved_providers)


def propose(
    image: npt.NDArray[np.uint8],
    config: BaseModel,
    *,
    backend: ProposalBackend | None = None,
) -> list[Proposal]:
    """Return class-agnostic region proposals for ``image`` -- the independently callable unit.

    This is the Milestone 2 seam: it takes the raw scene and a config, and returns proposals. It
    does not know about exemplars, embeddings, or retrieval; ``propose_retrieve.py`` composes it
    with the embedding stage.

    Args:
        image: The BGR scene to propose regions in.
        config: The backend's decoding config (e.g. :class:`FastSAMConfig`).
        backend: The proposal backend to delegate to. ``None`` constructs the default FastSAM
            backend from the registry (which requires the exported weight). Tests inject a stub
            backend here to exercise the callable-unit contract without the weight.

    Returns:
        Proposals ordered by descending objectness, each carrying a box and objectness (and a mask
        when the config requested one).
    """
    resolved = backend if backend is not None else default_backend()
    return resolved.propose(image, config)


__all__ = [
    "FastSAMConfig",
    "Proposal",
    "ProposalBackend",
    "default_backend",
    "propose",
]
