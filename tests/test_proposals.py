"""Tests for :mod:`object_search.search.proposals` -- the independently callable proposal unit.

The load-bearing assertion of Phase 7 is that ``propose()`` is a standalone unit: it is called
**directly here, never through ``search()``**. Model-free where possible (a stub backend proves the
callable-unit contract without the gitignored AGPL weight); the real-inference assertion is skipped
when ``fastsam_s.onnx`` is absent.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt
import pytest
from pydantic import BaseModel

from object_search.inference import FastSAMConfig, Proposal, models
from object_search.schemas import BBox
from object_search.search.proposals import (
    ProposalBackend,
    default_backend,
    propose,
)

_CPU = ["CPUExecutionProvider"]
_MODEL_PATH: Path = models.models_dir() / models.MODEL_REGISTRY["fastsam-s"].dest
_HAVE_MODEL: bool = _MODEL_PATH.is_file()
_needs_model = pytest.mark.skipif(
    not _HAVE_MODEL,
    reason=(
        f"fastsam-s weight absent at {_MODEL_PATH} "
        f"(gitignored AGPL export; run pixi run -e export export-fastsam)"
    ),
)

_CHIPSET_IMAGE = (
    Path(__file__).resolve().parent.parent / "assets" / "demo" / "chipset" / "chipset-01.png"
)


class _StubBackend:
    """A minimal :class:`ProposalBackend` that records its calls and returns fixed proposals.

    Proves ``propose()`` is an independently callable unit: it delegates to a backend and returns
    ``Proposal`` objects, with no reference to exemplars or retrieval and no weight required.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[tuple[int, ...], BaseModel]] = []

    def propose(
        self,
        image: npt.NDArray[np.uint8],
        config: BaseModel,
    ) -> list[Proposal]:
        self.calls.append((image.shape, config))
        return [
            Proposal(box=BBox(x=0, y=0, w=10, h=10), mask=None, objectness=0.9),
            Proposal(box=BBox(x=20, y=20, w=15, h=15), mask=None, objectness=0.7),
        ]


def test_stub_backend_satisfies_the_protocol() -> None:
    """The protocol is runtime-checkable, so a structural implementation is recognised."""
    assert isinstance(_StubBackend(), ProposalBackend)


def test_propose_is_callable_standalone_and_returns_proposals() -> None:
    """Call propose() DIRECTLY (not via search) and assert boxes + objectness come back."""
    assert _CHIPSET_IMAGE.is_file(), f"committed chipset image missing at {_CHIPSET_IMAGE}"
    image = cv2.imread(str(_CHIPSET_IMAGE))
    assert image is not None

    stub = _StubBackend()
    proposals = propose(image, FastSAMConfig(), backend=stub)

    # The callable-unit contract: a non-empty list of Proposals, each with a box and objectness.
    assert len(proposals) >= 1
    assert all(isinstance(p, Proposal) for p in proposals)
    for p in proposals:
        assert isinstance(p.box, BBox)
        assert 0.0 <= p.objectness <= 1.0

    # It delegated to the injected backend, passing the raw scene through untouched.
    assert len(stub.calls) == 1
    assert stub.calls[0][0] == image.shape


def test_propose_passes_config_through_to_backend() -> None:
    stub = _StubBackend()
    cfg = FastSAMConfig(conf_thres=0.25, max_proposals=5)
    propose(np.zeros((32, 32, 3), dtype=np.uint8), cfg, backend=stub)
    assert stub.calls[0][1] is cfg


def test_default_backend_without_weight_raises() -> None:
    """With the weight absent, constructing the default backend surfaces a loud error.

    (When the weight IS present this path is covered by the real-model test below.)
    """
    if _HAVE_MODEL:
        pytest.skip("weight present; the missing-weight path is exercised only when absent")
    with pytest.raises(FileNotFoundError):
        default_backend()


@_needs_model
def test_real_fastsam_backend_is_a_proposal_backend() -> None:
    backend = default_backend(_MODEL_PATH, providers=_CPU)
    assert isinstance(backend, ProposalBackend)


@_needs_model
def test_propose_with_real_backend_returns_nonempty() -> None:
    image = cv2.imread(str(_CHIPSET_IMAGE))
    assert image is not None
    backend = default_backend(_MODEL_PATH, providers=_CPU)
    proposals = propose(image, FastSAMConfig(conf_thres=0.3), backend=backend)
    assert len(proposals) >= 1
    for p in proposals:
        assert isinstance(p.box, BBox)
        assert 0.0 <= p.objectness <= 1.0
