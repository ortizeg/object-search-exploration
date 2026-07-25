"""The marker-conditioned exploration, tested model-free.

Neither a marker-finding model nor FastSAM is needed here: a throwaway marker *method* returns
matches at the synthetic markers' known boxes, and a stub proposal *backend* returns a fixed
proposal set. That isolates the orientation + scoring + propose-once logic from any ONNX weight;
the real end-to-end path is covered by a skip-when-absent test in the API suite.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import numpy.typing as npt
import pytest
from pydantic import BaseModel, ConfigDict

from object_search.explorations.marker_conditioned import MarkerConditionedConfig, run
from object_search.inference import FastSAMConfig, Proposal
from object_search.schemas.geometry import BBox, ExemplarBox
from object_search.schemas.search import (
    LatencyBreakdown,
    Match,
    SearchOutcome,
    SearchResult,
)
from object_search.search import register_method, unregister
from object_search.synthetic.generator import MarkerImage, MarkerSpec, synthesize_markers

_STUB_MARKER_METHOD = "stub-marker-for-tests"


class _StubMarkerConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class _CountingBackend:
    """A proposal backend that returns a fixed set and counts how often it is asked."""

    def __init__(self, proposals: list[Proposal]) -> None:
        self._proposals = proposals
        self.calls = 0

    def propose(self, image: npt.NDArray[np.uint8], config: BaseModel) -> list[Proposal]:
        self.calls += 1
        return list(self._proposals)


def _register_stub_marker_method(image: MarkerImage) -> None:
    """Register a marker method that returns a Match at each synthetic marker's box."""
    matches = tuple(Match(box=m.box, score=1.0 - 0.01 * i) for i, m in enumerate(image.markers))

    @register_method(
        name=_STUB_MARKER_METHOD,
        description="Returns the synthetic markers directly; no model.",
        version="0.0.0",
        config_model=_StubMarkerConfig,
    )
    def _search(
        img: npt.NDArray[np.uint8],
        exemplar: ExemplarBox,
        config: BaseModel,
    ) -> SearchResult:
        if not matches:
            return SearchResult(
                method=_STUB_MARKER_METHOD,
                method_version="0.0.0",
                outcome=SearchOutcome.EMPTY,
                matches=(),
                latency=LatencyBreakdown(preprocess_ms=0.0, inference_ms=0.0, postprocess_ms=0.0),
                threshold_applied=None,
            )
        return SearchResult(
            method=_STUB_MARKER_METHOD,
            method_version="0.0.0",
            outcome=SearchOutcome.OK,
            matches=matches,
            latency=LatencyBreakdown(preprocess_ms=0.0, inference_ms=0.0, postprocess_ms=0.0),
            threshold_applied=None,
        )


@pytest.fixture
def marker_scene() -> Iterator[MarkerImage]:
    """A 3-arrow scene with a stub marker method registered for it, cleaned up after."""
    image = synthesize_markers(MarkerSpec(seed=31, marker="arrow", n_markers=3, arrow_len=64))
    _register_stub_marker_method(image)
    try:
        yield image
    finally:
        unregister(_STUB_MARKER_METHOD)


def _proposals_at_the_targets(image: MarkerImage) -> list[Proposal]:
    """One well-placed proposal just past each marker's tip, plus a couple of decoys."""
    proposals: list[Proposal] = []
    for marker in image.markers:
        assert marker.direction is not None
        dx, dy = marker.direction
        # A box centred ~40 px past the tip along the pointing direction.
        cx = marker.tip.x + dx * 40.0
        cy = marker.tip.y + dy * 40.0
        proposals.append(
            Proposal(
                box=BBox(x=max(0, int(cx) - 16), y=max(0, int(cy) - 16), w=32, h=32),
                mask=None,
                objectness=0.9,
            )
        )
    # Two decoys far from every marker, in a corner.
    proposals.append(Proposal(box=BBox(x=1, y=1, w=20, h=20), mask=None, objectness=0.95))
    proposals.append(Proposal(box=BBox(x=1, y=40, w=20, h=20), mask=None, objectness=0.5))
    return proposals


def _config() -> MarkerConditionedConfig:
    return MarkerConditionedConfig(
        marker_method=_STUB_MARKER_METHOD, marker_config={}, proposal=FastSAMConfig()
    )


def test_three_markers_yield_three_matches_at_their_targets(marker_scene: MarkerImage) -> None:
    proposals = _proposals_at_the_targets(marker_scene)
    backend = _CountingBackend(proposals)
    result = run(
        marker_scene.image, ExemplarBox(box=marker_scene.markers[0].box), _config(), backend=backend
    )

    assert result.outcome is SearchOutcome.OK
    assert len(result.matches) == 3
    # Each marker should pick its own well-aligned proposal (index i), not a corner decoy.
    target_boxes = {(p.box.x, p.box.y) for p in proposals[:3]}
    for match in result.matches:
        assert (match.box.x, match.box.y) in target_boxes


def test_propose_is_called_exactly_once(marker_scene: MarkerImage) -> None:
    backend = _CountingBackend(_proposals_at_the_targets(marker_scene))
    run(
        marker_scene.image, ExemplarBox(box=marker_scene.markers[0].box), _config(), backend=backend
    )
    assert backend.calls == 1


def test_diagnostics_carry_markers_reference_points_directions(marker_scene: MarkerImage) -> None:
    backend = _CountingBackend(_proposals_at_the_targets(marker_scene))
    result = run(
        marker_scene.image, ExemplarBox(box=marker_scene.markers[0].box), _config(), backend=backend
    )
    diag = result.diagnostics
    assert diag.proposals is not None and len(diag.proposals) == 5
    assert diag.markers is not None and len(diag.markers) == 3
    assert diag.marker_reference_points is not None and len(diag.marker_reference_points) == 3
    assert diag.marker_directions is not None and len(diag.marker_directions) == 3
    assert diag.metrics["n_markers"] == 3.0


def test_direction_none_marker_still_scores_by_distance() -> None:
    """A dot (no direction) still resolves to the nearest proposal."""
    image = synthesize_markers(MarkerSpec(seed=32, marker="dot", n_markers=1))
    _register_stub_marker_method(image)
    try:
        dot = image.markers[0]
        near = Proposal(
            box=BBox(
                x=max(0, int(dot.centroid.x) + 10), y=max(0, int(dot.centroid.y) + 10), w=24, h=24
            ),
            mask=None,
            objectness=0.6,
        )
        far = Proposal(box=BBox(x=1, y=1, w=24, h=24), mask=None, objectness=0.6)
        backend = _CountingBackend([far, near])
        result = run(image.image, ExemplarBox(box=dot.box), _config(), backend=backend)
        assert result.outcome is SearchOutcome.OK
        assert len(result.matches) == 1
        assert (result.matches[0].box.x, result.matches[0].box.y) == (near.box.x, near.box.y)
        # The dot reported no direction.
        assert result.diagnostics.marker_directions == (None,)
    finally:
        unregister(_STUB_MARKER_METHOD)


def test_no_markers_found_is_empty_with_note() -> None:
    empty = MarkerImage(image=np.zeros((64, 64, 3), dtype=np.uint8), markers=(), spec=MarkerSpec())
    _register_stub_marker_method(empty)
    try:
        backend = _CountingBackend([])
        result = run(
            empty.image, ExemplarBox(box=BBox(x=0, y=0, w=8, h=8)), _config(), backend=backend
        )
        assert result.outcome is SearchOutcome.EMPTY
        assert result.matches == ()
        assert result.diagnostics.notes
        assert backend.calls == 0  # propose is never reached when there are no markers
    finally:
        unregister(_STUB_MARKER_METHOD)


def test_markers_but_no_proposals_is_empty_with_note(marker_scene: MarkerImage) -> None:
    backend = _CountingBackend([])  # markers found, but the proposal stage returns nothing
    result = run(
        marker_scene.image, ExemplarBox(box=marker_scene.markers[0].box), _config(), backend=backend
    )
    assert result.outcome is SearchOutcome.EMPTY
    assert result.diagnostics.notes
    assert backend.calls == 1


def test_deterministic_under_fixed_inputs(marker_scene: MarkerImage) -> None:
    exemplar = ExemplarBox(box=marker_scene.markers[0].box)
    r1 = run(
        marker_scene.image,
        exemplar,
        _config(),
        backend=_CountingBackend(_proposals_at_the_targets(marker_scene)),
    )
    r2 = run(
        marker_scene.image,
        exemplar,
        _config(),
        backend=_CountingBackend(_proposals_at_the_targets(marker_scene)),
    )
    assert [m.box for m in r1.matches] == [m.box for m in r2.matches]
    assert [m.score for m in r1.matches] == [m.score for m in r2.matches]


def test_wrong_config_type_raises(marker_scene: MarkerImage) -> None:
    with pytest.raises(TypeError, match="MarkerConditionedConfig"):
        run(marker_scene.image, ExemplarBox(box=marker_scene.markers[0].box), FastSAMConfig())
