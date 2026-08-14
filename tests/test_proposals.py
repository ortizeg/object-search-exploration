"""Tests for :mod:`object_search.search.proposals` -- the independently callable proposal unit.

The load-bearing assertion of Phase 7 is that ``propose()`` is a standalone unit: it is called
**directly here, never through ``search()``**. Model-free where possible (a stub backend proves the
callable-unit contract without the gitignored AGPL weight); the real-inference assertion is skipped
when ``fastsam_s.onnx`` is absent.
"""

from __future__ import annotations

from itertools import pairwise
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
    _ios,
    _merge_tiled_proposals,
    _tile_origins,
    default_backend,
    propose,
    propose_tiled,
    propose_tiled_with_stats,
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


# ============================================ SAHI-style tiling (model-free: pure arithmetic)
#
# Every test below runs without the gitignored AGPL weight -- tile geometry and the cross-tile
# merge are pure arithmetic, and the backend is a stub. CI has no weights, so this is the only way
# tiling gets covered at all (the `decode_fastsam` precedent).


class _TileRecordingBackend:
    """A stub backend that returns FIXED tile-local boxes and records the crop shape of each call.

    Tile-local is the point: `propose_tiled` must offset these into full-image coordinates, and a
    stub returning the same box every call makes a missing offset immediately visible (every
    returned box would be identical).
    """

    def __init__(self, boxes: list[BBox] | None = None) -> None:
        self.shapes: list[tuple[int, int]] = []
        self._boxes = boxes if boxes is not None else [BBox(x=2, y=3, w=10, h=10)]

    def propose(
        self,
        image: npt.NDArray[np.uint8],
        config: BaseModel,
    ) -> list[Proposal]:
        self.shapes.append((int(image.shape[0]), int(image.shape[1])))
        return [
            Proposal(box=box, mask=None, objectness=0.9 - 0.01 * i)
            for i, box in enumerate(self._boxes)
        ]


def test_tile_origins_is_an_identity_on_a_scene_that_fits_in_one_tile() -> None:
    """THE regression guard: w <= tile_side and h <= tile_side => exactly ONE full-image tile.

    This is what makes tiling a no-op on the chipset/textured/synthetic regimes *by construction*
    rather than by measurement -- those scenes are all well under a 1024 tile.
    """
    assert _tile_origins(800, 600, 1024, 0.2) == [(0, 0, 800, 600)]
    assert _tile_origins(1024, 1024, 1024, 0.2) == [(0, 0, 1024, 1024)]  # boundary: <=, not <


def test_tile_origins_steps_by_one_minus_overlap_and_clamps_the_final_tile() -> None:
    tiles = _tile_origins(2000, 512, 512, 0.2)
    step = round(512 * 0.8)  # 410
    xs = [x0 for x0, _, _, _ in tiles]

    assert xs[:3] == [0, step, 2 * step]
    # The final tile is CLAMPED to end at the image edge, not padded past it.
    assert tiles[-1][2] == 2000
    assert xs[-1] == 2000 - 512
    assert all(x1 <= 2000 and y1 <= 512 for _, _, x1, y1 in tiles)
    # Every tile is a full tile_side wide here (the clamp shifts the origin, it does not shrink).
    assert all(x1 - x0 == 512 for x0, _, x1, _ in tiles)


def test_tile_origins_consecutive_tiles_overlap_by_at_least_the_requested_band() -> None:
    for side, overlap in ((512, 0.2), (768, 0.2), (512, 0.3), (1024, 0.3)):
        tiles = _tile_origins(3000, 2000, side, overlap)
        xs = sorted({x0 for x0, _, _, _ in tiles})
        band = int(overlap * side)
        gaps = [(a + side) - b for a, b in pairwise(xs)]
        assert all(gap >= band for gap in gaps), (side, overlap, gaps, band)


def test_tile_origins_is_deterministic_and_ordered_by_y_then_x() -> None:
    tiles = _tile_origins(2000, 1500, 512, 0.2)
    assert tiles == sorted(tiles, key=lambda t: (t[1], t[0]))
    assert tiles == _tile_origins(2000, 1500, 512, 0.2)  # same input, same output


@pytest.mark.parametrize(
    ("side", "overlap"),
    [(0, 0.2), (-1, 0.2), (512, -0.1), (512, 0.9), (512, 1.0)],
)
def test_tile_origins_rejects_impossible_geometry(side: int, overlap: float) -> None:
    with pytest.raises(ValueError, match=r"tile_side|overlap"):
        _tile_origins(2000, 2000, side, overlap)


def test_merge_suppresses_a_contained_fragment_that_iou_nms_would_keep() -> None:
    """The load-bearing IoS test: a tile-edge-truncated fragment must NOT survive beside its whole.

    Plain IoU-NMS would keep both -- the fragment's IoU with the whole box is far below any sane
    threshold -- and the merged set would carry a duplicate for every instance straddling a tile
    edge. IoS (intersection / min-area) is 1.0 for a contained box, so it is suppressed.
    """
    whole = BBox(x=100, y=100, w=100, h=100)
    fragment = BBox(x=100, y=100, w=20, h=100)  # the sliver a tile edge cut off, fully inside

    assert fragment.iou(whole) == pytest.approx(0.2)  # an IoU merge at 0.5 keeps BOTH
    assert _ios(fragment, whole) == pytest.approx(1.0)  # contained => IoS is 1.0

    merged = _merge_tiled_proposals(
        [
            Proposal(box=fragment, mask=None, objectness=0.8),
            Proposal(box=whole, mask=None, objectness=0.9),
        ],
        merge_ios=0.5,
    )
    assert [p.box for p in merged] == [whole]


def test_merge_keeps_genuinely_distinct_instances() -> None:
    """Two disjoint instances both survive: the merge collapses duplicates, not neighbours."""
    a = BBox(x=0, y=0, w=50, h=50)
    b = BBox(x=200, y=200, w=50, h=50)
    merged = _merge_tiled_proposals(
        [Proposal(box=a, mask=None, objectness=0.9), Proposal(box=b, mask=None, objectness=0.8)],
        merge_ios=0.5,
    )
    assert {(p.box.x, p.box.y) for p in merged} == {(0, 0), (200, 200)}


def test_merge_is_order_independent_and_uses_the_canonical_total_order() -> None:
    """Shuffled input, byte-identical output: the merge order is (-objectness, y, x), a TOTAL order.

    Includes a deliberate objectness TIE, which is the only case where the (y, x) tail decides --
    the situation that makes the difference between deterministic and merely usually-deterministic.
    """
    proposals = [
        Proposal(box=BBox(x=0, y=0, w=20, h=20), mask=None, objectness=0.5),
        Proposal(box=BBox(x=300, y=0, w=20, h=20), mask=None, objectness=0.5),  # tie
        Proposal(box=BBox(x=0, y=300, w=20, h=20), mask=None, objectness=0.5),  # tie
        Proposal(box=BBox(x=600, y=600, w=20, h=20), mask=None, objectness=0.9),
    ]
    expected = [(p.box.x, p.box.y) for p in _merge_tiled_proposals(proposals, merge_ios=0.5)]

    for shuffled in ([*proposals[::-1]], [proposals[2], proposals[0], proposals[3], proposals[1]]):
        assert [
            (p.box.x, p.box.y) for p in _merge_tiled_proposals(shuffled, merge_ios=0.5)
        ] == expected
    # ... and it really is descending objectness first, then y, then x.
    assert expected == [(600, 600), (0, 0), (300, 0), (0, 300)]


def test_propose_tiled_offsets_tile_boxes_into_full_image_coordinates() -> None:
    """Boxes from a tile at (x0, y0) come back offset by (x0, y0) -- and are all distinct."""
    backend = _TileRecordingBackend([BBox(x=2, y=3, w=10, h=10)])
    image = np.zeros((600, 2000, 3), dtype=np.uint8)

    result = propose_tiled_with_stats(
        image,
        FastSAMConfig(),
        backend=backend,
        tile_side=512,
        overlap=0.2,
        include_full_image=False,
    )

    origins = _tile_origins(2000, 600, 512, 0.2)
    assert result.n_tiles == len(origins)
    assert len(backend.shapes) == len(origins)
    expected = sorted((x0 + 2, y0 + 3) for x0, y0, _, _ in origins)
    assert sorted((p.box.x, p.box.y) for p in result.proposals) == expected
    # Every box lies inside the scene.
    assert all(p.box.x2 <= 2000 and p.box.y2 <= 600 for p in result.proposals)


def test_propose_tiled_clips_a_box_that_runs_off_the_scene() -> None:
    """A tile-local box extending past the image edge is clipped, never emitted out of bounds."""
    backend = _TileRecordingBackend([BBox(x=500, y=500, w=200, h=200)])
    image = np.zeros((600, 2000, 3), dtype=np.uint8)

    proposals = propose_tiled(
        image,
        FastSAMConfig(),
        backend=backend,
        tile_side=512,
        overlap=0.2,
        include_full_image=False,
    )
    assert proposals  # not everything was clipped away
    assert all(p.box.x2 <= 2000 and p.box.y2 <= 600 for p in proposals)


def test_propose_tiled_unions_the_full_image_pass_when_asked() -> None:
    """SAHI + FI: one extra whole-image pass, counted in n_tiles -- and skipped when turned off."""
    image = np.zeros((600, 2000, 3), dtype=np.uint8)
    n_geom = len(_tile_origins(2000, 600, 512, 0.2))

    with_fi = _TileRecordingBackend()
    propose_tiled(
        image,
        FastSAMConfig(),
        backend=with_fi,
        tile_side=512,
        overlap=0.2,
        include_full_image=True,
    )
    without_fi = _TileRecordingBackend()
    propose_tiled(
        image,
        FastSAMConfig(),
        backend=without_fi,
        tile_side=512,
        overlap=0.2,
        include_full_image=False,
    )

    assert len(with_fi.shapes) == n_geom + 1
    assert len(without_fi.shapes) == n_geom
    assert (600, 2000) in with_fi.shapes  # the whole-image pass really saw the whole image
    assert (600, 2000) not in without_fi.shapes


def test_propose_tiled_skips_the_duplicate_full_image_pass_on_a_single_tile_scene() -> None:
    """A scene that fits in one tile gets ONE pass even with SAHI + FI on -- the pass would be
    a byte-identical duplicate of the single tile, so paying for it twice would be a bug."""
    backend = _TileRecordingBackend()
    result = propose_tiled_with_stats(
        np.zeros((400, 400, 3), dtype=np.uint8),
        FastSAMConfig(),
        backend=backend,
        tile_side=1024,
        include_full_image=True,
    )
    assert result.n_tiles == 1
    assert backend.shapes == [(400, 400)]


def test_propose_tiled_applies_max_proposals_globally_after_the_merge() -> None:
    """max_proposals caps the MERGED set, never each tile -- a per-tile cap would silently
    re-impose the fixed proposal budget that tiling exists to defeat."""
    backend = _TileRecordingBackend([BBox(x=5, y=5, w=20, h=20), BBox(x=200, y=200, w=20, h=20)])
    image = np.zeros((600, 2000, 3), dtype=np.uint8)

    result = propose_tiled_with_stats(
        image,
        FastSAMConfig(max_proposals=3),
        backend=backend,
        tile_side=512,
        overlap=0.2,
        include_full_image=False,
    )
    assert len(result.proposals) == 3
    assert result.n_pre_merge > 3  # the cap was applied after the merge, not per tile


def test_propose_tiled_reports_the_merge_it_performed() -> None:
    """n_pre_merge >= len(proposals): the tiling cost is reported, never hidden (EVAL-11)."""
    # Every tile returns a box filling that tile; at overlap 0.6 consecutive tiles share 60% of
    # their extent, so those full-image boxes have IoS > 0.5 and the merge collapses them.
    backend = _TileRecordingBackend([BBox(x=0, y=0, w=512, h=512)])
    result = propose_tiled_with_stats(
        np.zeros((1024, 1024, 3), dtype=np.uint8),
        FastSAMConfig(),
        backend=backend,
        tile_side=512,
        overlap=0.6,
        include_full_image=False,
    )
    assert result.n_pre_merge == result.n_tiles
    assert len(result.proposals) < result.n_pre_merge


def test_propose_tiled_rejects_masks_rather_than_returning_tile_local_ones() -> None:
    """Masks come back in TILE-local coordinates; silently returning them would be a correctness
    bug, so the unsupported combination is loud."""
    with pytest.raises(ValueError, match="return_masks"):
        propose_tiled(
            np.zeros((2000, 2000, 3), dtype=np.uint8),
            FastSAMConfig(return_masks=True),
            backend=_TileRecordingBackend(),
            tile_side=512,
        )


def test_propose_tiled_returns_descending_objectness_like_propose() -> None:
    """Same contract as propose(): a caller cannot tell tiled from untiled by the return value."""
    backend = _TileRecordingBackend(
        [BBox(x=5, y=5, w=20, h=20), BBox(x=200, y=200, w=20, h=20), BBox(x=400, y=5, w=20, h=20)]
    )
    proposals = propose_tiled(
        np.zeros((600, 2000, 3), dtype=np.uint8),
        FastSAMConfig(),
        backend=backend,
        tile_side=512,
        overlap=0.2,
    )
    scores = [p.objectness for p in proposals]
    assert scores == sorted(scores, reverse=True)
    assert all(isinstance(p.box, BBox) for p in proposals)
