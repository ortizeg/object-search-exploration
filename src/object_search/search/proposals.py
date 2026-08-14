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

SAHI-style tiling lives here, as a peer of :func:`propose`
-----------------------------------------------------------
:func:`propose_tiled` runs the *same* backend over overlapping tiles of the scene and merges the
results. It sits beside :func:`propose` rather than inside ``propose_retrieve.py`` because
``propose`` already has THREE callers (``search/propose_retrieve.py``,
``explorations/marker_conditioned.py``, ``synthetic/real_insertion.py``), so the Rule of Three is
already satisfied for this module -- tiling is a property of the proposal stage, not of one method.
It is kept in the same register as ``propose``: a wrapper and a merge, not a framework.

Its motivation is measured, not assumed. On floor plans, FastSAM's everything-mode proposal budget
scales with image **area** (r = +0.59) and barely with instance count (r = +0.22), so a crowded plan
gets ~40 proposals for ~15 symbols and the proposal stage caps recall at 0.27 before any retrieval
runs (quick task ``260812-m8m``, EXPERIMENTS.md B0). N tiles buy roughly N x the budget, and each
tile is upscaled by FastSAM's fixed 1024 letterbox, so a small symbol also arrives magnified.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel

from object_search.inference import (
    FastSAMConfig,
    FastSAMInferencer,
    Proposal,
    models,
    resolve_providers,
)
from object_search.schemas import BBox


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
        providers: ONNX Runtime execution providers. ``None`` uses :func:`resolve_providers`
            (``CPUExecutionProvider`` by default, so a run is bit-identical across machines --
            NOT the runtime default, which puts CoreML first on macOS: its kernels are
            non-deterministic and empirically fail to build a plan for some shapes). Set the
            ``OS_ONNX_PROVIDERS`` env var (or pass an explicit list) to opt into GPU.

    Raises:
        FileNotFoundError: If the weight is absent -- surfaced here rather than swallowed, so the
            "export the AGPL weight first" step is loud.
    """
    path = (
        Path(model_path)
        if model_path is not None
        else (models.models_dir() / models.MODEL_REGISTRY["fastsam-s"].dest)
    )
    resolved_providers = providers if providers is not None else resolve_providers()
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


# ------------------------------------------------------------------ SAHI-style tiled proposals
# Pure arithmetic below (no model, no RNG), so CI gates it fully without the gitignored AGPL weight
# -- the same discipline `decode_fastsam` follows. Tiling adds NO randomness: tile order and the
# merge use a deterministic total order, so there is deliberately no seed parameter here (an
# advertised-but-inert control is worse than no control).


def _axis_starts(extent: int, side: int, step: int) -> list[int]:
    """Tile start offsets along one axis: fixed ``step``, final tile CLAMPED to the edge.

    Clamped rather than padded, matching SAHI (``slicing.py``: ``x_max = min(image_width, x_max)``).
    Padding a short final tile would letterbox grey into the model's field of view and shift the
    effective magnification for the symbols that happen to fall there.
    """
    if extent <= side:
        return [0]
    starts: list[int] = []
    pos = 0
    while pos + side < extent:
        starts.append(pos)
        pos += step
    starts.append(extent - side)  # the clamped final tile
    return starts


def _tile_origins(
    width: int,
    height: int,
    tile_side: int,
    overlap: float,
) -> list[tuple[int, int, int, int]]:
    """Return the tile rectangles ``(x0, y0, x1, y1)`` covering a ``width`` x ``height`` scene.

    Deterministic and image-clamped. **When the scene already fits inside one tile
    (``width <= tile_side and height <= tile_side``) exactly ONE tile is returned, equal to the
    whole image** -- so tiling is an identity on small scenes and cannot perturb them. That property
    is the regression guard: it is why enabling tiling on the chipset/textured regimes (all well
    under a 1024 tile) is a no-op by construction rather than by measurement.

    Step is ``round(tile_side * (1 - overlap))``, so consecutive tiles overlap by
    ``tile_side - step >= int(overlap * tile_side)`` pixels. Tiles are ordered by ``(y0, x0)``.

    Args:
        width: Scene width in pixels.
        height: Scene height in pixels.
        tile_side: Tile edge length in **native image pixels** (not model input pixels).
        overlap: Fraction of ``tile_side`` that consecutive tiles share, in ``[0, 0.9)``.

    Raises:
        ValueError: If ``tile_side`` is not positive or ``overlap`` is outside ``[0, 0.9)``.
    """
    if tile_side <= 0:
        raise ValueError(f"tile_side must be positive, got {tile_side}")
    if not 0.0 <= overlap < 0.9:
        raise ValueError(f"overlap must be in [0.0, 0.9), got {overlap}")
    if width <= tile_side and height <= tile_side:
        return [(0, 0, width, height)]

    step = max(1, round(tile_side * (1.0 - overlap)))
    return [
        (x0, y0, min(x0 + tile_side, width), min(y0 + tile_side, height))
        for y0 in _axis_starts(height, tile_side, step)
        for x0 in _axis_starts(width, tile_side, step)
    ]


def _ios(a: BBox, b: BBox) -> float:
    """Intersection over the SMALLER box's area -- the SAHI match metric, not IoU.

    A symbol truncated by a tile edge is nearly *contained* in the whole-object box that an
    overlapping tile (or the full-image pass) found. Contained boxes have **high IoS and low IoU**,
    so plain IoU-NMS keeps both and the merged set carries a duplicate fragment for every instance
    straddling a tile edge. This is exactly why ``search/common/nms.py`` is NOT imported here: it is
    IoU-based, which is the right metric for collapsing over-segmentation *after* retrieval and the
    wrong one for merging across tiles. Shared helpers are offerings, not requirements.
    """
    ix = max(0, min(a.x2, b.x2) - max(a.x, b.x))
    iy = max(0, min(a.y2, b.y2) - max(a.y, b.y))
    intersection = ix * iy
    smaller = min(a.area, b.area)
    return intersection / smaller if smaller > 0 else 0.0


def _merge_tiled_proposals(
    proposals: list[Proposal],
    *,
    merge_ios: float,
    max_proposals: int | None = None,
) -> list[Proposal]:
    """Greedy merge of cross-tile proposals by **IoS**, in the project's canonical order.

    Sorts by ``(-objectness, y, x)`` -- the deterministic order used everywhere in this repo --
    **refined by ``(h, w)``**, then keeps a proposal unless its IoS against an already-kept one
    **exceeds** ``merge_ios`` (SAHI matches on a strict ``>``). The size refinement is needed here
    and nowhere else: tiling routinely produces a truncated fragment and its whole-object box
    sharing a top-left corner, so ``(-objectness, y, x)`` alone is not a *total* order on this input
    and the result would depend on tile visit order. With the refinement the same input in a
    shuffled order yields byte-identical output.

    ``max_proposals`` is applied **after** the merge, never per tile, so the budget is global: a
    per-tile cap would silently re-impose the fixed-budget failure mode tiling exists to defeat.
    """
    ordered = sorted(proposals, key=lambda p: (-p.objectness, p.box.y, p.box.x, p.box.h, p.box.w))
    kept: list[Proposal] = []
    for candidate in ordered:
        if any(_ios(candidate.box, k.box) > merge_ios for k in kept):
            continue
        kept.append(candidate)
        if max_proposals is not None and len(kept) >= max_proposals:
            break
    return kept


class TiledProposals(NamedTuple):
    """:func:`propose_tiled`'s result plus the two cost numbers EVAL-11 requires be reported.

    ``propose_tiled`` returns only ``proposals`` (the same contract as :func:`propose`, so callers
    cannot tell tiled from untiled); a caller that must *report* the tiling cost -- as
    ``propose_retrieve.search`` does, into ``diagnostics.metrics`` -- calls
    :func:`propose_tiled_with_stats` instead. The cost of tiling is never hidden.
    """

    proposals: list[Proposal]
    n_tiles: int
    n_pre_merge: int


def propose_tiled_with_stats(
    image: npt.NDArray[np.uint8],
    config: BaseModel,
    *,
    backend: ProposalBackend | None = None,
    tile_side: int = 1024,
    overlap: float = 0.2,
    merge_ios: float = 0.5,
    include_full_image: bool = True,
) -> TiledProposals:
    """:func:`propose_tiled`, additionally returning the tile count and the pre-merge count.

    Raises:
        ValueError: If ``config.return_masks`` is true. Masks come back in **tile-local**
            coordinates and mapping them into full-image coordinates is out of scope; silently
            returning tile-local masks would be a correctness bug, so this is loud.
    """
    # The ProposalBackend contract types config as BaseModel, so the two decoding fields this
    # wrapper must honour are read defensively -- a backend config lacking them still tiles.
    if getattr(config, "return_masks", False):
        raise ValueError(
            "propose_tiled does not support return_masks=True: FastSAM masks come back in "
            "tile-local coordinates and are not mapped back to the full image. Use propose() for "
            "masks, or set return_masks=False."
        )
    max_proposals: int | None = getattr(config, "max_proposals", None)

    resolved = backend if backend is not None else default_backend()
    height, width = int(image.shape[0]), int(image.shape[1])
    tiles = _tile_origins(width, height, tile_side, overlap)

    # 0. Single-pass short circuit -- what makes tiling an EXACT identity on a scene that already
    #    fits in one tile. Without it, the IoS merge would still run and would suppress the
    #    *contained* proposals that "everything mode" deliberately emits: the merge is a CROSS-TILE
    #    deduplicator, and with one pass there is nothing cross-tile to deduplicate. Collapsing
    #    over-segmentation is the post-retrieval NMS's job (propose_retrieve step 6), not this
    #    one's.
    if len(tiles) == 1 and tiles[0] == (0, 0, width, height):
        single = resolved.propose(image, config)
        return TiledProposals(proposals=single, n_tiles=1, n_pre_merge=len(single))

    # 1. One backend pass per tile; boxes come back tile-local and are offset into full-image
    #    coordinates, then clipped to the scene.
    collected: list[Proposal] = []
    for x0, y0, x1, y1 in tiles:
        crop = np.ascontiguousarray(image[y0:y1, x0:x1])
        for proposal in resolved.propose(crop, config):
            shifted = _shift_and_clip(proposal.box, x0, y0, width, height)
            if shifted is not None:
                collected.append(Proposal(box=shifted, mask=None, objectness=proposal.objectness))

    # 2. SAHI + FI: union the whole-image pass in, so a symbol too large for any tile's overlap
    #    band (and the whole-plan context) still comes back. (The single-tile scene never reaches
    #    here -- step 0 returned it -- so this pass is never a duplicate of the loop above.)
    n_tiles = len(tiles)
    if include_full_image:
        collected.extend(resolved.propose(image, config))
        n_tiles += 1

    # 3. Merge by IoS in the canonical order, then apply the GLOBAL max_proposals cap.
    n_pre_merge = len(collected)
    merged = _merge_tiled_proposals(collected, merge_ios=merge_ios, max_proposals=max_proposals)
    return TiledProposals(proposals=merged, n_tiles=n_tiles, n_pre_merge=n_pre_merge)


def _shift_and_clip(box: BBox, x0: int, y0: int, width: int, height: int) -> BBox | None:
    """Offset a tile-local box into full-image coordinates and clip it.

    Returns ``None`` when the clip leaves nothing (a box entirely outside the scene).
    """
    left = max(0, min(box.x + x0, width))
    top = max(0, min(box.y + y0, height))
    right = max(0, min(box.x2 + x0, width))
    bottom = max(0, min(box.y2 + y0, height))
    if right - left < 1 or bottom - top < 1:
        return None
    return BBox(x=left, y=top, w=right - left, h=bottom - top)


def propose_tiled(
    image: npt.NDArray[np.uint8],
    config: BaseModel,
    *,
    backend: ProposalBackend | None = None,
    tile_side: int = 1024,
    overlap: float = 0.2,
    merge_ios: float = 0.5,
    include_full_image: bool = True,
) -> list[Proposal]:
    """SAHI-style tiled proposals -- a peer of :func:`propose` with the identical return contract.

    Runs ``backend`` over overlapping ``tile_side`` tiles of the scene, offsets each tile's boxes
    into full-image coordinates, optionally unions in a whole-image pass ("SAHI + FI"), and merges
    the union by **IoS** (see :func:`_ios` for why not IoU). Returns proposals ordered by descending
    objectness -- exactly what :func:`propose` returns, so a caller cannot tell the difference.

    On a scene that already fits inside one tile this is an identity: one tile equal to the whole
    image, no full-image duplicate pass, and the merge sees a single backend call's output.

    Args:
        image: The BGR scene to propose regions in.
        config: The backend's decoding config (e.g. :class:`FastSAMConfig`). ``max_proposals`` is
            applied **after** the merge, so the budget is global rather than per tile.
        backend: The proposal backend to delegate to (``None`` constructs the default FastSAM one).
        tile_side: Tile edge in **native image pixels**, not model input pixels. FastSAM letterboxes
            every tile to a fixed 1024 square, so a tile of side S magnifies each symbol by 1024/S.
        overlap: Fraction of ``tile_side`` shared by consecutive tiles (SAHI's default is 0.2).
        merge_ios: IoS match threshold for the cross-tile merge (SAHI's default is 0.5).
        include_full_image: Union the untiled whole-image pass in before merging ("SAHI + FI",
            SAHI's ``perform_standard_pred``, default True). Costs exactly one extra pass.

    Returns:
        Merged proposals ordered by descending objectness.
    """
    return propose_tiled_with_stats(
        image,
        config,
        backend=backend,
        tile_side=tile_side,
        overlap=overlap,
        merge_ios=merge_ios,
        include_full_image=include_full_image,
    ).proposals


__all__ = [
    "FastSAMConfig",
    "Proposal",
    "ProposalBackend",
    "TiledProposals",
    "default_backend",
    "propose",
    "propose_tiled",
    "propose_tiled_with_stats",
]
