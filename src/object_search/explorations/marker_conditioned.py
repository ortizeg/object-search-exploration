"""Marker-conditioned region proposal -- the Milestone 2 exploration (M2-01..M2-04).

The gesture, not the object, is the query: find every instance of a drawn *marker* (arrow, dot,
caret) in the image, and for each one return the best object-region proposal it points at. This
one self-contained module composes four already-shipped seams and adds nothing they cannot do
alone:

  1. Find markers    -- any Milestone 1 method, unchanged (``get_method(...).search``).
  2. Orient each     -- ``estimate_geometry`` (transform path when the method fitted an affine,
                        else PCA on the marker mask); a symmetric marker yields no direction.
  3. Propose objects -- ``propose(image, ...)`` ONCE for the whole scene (the proposal stage
                        dominates latency, so it is never called per marker).
  4. Score + pick    -- a documented weighted sum of proximity, direction-alignment, objectness
                        and a size prior; the best proposal per marker becomes a ``Match``.

The result is an ordinary :class:`SearchResult`, so it persists and is scored through the
Milestone 1 store/stats layer with no schema migration.

Pre-processing (explicit)
    The marker crop for orientation is ``image[box.y:box.y2, box.x:box.x2]`` in BGR; the marker
    method does its own pre-processing. ``propose`` letterboxes and normalises the scene per the
    FastSAM inferencer's documented contract. No colour conversion or resize happens in this
    module beyond cropping.

Post-processing (explicit)
    Proposals arrive ordered by descending objectness. For each marker every proposal is scored
    and the single best is kept (METHOD-12 analogue: one best proposal *per marker*, never a
    single global result). Ties are broken by ``(-score, box.y, box.x)`` so the pick is
    deterministic. No NMS is applied across markers -- two markers may legitimately point at the
    same object.

ROBUSTNESS BACKLOG
    - Direction alignment uses the proposal *centre*; an elongated object off to one side of the
      pointing ray can be penalised. A ray-to-box distance would be fairer.
    - The size prior is a single global ``size_prior_frac``; a per-marker prior learned from the
      exemplar's own object would be better.
    - Proximity and size use the marker's box diagonal as the length scale; a marker whose box is
      dominated by a long thin shaft has a larger scale than its head warrants.
    - A proposal is scored against every marker independently; a global assignment (each proposal
      to at most one marker) would avoid double-claiming when markers are close.
    - Symmetric markers score with the direction term zeroed; a weak orientation prior from
      nearby object density could still help without guessing a hard direction.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from time import perf_counter

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field

from object_search.explorations.markers import estimate_geometry
from object_search.explorations.registry import register_exploration
from object_search.inference import FastSAMConfig, Proposal
from object_search.schemas.geometry import ExemplarBox, Point
from object_search.schemas.search import (
    Diagnostics,
    LatencyBreakdown,
    Match,
    SearchOutcome,
    SearchResult,
)
from object_search.search import get_method
from object_search.search.proposals import ProposalBackend, propose

_EXPLORATION_VERSION = "1.0.0"
_EPS = 1e-9


class MarkerConditionedConfig(BaseModel):
    """Config for the marker-conditioned exploration -- every weight is Field-described so it
    drives the UI form directly.

    The four ``w_*`` weights and the size prior are the real design surface (they are *not* buried
    magic constants): ``proximity`` prefers proposals near the marker's reference point,
    ``direction`` prefers proposals along its pointing vector, ``objectness`` trusts the proposal
    stage, and ``size`` prefers proposals near the expected object size.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    marker_method: str = Field(
        default="sparse-geo", description="Milestone 1 method used to find marker instances."
    )
    marker_config: Mapping[str, object] = Field(
        default_factory=dict,
        description="Config for the marker-finding method; validated against its schema.",
    )
    proposal: FastSAMConfig = Field(
        default_factory=FastSAMConfig, description="Class-agnostic proposal-stage config."
    )
    w_distance: float = Field(
        default=1.0, ge=0.0, description="Weight on proximity to the reference point."
    )
    w_direction: float = Field(
        default=0.5, ge=0.0, description="Weight on alignment with the marker's pointing vector."
    )
    w_objectness: float = Field(
        default=0.5, ge=0.0, description="Weight on the proposal's own objectness."
    )
    w_size: float = Field(
        default=0.3, ge=0.0, description="Weight on how well the proposal matches the size prior."
    )
    size_prior_frac: float = Field(
        default=2.0,
        gt=0.0,
        description="Expected object size as a multiple of the marker's box diagonal.",
    )
    max_markers: int = Field(
        default=20, ge=1, description="Cap on how many marker instances to resolve."
    )


def _score_proposal(
    proposal: Proposal,
    reference: Point,
    direction: tuple[float, float] | None,
    length_scale: float,
    expected_diag: float,
    config: MarkerConditionedConfig,
) -> float:
    """The documented weighted sum: proximity + direction + objectness + size prior.

    The direction term is dropped (contributes zero) when ``direction`` is ``None`` -- a symmetric
    marker scores on distance, objectness and size only, never on a guessed direction.
    """
    cx, cy = proposal.box.cx, proposal.box.cy
    vx, vy = cx - reference.x, cy - reference.y
    dist = math.hypot(vx, vy)

    # Proximity: decays with distance from the reference point over the marker's length scale.
    proximity = math.exp(-dist / (length_scale + _EPS))

    # Direction alignment: cosine between the pointing vector and reference -> proposal centre.
    alignment = 0.0
    if direction is not None and dist > _EPS:
        cos = (direction[0] * vx + direction[1] * vy) / dist
        alignment = max(0.0, cos)

    # Size prior: 1.0 when the proposal diagonal matches the expected object size, decaying either
    # side (symmetric in log-size via the min-of-ratio-and-inverse form).
    prop_diag = math.hypot(proposal.box.w, proposal.box.h)
    ratio = prop_diag / (expected_diag + _EPS)
    size_fit = min(ratio, 1.0 / ratio) if ratio > 0.0 else 0.0

    return (
        config.w_distance * proximity
        + config.w_direction * alignment
        + config.w_objectness * proposal.objectness
        + config.w_size * size_fit
    )


def _empty_result(note: str, metrics: dict[str, float]) -> SearchResult:
    """An honest EMPTY outcome carrying a diagnostic note -- never a silent empty (METHOD-04c)."""
    return SearchResult(
        method="marker-conditioned",
        method_version=_EXPLORATION_VERSION,
        outcome=SearchOutcome.EMPTY,
        matches=(),
        latency=LatencyBreakdown(preprocess_ms=0.0, inference_ms=0.0, postprocess_ms=0.0),
        threshold_applied=None,
        diagnostics=Diagnostics(notes=(note,), metrics=metrics),
    )


def run(
    image: npt.NDArray[np.uint8],
    exemplar: ExemplarBox,
    config: BaseModel,
    *,
    backend: ProposalBackend | None = None,
) -> SearchResult:
    """Resolve every marker to its best-pointed-at object proposal.

    ``backend`` injects a proposal backend for tests (a stub, or a counting stub that asserts
    ``propose`` is called exactly once); production leaves it ``None`` to build the default
    FastSAM backend.
    """
    if not isinstance(config, MarkerConditionedConfig):
        raise TypeError(
            f"marker-conditioned requires a MarkerConditionedConfig, got {type(config).__name__}"
        )

    # 1. Find markers: reuse a Milestone 1 method wholesale; the exemplar box is the marker crop.
    marker_spec = get_method(config.marker_method)
    marker_cfg = marker_spec.config_model.model_validate(config.marker_config)
    t0 = perf_counter()
    marker_result = marker_spec.fn(image, exemplar, marker_cfg)
    find_ms = (perf_counter() - t0) * 1000.0
    markers = marker_result.matches[: config.max_markers]
    if not markers:
        return _empty_result(
            f"marker method {config.marker_method!r} found no markers "
            f"(outcome={marker_result.outcome.value})",
            {"n_markers": 0.0, "n_proposals": 0.0},
        )

    # 2. Per marker: reference point + orientation. The exemplar's own tip (estimated once by PCA)
    #    resolves the 180-degree flip on the transform path.
    height, width = image.shape[:2]
    ex_box = exemplar.box
    exemplar_geom = estimate_geometry(image[ex_box.y : ex_box.y2, ex_box.x : ex_box.x2], ex_box)
    exemplar_tip = exemplar_geom.reference_point

    references: list[Point] = []
    directions: list[tuple[float, float] | None] = []
    length_scales: list[float] = []
    expected_diags: list[float] = []
    for match in markers:
        box = match.box
        crop = image[box.y : box.y2, box.x : box.x2]
        geom = estimate_geometry(crop, box, transform=match.transform, exemplar_tip=exemplar_tip)
        references.append(geom.reference_point)
        directions.append(geom.direction)
        marker_diag = math.hypot(box.w, box.h)
        length_scales.append(marker_diag * config.size_prior_frac)
        expected_diags.append(marker_diag * config.size_prior_frac)

    # 3. Propose objects ONCE for the whole image -- shared across all markers (never per marker).
    t1 = perf_counter()
    proposals = propose(image, config.proposal, backend=backend)
    propose_ms = (perf_counter() - t1) * 1000.0
    if not proposals:
        return _empty_result(
            f"{len(markers)} marker(s) found but the proposal stage returned nothing",
            {"n_markers": float(len(markers)), "n_proposals": 0.0},
        )

    # 4. Per marker, score every proposal and keep the single best (one best proposal per marker).
    #    Ties break on (-score, box.y, box.x), so the pick is deterministic.
    t2 = perf_counter()
    matches: list[Match] = []
    for i in range(len(markers)):

        def _key(proposal: Proposal, i: int = i) -> tuple[float, int, int]:
            score = _score_proposal(
                proposal,
                references[i],
                directions[i],
                length_scales[i],
                expected_diags[i],
                config,
            )
            return (-score, proposal.box.y, proposal.box.x)

        best = min(proposals, key=_key)  # proposals is non-empty (checked above)
        matches.append(Match(box=best.box.clipped_to(width, height), score=-_key(best)[0]))
    score_ms = (perf_counter() - t2) * 1000.0

    # 5. Diagnostics: the full proposal set for the overlay, plus per-marker reference points and
    #    directions so the UI can draw the pointing arrows. Latency attributes the three stages.
    diagnostics = Diagnostics(
        notes=(f"resolved {len(matches)} marker(s) against {len(proposals)} proposal(s)",),
        metrics={"n_markers": float(len(markers)), "n_proposals": float(len(proposals))},
        proposals=tuple(p.box for p in proposals),
        markers=tuple(m.box for m in markers),
        marker_reference_points=tuple(references),
        marker_directions=tuple(directions),
    )
    return SearchResult(
        method="marker-conditioned",
        method_version=_EXPLORATION_VERSION,
        outcome=SearchOutcome.OK,
        matches=tuple(matches),
        latency=LatencyBreakdown(
            preprocess_ms=find_ms, inference_ms=propose_ms, postprocess_ms=score_ms
        ),
        threshold_applied=None,
        diagnostics=diagnostics,
    )


register_exploration(
    name="marker-conditioned",
    description="Find every marker instance and the best object region it points at (Milestone 2).",
    version=_EXPLORATION_VERSION,
    config_model=MarkerConditionedConfig,
)(run)
