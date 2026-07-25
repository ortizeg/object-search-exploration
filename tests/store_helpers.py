"""Builders for store round-trip tests.

Records are constructed with fixed, explicit values (a pinned ``created_at`` and a
hand-built :class:`Provenance`) rather than via ``Provenance.capture`` so equality
assertions after a database round trip are deterministic and do not depend on the
environment the test happens to run in.
"""

from __future__ import annotations

from datetime import UTC, datetime

from object_search.schemas.geometry import BBox, ExemplarBox
from object_search.schemas.records import Provenance, RunRecord, SliceMetadata
from object_search.schemas.search import (
    Candidate,
    Diagnostics,
    LatencyBreakdown,
    Match,
    MethodError,
    SearchOutcome,
    SearchResult,
)

FIXED_TIME = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)


def make_provenance(*, config_hash: str = "cafe", method_version: str = "1.0.0") -> Provenance:
    """A fully specified provenance record with a pinned timestamp."""
    return Provenance(
        git_sha="0123456789abcdef",
        method_version=method_version,
        config_hash=config_hash,
        model_hashes={"backbone": "deadbeef"},
        python_version="3.12.0",
        numpy_version="2.1.0",
        cv2_version="4.10.0",
        onnxruntime_version="1.23.2",
        ort_providers="CPUExecutionProvider",
        pixi_lock_sha256="feedface",
        created_at=FIXED_TIME,
    )


def make_search_result(
    *,
    method: str = "ncc",
    outcome: SearchOutcome = SearchOutcome.OK,
    n_matches: int = 3,
    n_candidates: int = 0,
    error: MethodError | None = None,
    diagnostics: Diagnostics | None = None,
    threshold_applied: float | None = 0.7,
) -> SearchResult:
    """A search result with ``n_matches`` matches and ``n_candidates`` candidates."""
    matches = tuple(
        Match(
            box=BBox(x=10 + i, y=20 + i, w=8, h=8),
            score=0.9 - 0.01 * i,
            is_exemplar=(i == 0),
            transform=(1.0, 0.0, float(i), 0.0, 1.0, 0.0) if i == 1 else None,
        )
        for i in range(n_matches if outcome is SearchOutcome.OK else 0)
    )
    candidates = tuple(
        Candidate(box=BBox(x=100 + i, y=100 + i, w=5, h=5), score=0.5 - 0.001 * i)
        for i in range(n_candidates)
    )
    return SearchResult(
        method=method,
        method_version="1.0.0",
        outcome=outcome,
        matches=matches,
        latency=LatencyBreakdown(preprocess_ms=1.5, inference_ms=12.0, postprocess_ms=0.5),
        threshold_applied=threshold_applied,
        candidates=candidates,
        diagnostics=diagnostics if diagnostics is not None else Diagnostics(),
        error=error,
    )


def make_run_record(
    *,
    method: str = "ncc",
    image_id: str = "synthetic/lattice-plain.png",
    outcome: SearchOutcome = SearchOutcome.OK,
    n_matches: int = 3,
    n_candidates: int = 0,
    error: MethodError | None = None,
    slice_metadata: SliceMetadata | None = None,
    diagnostics: Diagnostics | None = None,
    threshold_applied: float | None = 0.7,
) -> RunRecord:
    """A run record ready for :func:`object_search.store.runs.insert_run`."""
    result = make_search_result(
        method=method,
        outcome=outcome,
        n_matches=n_matches,
        n_candidates=n_candidates,
        error=error,
        diagnostics=diagnostics,
        threshold_applied=threshold_applied,
    )
    return RunRecord(
        image_id=image_id,
        exemplar=ExemplarBox(box=BBox(x=1, y=1, w=4, h=4), label="widget"),
        method=method,
        config_json='{"threshold":0.7}',
        config_hash="cafe",
        result=result,
        slice_metadata=(slice_metadata if slice_metadata is not None else SliceMetadata()),
        provenance=make_provenance(),
    )
