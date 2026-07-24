"""Tests for the frozen schema set.

The tests here are not "does Pydantic work" tests. Each one pins a decision that the rest
of the system relies on and that a well-meaning edit could quietly undo:

* boxes are half-open, so ``w == x2 - x`` with no ``+1``;
* every contract is frozen and rejects unknown fields;
* an inconsistent :class:`SearchResult` cannot be constructed at all;
* ``Rating`` count fields stay ``None`` when unassessed (EVAL-17).
"""

import pytest
from pydantic import ValidationError

from object_search.schemas import (
    BBox,
    Candidate,
    Correspondence,
    Diagnostics,
    ExemplarBox,
    FPSource,
    HeatmapPayload,
    HoughPeak,
    LatencyBreakdown,
    Match,
    MatchVerdict,
    MethodError,
    Point,
    Provenance,
    Rating,
    RatingCompleteness,
    RunRecord,
    SearchOutcome,
    SearchResult,
    SliceMetadata,
)

ZERO_LATENCY = LatencyBreakdown(preprocess_ms=0.0, inference_ms=0.0, postprocess_ms=0.0)
ONE_MATCH = (Match(box=BBox(x=0, y=0, w=4, h=4), score=0.9),)


def _result(**overrides: object) -> SearchResult:
    """Build a valid SearchResult, then let each test break exactly one thing."""
    kwargs: dict[str, object] = {
        "method": "ncc",
        "method_version": "1.0.0",
        "outcome": SearchOutcome.OK,
        "matches": ONE_MATCH,
        "latency": ZERO_LATENCY,
        "threshold_applied": 0.7,
    }
    kwargs.update(overrides)
    return SearchResult(**kwargs)  # type: ignore[arg-type]


# --------------------------------------------------------------------------------- BBox


def test_bbox_derived_values_follow_the_exclusive_x2_convention():
    box = BBox(x=10, y=20, w=30, h=40)
    assert box.x2 == 40  # x + w, exclusive
    assert box.y2 == 60
    assert box.w == box.x2 - box.x  # the convention, restated as an assertion
    assert box.h == box.y2 - box.y
    assert box.area == 1200  # no "+1" terms
    assert box.xyxy == (10, 20, 40, 60)
    assert box.cx == 25.0
    assert box.cy == 40.0


def test_bbox_slices_a_numpy_array_with_no_off_by_one():
    """The whole point of the half-open convention: the box IS the slice."""
    np = pytest.importorskip("numpy")
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    box = BBox(x=10, y=20, w=30, h=40)
    crop = image[box.y : box.y2, box.x : box.x2]
    assert crop.shape == (box.h, box.w, 3)


def test_bbox_is_frozen():
    box = BBox(x=0, y=0, w=1, h=1)
    with pytest.raises(ValidationError):
        box.x = 5  # type: ignore[misc]


def test_bbox_forbids_unknown_fields():
    with pytest.raises(ValidationError):
        BBox(x=0, y=0, w=1, h=1, score=0.9)  # type: ignore[call-arg]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"x": -1, "y": 0, "w": 1, "h": 1},
        {"x": 0, "y": -1, "w": 1, "h": 1},
        {"x": 0, "y": 0, "w": 0, "h": 1},
        {"x": 0, "y": 0, "w": 1, "h": 0},
    ],
)
def test_bbox_rejects_out_of_range_dimensions(kwargs):
    with pytest.raises(ValidationError):
        BBox(**kwargs)


def test_bbox_iou_identical_is_one_and_disjoint_is_zero():
    a = BBox(x=0, y=0, w=10, h=10)
    assert a.iou(a) == pytest.approx(1.0)
    far = BBox(x=100, y=100, w=10, h=10)
    assert a.iou(far) == 0.0


def test_bbox_iou_touching_boxes_do_not_overlap():
    """Half-open means ``a.x2 == b.x`` is adjacency, not overlap."""
    a = BBox(x=0, y=0, w=10, h=10)
    b = BBox(x=10, y=0, w=10, h=10)
    assert a.iou(b) == 0.0


def test_bbox_iou_half_overlap():
    a = BBox(x=0, y=0, w=10, h=10)
    b = BBox(x=5, y=0, w=10, h=10)
    # intersection 5x10 = 50, union 100 + 100 - 50 = 150
    assert a.iou(b) == pytest.approx(50 / 150)


def test_bbox_clipped_to_trims_and_preserves_the_convention():
    box = BBox(x=90, y=90, w=50, h=50)
    clipped = box.clipped_to(100, 120)
    assert clipped.xyxy == (90, 90, 100, 120)
    assert clipped.w == 10
    assert clipped.h == 30


def test_bbox_clipped_to_is_a_noop_when_already_inside():
    box = BBox(x=1, y=2, w=3, h=4)
    assert box.clipped_to(100, 100) == box


def test_bbox_clipped_to_raises_when_fully_outside():
    box = BBox(x=200, y=200, w=10, h=10)
    with pytest.raises(ValueError, match="does not intersect"):
        box.clipped_to(100, 100)


# -------------------------------------------------------------------------- ExemplarBox


def test_exemplar_box_wraps_a_bbox_and_defaults_to_no_label():
    exemplar = ExemplarBox(box=BBox(x=1, y=1, w=4, h=4))
    assert exemplar.label is None
    assert exemplar.box.area == 16


def test_exemplar_box_is_frozen():
    exemplar = ExemplarBox(box=BBox(x=1, y=1, w=4, h=4), label="player")
    with pytest.raises(ValidationError):
        exemplar.label = "ball"  # type: ignore[misc]


# --------------------------------------------------------------------------------- Point


def test_point_is_subpixel_and_frozen():
    point = Point(x=1.5, y=2.25)
    assert point.x == 1.5
    with pytest.raises(ValidationError):
        point.x = 0.0  # type: ignore[misc]


# --------------------------------------------------------------------------------- Match


def test_match_defaults_are_not_an_exemplar_and_carry_no_transform():
    match = Match(box=BBox(x=0, y=0, w=4, h=4), score=0.5)
    assert match.is_exemplar is False
    assert match.transform is None


def test_match_accepts_a_flattened_2x3_affine():
    match = Match(
        box=BBox(x=0, y=0, w=4, h=4),
        score=0.5,
        transform=(1.0, 0.0, 3.0, 0.0, 1.0, 4.0),
    )
    assert match.transform is not None
    assert len(match.transform) == 6


@pytest.mark.parametrize("bad", [(1.0, 0.0, 3.0), (1.0,) * 9])
def test_match_rejects_a_transform_that_is_not_six_floats(bad):
    with pytest.raises(ValidationError, match="flattened 2x3 affine"):
        Match(box=BBox(x=0, y=0, w=4, h=4), score=0.5, transform=bad)


def test_candidate_cannot_masquerade_as_a_match():
    """A Candidate has no is_exemplar and no transform -- EVAL-08 keeps the types apart."""
    candidate = Candidate(box=BBox(x=0, y=0, w=4, h=4), score=0.1)
    assert not hasattr(candidate, "is_exemplar")
    with pytest.raises(ValidationError):
        Candidate(box=BBox(x=0, y=0, w=4, h=4), score=0.1, is_exemplar=True)  # type: ignore[call-arg]


# ---------------------------------------------------------------------- LatencyBreakdown


def test_total_ms_is_a_property_not_a_field():
    latency = LatencyBreakdown(preprocess_ms=1.0, inference_ms=2.5, postprocess_ms=0.5)
    assert latency.total_ms == pytest.approx(4.0)
    # Not a field: it is absent from the schema and cannot be set independently, so it can
    # never disagree with the three parts (EVAL-11).
    assert "total_ms" not in LatencyBreakdown.model_fields
    assert "total_ms" not in latency.model_dump()
    with pytest.raises(ValidationError):
        LatencyBreakdown(  # type: ignore[call-arg]
            preprocess_ms=1.0, inference_ms=2.5, postprocess_ms=0.5, total_ms=99.0
        )


def test_latency_rejects_negative_stages():
    with pytest.raises(ValidationError):
        LatencyBreakdown(preprocess_ms=-1.0, inference_ms=0.0, postprocess_ms=0.0)


# --------------------------------------------------------------------------- Diagnostics


def test_diagnostics_defaults_to_everything_absent():
    diagnostics = Diagnostics()
    assert diagnostics.notes == ()
    assert dict(diagnostics.metrics) == {}
    assert diagnostics.similarity_heatmap is None
    assert diagnostics.keypoints is None
    assert diagnostics.correspondences is None
    assert diagnostics.hough_peaks is None
    assert diagnostics.proposals is None


def test_diagnostics_default_metrics_are_not_shared_between_instances():
    first = Diagnostics(metrics={"n_keypoints": 12.0})
    second = Diagnostics()
    assert dict(second.metrics) == {}
    assert dict(first.metrics) == {"n_keypoints": 12.0}


def test_diagnostics_carries_every_named_payload_the_ui_can_render():
    diagnostics = Diagnostics(
        notes=("low texture: crop std 0.4 < 2.0",),
        metrics={"n_keypoints": 3.0},
        similarity_heatmap=HeatmapPayload(
            png_b64="aGVsbG8=", width=8, height=4, vmin=-0.1, vmax=0.93
        ),
        keypoints=(Point(x=1.0, y=2.0),),
        correspondences=(
            Correspondence(src=Point(x=1.0, y=2.0), dst=Point(x=9.0, y=8.0), distance=0.2, rank=0),
        ),
        hough_peaks=(HoughPeak(dx=1.0, dy=2.0, log_scale=0.0, theta_deg=0.0, votes=4.5),),
        proposals=(BBox(x=0, y=0, w=2, h=2),),
    )
    assert diagnostics.notes[0].startswith("low texture")
    assert diagnostics.hough_peaks is not None
    assert diagnostics.hough_peaks[0].n_inliers is None  # unverified, not rejected


def test_heatmap_payload_requires_real_content_and_a_value_range():
    with pytest.raises(ValidationError):
        HeatmapPayload(png_b64="", width=8, height=4, vmin=0.0, vmax=1.0)
    with pytest.raises(ValidationError):
        HeatmapPayload(png_b64="aGVsbG8=", width=0, height=4, vmin=0.0, vmax=1.0)


# -------------------------------------------------------------------------- SearchResult


def test_search_outcome_has_exactly_ok_empty_error():
    assert [member.value for member in SearchOutcome] == ["ok", "empty", "error"]
    assert SearchOutcome.OK == "ok"  # StrEnum: JSON-serialises as the plain string


def test_valid_ok_result_constructs_and_defaults_are_empty():
    result = _result()
    assert result.outcome is SearchOutcome.OK
    assert result.candidates == ()
    assert result.diagnostics.notes == ()
    assert result.error is None


def test_error_outcome_without_an_error_payload_is_rejected():
    with pytest.raises(ValidationError, match="outcome/error disagree"):
        _result(outcome=SearchOutcome.ERROR, matches=(), error=None)


def test_error_payload_without_the_error_outcome_is_rejected():
    with pytest.raises(ValidationError, match="outcome/error disagree"):
        _result(error=MethodError(kind="boom", message="exploded"))


def test_error_result_constructs_when_both_agree():
    result = _result(
        outcome=SearchOutcome.ERROR,
        matches=(),
        error=MethodError(kind="exemplar_out_of_bounds", message="box outside image"),
    )
    assert result.error is not None
    assert result.error.kind == "exemplar_out_of_bounds"


def test_empty_outcome_carrying_matches_is_rejected():
    with pytest.raises(ValidationError, match="cannot carry matches"):
        _result(outcome=SearchOutcome.EMPTY, matches=ONE_MATCH)


def test_empty_outcome_with_no_matches_and_a_note_constructs():
    result = _result(
        outcome=SearchOutcome.EMPTY,
        matches=(),
        diagnostics=Diagnostics(notes=("crop is textureless; abstaining",)),
    )
    assert result.matches == ()
    assert result.diagnostics.notes  # METHOD-04c: never silently empty


def test_ok_outcome_with_zero_matches_is_rejected_as_ambiguous():
    with pytest.raises(ValidationError, match="ambiguous"):
        _result(matches=())


def test_search_result_is_frozen():
    result = _result()
    with pytest.raises(ValidationError):
        result.method = "other"  # type: ignore[misc]


# -------------------------------------------------------------------------- SliceMetadata


def test_slice_metadata_is_all_none_by_default_because_unknown_is_not_zero():
    meta = SliceMetadata()
    assert meta.model_dump() == dict.fromkeys(SliceMetadata.model_fields, None)


def test_slice_metadata_is_exact_for_a_synthetic_image():
    meta = SliceMetadata(
        true_instance_count=12,
        instance_scale_min=1.0,
        instance_scale_max=1.0,
        rotation_min_deg=0.0,
        rotation_max_deg=0.0,
        clutter_level=0.0,
    )
    assert meta.true_instance_count == 12
    assert meta.exemplar_keypoint_count is None  # not known until a method runs


# ----------------------------------------------------------------------------- Provenance


def test_provenance_capture_fills_code_config_and_environment():
    record = Provenance.capture(
        method_version="1.0.0", config_hash="deadbeef", model_hashes={"dinov2-small": "abc"}
    )
    assert record.method_version == "1.0.0"
    assert record.config_hash == "deadbeef"
    assert dict(record.model_hashes) == {"dinov2-small": "abc"}
    # The environment half -- the fields a git SHA does not capture, and without which
    # ratings from before and after a dependency bump would be silently pooled.
    assert record.git_sha  # a real sha here, or "unknown" outside a checkout
    assert record.cv2_version.startswith("4.")
    assert record.onnxruntime_version
    assert record.numpy_version.startswith("2.")
    assert "ExecutionProvider" in record.ort_providers
    assert len(record.pixi_lock_sha256) == 64  # sha256 hex of the committed lockfile
    assert record.created_at.tzinfo is not None  # tz-aware, so it sorts correctly


def test_provenance_environment_fields_cannot_be_forgotten():
    """They are required, so a hand-built Provenance that omits them fails loudly."""
    with pytest.raises(ValidationError):
        Provenance(git_sha="abc", method_version="1.0.0", config_hash="d")  # type: ignore[call-arg]


def test_provenance_is_frozen():
    record = Provenance.capture(method_version="1.0.0", config_hash="d")
    with pytest.raises(ValidationError):
        record.git_sha = "tampered"  # type: ignore[misc]


# ------------------------------------------------------------------------------ RunRecord


def test_run_record_has_no_id_before_insert():
    record = RunRecord(
        image_id="synthetic/lattice-plain.png",
        exemplar=ExemplarBox(box=BBox(x=1, y=1, w=4, h=4)),
        method="ncc",
        config_json='{"threshold":0.7}',
        config_hash="cafe",
        result=_result(),
        provenance=Provenance.capture(method_version="1.0.0", config_hash="cafe"),
    )
    assert record.id is None
    assert record.slice_metadata.true_instance_count is None


# --------------------------------------------------------------------------------- Rating
# EVAL-17 lives or dies here.


def test_rating_counts_are_none_when_unassessed():
    """The single most important assertion in the schema suite (EVAL-17).

    If this ever fails because someone added a numeric default, every unreviewed run in the
    store starts claiming perfect precision and recall.
    """
    rating = Rating(run_id=1, thumbs_up=True)
    assert rating.wrong_count is None
    assert rating.missed_count is None
    assert rating.false_positives is None
    assert rating.fp_source is None
    assert rating.completeness is RatingCompleteness.NONE


def test_rating_counts_survive_a_json_round_trip_as_null_not_zero():
    """A serialisation layer is a classic place for a 0 to appear from nowhere."""
    rating = Rating(run_id=1, thumbs_up=False)
    payload = rating.model_dump(mode="json")
    assert payload["wrong_count"] is None
    assert payload["missed_count"] is None
    assert Rating.model_validate_json(rating.model_dump_json()).wrong_count is None


def test_rating_schema_declares_no_numeric_default_for_either_count():
    """Guards the JSON Schema too -- the UI form is generated from it (EVAL-17)."""
    schema = Rating.model_json_schema()
    for field in ("wrong_count", "missed_count"):
        assert schema["properties"][field]["default"] is None
        assert field not in schema.get("required", [])


def test_rating_zero_is_a_real_assessment_distinct_from_none():
    assessed = Rating(run_id=1, thumbs_up=True, wrong_count=0, missed_count=0)
    assert assessed.false_positives == 0
    assert assessed.fp_source is FPSource.COUNT
    assert assessed.completeness is RatingCompleteness.COMPLETE


def test_rating_rejects_negative_counts():
    with pytest.raises(ValidationError):
        Rating(run_id=1, thumbs_up=True, wrong_count=-1)
    with pytest.raises(ValidationError):
        Rating(run_id=1, thumbs_up=True, missed_count=-1)


@pytest.mark.parametrize(
    ("wrong", "missed", "expected"),
    [
        (None, None, RatingCompleteness.NONE),
        (2, None, RatingCompleteness.PRECISION_ONLY),
        (None, 3, RatingCompleteness.RECALL_ONLY),
        (2, 3, RatingCompleteness.COMPLETE),
    ],
)
def test_rating_completeness_tiers(wrong, missed, expected):
    rating = Rating(run_id=1, thumbs_up=True, wrong_count=wrong, missed_count=missed)
    assert rating.completeness is expected


def test_unconfirmed_per_match_verdicts_are_ignored_entirely():
    """UI-08: an untouched checkbox grid is not a judgement.

    "Looked at every box and approved them" must stay distinguishable from "never opened
    the panel", so unconfirmed verdicts contribute nothing at all.
    """
    rating = Rating(
        run_id=1,
        thumbs_up=True,
        per_match_verdicts=(
            MatchVerdict(match_index=0, correct=False),
            MatchVerdict(match_index=1, correct=True),
        ),
        verdicts_confirmed=False,
    )
    assert rating.fp_source is None
    assert rating.false_positives is None
    assert rating.completeness is RatingCompleteness.NONE
    assert rating.has_fp_discrepancy is False


def test_confirmed_per_match_verdicts_provide_false_positives():
    rating = Rating(
        run_id=1,
        thumbs_up=True,
        per_match_verdicts=(
            MatchVerdict(match_index=0, correct=False),
            MatchVerdict(match_index=1, correct=True),
            MatchVerdict(match_index=2, correct=False),
        ),
        verdicts_confirmed=True,
    )
    assert rating.fp_source is FPSource.PER_MATCH
    assert rating.false_positives == 2
    assert rating.completeness is RatingCompleteness.PRECISION_ONLY


def test_per_match_verdicts_win_over_a_conflicting_wrong_count_and_the_clash_is_flagged():
    """EVAL-18: per-match wins, and the disagreement is surfaced rather than reconciled."""
    rating = Rating(
        run_id=1,
        thumbs_up=True,
        wrong_count=1,
        per_match_verdicts=(
            MatchVerdict(match_index=0, correct=False),
            MatchVerdict(match_index=1, correct=False),
        ),
        verdicts_confirmed=True,
    )
    assert rating.fp_source is FPSource.PER_MATCH
    assert rating.false_positives == 2  # not the typed 1
    assert rating.has_fp_discrepancy is True


def test_agreeing_per_match_and_count_are_not_a_discrepancy():
    rating = Rating(
        run_id=1,
        thumbs_up=True,
        wrong_count=1,
        per_match_verdicts=(
            MatchVerdict(match_index=0, correct=False),
            MatchVerdict(match_index=1, correct=True),
        ),
        verdicts_confirmed=True,
    )
    assert rating.has_fp_discrepancy is False


def test_discrepancy_needs_both_sides_present():
    only_verdicts = Rating(
        run_id=1,
        thumbs_up=True,
        per_match_verdicts=(MatchVerdict(match_index=0, correct=False),),
        verdicts_confirmed=True,
    )
    assert only_verdicts.has_fp_discrepancy is False


def test_rating_is_frozen_and_forbids_extra_fields():
    rating = Rating(run_id=1, thumbs_up=True)
    with pytest.raises(ValidationError):
        rating.wrong_count = 0  # type: ignore[misc]
    with pytest.raises(ValidationError):
        Rating(run_id=1, thumbs_up=True, precision=1.0)  # type: ignore[call-arg]


def test_unratable_is_the_explicit_skip_path():
    rating = Rating(run_id=1, thumbs_up=False, unratable=True)
    assert rating.unratable is True
    assert rating.completeness is RatingCompleteness.NONE


# ------------------------------------------------------- Rating.validate_against_retrieved


def test_validate_against_retrieved_accepts_a_consistent_rating():
    rating = Rating(
        run_id=1,
        thumbs_up=True,
        wrong_count=2,
        per_match_verdicts=(
            MatchVerdict(match_index=0, correct=False),
            MatchVerdict(match_index=1, correct=False),
        ),
        verdicts_confirmed=True,
    )
    assert rating.validate_against_retrieved(3) == (True, None)


def test_validate_against_retrieved_rejects_wrong_count_above_the_returned_count():
    rating = Rating(run_id=1, thumbs_up=True, wrong_count=5)
    ok, reason = rating.validate_against_retrieved(3)
    assert ok is False
    assert reason is not None
    assert "exceeds" in reason


def test_validate_against_retrieved_rejects_too_many_verdicts():
    rating = Rating(
        run_id=1,
        thumbs_up=True,
        per_match_verdicts=tuple(MatchVerdict(match_index=i, correct=True) for i in range(4)),
    )
    ok, reason = rating.validate_against_retrieved(2)
    assert ok is False
    assert reason is not None
    assert "per-match verdicts" in reason


def test_validate_against_retrieved_rejects_an_out_of_range_index():
    rating = Rating(
        run_id=1, thumbs_up=True, per_match_verdicts=(MatchVerdict(match_index=7, correct=True),)
    )
    ok, reason = rating.validate_against_retrieved(3)
    assert ok is False
    assert reason is not None
    assert "outside" in reason


def test_validate_against_retrieved_rejects_duplicate_indices():
    rating = Rating(
        run_id=1,
        thumbs_up=True,
        per_match_verdicts=(
            MatchVerdict(match_index=0, correct=True),
            MatchVerdict(match_index=0, correct=False),
        ),
    )
    ok, reason = rating.validate_against_retrieved(2)
    assert ok is False
    assert reason == "duplicate per-match verdict indices"


def test_validate_against_retrieved_is_fine_when_nothing_was_assessed():
    assert Rating(run_id=1, thumbs_up=True).validate_against_retrieved(0) == (True, None)
