"""The contract every search method returns, and the diagnostics payload the UI renders.

Three decisions in this module are load-bearing and are enforced by validators rather than
documented and hoped for:

1. **A match is not a candidate.** :class:`Match` is something the method is willing to
   claim; :class:`Candidate` is a sub-threshold observation kept only so an offline
   threshold sweep can rebuild a full precision/recall curve (EVAL-08). A candidate must
   never be renderable as a match, which is why it is a separate type with no ``transform``
   and no ``is_exemplar``.

2. **"Found nothing" and "blew up" are different outcomes** (:class:`SearchOutcome`,
   EVAL-12). Neither is a zero-precision run. Collapsing them makes a crashing method look
   merely unlucky and an honest abstention look like a failure.

3. **Dense arrays are transported as PNG, not as JSON numbers** (:class:`HeatmapPayload`).
   A 1920x1080 float32 similarity map is ~8 MB raw and ~41 MB as a JSON array of floats;
   the same map quantised to uint8 and PNG-encoded is ~2 MB. The UI draws it to a canvas as
   an image anyway, so the array form buys nothing and costs the response.

No field in this module is an ``Any``-valued mapping. ``Any`` would defeat mypy strict, and
the *named* diagnostic fields are what let the UI render a diagnostic from a method it has
never seen -- a free-form blob would push that knowledge into the front end.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from object_search.schemas.geometry import BBox, Point

# ---------------------------------------------------------------------------- results


class Match(BaseModel):
    """One instance the method claims to have found.

    Attributes:
        box: Where it is, in scene pixels.
        score: The method's own confidence. Comparable *within* one method and one config;
            never assume cross-method comparability.
        is_exemplar: True when this match is the exemplar's own region. The self-match is a
            genuine instance and is labelled rather than dropped or double-counted
            (METHOD-04c): dropping it understates recall, counting it silently as a
            discovery overstates the method.
        transform: Optional flattened 2x3 affine (6 floats, row-major) mapping the exemplar
            crop onto this instance. Method 2 fills it from its per-peak RANSAC estimate;
            appearance-only methods leave it ``None``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    box: BBox
    score: float
    is_exemplar: bool = False
    transform: tuple[float, ...] | None = None

    @field_validator("transform")
    @classmethod
    def _transform_is_a_flattened_2x3(
        cls, value: tuple[float, ...] | None
    ) -> tuple[float, ...] | None:
        if value is not None and len(value) != 6:
            raise ValueError(
                f"transform must be a flattened 2x3 affine (6 floats), got {len(value)}"
            )
        return value


class Candidate(BaseModel):
    """A sub-threshold observation, kept for offline threshold sweeps (EVAL-08).

    Deliberately *not* a :class:`Match`. Persisting the top-N candidates with their raw
    scores alongside the threshold that was applied is what makes a full PR curve
    recoverable after the fact, without re-running the method. Giving candidates the same
    type as matches would make it one careless ``+`` away from rendering rejected
    observations as claims.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    box: BBox
    score: float


class LatencyBreakdown(BaseModel):
    """Per-stage wall-clock timing in milliseconds (EVAL-11).

    A single total is explicitly insufficient: "slow" means something different when it is
    the ONNX forward pass than when it is a Python post-processing loop, and only the
    breakdown tells a practitioner which knob to reach for.

    ``total_ms`` is a **property**, not a stored field, so it can never disagree with its
    parts.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    preprocess_ms: float = Field(ge=0.0)
    inference_ms: float = Field(ge=0.0)
    postprocess_ms: float = Field(ge=0.0)

    @property
    def total_ms(self) -> float:
        """Sum of the three stages. Derived, never stored."""
        return self.preprocess_ms + self.inference_ms + self.postprocess_ms


# ------------------------------------------------------------------------ diagnostics


class HeatmapPayload(BaseModel):
    """A dense 2-D map transported as a base64 PNG plus its real value range.

    ``vmin``/``vmax`` are the *pre-quantisation* extremes of the underlying float map, so
    the UI can label the colour scale with real numbers instead of "0..255". Without them a
    similarity map is unreadable: every map would look like it spans the full range.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    png_b64: str = Field(min_length=1)
    width: int = Field(ge=1)
    height: int = Field(ge=1)
    vmin: float
    vmax: float


class Correspondence(BaseModel):
    """One exemplar-keypoint to scene-keypoint match (Method 2 diagnostics).

    Attributes:
        src: Keypoint in the exemplar crop, in **scene** pixel coordinates.
        dst: Matched keypoint in the scene.
        distance: Descriptor distance (lower is more similar).
        rank: Which nearest neighbour this was, ``0`` for the closest. Kept so the overlay
            can show whether a vote came from a confident first neighbour or from the tail
            -- the standard Lowe ratio test is disabled for this project, so the rank is
            the honest replacement for the discarded ratio signal.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    src: Point
    dst: Point
    distance: float
    rank: int = Field(ge=0)


class HoughPeak(BaseModel):
    """One peak in Method 2's 4-DoF pose vote space.

    Attributes:
        dx: Translation in x, scene pixels.
        dy: Translation in y, scene pixels.
        log_scale: Natural log of the scale factor, which is the space the votes are binned
            in (scale is multiplicative, so linear scale bins are the wrong shape).
        theta_deg: Rotation in degrees.
        votes: Accumulated vote **weight**, not a count -- soft binning spreads one
            correspondence across neighbouring bins with fractional weight.
        n_inliers: RANSAC inlier count for this peak once it has been verified, ``None``
            before verification. ``None`` and ``0`` differ: unverified versus rejected.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    dx: float
    dy: float
    log_scale: float
    theta_deg: float
    votes: float
    n_inliers: int | None = None


class Diagnostics(BaseModel):
    """Why a method returned what it returned -- the debug overlay's data source.

    Every field is optional, because each method produces a different subset. The fields
    are *named* rather than a free-form blob so the UI can render a diagnostic from a
    method it has never seen: a new method that fills ``similarity_heatmap`` gets the
    heatmap overlay for free.

    ``notes`` is what satisfies METHOD-04c's "never silently return an empty result": the
    low-texture and low-keypoint guards return ``outcome=EMPTY`` *with* a note saying why,
    so an abstention is legible instead of looking like a bug.

    ``Mapping`` and ``tuple`` are used instead of ``dict`` and ``list`` so that "frozen"
    means something at the type level rather than only at the top level.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    notes: tuple[str, ...] = ()
    metrics: Mapping[str, float] = Field(default_factory=dict)
    similarity_heatmap: HeatmapPayload | None = None
    keypoints: tuple[Point, ...] | None = None
    correspondences: tuple[Correspondence, ...] | None = None
    hough_peaks: tuple[HoughPeak, ...] | None = None
    proposals: tuple[BBox, ...] | None = None


# ---------------------------------------------------------------------------- outcome


class SearchOutcome(StrEnum):
    """Whether a search succeeded, honestly found nothing, or failed (EVAL-12).

    ``EMPTY`` and ``ERROR`` are deliberately distinct and **neither is a zero-precision
    run**. A method that abstains on a textureless crop is behaving correctly; a method that
    raised is not measurable at all. Pooling either into "0 correct out of 0 returned"
    invents a data point that does not exist.
    """

    OK = "ok"
    EMPTY = "empty"
    ERROR = "error"


class MethodError(BaseModel):
    """A structured failure, so the API can return a typed error (API-08).

    Attributes:
        kind: Short machine-readable tag, e.g. ``"exemplar_out_of_bounds"``. Grouping
            failures by kind is what turns "it crashed sometimes" into a fixable list.
        message: Human-readable detail for the UI.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: str = Field(min_length=1)
    message: str


class SearchResult(BaseModel):
    """Everything one method run produced. The return type of the ``SearchMethod`` protocol.

    The model validator makes an inconsistent result **unconstructible**, rather than
    catching it at review time or in the store:

    * ``outcome == ERROR`` if and only if ``error is not None`` -- no silent failures, and
      no error payload attached to a successful run;
    * ``outcome == EMPTY`` requires ``matches == ()`` -- "found nothing" cannot ship matches;
    * ``outcome == OK`` requires at least one match -- a zero-match success is exactly the
      ambiguity ``EMPTY`` exists to remove.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    method: str = Field(min_length=1)
    method_version: str = Field(min_length=1)
    outcome: SearchOutcome
    matches: tuple[Match, ...]
    latency: LatencyBreakdown
    threshold_applied: float | None
    candidates: tuple[Candidate, ...] = ()
    diagnostics: Diagnostics = Field(default_factory=Diagnostics)
    error: MethodError | None = None

    @model_validator(mode="after")
    def _outcome_agrees_with_payload(self) -> SearchResult:
        if (self.outcome is SearchOutcome.ERROR) != (self.error is not None):
            raise ValueError(
                f"outcome/error disagree: outcome={self.outcome.value!r} with "
                f"error={'set' if self.error is not None else 'None'}. "
                f"outcome must be 'error' if and only if an error payload is present."
            )
        if self.outcome is SearchOutcome.EMPTY and self.matches:
            raise ValueError(
                f"outcome='empty' cannot carry matches, got {len(self.matches)}. "
                f"Use outcome='ok' when there is at least one match."
            )
        if self.outcome is SearchOutcome.OK and not self.matches:
            raise ValueError(
                "outcome='ok' with zero matches is ambiguous; use outcome='empty' with a "
                "diagnostics note explaining why nothing was found (METHOD-04c)."
            )
        return self
