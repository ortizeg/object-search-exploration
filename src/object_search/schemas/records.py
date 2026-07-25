"""Persisted records: what was run, in what environment, and how well it did.

**This module is where the project's most-warned-about bug is prevented. Read
:class:`Rating` before editing anything here.**

Two rules govern the whole evaluation layer and both are easy to regress:

1. **Human count fields are nullable and stored empty.** ``null`` means "not assessed";
   ``0`` means "assessed, none". Defaulting either to ``0`` at *any* layer -- form, API,
   schema, database column -- makes every unreviewed run claim perfect precision and
   recall.
2. **Derived metrics are never stored.** Precision, recall, F1 and false-positive counts
   are properties here and views in the store, so ``NULL`` propagates instead of being
   quietly filled in.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from object_search import provenance as prov
from object_search.schemas.geometry import ExemplarBox
from object_search.schemas.search import SearchResult


def _utc_now() -> datetime:
    """Timezone-aware UTC timestamp. Naive datetimes sort wrong across a DST boundary."""
    return datetime.now(UTC)


class SliceMetadata(BaseModel):
    """What kind of image this run was on, for per-slice failure analysis (EVAL-10).

    Every field is nullable and **unknown must never read as zero**. These values are
    *exact* for synthetic images -- the generator knows the true instance count, the scale
    range and the rotation range it drew -- and best-effort or absent for photographs. A
    ``true_instance_count`` of ``None`` means "nobody has labelled this image"; a value of
    ``0`` would mean "this image genuinely contains no instances", which is a completely
    different claim and would make recall undefined rather than unknown.

    Attributes:
        true_instance_count: Ground-truth number of instances in the image.
        instance_scale_min: Smallest instance scale factor relative to the exemplar.
        instance_scale_max: Largest instance scale factor relative to the exemplar.
        rotation_min_deg: Smallest in-plane rotation across instances, degrees.
        rotation_max_deg: Largest in-plane rotation across instances, degrees.
        clutter_level: Background texture strength in ``[0, 1]`` for synthetic images.
        exemplar_keypoint_count: Keypoints detected on the exemplar crop. The variable that
            best predicts whether Method 2 could have worked at all (METHOD-04c).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    true_instance_count: int | None = None
    instance_scale_min: float | None = None
    instance_scale_max: float | None = None
    rotation_min_deg: float | None = None
    rotation_max_deg: float | None = None
    clutter_level: float | None = None
    exemplar_keypoint_count: int | None = None


class Provenance(BaseModel):
    """Everything needed to reproduce, or to refuse to compare, a run (EVAL-09).

    The environment fields are not padding. Measured: OpenCV 4.10.0 and 5.0.0 return
    different ``estimateAffinePartial2D`` results for identical input and opposite constants
    for the flat-template NCC case; on macOS the CoreML execution provider is available by
    default, and CoreML versus CPU changes the numbers. A git SHA captures none of that, so
    ratings collected before and after a ``pixi update`` would be pooled and a method's
    score would move for no code reason.

    Build these with :meth:`capture` rather than by hand -- the environment fields are
    required precisely so that a caller cannot forget them.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    git_sha: str
    method_version: str
    config_hash: str
    model_hashes: Mapping[str, str] = Field(default_factory=dict)
    python_version: str
    numpy_version: str
    cv2_version: str
    onnxruntime_version: str
    ort_providers: str
    pixi_lock_sha256: str
    created_at: datetime = Field(default_factory=_utc_now)

    @classmethod
    def capture(
        cls,
        *,
        method_version: str,
        config_hash: str,
        model_hashes: Mapping[str, str] | None = None,
    ) -> Provenance:
        """Snapshot the current code, config and environment.

        Args:
            method_version: The running method's ``version`` from its registry entry.
            config_hash: Output of :func:`object_search.provenance.config_hash`.
            model_hashes: ``{model_key: sha256}`` for every weight file the run loaded.

        Returns:
            A fully populated, frozen record.
        """
        environment = prov.environment_identity()
        return cls(
            git_sha=prov.current_git_sha(),
            method_version=method_version,
            config_hash=config_hash,
            model_hashes=dict(model_hashes or {}),
            python_version=environment["python_version"],
            numpy_version=environment["numpy_version"],
            cv2_version=environment["cv2_version"],
            onnxruntime_version=environment["onnxruntime_version"],
            ort_providers=environment["ort_providers"],
            pixi_lock_sha256=environment["pixi_lock_sha256"],
        )


class RunRecord(BaseModel):
    """One search execution, as persisted.

    Attributes:
        id: Database row id. ``None`` before insert -- the store assigns it.
        image_id: Stable identifier for the scene image (relative asset path or hash).
        exemplar: The box the user drew.
        method: Registry key of the method that ran.
        config_json: The **canonical** config JSON that was hashed, stored verbatim so a
            hash mismatch can be diffed rather than guessed at.
        config_hash: SHA-256 of ``config_json``.
        result: What the method returned, including latency and diagnostics.
        slice_metadata: What kind of image this was (EVAL-10).
        provenance: Code, config and environment identity (EVAL-09).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: int | None = None
    image_id: str = Field(min_length=1)
    exemplar: ExemplarBox
    method: str = Field(min_length=1)
    config_json: str
    config_hash: str
    result: SearchResult
    slice_metadata: SliceMetadata = Field(default_factory=SliceMetadata)
    provenance: Provenance


class MatchVerdict(BaseModel):
    """A human's per-match judgement (UI-08, EVAL-18).

    Attributes:
        match_index: Index into ``SearchResult.matches``.
        correct: True if that specific box is a genuine instance of the exemplar object.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    match_index: int = Field(ge=0)
    correct: bool


class RatingCompleteness(StrEnum):
    """How much of a rating was actually assessed.

    Statistics must be reported per completeness tier, never pooled: a precision figure
    computed over ratings that only ever assessed precision is honest, while the same
    figure presented as though recall had been checked is not.
    """

    NONE = "none"
    PRECISION_ONLY = "precision_only"
    RECALL_ONLY = "recall_only"
    COMPLETE = "complete"


class FPSource(StrEnum):
    """Where a false-positive count came from.

    ``PER_MATCH`` is stronger evidence than ``COUNT``: it says *which* boxes were wrong, so
    it can be re-audited. ``COUNT`` is a single number a human typed.
    """

    PER_MATCH = "per_match"
    COUNT = "count"


class Rating(BaseModel):
    """A human's assessment of one run.

    **Why ``wrong_count`` and ``missed_count`` default to ``None`` and must never default
    to ``0`` (EVAL-17).**

    ``null`` means *"not assessed"*. ``0`` means *"assessed, and there were none"*. These
    are different facts and only one of them is evidence. If an unreviewed rating arrives
    with ``wrong_count = 0`` and ``missed_count = 0``, then every unreviewed run silently
    claims **perfect precision and perfect recall**, the scoreboard fills up with fabricated
    100% scores, and the more runs go unreviewed the better every method looks. That is the
    exact opposite of what this project exists to measure.

    So: no numeric default here, no ``= 0`` in the API request model, no ``DEFAULT 0`` on
    the database column, and no prepopulated ``0`` in the rating form. The fields are
    submitted **empty** and stay empty until a human puts a number in one. Anyone tempted to
    add a convenience default should read this paragraph first -- it is here rather than in
    a design document for exactly that reason.

    ``thumbs_up`` is the only always-required judgement (Tier 0): it costs one click, so it
    is the one signal that will actually accumulate.

    Attributes:
        run_id: The run being rated.
        thumbs_up: Tier 0. Did this run look useful, overall?
        wrong_count: Number of returned boxes that are wrong. ``None`` = not assessed.
        missed_count: Number of true instances missed. ``None`` = not assessed.
        per_match_verdicts: Optional per-box judgements. ``None`` = not offered/not done.
        verdicts_confirmed: Per-match verdicts count only after an explicit confirm
            (UI-08). An untouched checkbox grid is not a judgement that every box is wrong,
            so unconfirmed verdicts are ignored entirely.
        unratable: The explicit skip path -- "I cannot judge this one". Distinct from a
            thumbs-down, and distinct from never having been shown the run.
        note: Free text.
        created_at: When the rating was submitted.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: int
    thumbs_up: bool
    # EVAL-17: NO numeric default. EVER. See the class docstring.
    wrong_count: int | None = Field(default=None, ge=0)
    # EVAL-17: NO numeric default. EVER. See the class docstring.
    missed_count: int | None = Field(default=None, ge=0)
    per_match_verdicts: tuple[MatchVerdict, ...] | None = None
    verdicts_confirmed: bool = False
    unratable: bool = False
    note: str | None = None
    created_at: datetime = Field(default_factory=_utc_now)

    # -- derived values: properties, never columns (EVAL-07) ---------------------------

    @property
    def _confirmed_verdicts(self) -> tuple[MatchVerdict, ...] | None:
        """The per-match verdicts, but only once a human explicitly confirmed them (UI-08).

        Returns the tuple rather than a bool so every caller is type-narrowed by the same
        check and none of them can read unconfirmed verdicts by accident.
        """
        if self.per_match_verdicts is not None and self.verdicts_confirmed:
            return self.per_match_verdicts
        return None

    @property
    def fp_source(self) -> FPSource | None:
        """Which false-positive evidence is available, if any.

        Per-match verdicts win when both are present (EVAL-18), because they say *which*
        boxes were wrong and can be re-audited against the stored matches.
        """
        if self._confirmed_verdicts is not None:
            return FPSource.PER_MATCH
        if self.wrong_count is not None:
            return FPSource.COUNT
        return None

    @property
    def false_positives(self) -> int | None:
        """Number of wrong boxes, or ``None`` when precision was never assessed."""
        verdicts = self._confirmed_verdicts
        if verdicts is not None:
            return sum(1 for verdict in verdicts if not verdict.correct)
        return self.wrong_count

    @property
    def has_fp_discrepancy(self) -> bool:
        """True when confirmed per-match verdicts and ``wrong_count`` disagree.

        Flagged, never silently reconciled. A disagreement means the human's two answers
        contradict each other, which is information -- possibly about the UI, possibly about
        an ambiguous instance -- and averaging it away destroys that information.
        """
        verdicts = self._confirmed_verdicts
        if verdicts is None or self.wrong_count is None:
            return False
        from_verdicts = sum(1 for verdict in verdicts if not verdict.correct)
        return from_verdicts != self.wrong_count

    @property
    def completeness(self) -> RatingCompleteness:
        """Which halves of the precision/recall pair this rating actually supports."""
        has_precision = self.false_positives is not None
        has_recall = self.missed_count is not None
        if has_precision and has_recall:
            return RatingCompleteness.COMPLETE
        if has_precision:
            return RatingCompleteness.PRECISION_ONLY
        if has_recall:
            return RatingCompleteness.RECALL_ONLY
        return RatingCompleteness.NONE

    def validate_against_retrieved(self, retrieved: int) -> tuple[bool, str | None]:
        """Check the rating against the run's actual returned-box count (EVAL-18).

        The upper bound ``wrong_count <= retrieved`` cannot live on the model as a field
        constraint, because ``retrieved`` is a property of the *run*, not of the rating. The
        API layer therefore calls this after loading the run.

        Note that a database ``CHECK`` constraint is not a substitute: research verified
        that ``CHECK (wrong_count <= retrieved_count)`` silently accepts ``wrong_count = 99``
        whenever the other side is ``NULL``, because SQL three-valued logic treats an
        unknown comparison as satisfied.

        Args:
            retrieved: ``len(run.result.matches)``.

        Returns:
            ``(True, None)`` when consistent, otherwise ``(False, reason)`` with a message
            fit to show a user.
        """
        if self.wrong_count is not None and self.wrong_count > retrieved:
            return (
                False,
                f"wrong_count={self.wrong_count} exceeds the {retrieved} box(es) this run returned",
            )
        if self.per_match_verdicts is not None:
            if len(self.per_match_verdicts) > retrieved:
                return (
                    False,
                    f"{len(self.per_match_verdicts)} per-match verdicts for a run that "
                    f"returned {retrieved} box(es)",
                )
            out_of_range = [
                verdict.match_index
                for verdict in self.per_match_verdicts
                if verdict.match_index >= retrieved
            ]
            if out_of_range:
                return (
                    False,
                    f"per-match verdict index(es) {sorted(out_of_range)} are outside the "
                    f"{retrieved} box(es) this run returned",
                )
            indices = [verdict.match_index for verdict in self.per_match_verdicts]
            if len(set(indices)) != len(indices):
                return (False, "duplicate per-match verdict indices")
        return (True, None)
