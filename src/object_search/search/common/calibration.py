"""Threshold calibration -- four strategies, each returning its *reasoning*, one offering.

Where to put the accept/reject line on a score distribution is, per the source research, one
of the two places these methods actually succeed or fail (peak extraction being the other).
So this module ships four *selectable* strategies rather than a magic number buried in each
method, and -- crucially -- every strategy returns a :class:`CalibrationResult` carrying both
the threshold **and** a human ``reason`` string. That ``reason`` is what makes Phase 2 success
criterion 4 ("different, **inspectable** thresholds") checkable: a bare float is not
inspectable, so a practitioner comparing methods cannot see *why* one cut where it did.

The strategies:

- ``"fixed"`` -- passthrough of an explicit threshold, so a method like ``ncc`` has a single
  code path whether the user pinned a number or asked for calibration.
- ``"self-similarity"`` -- calibrate relative to the exemplar's own self-match score. NCC of
  the exemplar against itself is ~1.0, so a match is accepted above ``self_score * retain_frac``.
- ``"ratio"`` -- cut at the largest relative gap in the sorted top scores; report the gap.
- ``"gmm"`` -- fit a two-component Gaussian mixture and cut between the modes. Its
  ``random_state`` is a **genuine** seed: GMM initialisation really is stochastic (unlike
  OpenCV's RANSAC, PITFALLS.md 3.2), so pinning it is real reproducibility work, not theatre.
  A **degeneracy guard** detects a single-mode distribution and falls back to ``ratio`` with
  ``degenerate=True`` rather than returning a confident-looking threshold from a fit that does
  not describe the data.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
import numpy.typing as npt
from loguru import logger
from pydantic import BaseModel, ConfigDict
from sklearn.mixture import GaussianMixture

CalibrationStrategy = Literal["fixed", "self-similarity", "ratio", "gmm"]


class CalibrationResult(BaseModel):
    """A calibrated threshold together with the reasoning that produced it.

    Frozen, like every cross-boundary value here. The ``reason`` field is the point of the
    type: it is what a practitioner reads to understand *why* the line landed where it did,
    and what makes two strategies' thresholds comparable rather than two opaque floats.

    Attributes:
        threshold: The accept/reject cut. A score strictly above it is accepted; the exact
            comparison is the calling method's, but the calibrators all treat it as a lower cut.
        strategy: Which strategy produced this result. When ``gmm`` falls back this is still
            ``"gmm"`` -- the ``reason`` and ``degenerate`` fields record the fallback.
        reason: Human-readable justification (the gap size, the self-score, the mode means).
        degenerate: True when the requested strategy could not be applied and a fallback was
            used -- currently only ``gmm`` on a single-mode distribution.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    threshold: float
    strategy: str
    reason: str
    degenerate: bool = False


def calibrate(
    scores: Sequence[float] | npt.NDArray[np.floating],
    *,
    strategy: CalibrationStrategy,
    fixed_threshold: float | None = None,
    self_score: float | None = None,
    retain_frac: float = 0.7,
    ratio_top_n: int = 10,
    seed: int = 0,
    min_mode_separation_frac: float = 2.0,
    min_component_weight: float = 0.05,
) -> CalibrationResult:
    """Compute an accept/reject threshold from a score distribution.

    Args:
        scores: The observed match scores (e.g. per-peak NCC responses).
        strategy: One of ``fixed | self-similarity | ratio | gmm``.
        fixed_threshold: The threshold to pass through when ``strategy == "fixed"``.
        self_score: The exemplar's self-match score; required for ``"self-similarity"``.
        retain_frac: For ``"self-similarity"``, accept above ``self_score * retain_frac``.
        ratio_top_n: For ``"ratio"``, how many of the top sorted scores to search for the gap.
        seed: ``random_state`` for the GMM. A real seed -- GMM init is stochastic.
        min_mode_separation_frac: For ``"gmm"``, the two component means must be at least this
            many pooled within-component standard deviations apart, else the fit is called
            degenerate.
        min_component_weight: For ``"gmm"``, if either component's weight is below this the
            two-mode assumption does not hold and the fit is called degenerate.

    Returns:
        A :class:`CalibrationResult`. For ``"gmm"`` on a single-mode distribution, a
        ``ratio`` fallback with ``degenerate=True``.

    Raises:
        ValueError: If a strategy's required input is missing (``fixed_threshold`` for
            ``"fixed"``, ``self_score`` for ``"self-similarity"``), or the strategy is
            unknown. Missing inputs raise -- naming the strategy -- rather than silently
            defaulting, because a silent default is exactly how a wrong threshold ships.
    """
    arr = np.asarray(scores, dtype=np.float64).ravel()

    if strategy == "fixed":
        if fixed_threshold is None:
            raise ValueError("strategy 'fixed' requires an explicit fixed_threshold")
        return CalibrationResult(
            threshold=float(fixed_threshold),
            strategy="fixed",
            reason=f"fixed threshold {fixed_threshold:.4f} supplied by caller",
        )

    if strategy == "self-similarity":
        if self_score is None:
            raise ValueError(
                "strategy 'self-similarity' requires self_score (the exemplar's self-match "
                "score); it will not silently default"
            )
        threshold = float(self_score) * retain_frac
        return CalibrationResult(
            threshold=threshold,
            strategy="self-similarity",
            reason=(
                f"self_score {self_score:.4f} x retain_frac {retain_frac:.2f} = {threshold:.4f}"
            ),
        )

    if strategy == "ratio":
        threshold, reason = _ratio_threshold(arr, ratio_top_n)
        return CalibrationResult(threshold=threshold, strategy="ratio", reason=reason)

    if strategy == "gmm":
        return _gmm_threshold(
            arr,
            seed=seed,
            ratio_top_n=ratio_top_n,
            min_mode_separation_frac=min_mode_separation_frac,
            min_component_weight=min_component_weight,
        )

    raise ValueError(
        f"unknown calibration strategy {strategy!r}; expected fixed | self-similarity | ratio | gmm"
    )


def _ratio_threshold(arr: npt.NDArray[np.float64], top_n: int) -> tuple[float, str]:
    """Cut at the largest gap between consecutive scores within the top ``top_n``.

    The intuition: a well-separated set of true matches sits at the top with an empty band
    below it before the noise floor begins. The widest gap in the sorted top scores is that
    band, and its midpoint is the natural cut. The gap size is returned in the reason so the
    confidence of the cut is visible.
    """
    if arr.size == 0:
        return 0.0, "no scores; threshold defaulted to 0.0"
    ordered = np.sort(arr)[::-1]
    if ordered.size == 1:
        # One score: no gap to cut at. Sit just below it so the single observation is accepted.
        return float(ordered[0]) - 1e-9, "single score; threshold placed just below it"

    top = ordered[: max(2, top_n)]
    gaps = top[:-1] - top[1:]
    i = int(np.argmax(gaps))  # first-occurrence tie-break: the highest such gap wins
    threshold = float((top[i] + top[i + 1]) / 2.0)
    return threshold, (
        f"largest gap {float(gaps[i]):.4f} between {float(top[i]):.4f} and "
        f"{float(top[i + 1]):.4f} (top {top.size}); cut at midpoint {threshold:.4f}"
    )


def _gmm_threshold(
    arr: npt.NDArray[np.float64],
    *,
    seed: int,
    ratio_top_n: int,
    min_mode_separation_frac: float,
    min_component_weight: float,
) -> CalibrationResult:
    """Fit a two-component GMM and cut between the modes, or fall back if it is degenerate."""

    def _degenerate_fallback(why: str) -> CalibrationResult:
        threshold, ratio_reason = _ratio_threshold(arr, ratio_top_n)
        logger.debug("gmm calibration degenerate ({}); falling back to ratio", why)
        return CalibrationResult(
            threshold=threshold,
            strategy="gmm",
            reason=f"gmm degenerate ({why}); fell back to ratio -> {ratio_reason}",
            degenerate=True,
        )

    # A GMM needs at least two distinct points to fit two components at all.
    if np.unique(arr).size < 2:
        return _degenerate_fallback("fewer than two distinct scores")

    gm = GaussianMixture(n_components=2, random_state=seed)
    try:
        gm.fit(arr.reshape(-1, 1))
    except (ValueError, FloatingPointError) as exc:  # singular covariance, etc.
        return _degenerate_fallback(f"fit failed: {exc}")

    means = gm.means_.ravel()
    weights = gm.weights_.ravel()
    variances = gm.covariances_.ravel()
    pooled_std = float(np.sqrt(np.average(variances, weights=weights))) + 1e-12
    separation = float(abs(means[0] - means[1]))

    # Degeneracy guard: the two-mode story only holds if the means are well separated AND
    # both components carry real weight. Either failing means the confident-looking cut is a
    # fiction, so admit it and fall back rather than emit it (PITFALLS.md / CONTEXT risk).
    if separation < min_mode_separation_frac * pooled_std:
        return _degenerate_fallback(
            f"modes {separation:.4f} apart < {min_mode_separation_frac:.1f} x pooled std "
            f"{pooled_std:.4f}"
        )
    if float(weights.min()) < min_component_weight:
        return _degenerate_fallback(
            f"minority component weight {float(weights.min()):.3f} < {min_component_weight:.3f}"
        )

    lo, hi = float(min(means)), float(max(means))
    hi_idx = int(np.argmax(means))
    # The decision boundary is where the posterior for the higher-mean component reaches 0.5.
    # It is monotone between the two means, so a linear scan is exact and deterministic.
    grid = np.linspace(lo, hi, 512).reshape(-1, 1)
    posterior_hi = gm.predict_proba(grid)[:, hi_idx]
    crossing_idx = int(np.searchsorted(posterior_hi, 0.5))
    crossing_idx = min(crossing_idx, grid.size - 1)
    threshold = float(grid[crossing_idx, 0])
    return CalibrationResult(
        threshold=threshold,
        strategy="gmm",
        reason=(
            f"two modes at {lo:.4f} / {hi:.4f} (weights "
            f"{weights[0]:.2f}/{weights[1]:.2f}); posterior cut at {threshold:.4f}"
        ),
    )
