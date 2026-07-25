"""Tests for the threshold-calibration offering.

The two load-bearing tests here are Phase 2 success criterion 4:

* the three real strategies produce *different* thresholds on the same bimodal input, each
  with a stated reason (inspectable, not just a float);
* ``gmm`` recognises a single-mode distribution as degenerate and falls back rather than
  emitting a confident-looking bad cut.
"""

import numpy as np
import pytest

from object_search.search.common.calibration import calibrate

LOW_MODE = 0.2
HIGH_MODE = 0.9


def _bimodal() -> np.ndarray:
    rng = np.random.default_rng(0)
    low = rng.normal(LOW_MODE, 0.03, 80)
    high = rng.normal(HIGH_MODE, 0.03, 80)
    return np.clip(np.concatenate([low, high]), 0.0, 1.0)


def _unimodal() -> np.ndarray:
    rng = np.random.default_rng(0)
    return np.clip(rng.normal(0.5, 0.02, 300), 0.0, 1.0)


def test_three_strategies_differ_on_a_bimodal_input():
    # Phase 2 success criterion 4. Same scores, three real strategies, thresholds that
    # separate the modes but are NOT all equal -- and each carries a reason.
    scores = _bimodal()

    self_sim = calibrate(scores, strategy="self-similarity", self_score=1.0, retain_frac=0.7)
    # top_n spans both modes so ratio finds the between-mode gap, not a within-cluster one.
    ratio = calibrate(scores, strategy="ratio", ratio_top_n=scores.size)
    gmm = calibrate(scores, strategy="gmm", seed=0)

    thresholds = [self_sim.threshold, ratio.threshold, gmm.threshold]
    # Every strategy cuts between the two modes.
    for result in (self_sim, ratio, gmm):
        assert LOW_MODE < result.threshold < HIGH_MODE, f"{result.strategy}: {result.threshold}"
        assert result.reason  # inspectable, non-empty reason
    # And they do not all land on the same number.
    assert len({round(t, 6) for t in thresholds}) > 1
    assert not gmm.degenerate


def test_gmm_flags_a_unimodal_distribution_as_degenerate_and_falls_back():
    result = calibrate(_unimodal(), strategy="gmm", seed=0)
    assert result.degenerate is True
    assert "degenerate" in result.reason.lower()
    assert "ratio" in result.reason.lower()  # the reason names the fallback


def test_self_similarity_without_self_score_raises():
    with pytest.raises(ValueError, match="self-similarity"):
        calibrate(_bimodal(), strategy="self-similarity")


def test_gmm_is_deterministic_for_a_fixed_seed():
    scores = _bimodal()
    first = calibrate(scores, strategy="gmm", seed=7)
    second = calibrate(scores, strategy="gmm", seed=7)
    assert first.threshold == second.threshold
    assert first.reason == second.reason


def test_fixed_strategy_passes_the_threshold_through():
    result = calibrate([0.1, 0.5, 0.9], strategy="fixed", fixed_threshold=0.42)
    assert result.threshold == 0.42
    assert result.degenerate is False


def test_fixed_without_threshold_raises():
    with pytest.raises(ValueError, match="fixed"):
        calibrate([0.1, 0.5], strategy="fixed")


def test_ratio_reports_the_gap_in_its_reason():
    # A clean split: a cluster near 0.9 and a cluster near 0.1, big empty band between.
    scores = [0.92, 0.90, 0.88, 0.12, 0.10, 0.08]
    result = calibrate(scores, strategy="ratio")
    assert 0.12 < result.threshold < 0.88
    assert "gap" in result.reason.lower()


def test_unknown_strategy_raises():
    with pytest.raises(ValueError, match="unknown calibration strategy"):
        calibrate([0.1, 0.2], strategy="bogus")  # type: ignore[arg-type]
