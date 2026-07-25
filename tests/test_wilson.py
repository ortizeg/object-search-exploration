"""Wilson interval tests -- numeric-exact against the verified closed forms (PITFALLS §8.2).

Every expected value here was computed and pinned during research. If one drifts, the
implementation changed, not the mathematics.
"""

from __future__ import annotations

import math
from pathlib import Path
from statistics import NormalDist

import pytest

from object_search.store.wilson import wilson_interval


def test_z_at_95_percent_is_exact() -> None:
    # The 95% z is 1.9599639845400534. NormalDist().inv_cdf is pure Python but calls
    # math.log/math.sqrt, whose last ULP is libm-dependent: macOS arm64 yields ...534,
    # Linux x86_64 yields ...536. Pin to 1e-12 -- 12 significant figures, three orders of
    # magnitude below the ~4e-6 gap to the textbook 1.96, but robust to platform libm.
    z = NormalDist().inv_cdf(1 - (1 - 0.95) / 2)
    assert z == pytest.approx(1.9599639845400534, abs=1e-12)


def test_wilson_source_computes_z_and_never_hardcodes_1_96() -> None:
    source = Path("src/object_search/store/wilson.py").read_text()
    assert "NormalDist().inv_cdf" in source
    assert "1.96" not in source


def test_n_zero_returns_none_not_zero_one() -> None:
    """[0, 1] is 'no information' and must be distinct from an estimate (EVAL-17 analogue)."""
    assert wilson_interval(0, 0) is None


def test_zero_of_ten_closed_form() -> None:
    result = wilson_interval(0, 10)
    # 1e-12 tolerance on the upper bound absorbs libm last-ULP variance (see z test).
    assert result == pytest.approx((0.0, 0.2775327998628891), abs=1e-12)
    # Lower bound is exactly 0.0, and NOT -0.0 -- an exact requirement, not approximate.
    assert result is not None
    lower = result[0]
    assert lower == 0.0
    assert math.copysign(1.0, lower) == 1.0  # positive zero, guards the JSON -0.0 artifact


def test_ten_of_ten_closed_form() -> None:
    # Lower bound n/(n+z^2); 1e-12 tolerance for libm last-ULP variance. Upper is exact 1.0.
    result = wilson_interval(10, 10)
    assert result == pytest.approx((0.722467200137111, 1.0), abs=1e-12)
    assert result is not None
    assert result[1] == 1.0  # closed form pins the upper bound exactly


def test_zero_of_three_lower_bound_is_positive_zero() -> None:
    lower, _ = wilson_interval(0, 3)  # type: ignore[misc]
    assert lower == 0.0
    assert math.copysign(1.0, lower) == 1.0  # not -0.0


def test_lower_bound_ranking_beats_raw_rate() -> None:
    """The point of the interval: 1/1 must rank below 50/100 by lower bound."""
    one_of_one = wilson_interval(1, 1)
    fifty_of_hundred = wilson_interval(50, 100)
    assert one_of_one is not None and fifty_of_hundred is not None
    assert one_of_one[0] < fifty_of_hundred[0]


@pytest.mark.parametrize(
    ("successes", "n"),
    [(0, 1), (1, 1), (0, 10), (3, 10), (7, 10), (10, 10), (1, 3), (50, 100)],
)
def test_bounds_stay_within_unit_interval(successes: int, n: int) -> None:
    result = wilson_interval(successes, n)
    assert result is not None
    lo, hi = result
    assert 0.0 <= lo <= hi <= 1.0


@pytest.mark.parametrize(
    ("successes", "n"),
    [(-1, 5), (6, 5), (3, -1)],
)
def test_invalid_counts_raise(successes: int, n: int) -> None:
    with pytest.raises(ValueError, match="invalid counts"):
        wilson_interval(successes, n)


def test_confidence_argument_changes_z() -> None:
    narrow = wilson_interval(3, 10, confidence=0.90)
    wide = wilson_interval(3, 10, confidence=0.99)
    assert narrow is not None and wide is not None
    # A higher confidence must widen the interval.
    assert (wide[1] - wide[0]) > (narrow[1] - narrow[0])
