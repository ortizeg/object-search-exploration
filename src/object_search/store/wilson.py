"""The Wilson score interval for a thumbs-up rate (EVAL-14), dependency-free.

Why not the default. The Wald interval -- ``phat +/- z*sqrt(phat*(1-phat)/n)`` --
collapses to *zero width* at ``phat`` of 0 or 1, so 3 thumbs-up out of 3 reads as
"100% +/- 0%" and 0/10 as "0% +/- 0%", falsely implying certainty from almost no data.
That is the exact failure
EVAL-14 exists to prevent, and it is ``statsmodels``' default method, so this project
computes Wilson itself rather than risk a stray ``method='normal'`` (PITFALLS §8.1).

Three details that are easy to get wrong (all verified numerically, PITFALLS §8.2):

* **z is computed, never hardcoded.** ``NormalDist().inv_cdf`` gives
  ``1.9599639845400534`` at 95% -- the textbook two-decimal constant is wrong in the
  fourth decimal and cannot follow the ``confidence`` argument.
* **``n == 0`` returns ``None``, not ``(0.0, 1.0)``.** ``[0, 1]`` is "no information", the
  same not-assessed-vs-assessed distinction as EVAL-17 one layer up; the caller must
  render it distinctly and never as an estimate.
* **The ``-0.0`` artifact.** At ``p̂ = 0`` the subtraction produces ``-0.0``, which
  serialises into JSON as ``-0.0`` and looks like a bug. ``max(0.0, ...)`` erases it.

Ranking uses the interval's **lower bound**, not ``p̂`` -- that is the whole point of
computing it: 1/1 (lower ≈ 0.21) must rank below 50/100 (lower ≈ 0.40).
"""

from __future__ import annotations

import math
from statistics import NormalDist


def wilson_interval(successes: int, n: int, confidence: float = 0.95) -> tuple[float, float] | None:
    """Wilson score interval for ``successes`` out of ``n`` at the given confidence.

    Args:
        successes: Number of successes (thumbs-up), ``0 <= successes <= n``.
        n: Number of trials (rated runs). ``0`` returns ``None``.
        confidence: Two-sided confidence level, e.g. ``0.95``.

    Returns:
        ``(lower, upper)`` with ``0.0 <= lower <= upper <= 1.0``, or ``None`` when
        ``n == 0`` -- which the caller **must** render distinctly from ``(0.0, 1.0)``.

    Raises:
        ValueError: If ``n < 0`` or ``successes`` is outside ``[0, n]``.
    """
    if n < 0 or not (0 <= successes <= n):
        raise ValueError(f"invalid counts: {successes}/{n}")
    if n == 0:
        return None

    z = NormalDist().inv_cdf(1 - (1 - confidence) / 2)
    z2 = z * z
    denom = n + z2

    # Closed forms at the extremes: exact bounds, no division-in-the-±-term rounding, and
    # an explicit 0.0 that can never be the -0.0 the general subtraction would produce.
    if successes == 0:
        return (0.0, z2 / denom)
    if successes == n:
        return (n / denom, 1.0)

    centre = (successes + z2 / 2) / denom
    half = (z / denom) * math.sqrt(successes * (n - successes) / n + z2 / 4)
    return (max(0.0, centre - half), min(1.0, centre + half))
