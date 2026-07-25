"""The per-method scoreboard, assembled from the derived-metric views.

Every rate here ships with the ``n`` it was actually computed over, and each ``n`` is
reported **separately** (EVAL-13): the thumbs sample, the precision sample and the recall
sample are different subsets, because a run rated with only a bare thumbs-up feeds none of
precision/recall, and one rated with only a ``wrong_count`` feeds precision but not the
threshold sweep. Collapsing them onto one shared ``n`` is the §7.2 "average over an
unstated subset" bug.

The four SQLite traps that would silently corrupt these numbers are handled in the views
(:mod:`object_search.store.schema`): ``CAST(... AS REAL)`` before every division,
``SUM`` never ``TOTAL``, ``NULLIF`` on every denominator, and ``NULL`` propagating so an
unassessed run contributes to nothing rather than to a zero. This module only aggregates
the already-correct per-run rows and never re-introduces a default.

Methods are ranked by the **lower bound** of the thumbs Wilson interval, not by the raw
rate -- so a 1/1 does not outrank a 50/100 (EVAL-14). A method with no rated runs (Wilson
``None``) sorts last.
"""

from __future__ import annotations

import sqlite3

import numpy as np
from pydantic import BaseModel, ConfigDict

from object_search.store.wilson import wilson_interval


class MethodStats(BaseModel):
    """One method's row on the scoreboard.

    Every optional rate is ``None`` -- never ``0`` -- when its sample is empty, so the UI
    can render an em dash instead of a fabricated score.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    method: str

    thumbs_n: int
    thumbs_n_up: int
    thumbs_rate: float | None
    thumbs_ci_lower: float | None
    thumbs_ci_upper: float | None

    precision_n: int
    precision_mean: float | None

    recall_n: int
    recall_mean: float | None

    latency_p50_ms: float | None
    latency_p90_ms: float | None
    latency_p99_ms: float | None

    abstention_count: int
    error_count: int
    threshold_sweep_eligible_count: int


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def scoreboard(conn: sqlite3.Connection, confidence: float = 0.95) -> list[MethodStats]:
    """Assemble the per-method scoreboard, ranked by the thumbs Wilson lower bound.

    Args:
        conn: An open, migrated connection.
        confidence: Confidence level for the thumbs Wilson interval.

    Returns:
        One :class:`MethodStats` per method that has at least one run, ordered best-first
        by Wilson lower bound (methods with no rated runs last).
    """
    # Thumbs: n = COUNT(thumbs_up), never COUNT(*) -- the view enforces that.
    thumbs = {
        row["method"]: (row["n_up"] or 0, row["n_rated"] or 0)
        for row in conn.execute("SELECT method, n_up, n_rated FROM method_thumbs")
    }

    # Per-run derived metrics. precision/recall are already NULL where unavailable, so a
    # simple "is not None" filter yields each metric's honest sample.
    precision: dict[str, list[float]] = {}
    recall: dict[str, list[float]] = {}
    sweep_eligible: dict[str, int] = {}
    for row in conn.execute("SELECT method, precision, recall, fp_source FROM run_metrics"):
        method = row["method"]
        if row["precision"] is not None:
            precision.setdefault(method, []).append(float(row["precision"]))
        if row["recall"] is not None:
            recall.setdefault(method, []).append(float(row["recall"]))
        if row["fp_source"] == "per_match":
            sweep_eligible[method] = sweep_eligible.get(method, 0) + 1

    # Latency and outcome counts come straight off runs, so a run with two ratings cannot
    # double-count its latency.
    latency: dict[str, list[float]] = {}
    abstentions: dict[str, int] = {}
    errors: dict[str, int] = {}
    methods: set[str] = set()
    for row in conn.execute(
        "SELECT method, outcome, preprocess_ms + inference_ms + postprocess_ms AS total_ms "
        "FROM runs"
    ):
        method = row["method"]
        methods.add(method)
        latency.setdefault(method, []).append(float(row["total_ms"]))
        if row["outcome"] == "empty":
            abstentions[method] = abstentions.get(method, 0) + 1
        elif row["outcome"] == "error":
            errors[method] = errors.get(method, 0) + 1

    board: list[MethodStats] = []
    for method in methods:
        n_up, n_rated = thumbs.get(method, (0, 0))
        ci = wilson_interval(n_up, n_rated, confidence) if n_rated > 0 else None
        prec_values = precision.get(method, [])
        rec_values = recall.get(method, [])
        lat_values = latency.get(method, [])

        board.append(
            MethodStats(
                method=method,
                thumbs_n=n_rated,
                thumbs_n_up=n_up,
                thumbs_rate=(n_up / n_rated) if n_rated > 0 else None,
                thumbs_ci_lower=ci[0] if ci is not None else None,
                thumbs_ci_upper=ci[1] if ci is not None else None,
                precision_n=len(prec_values),
                precision_mean=_mean(prec_values),
                recall_n=len(rec_values),
                recall_mean=_mean(rec_values),
                latency_p50_ms=_percentile(lat_values, 50),
                latency_p90_ms=_percentile(lat_values, 90),
                latency_p99_ms=_percentile(lat_values, 99),
                abstention_count=abstentions.get(method, 0),
                error_count=errors.get(method, 0),
                threshold_sweep_eligible_count=sweep_eligible.get(method, 0),
            )
        )

    # Rank by Wilson lower bound; a method with no rated runs (None) sorts last.
    board.sort(
        key=lambda s: s.thumbs_ci_lower if s.thumbs_ci_lower is not None else -1.0,
        reverse=True,
    )
    return board
