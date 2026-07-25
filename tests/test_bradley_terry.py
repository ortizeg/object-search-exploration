"""Bradley-Terry with the complete-separation guard (EVAL-15, PITFALLS §8.3).

The load-bearing assertions: an undefeated method yields a FINITE score, ``strongly_connected``
detects a disconnected win-graph (and regularisation repairs it), and two refits of the same data
return comparable strengths because the scale is pinned.
"""

from __future__ import annotations

import math

import pytest

from object_search.eval.bradley_terry import (
    EPS,
    WinGraph,
    fit_bradley_terry,
    regularise,
    strongly_connected,
)

_METHODS = ["ncc", "sparse-geo", "dino-dense", "propose-retrieve"]


def _undefeated_graph() -> WinGraph:
    """A win-graph where 'D' never loses: a 3-cycle among the others, D beats all three."""
    a, b, c, d = "ncc", "sparse-geo", "dino-dense", "propose-retrieve"
    return {
        (a, b): 1.0,
        (b, c): 1.0,
        (c, a): 1.0,  # 3-cycle keeps a, b, c mutually reachable
        (d, a): 1.0,
        (d, b): 1.0,
        (d, c): 1.0,  # d beats everyone, loses to no one
    }


def test_undefeated_method_yields_finite_score() -> None:
    # The complete-separation guard: without regularisation D's MLE strength diverges to +inf.
    wins = _undefeated_graph()
    d = "propose-retrieve"

    # The raw graph is NOT strongly connected -- nothing reaches D.
    assert strongly_connected(wins, _METHODS) is False

    result = fit_bradley_terry(wins, _METHODS)

    # Every strength is finite and positive -- the whole point of the guard.
    assert all(math.isfinite(v) for v in result.strengths.values())
    assert all(v > 0.0 for v in result.strengths.values())
    assert all(math.isfinite(b) for b in result.log_strengths.values())

    # D is undefeated, so it ranks first, but with a finite strength.
    assert result.ranking[0] == d
    assert math.isfinite(result.strengths[d])

    # The result is honest about being regularised-only, and surfaces the per-pair evidence.
    assert result.regularised_only is True
    assert result.strongly_connected is False
    assert result.converged is True
    assert result.pairwise_counts[f"{d}>ncc"] == 1.0


def test_strongly_connected_detects_disconnected_graph() -> None:
    # Two islands: {ncc, sparse-geo} and {dino-dense, propose-retrieve}, no cross edges.
    wins: WinGraph = {
        ("ncc", "sparse-geo"): 1.0,
        ("sparse-geo", "ncc"): 1.0,
        ("dino-dense", "propose-retrieve"): 1.0,
        ("propose-retrieve", "dino-dense"): 1.0,
    }
    assert strongly_connected(wins, _METHODS) is False

    # Regularisation adds a pseudo-game to every ordered pair, which repairs connectivity.
    padded = regularise(wins, _METHODS)
    assert strongly_connected(padded, _METHODS) is True


def test_strongly_connected_true_on_a_cycle() -> None:
    wins: WinGraph = {("ncc", "sparse-geo"): 1.0, ("sparse-geo", "dino-dense"): 1.0}
    three = ["ncc", "sparse-geo", "dino-dense"]
    assert strongly_connected(wins, three) is False  # no edge back to ncc
    wins[("dino-dense", "ncc")] = 1.0
    assert strongly_connected(wins, three) is True  # now a full cycle


def test_regularise_adds_eps_to_every_ordered_pair() -> None:
    wins: WinGraph = {("ncc", "sparse-geo"): 3.0}
    padded = regularise(wins, ["ncc", "sparse-geo"], eps=EPS)
    # The one real edge gains eps; the reverse (unseen) edge is exactly eps.
    assert padded[("ncc", "sparse-geo")] == pytest.approx(3.0 + EPS)
    assert padded[("sparse-geo", "ncc")] == pytest.approx(EPS)
    # Every ordered pair is present.
    assert set(padded) == {("ncc", "sparse-geo"), ("sparse-geo", "ncc")}


def test_scale_is_pinned_so_two_refits_are_comparable() -> None:
    # Same data, two different input dict orderings. A scale-pinned fit must return identical
    # strengths (geometric mean 1), which is what makes successive fits comparable.
    wins_a: WinGraph = {("ncc", "sparse-geo"): 3.0, ("sparse-geo", "ncc"): 1.0}
    wins_b: WinGraph = {("sparse-geo", "ncc"): 1.0, ("ncc", "sparse-geo"): 3.0}
    two = ["ncc", "sparse-geo"]

    first = fit_bradley_terry(wins_a, two)
    second = fit_bradley_terry(wins_b, two)
    for m in two:
        assert first.strengths[m] == pytest.approx(second.strengths[m])

    # Geometric mean of the strengths is pinned to 1.
    log_gm = sum(math.log(v) for v in first.strengths.values()) / len(first.strengths)
    assert math.exp(log_gm) == pytest.approx(1.0)


def test_stronger_method_ranks_higher() -> None:
    # A connected, decisive graph: ncc beats sparse-geo 9:1. ncc must be stronger.
    wins: WinGraph = {("ncc", "sparse-geo"): 9.0, ("sparse-geo", "ncc"): 1.0}
    two = ["ncc", "sparse-geo"]
    result = fit_bradley_terry(wins, two)
    assert result.strongly_connected is True
    assert result.regularised_only is False
    assert result.ranking[0] == "ncc"
    assert result.strengths["ncc"] > result.strengths["sparse-geo"]


def test_balanced_graph_gives_equal_strengths() -> None:
    wins: WinGraph = {("ncc", "sparse-geo"): 5.0, ("sparse-geo", "ncc"): 5.0}
    result = fit_bradley_terry(wins, ["ncc", "sparse-geo"])
    assert result.strengths["ncc"] == pytest.approx(result.strengths["sparse-geo"])
    assert result.strengths["ncc"] == pytest.approx(1.0)  # geo-mean-1, equal -> both 1


def test_fit_requires_at_least_two_methods() -> None:
    with pytest.raises(ValueError, match=">= 2 methods"):
        fit_bradley_terry({}, ["ncc"])
