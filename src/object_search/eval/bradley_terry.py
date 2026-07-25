"""Bradley-Terry strengths over paired comparisons, with the complete-separation guard (EVAL-15).

Bradley-Terry models ``Pr(i beats j) = p_i / (p_i + p_j)``. Its maximum-likelihood fit is finite
**iff the directed win-graph is strongly connected** (Ford 1957 / Zermelo 1929): if one method
never loses there is no path out of its node, the likelihood increases without bound, and its
strength diverges to ``+inf`` while the optimiser still reports "converged". With four methods and
a handful of human comparisons a strongly-connected graph is the *exception*, so this is the
expected state of Phase 8, not a rare edge case (PITFALLS §8.3).

Three layers, all applied here:

1. **Check connectivity before trusting absolute strengths.** :func:`strongly_connected` reports
   whether the *raw* win-graph is strongly connected; when it is not, the fit is flagged
   ``regularised_only`` and the per-pair comparison counts are always surfaced alongside, so a
   ranking built on four comparisons cannot be read as if it were built on four hundred.
2. **Regularise with pseudo-games.** :func:`regularise` adds ``EPS`` wins *and* ``EPS`` losses to
   every ordered pair. This injects mild evidence of equality into every matchup, **guarantees**
   strong connectivity (so the MLE is finite), and shrinks strength differences toward zero.
   ``EPS = 0.5`` is a reported convention, not a measured value.
3. **Pin the scale.** Bradley-Terry is identified only up to a multiplicative constant on the
   strengths (an additive constant on the log-strengths), so successive fits are incomparable
   unless the scale is fixed. Strengths are normalised to **geometric mean 1**, which is what lets
   two refits of the same data be compared directly.

Ties are handled by the caller as **half a win to each side** (the pragmatic scoring) but are kept
as a distinct ``'tie'`` outcome in the ``paired_comparisons`` table (:mod:`object_search.eval.
paired`) so the modelling choice can be revisited without re-collecting data.

The ranking is reported **with uncertainty**, never as a bare ordering: each log-strength carries
an approximate standard error from the diagonal of the Fisher information, and the connectivity
flag plus per-pair counts say how much evidence stands behind the numbers.
"""

from __future__ import annotations

import math

from pydantic import BaseModel, ConfigDict

# The pseudo-game strength. Reported in the result so a reader knows the ranking is regularised,
# and can see how much shrinkage was applied. A convention, not a verified value (PITFALLS §8.3).
EPS = 0.5

WinGraph = dict[tuple[str, str], float]
"""``(winner, loser) -> weight``. A tie is two half-weight entries, one each direction, added by
the caller so this module never has to know about ties -- it only ever sees directed win weight."""


class BradleyTerryResult(BaseModel):
    """Fitted strengths plus everything needed to read them honestly."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    strengths: dict[str, float]
    """``p_i``, normalised to geometric mean 1 so refits are directly comparable."""

    log_strengths: dict[str, float]
    """``beta_i = log p_i``; sums to ~0 given the geometric-mean-1 pin."""

    log_strength_stderr: dict[str, float]
    """Approximate marginal standard error of each ``beta_i`` (diagonal Fisher information)."""

    ranking: tuple[str, ...]
    """Method names best-first by strength -- reported *with* the uncertainty fields, not alone."""

    strongly_connected: bool
    """Whether the **raw** win-graph was strongly connected. False => absolute strengths are
    regularised-only and must not be over-interpreted."""

    regularised_only: bool
    """True when the ranking exists only because of the pseudo-games (raw graph disconnected)."""

    eps: float
    """The pseudo-game weight applied to every ordered pair."""

    n_comparisons: float
    """Total raw comparison weight (ties count as one game), before regularisation."""

    pairwise_counts: dict[str, float]
    """``"i>j" -> raw wins of i over j`` -- the evidence behind each edge, always surfaced."""

    converged: bool
    iterations: int


def strongly_connected(wins: WinGraph, methods: list[str]) -> bool:
    """Whether every method is reachable from every other along strictly-positive win edges.

    Iterates ``sorted(...)`` so the traversal order is deterministic across processes (a ``set``'s
    order is not, PITFALLS §6.3). This is the cheap pre-check (``n`` is 4) that decides whether the
    Bradley-Terry MLE can be finite without regularisation.

    Args:
        wins: The directed win-graph; only strictly-positive weights count as edges.
        methods: The full method set the graph is over.

    Returns:
        ``True`` iff the win-graph is strongly connected.
    """
    universe = set(methods)
    adjacency = {m: {j for j in methods if wins.get((m, j), 0.0) > 0.0} for m in methods}

    def reachable_from(start: str) -> set[str]:
        seen = {start}
        stack = [start]
        while stack:
            for nxt in sorted(adjacency[stack.pop()] - seen):
                seen.add(nxt)
                stack.append(nxt)
        return seen

    return all(reachable_from(m) == universe for m in methods)


def regularise(wins: WinGraph, methods: list[str], eps: float = EPS) -> WinGraph:
    """Add ``eps`` wins and ``eps`` losses to every ordered pair (the pseudo-games).

    Every ordered pair ends with weight ``>= eps``, which guarantees the regularised graph is
    strongly connected (hence a finite MLE) and shrinks strength differences toward zero.

    Args:
        wins: The raw directed win-graph.
        methods: The full method set.
        eps: Pseudo-game weight per ordered pair.

    Returns:
        A new win-graph with ``eps`` added to every ordered pair.
    """
    padded: WinGraph = {}
    for i in methods:
        for j in methods:
            if i != j:
                padded[(i, j)] = wins.get((i, j), 0.0) + eps
    return padded


def _fisher_stderr(
    strengths: dict[str, float], games: dict[tuple[str, str], float], methods: list[str]
) -> dict[str, float]:
    """Approximate marginal standard error of each log-strength from the diagonal Fisher info.

    ``I_ii = sum_{j!=i} N_ij * p_i p_j / (p_i + p_j)^2``; ``SE_i ~= 1/sqrt(I_ii)``. Off-diagonal
    terms are ignored, so this is an approximation reported as such -- honest context on the
    ranking, not a claim of exact posterior width.
    """
    stderr: dict[str, float] = {}
    for i in methods:
        info = 0.0
        p_i = strengths[i]
        for j in methods:
            if i == j:
                continue
            n_ij = games.get((i, j), 0.0)
            p_j = strengths[j]
            info += n_ij * p_i * p_j / (p_i + p_j) ** 2
        stderr[i] = float("inf") if info <= 0.0 else 1.0 / math.sqrt(info)
    return stderr


def fit_bradley_terry(
    wins: WinGraph,
    methods: list[str],
    eps: float = EPS,
    max_iter: int = 1000,
    tol: float = 1e-9,
) -> BradleyTerryResult:
    """Fit regularised Bradley-Terry strengths, scale-pinned to geometric mean 1.

    The fit runs the standard minorisation-maximisation update on the **regularised** win-graph, so
    it is always finite -- an undefeated method gets a large but finite strength, never ``+inf``.
    The connectivity of the **raw** graph is recorded separately so the caller knows whether the
    absolute strengths are trustworthy or regularised-only.

    Args:
        wins: Raw directed win-graph, ``(winner, loser) -> weight`` (ties pre-split by the caller).
        methods: The full method set to rank. Sorted internally for deterministic iteration.
        eps: Pseudo-game weight (see :func:`regularise`).
        max_iter: Maximum MM iterations.
        tol: Convergence tolerance on the maximum strength change between iterations.

    Returns:
        A :class:`BradleyTerryResult` with strengths, log-strengths, approximate standard errors,
        the ranking, the connectivity/regularisation flags, and the per-pair evidence counts.

    Raises:
        ValueError: If fewer than two methods are given -- a comparison needs two competitors.
    """
    ordered = sorted(set(methods))
    if len(ordered) < 2:
        raise ValueError(f"Bradley-Terry needs >= 2 methods, got {len(ordered)}")

    raw_connected = strongly_connected(wins, ordered)

    # Regularised win-graph and the symmetric per-pair game counts N_ij used by the MM update.
    padded = regularise(wins, ordered, eps)
    games: dict[tuple[str, str], float] = {}
    total_wins: dict[str, float] = {}
    for i in ordered:
        total_wins[i] = sum(padded[(i, j)] for j in ordered if j != i)
        for j in ordered:
            if i != j:
                games[(i, j)] = padded[(i, j)] + padded[(j, i)]

    strengths = dict.fromkeys(ordered, 1.0)
    converged = False
    iterations = 0
    for step in range(1, max_iter + 1):
        iterations = step
        updated: dict[str, float] = {}
        for i in ordered:
            denom = sum(games[(i, j)] / (strengths[i] + strengths[j]) for j in ordered if j != i)
            updated[i] = total_wins[i] / denom
        # Pin the scale every iteration (geometric mean 1) so the values cannot drift off to a
        # degenerate magnitude, and so the convergence test is on a stable scale.
        log_gm = sum(math.log(v) for v in updated.values()) / len(updated)
        gm = math.exp(log_gm)
        updated = {m: v / gm for m, v in updated.items()}
        max_delta = max(abs(updated[m] - strengths[m]) for m in ordered)
        strengths = updated
        if max_delta < tol:
            converged = True
            break

    log_strengths = {m: math.log(strengths[m]) for m in ordered}
    stderr = _fisher_stderr(strengths, games, ordered)
    ranking = tuple(sorted(ordered, key=lambda m: strengths[m], reverse=True))

    raw_win_weight = sum(wins.values())
    pairwise_counts = {
        f"{i}>{j}": wins.get((i, j), 0.0) for i in ordered for j in ordered if i != j
    }

    return BradleyTerryResult(
        strengths=strengths,
        log_strengths=log_strengths,
        log_strength_stderr=stderr,
        ranking=ranking,
        strongly_connected=raw_connected,
        regularised_only=not raw_connected,
        eps=eps,
        n_comparisons=raw_win_weight,
        pairwise_counts=pairwise_counts,
        converged=converged,
        iterations=iterations,
    )
