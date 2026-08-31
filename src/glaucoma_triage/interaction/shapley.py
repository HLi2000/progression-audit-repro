"""Exact coalition-game Shapley values and pairwise interaction indices.

All functions operate on a :class:`ValueTable` — a mapping from a coalition (``frozenset`` of
modality names) to its characteristic value ``v(S)`` (here, a cross-validated performance score such
as AUC). With only a handful of modalities the powerset is tiny, so every quantity is computed
**exactly** (no Monte-Carlo sampling), which keeps the interaction signs deterministic and testable.

Conventions
-----------
* Shapley value ``φ_i`` — modality *i*'s average marginal contribution across all orderings.
  Efficiency: ``Σ_i φ_i == v(N) − v(∅)``.
* Shapley interaction index ``I_{ij}`` (Grabisch & Roubens) — the average of the second-order
  discrete derivative ``δ_{ij}(S) = v(S∪{i,j}) − v(S∪{i}) − v(S∪{j}) + v(S)``. **Positive ⇒ synergy**
  (the two modalities are complementary — together they beat the sum of their parts); **negative ⇒
  redundancy** (each carries much of the other's information). This is the sign convention used by
  InterSHAP and is the metric not previously reported for glaucoma progression.
"""

from __future__ import annotations

from itertools import chain, combinations
from math import factorial

# A coalition is a frozenset of modality names; the table gives v(S) for (ideally) every subset.
ValueTable = dict[frozenset[str], float]


def coalitions(players: list[str]) -> list[frozenset[str]]:
    """Every subset of ``players`` (the full powerset, including ∅ and the grand coalition)."""
    return [frozenset(c) for c in chain.from_iterable(
        combinations(players, r) for r in range(len(players) + 1))]


def _v(table: ValueTable, s: frozenset[str]) -> float:
    """``v(S)`` with a clear error if the coalition is absent from the table."""
    if s not in table:
        raise KeyError(f"value table is missing coalition {set(s)!r}")
    return float(table[s])


def shapley_values(table: ValueTable, players: list[str]) -> dict[str, float]:
    """Exact Shapley value per player. Requires ``table`` to cover the full powerset of ``players``.

    ``φ_i = Σ_{S⊆N∖{i}} |S|!·(n−|S|−1)! / n! · (v(S∪{i}) − v(S))``. The values satisfy the efficiency
    identity ``Σ_i φ_i == v(N) − v(∅)`` (asserted by callers/tests, not here)."""
    n = len(players)
    phi: dict[str, float] = {}
    for i in players:
        rest = [p for p in players if p != i]
        total = 0.0
        for r in range(len(rest) + 1):
            weight = factorial(r) * factorial(n - r - 1) / factorial(n)
            for combo in combinations(rest, r):
                s = frozenset(combo)
                total += weight * (_v(table, s | {i}) - _v(table, s))
        phi[i] = total
    return phi


def shapley_interaction(table: ValueTable, i: str, j: str, players: list[str]) -> float:
    """Exact pairwise Shapley interaction index ``I_{ij}`` (Grabisch & Roubens).

    ``I_{ij} = Σ_{S⊆N∖{i,j}} |S|!·(n−|S|−2)! / (n−1)! · δ_{ij}(S)`` with
    ``δ_{ij}(S) = v(S∪{i,j}) − v(S∪{i}) − v(S∪{j}) + v(S)``. Positive ⇒ synergy/complementarity,
    negative ⇒ redundancy. Undefined for ``i == j``."""
    if i == j:
        raise ValueError("interaction index is defined for a pair of distinct modalities")
    n = len(players)
    rest = [p for p in players if p not in (i, j)]
    total = 0.0
    for r in range(len(rest) + 1):
        weight = factorial(r) * factorial(n - r - 2) / factorial(n - 1)
        for combo in combinations(rest, r):
            s = frozenset(combo)
            delta = (_v(table, s | {i, j}) - _v(table, s | {i})
                     - _v(table, s | {j}) + _v(table, s))
            total += weight * delta
    return total


def interaction_matrix(table: ValueTable, players: list[str]) -> dict[tuple[str, str], float]:
    """All unordered pairwise interaction indices, keyed by a sorted ``(i, j)`` tuple.

    A negative entry flags a **redundant** modality pair (safe to drop one), a positive entry a
    **synergistic** pair (both needed) — the per-pair redundancy/synergy read for the atlas."""
    out: dict[tuple[str, str], float] = {}
    for a, b in combinations(players, 2):
        out[(a, b)] = shapley_interaction(table, a, b, players)
    return out
