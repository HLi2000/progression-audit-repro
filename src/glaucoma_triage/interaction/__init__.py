"""Modality-level interaction/redundancy decomposition (P1 progression-interaction plan).

Exact coalition-game Shapley analysis over a small set of *modalities* (not features). The
characteristic function ``v(S)`` is the cross-validated predictive performance of a model trained on
modality subset ``S`` (see :mod:`glaucoma_triage.models.fusion`); this module is the dependency-free
game-theory layer that turns a ``{frozenset: value}`` table into:

* Shapley **values** — each modality's marginal contribution (the standard, already-common metric);
* Shapley **interaction indices** — pairwise synergy (``> 0``) vs redundancy (``< 0``), the headline
  contribution (InterSHAP convention; not previously reported for glaucoma progression).

Kept free of pandas/sklearn so the coalition math is unit-testable against synthetic value tables
(a known-redundant pair must score negative; a known-synergistic XOR pair positive).
"""

from __future__ import annotations

from .shapley import (
    ValueTable,
    coalitions,
    interaction_matrix,
    shapley_interaction,
    shapley_values,
)

__all__ = [
    "ValueTable",
    "coalitions",
    "interaction_matrix",
    "shapley_interaction",
    "shapley_values",
]
