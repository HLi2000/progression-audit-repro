"""Light CPU ML layer for the P1 progression-interaction plan.

Additive to (never a replacement for) the repo's rule-based arms. Provides the modality-subset
estimator + patient-grouped cross-validated value function that feeds the coalition-Shapley
interaction decomposition in :mod:`glaucoma_triage.interaction`.
"""

from __future__ import annotations

from .fusion import ModalitySpec, coalition_values, subset_auc

__all__ = ["ModalitySpec", "coalition_values", "subset_auc"]
