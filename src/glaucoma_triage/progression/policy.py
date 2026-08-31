"""Per-eye order-or-skip acquisition policy for a candidate test, given an already-acquired baseline.

The clinical question (P1's acquisition-policy goal, non-circular framing): a cheap baseline is already
in hand at every glaucoma visit — **fundus-disc photo + IOP + CCT + age**. A *candidate* test (a visual
field, or an OCT) is costly / burdensome. For which eyes does ordering it add enough *decision* value to
justify it — and for which can we act on the baseline alone?

The deployable rule is the **value-of-information "uncertainty band"** (Vickers-style): order the test
only when the baseline model's predicted risk sits within a band ``|p_base − p_t| < margin`` around the
decision threshold ``p_t`` — the eyes where the current decision is genuinely in doubt. Eyes far from the
threshold are decided on the baseline alone (ordering the test cannot flip the decision). Crucially the
band is computed from **baseline information only**, so the policy is deployable before the test is
ordered. The policy's fused probability then uses the richer baseline+test model *only for ordered eyes*.

Evaluated against **always-order** (everyone gets the test) and **baseline-only** (no one does) via the
decision-curve / EVPI machinery in :mod:`glaucoma_triage.models.decision`: a good policy retains most of
always-order's net benefit while ordering far fewer tests (a lower ``order_rate``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PolicyResult:
    """Outcome of the uncertainty-band order/skip policy at one ``(threshold, margin)``."""

    margin: float
    threshold: float
    order_mask: np.ndarray      # bool per eye — True = order the candidate test
    order_rate: float           # fraction of eyes for which the test is ordered
    fused_proba: np.ndarray     # p_base+test where ordered, else p_base


def uncertainty_band_policy(
    p_base: np.ndarray, p_full: np.ndarray, threshold: float, margin: float,
) -> PolicyResult:
    """Order the candidate test iff the baseline risk is within ``margin`` of ``threshold``.

    ``p_base`` = baseline-only predicted risk (available at decision time); ``p_full`` = the
    baseline+candidate-test risk (used only where the policy chooses to order). Returns the order mask,
    the order rate, and the fused per-eye probability the policy would act on."""
    p_base = np.asarray(p_base, dtype=float)
    p_full = np.asarray(p_full, dtype=float)
    order = np.abs(p_base - threshold) < margin
    fused = np.where(order, p_full, p_base)
    return PolicyResult(float(margin), float(threshold), order,
                        float(order.mean()) if len(order) else 0.0, fused)


def order_rate_tradeoff(
    p_base: np.ndarray, p_full: np.ndarray, threshold: float, margins: np.ndarray,
) -> list[PolicyResult]:
    """The policy across a sweep of band widths — the order-rate vs decision-value trade-off curve."""
    return [uncertainty_band_policy(p_base, p_full, threshold, m) for m in margins]
